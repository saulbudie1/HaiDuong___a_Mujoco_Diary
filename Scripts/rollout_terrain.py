"""Roll out one policy for one 20 s episode on a chosen ground (no video) and print how it did.

    python rollout_terrain.py flat  1 0 0 [checkpoint]      # stock flat terrain
    python rollout_terrain.py rough 1 0 0 [checkpoint]      # stock rough terrain
    python rollout_terrain.py /path/to/ground_scene.xml 1 0 0 [checkpoint]   # any scene XML (a custom ground)
    add --cone to put the visual-only cone hat (Assets/Decos/go1_mjx_feetonly_cone.xml) on the robot; output names then end in _cone

The custom scene is loaded by pointing Playground's task name "rough_custom" at the XML (name starts with "rough", so it
gets the larger contact buffers the stock rough task gets). The policy network is built from the flat task's PPO settings,
the same as rollout.py. Same episode as rollout.py (1000 steps, PRNGKey(0)).
"""
import sys
import time

import jax
import jax.numpy as jp
import numpy as np
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from etils import epath
from mujoco_playground import registry
from mujoco_playground._src.locomotion.go1 import go1_constants as consts
from mujoco_playground._src.locomotion.go1 import joystick
from mujoco_playground.config import locomotion_params

argv = sys.argv[1:]
out_dir = "."
if "--out" in argv:  # --out DIR: where the motion data is saved (default: the current folder)
    i = argv.index("--out")
    out_dir = argv[i + 1]
    del argv[i:i + 2]
cone = "--cone" in argv
if cone:
    argv.remove("--cone")
args = argv
ground = args[0]
x_vel, y_vel, yaw_vel = (float(a) for a in args[1:4])
ckpt = args[4] if len(args) > 4 else "go1_final"

if cone:  # swap the robot file in Playground's asset dictionary for the one with the hat (+ the cone mesh)
    from pathlib import Path
    from mujoco_playground._src.locomotion.go1 import base
    _decos = Path(__file__).resolve().parent.parent / "Assets" / "Decos"
    _get_assets = base.get_assets

    def _assets_with_cone():
        assets = _get_assets()
        assets["go1_mjx_feetonly.xml"] = (_decos / "go1_mjx_feetonly_cone.xml").read_bytes()
        assets["cone_tophat.STL"] = (_decos / "cone_tophat.STL").read_bytes()
        return assets

    base.get_assets = _assets_with_cone

if ground == "flat":
    env = registry.load("Go1JoystickFlatTerrain")
elif ground == "rough":
    env = registry.load("Go1JoystickRoughTerrain")
else:
    xml = epath.Path(ground)
    _stock = consts.task_to_xml
    consts.task_to_xml = lambda t: xml if t == "rough_custom" else _stock(t)
    env = joystick.Joystick(task="rough_custom", config=joystick.default_config())
env_cfg = registry.get_default_config("Go1JoystickFlatTerrain")
ppo_params = locomotion_params.brax_ppo_config("Go1JoystickFlatTerrain")

network = ppo_networks.make_ppo_networks(
    env.observation_size, env.action_size,
    preprocess_observations_fn=running_statistics.normalize,
    **ppo_params.network_factory,
)
make_inference_fn = ppo_networks.make_inference_fn(network)
params = model.load_params(ckpt)
inference_fn = jax.jit(make_inference_fn(params, deterministic=True))
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

command = jp.array([x_vel, y_vel, yaw_vel])


@jax.jit
def measure(data):
    v = env.get_local_linvel(data)
    return (env._reward_tracking_lin_vel(command, v), v, data.qpos[0:3], data.qpos[3:7],
            env.get_gyro(data), env.get_accelerometer(data), data.qpos)


t0 = time.time()
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
state.info["command"] = command
track, linvel, pos, quat, done, gyro, accel, qpos = [], [], [], [], [], [], [], []
for _ in range(env_cfg.episode_length):
    act_rng, rng = jax.random.split(rng)
    ctrl, _ = inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    state.info["command"] = command
    t, v, p, q, g, a, qp = measure(state.data)
    track.append(t), linvel.append(v), pos.append(p), quat.append(q), done.append(state.done)
    gyro.append(g), accel.append(a), qpos.append(qp)
track = np.asarray(jp.stack(track))
linvel = np.asarray(jp.stack(linvel))
pos = np.asarray(jp.stack(pos))
quat = np.asarray(jp.stack(quat))
gyro = np.asarray(jp.stack(gyro))
accel = np.asarray(jp.stack(accel))
qpos = np.asarray(jp.stack(qpos))  # every joint angle per step, for render_qpos.py
done = np.asarray(jp.stack(done)) > 0.5
up = 1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)  # z component of the body up axis; 1 = upright, 0 = on its side

first_done = int(np.argmax(done)) if done.any() else -1
dt = float(env.dt)
print(f"ground: {ground}   command: {x_vel} {y_vel} {yaw_vel}   checkpoint: {ckpt}   ({time.time() - t0:.0f} s)")
print(f"mean track: {track.mean():.4f}")
print(f"mean linvel (body frame): {linvel.mean(0)}")
print(f"distance from start: {np.hypot(*(pos[-1, :2] - pos[0, :2])):.2f} m in {len(pos) * dt:.0f} s")
print(f"body height: min {pos[:, 2].min():.3f}  mean {pos[:, 2].mean():.3f}  max {pos[:, 2].max():.3f} m")
print(f"upright (1 = level): min {up.min():.3f}  at the end {up[-1]:.3f}")
print("fell (episode terminated):", "no" if first_done < 0 else f"yes, at step {first_done} ({first_done * dt:.1f} s)")
import os
os.makedirs(out_dir, exist_ok=True)
tag = ground if ground in ("flat", "rough") else os.path.splitext(os.path.basename(ground))[0]
base = f"x{x_vel}_y{y_vel}_yaw{yaw_vel}_{tag}_{ckpt}" + ("_cone" if cone else "")
# same keys as rollout.py's motion file (plot_motion.py and plot_trail.py read it), plus done and qpos
np.savez(f"{out_dir}/motion_{base}.npz", dt=dt, command=np.array([x_vel, y_vel, yaw_vel]), ckpt=ckpt,
         pos=pos, quat=quat, linvel=linvel, gyro=gyro, accel=accel, track=track, done=done, qpos=qpos)
print(f"saved {out_dir}/motion_{base}.npz")
