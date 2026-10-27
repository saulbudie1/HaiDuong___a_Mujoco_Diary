"""Roll out every checkpoint of one run in a single process (the policy is compiled once).

Same episode as rollout.py (1000 steps, PRNGKey(0), no video), so the numbers match
a single rollout. Writes one CSV row per checkpoint as it goes.

    python sweep.py 1 0 0                          # all ckpt_* (and go1_final) in the current folder
    python sweep.py 1 0 0 --folder ~/go1_seed2 --min-step 45e6 --max-step 70e6
    python sweep.py 1 0 0 --every 2                # every second checkpoint
"""
import argparse
import re
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import registry
from mujoco_playground.config import locomotion_params

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument("command", nargs=3, type=float, metavar=("X", "Y", "YAW"))
ap.add_argument("--folder", default=".", help="run folder with the ckpt_* files (default: current)")
ap.add_argument("--min-step", type=float, default=0)
ap.add_argument("--max-step", type=float, default=float("inf"))
ap.add_argument("--every", type=int, default=1, help="use every Nth checkpoint (default 1)")
ap.add_argument("--no-final", action="store_true", help="skip go1_final")
args = ap.parse_args()
x_vel, y_vel, yaw_vel = args.command

folder = Path(args.folder).expanduser()
ckpts = sorted((int(m[1]), f) for f in folder.iterdir() if (m := re.fullmatch(r"ckpt_(\d+)", f.name)))
ckpts = [(s, f) for s, f in ckpts if args.min_step <= s <= args.max_step][:: args.every]
if (folder / "go1_final").exists() and not args.no_final:
    ckpts.append((-1, folder / "go1_final"))  # step -1 marks the final policy
if not ckpts:
    sys.exit(f"no checkpoints found in {folder}")

env_name = "Go1JoystickFlatTerrain"
env = registry.load(env_name)
env_cfg = registry.get_default_config(env_name)
ppo_params = locomotion_params.brax_ppo_config(env_name)

network = ppo_networks.make_ppo_networks(
    env.observation_size, env.action_size,
    preprocess_observations_fn=running_statistics.normalize,
    **ppo_params.network_factory,
)
make_inference_fn = ppo_networks.make_inference_fn(network)


# params are an argument (not a closure), so this compiles once for all checkpoints
@jax.jit
def policy(params, obs, rng):
    return make_inference_fn(params, deterministic=True)(obs, rng)


jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
command = jp.array([x_vel, y_vel, yaw_vel])


@jax.jit
def measure(data):
    v = env.get_local_linvel(data)
    return env._reward_tracking_lin_vel(command, v), v, data.qpos[0:3], data.qpos[3:7]


def rollout(params):
    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    state.info["command"] = command
    track, linvel, pos, quat = [], [], [], []
    for _ in range(env_cfg.episode_length):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = policy(params, state.obs, act_rng)
        state = jit_step(state, ctrl)
        state.info["command"] = command
        t, v, p, q = measure(state.data)
        track.append(t), linvel.append(v), pos.append(p), quat.append(q)
    track = np.asarray(jp.stack(track))
    linvel = np.asarray(jp.stack(linvel))
    pos = np.asarray(jp.stack(pos))
    quat = np.asarray(jp.stack(quat))
    up_last = 1 - 2 * (quat[-1, 1] ** 2 + quat[-1, 2] ** 2)  # z component of the body up axis
    return (float(track.mean()), float(linvel[:, 0].mean()), float(np.hypot(*(pos[-1, :2] - pos[0, :2]))),
            float(pos[:, 2].min()), float(pos[:, 2].mean()), float(up_last))


base = f"x{x_vel}_y{y_vel}_yaw{yaw_vel}"
out = folder / f"sweep_{base}.csv"
header = "step,name,track,vx,distance,zmin,zmean,upright_last,seconds"
print(f"command: {x_vel} {y_vel} {yaw_vel}   {len(ckpts)} checkpoints in {folder}   -> {out}", flush=True)
with open(out, "w") as fh:
    fh.write(header + "\n")
    for step, path in ckpts:
        t0 = time.time()
        try:
            res = rollout(model.load_params(str(path)))
        except Exception as e:  # one bad checkpoint should not stop the sweep
            print(f"{path.name}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        secs = time.time() - t0
        fh.write(f"{step},{path.name}," + ",".join(f"{r:.5f}" for r in res) + f",{secs:.1f}\n")
        fh.flush()
        print(f"{path.name:>16}  track {res[0]:.3f}  vx {res[1]:+.3f}  dist {res[2]:6.2f} m  "
              f"z {res[3]:.2f}-{res[4]:.2f}  up {res[5]:.2f}  ({secs:.0f} s)", flush=True)
print("done")
