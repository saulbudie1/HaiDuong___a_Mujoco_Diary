"""Render top, side and track videos from the joint angles saved by rollout_terrain.py (no GPU needed for the physics).

    python render_qpos.py motion_x1.0_y0.0_yaw0.0_<tag>_go1_final.npz <ground> [out_dir] [--cone] [--cams top,side,track]

<ground> is flat, rough or the path of a scene XML (the same values rollout_terrain.py takes). The model is built on the
CPU, each saved step is put into it, and MuJoCo draws it with the same camera names and view options as rollout.py
(640 x 480, every second step, contact points shown). Contact points are recomputed by MuJoCo on the CPU, so they can
differ slightly from the ones the GPU simulation used.
"""
import os

os.environ["MUJOCO_GL"] = "egl"       # must be set before mujoco is imported
os.environ["JAX_PLATFORMS"] = "cpu"   # keep JAX (imported by Playground) off the GPU so the renderer has the memory

import sys
import time

import mediapy as media
import mujoco
import numpy as np
from etils import epath
from mujoco_playground._src.locomotion.go1 import base
from mujoco_playground._src.locomotion.go1 import go1_constants as consts

argv = sys.argv[1:]
cone = "--cone" in argv   # draw the robot with the visual-only cone hat (use on a motion file made with --cone)
if cone:
    argv.remove("--cone")
cams = ["top", "side", "track"]
if "--cams" in argv:      # --cams top,side: render only these cameras (so several can run in parallel)
    i = argv.index("--cams")
    cams = argv[i + 1].split(",")
    del argv[i:i + 2]
npz, ground = argv[0], argv[1]
out_dir = argv[2] if len(argv) > 2 else "."
os.makedirs(out_dir, exist_ok=True)

xml = {"flat": consts.FEET_ONLY_FLAT_TERRAIN_XML, "rough": consts.FEET_ONLY_ROUGH_TERRAIN_XML}.get(ground) or epath.Path(ground)
assets = base.get_assets()
if cone:
    decos = epath.Path(__file__).resolve().parent.parent / "Assets" / "Decos"
    assets["go1_mjx_feetonly.xml"] = (decos / "go1_mjx_feetonly_cone.xml").read_bytes()
    assets["cone_tophat.STL"] = (decos / "cone_tophat.STL").read_bytes()
model = mujoco.MjModel.from_xml_string(epath.Path(xml).read_text(), assets=assets)
data = mujoco.MjData(model)

d = np.load(npz, allow_pickle=True)
qpos, dt = d["qpos"], float(d["dt"])
name = os.path.basename(npz).replace("motion_", "").replace(".npz", "")  # already ends in _cone for a --cone motion file

scene_option = mujoco.MjvOption()
scene_option.geomgroup[2] = True
scene_option.geomgroup[3] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False

render_every = 2
fps = 1.0 / dt / render_every
renderer = mujoco.Renderer(model, height=480, width=640)
for cam in cams:
    t0 = time.time()
    frames = []
    for i in range(0, len(qpos), render_every):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=cam, scene_option=scene_option)
        frames.append(renderer.render())
    media.write_video(f"{out_dir}/{cam}_{name}.mp4", frames, fps=fps)
    print(f"{cam}: {len(frames)} frames at {fps:.0f} fps in {time.time() - t0:.0f} s -> {out_dir}/{cam}_{name}.mp4", flush=True)
print("done")
