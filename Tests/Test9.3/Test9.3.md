# Test Log 9.3 — Test 7 policy on the new sine ground (quick check, local RTX 5060)

**Date:** 2026-10-27
**Result:** Pass. The Test 7 policy, trained on flat ground only, walks on the new sine-wave ground: `track` 0.9205 against 0.9404 on flat ground, upright the whole time, no fall, 18.6 m covered in 20 s. The ground file loads and runs in Playground with the Warp backend. This says nothing yet about training on it.

**Purpose:** [Assets/Grounds/SimpleUniFormRoughGround_1.xml](../../Assets/Grounds/SimpleUniFormRoughGround_1.xml) (a 100 x 100 m sine-wave heightfield, written by Claude, see [Declaration.md](../../Declaration.md)) had only been compiled and rendered in MuJoCo on Windows. Before training on it, check with a policy known to walk that it loads in Playground, that the robot can stand and walk on it, and that nothing breaks. The Test 7 run was picked because it walks reliably ([Test 7](../Test7/Test7.md), `track` about 0.94 on 1 / 0 / 0).

## Setup

| Setting | Value |
|---|---|
| Policy | `go1_final` of `~/go1_test7` (the Test 7 run: flat ground, seed 0, 122.9M steps, `num_evals` 16) |
| Command | 1 / 0 / 0 (forward 1 m/s) |
| Episode | 1000 steps (20 s), no video, same start and random numbers as `rollout.py` (`PRNGKey(0)`) |
| Script | [Scripts/rollout_terrain.py](../../Scripts/rollout_terrain.py) (written by Claude), one run per ground |
| Grounds | stock flat; stock rough (the 20 m Playground patch); the new sine ground |
| How the new ground is loaded | the script points Playground's task name `rough_custom` at the XML (`go1_constants.task_to_xml`). The name starts with `rough`, so it gets the same larger contact buffers as the stock rough task. Nothing in the installed package is edited. |

The installed package (MuJoCo Playground 0.2.0) has the same structure as the GitHub main branch I read first: `task_to_xml` in `go1_constants.py`, `get_assets()` in `base.py` and the `rough` check in `joystick.py`.

## Rollout numbers

| Ground | `track` | Forward speed | Distance in 20 s | Body height (min / mean / max) | Most tilted (1 = level) | Fell |
|---|---|---|---|---|---|---|
| **New sine ground (100 m)** | **0.9205** | 0.929 m/s | 18.58 m | 0.323 / 0.350 / 0.381 m | 0.991 | no |
| Stock flat | 0.9404 | 0.962 m/s | 19.20 m | 0.288 / 0.315 / 0.328 m | 0.998 | no |
| Stock rough (20 m patch) | 0.5431 over the full 1000 steps, **0.8874 over the 612 steps before it fell** | not meaningful after the fall | 17.70 m | min -325 m | -0.868 | yes, at step 612 (12.2 s) |

Raw output: [rollout_terrain_results.txt](./rollout_terrain_results.txt).

- **The flat row reproduces Test 7.** `track` 0.9404 here against about 0.94 in Test 7, so the new script gives the same numbers as `rollout.py`.
- **The stock rough "fall" is the edge of its patch.** All three robots start at about (-0.5, -0.5) and walk to about (-16, -10) (from the saved motion data). The stock rough patch is 20 m wide (half-extent 10 m), and the robot crosses 10 m at step 589 (11.8 s) and is terminated at step 612, with a body height of -325 m. It walked off the edge, it did not stumble. Its `track` over the first 612 steps is 0.887, slightly below flat ground. The 0.5431 for the whole episode includes the fall, so it is not a fair comparison.
- **The new ground has no edge in reach.** The robot crosses 10 m at about 12.3 s and keeps walking on ground that extends to 50 m. The terrain was never left.
- **The new ground reads a little lower and bobs more.** `track` is 0.02 below flat, and the body height varies over 0.058 m against 0.040 m on flat. That fits the ground's own 0.075 m range. The mean height is about 0.035 m higher, matching the ground's average height.
- **Time cost of the huge heightfield looks small in this test.** The whole run (JIT and a 20 s episode) took 64 s on the new ground, 60 s on flat and 64 s on the stock rough patch. This is one environment, a single rollout, and it includes compile time, so it says nothing about the cost of training on it.

## Videos and plots (same policy, same command, rolled out again on the new ground)

A second rollout on the new ground, this time saving every joint angle, gave `track` **0.9199**, forward speed 0.927 m/s, 18.53 m in 20 s, upright (minimum 0.988), no fall. That matches the first run (0.9205) to GPU rounding. Videos, plots and the motion data are in [x1.0_y0.0_yaw0.0/](./x1.0_y0.0_yaw0.0/).

- **Videos (640 x 480, 20 s):** [side](./x1.0_y0.0_yaw0.0/side_x1.0_y0.0_yaw0.0_SimpleUniFormRoughGround_1_go1_final.mp4), [top](./x1.0_y0.0_yaw0.0/top_x1.0_y0.0_yaw0.0_SimpleUniFormRoughGround_1_go1_final.mp4), [track](./x1.0_y0.0_yaw0.0/track_x1.0_y0.0_yaw0.0_SimpleUniFormRoughGround_1_go1_final.mp4). Preview GIFs (first 6 s, top and track) are in [preview/](./preview/).
- **Motion graph:** [distance, velocity and rotation](./x1.0_y0.0_yaw0.0/motion_x1.0_y0.0_yaw0.0_SimpleUniFormRoughGround_1_go1_final.png). Forward speed sits around 1 m/s with a steady sideways wobble of about ±0.2 m/s from the gait. The body turns slowly, about -8 degrees in 20 s, the same kind of drift as in the earlier flat runs.
- **Trail map:** [bird's-eye view](./x1.0_y0.0_yaw0.0/trail_x1.0_y0.0_yaw0.0_SimpleUniFormRoughGround_1_go1_final.png), a nearly straight line of about 18.7 m of path for 18.5 m from the start.
- **What the videos show:** I looked at one frame from each camera. The robot is upright and stepping, with the orange foot markers on the surface, over the purple wavy ground. The colour looks darker than `#9c7590` under Playground's stock lighting. I did not watch the whole of every video.
- **How they were made:** `rollout_terrain.py` now also saves every joint angle per step, and [Scripts/render_qpos.py](../../Scripts/render_qpos.py) (written by Claude) draws them on the CPU with the same cameras and options as `rollout.py`. The contact points in the videos are recomputed by MuJoCo on the CPU, so they can differ slightly from the GPU simulation's.
- **Rendering this ground is slow:** about 17 minutes per camera (500 frames, about 2 s per frame), 53 minutes for the three videos. The 4-million-cell heightfield is the likely reason, but I did not test that. Plan for it before rendering more commands or policies on it.

## What this shows and what it does not

**Supported (in my runs):**
- The 2001 x 2001 inline heightfield compiles, loads and simulates in Playground with the Warp backend (the heightfield collision kernel loaded).
- A policy trained only on flat ground can walk straight over this ground (median slope about 12.5 degrees, steepest about 20 degrees by the generator's numbers), with `track` close to its flat value.
- The robot stays upright (never below 0.991).

**Not shown:**
- **One path.** The robot walked a roughly 19 m line through a 100 m patch, so it touched a small part of the terrain, and the slopes along that line are not measured separately.
- **One policy, one command, one rollout.** No other commands and no other policies were tried.
- **No training.** Nothing here says a policy can be trained on this ground, how long it would take or how much GPU memory it needs. Test 8 already ran out of memory on the stock rough terrain.
- **The videos were only spot-checked** (one frame per camera), so I cannot say much about the gait beyond "upright and stepping".
- The XLA "Failed to allocate device memory" lines at start-up appeared in all three runs, flat included, and the runs finished normally. They are the same start-up warnings as in earlier tests and are not specific to this ground.

## Next
1. ~~Roll out the same policy on the new ground with a video.~~ Done, see above.
2. Try other commands (1 / -0.5 / 0, 1 / 0 / 1) and the other Test 9 and 9.2 walking policies.
3. Short training run on the new ground (fewer environments, for example 2048; keep the reset interval the same as the walking runs, 80 steps), and compare with the flat run.
4. Decide whether the gentler amplitude is needed before training.

## Evidence
- Results: [rollout_terrain_results.txt](./rollout_terrain_results.txt)
- Ground: [SimpleUniFormRoughGround_1.xml](../../Assets/Grounds/SimpleUniFormRoughGround_1.xml) and its previews in `Assets/Grounds/`
- Scripts: [rollout_terrain.py](../../Scripts/rollout_terrain.py), [render_qpos.py](../../Scripts/render_qpos.py), plus the existing [plot_motion.py](../../Scripts/plot_motion.py) and [plot_trail.py](../../Scripts/plot_trail.py) (trail with `--head-length 2`)
- Videos, plots and motion data: [x1.0_y0.0_yaw0.0/](./x1.0_y0.0_yaw0.0/); preview GIFs: [preview/](./preview/)
- The first (three-ground) run's motion data (`terrain_*.npz`) and full console log: WSL `~/go1_test7`, not copied here
