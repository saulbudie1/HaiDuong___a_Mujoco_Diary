# Declaration: what I use and where it comes from

Everything this project stands on, so it is clear what is mine and what is not. Versions are the ones installed on my machine on 2026-10-23.

---

## 1. What this project is, and what is mine

- **Mine:** the questions, the test design, the run settings, the rollouts and plots in `Tests/`, `Diary.md`, `TestingAndFinding.md`, `RunGuide.md`, and the helper scripts in `Scripts/` (`rollout.py`, `plot_reward.py`, `plot_motion.py`, `plot_trail.py`).
- **Not mine:** the simulator, the robot model, the RL algorithm and the training code. I run them and test them. I did not write them.
- **Starting point:** `Doc/Notebook/haiduong_locomotion.py` is the Locomotion notebook from MuJoCo Playground (Apache 2.0) exported from Colab. `Scripts/train_test.py` is the same training recipe moved out of the notebook to run locally. The notebook needed a few fixes before it would train (Warp, Flax, a `device_put_replicated` shim). See `Diary.md`, Oct 20.
- All results are "in my runs". Most are one seed, and Test 9 uses five.

---

## 2. Libraries and software

### Simulation and training (inside WSL2, env `go1env`, Python 3.12.3)

| Library | Version | Used for | Source |
|---|---|---|---|
| MuJoCo | 3.16.0 | Physics engine | https://github.com/google-deepmind/mujoco |
| MuJoCo MJX | 3.16.0 | MuJoCo on the GPU through JAX | same repository |
| MuJoCo Playground (`playground`) | 0.2.0 | The Go1 tasks (`Go1JoystickFlatTerrain`, `Go1JoystickRoughTerrain`), reward, command sampling, registry | https://github.com/google-deepmind/mujoco_playground |
| Brax | 0.14.2 | PPO training loop, networks, checkpoint saving | https://github.com/google/brax |
| JAX / jaxlib | 0.12.0 | Array maths and compiling for the GPU | https://github.com/jax-ml/jax |
| JAX CUDA 12 plugin | 0.12.0 | Lets JAX use the NVIDIA GPU | same repository |
| Warp (`warp-lang`) | 1.18.0 | GPU backend the Go1 task needs | https://github.com/NVIDIA/warp |
| Flax | 0.12.10 | Neural network layers used by Brax | https://github.com/google/flax |
| Orbax checkpoint | 0.12.7 | Saving and loading checkpoints | https://github.com/google/orbax |
| ml_collections | 1.1.0 | Config objects (`ppo_params`) | https://github.com/google/ml_collections |
| etils | 1.14.0 | Small helpers imported by the notebook | https://github.com/google/etils |
| NumPy | 2.5.4 | Arrays, saving motion data | https://numpy.org |
| mediapy | 1.2.7 | Writing the rollout videos | https://github.com/google/mediapy |

### Plots and data

| Library | Version | Used for |
|---|---|---|
| Matplotlib | 3.11.2 (WSL and Windows) | Every figure: reward curves (`plot_reward.py`), distance / velocity / rotation graphs (`plot_motion.py`), bird's-eye trails (`plot_trail.py`) |
| pandas | 2.3.3 (Windows Python only) | Installed for tables and seed comparisons. **No script in this repo imports it yet**, and it is not in `go1env`. |

### Tools and system

| Thing | Details |
|---|---|
| Operating system | Windows 11 Home, with WSL2 running Ubuntu 24.04.5 LTS for all training since Test 5 |
| GPU | NVIDIA GeForce RTX 5060 Laptop (8 GB), driver 591.84 |
| Google Colab | Free tier, T4 GPU, Tests 1 to 4 (it ran out of compute) |
| tmux | Keeps a long run alive after the window closes |
| ffmpeg 6.1.1 | Cuts the preview GIFs shown in the README |
| Git and GitHub | Version control, repo at https://github.com/saulbudie1/HaiDuong |
| VS Code | Editing |

---

## 3. Robot model

- **Unitree Go1** model (`go1_mjx_feetonly.xml` and its scenes) comes through MuJoCo Playground, derived from the MuJoCo Menagerie `unitree_go1` model. Menagerie derived it from Unitree's public URDF. Licence: BSD-3-Clause (Unitree Robotics, 2016 to 2022).
  - MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie
  - Unitree URDF: https://github.com/unitreerobotics/unitree_ros
- `Doc/go1_mjx_feetonly.xml` is the Playground Go1 file as used in my runs.
- Rough terrain uses the heightfield image shipped with Playground (`assets/hfield.png`).

---

## 4. Use of AI

- **Claude (Anthropic)**, used through chat and Claude Code in VS Code.
- What it did: explained terms (RL, PPO, what a seed is), walked me through getting the RTX 5060 working with JAX and Warp in WSL2, helped debug errors (Colab setup, GPU out of memory, EGL rendering), reviewed my plans and results, and helped draft and tidy these write-ups and helper scripts.
- What it did not do: choose what the results mean without me checking the logs. The numbers come from my run logs and rollouts. When a number is not in a log, I write that it is unknown.
- I read the Playground source and the paper myself to check claims such as the reward terms and the command ranges.

---

## 5. References

**Code and models**
1. K. Zakka et al., *MuJoCo Playground*, arXiv:2502.08844, 2025. https://github.com/google-deepmind/mujoco_playground
2. E. Todorov, T. Erez, Y. Tassa, *MuJoCo: A physics engine for model-based control*, IROS 2012.
3. C. D. Freeman et al., *Brax: A Differentiable Physics Engine for Large Scale Rigid Body Simulation*, arXiv:2106.13281, 2021.
4. K. Zakka, Y. Tassa and MuJoCo Menagerie Contributors, *MuJoCo Menagerie: A collection of high-quality simulation models for MuJoCo*, 2022. https://github.com/google-deepmind/mujoco_menagerie
5. J. Bradbury et al., *JAX: composable transformations of Python+NumPy programs*, 2018. https://github.com/jax-ml/jax

**Method and papers**
6. J. Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347, 2017. (The algorithm Brax's PPO implements.)
7. J. Y. Luo, Y. Song, V. Klemm, F. Shi, D. Scaramuzza, M. Hutter, *Residual Policy Learning for Perceptive Quadruped Control Using Differentiable Simulation*, arXiv:2410.03076, 2024. (Read for the vision and terrain ideas, see `Diary.md`.)
8. J. D. Hunter, *Matplotlib: A 2D graphics environment*, Computing in Science & Engineering 9(3), 2007.

**Licences:** MuJoCo Playground and the notebook are Apache 2.0, the Go1 model is BSD-3-Clause. This repo keeps trained weights out of git (see `.gitignore`).
