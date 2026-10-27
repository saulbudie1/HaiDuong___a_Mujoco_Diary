# Diary

The short version of [`TestingAndFinding.md`](TestingAndFinding.md). Plain words, only the numbers that matter.

---

<details>
<summary><h2>Oct 20 — I first used Colab for simulation.</h2></summary>

**Background**I haven't used colab in ages (not since that deepfake video where you can input a photo to make it sing). So I spent the afternoon refamiliarizing with the website.

**Test 1 (6M steps).** The robot took a few steps for about 8 seconds, then fell over. It was about 6million steps that is too little training.

**Test 2 (18M).** Reran only the cells I changed. The robot stayed up but stood still, and the reward looked low and strange. Probably leftover settings from my last test (a guess, not proven).

**Test 3 (18M, clean restart).** Reward jumped to 17 to 19. It looked promisng but the render result disappointed me. the robot stood still when I told it to go forward(x_vel = 1.0), and only spun on the spot when I told it to turn(yaw_vel=1.0). 

**Test 4 (25M).** Same standing robot, and the reward dropped late in training. I decided "more steps makes it worse". That was wrong.

**Late night.** I read the notebook code and found the plot cuts off the data. My "18M" and "25M" runs both really ran about 29.5M steps. Then Colab ran out of compute while I was trying 100M.

</details>

---

<details>
<summary><h2>Oct 21 — Laptop integration and the first walking not talking</h2></summary>


**Switch to Ubuntu.** I installed Ubuntu inside Windows (WSL2), Claude guided my way to get my RTX 5060 working with JAX and Warp, and moved training off Colab. The laptop ran about twice as fast as Colab's free GPU.

**Test 5 (100M steps).** About 19 minutes to train. Reward went to 26, and this time the robot **walked**: I asked for 1 m/s at direction x and it moved at about 0.96 m/s(which is close enough). So concludes the first real walking motion.

**Test 6 (control).** I needed to know the walking came from the extra steps, not from migrating to Ubuntu and the laptop. So I redid the Colab-length run (18M in the file and about 29.5M steps in the terminal) on the laptop. It stood still, just like Tests 3 and 4. So the platform wasn't the cause.

**Test 7 (120M, diagonals).** A new training run. I asked for 120M and it ran 122.9M steps (30 minutes, 16 checkpoints), and the reward ended at 26.9. It sat near 19 from 25M to 57M steps (the standing level), then climbed. I rolled the final policy out on four commands: straight (1 / 0 / 0) and three diagonals (sideways speed 0.5, -0.5 and -1). All four followed the command (`track` 0.93 to 0.95). Even sideways -1, which is outside the range it trained on (up to 0.8).

**The laptop died.** I left it running the rollouts and the battery ran out. The training had already finished and saved, so I only redid the rollouts, not the 30 minutes. Plug in before leaving.

**Test 7.5 (the odd one out).** In Test 7 the new policy walked sideways -1, but the Test 5 policy had stood still on that command. So I tried the Test 5 policy on -0.5: it walked (`track` 0.935). It can walk sideways, so the failure at -1 most likely comes from that command being out of range. Not proven, because the two runs also differ in seed, steps and evaluations.

**It still turns a little.** With the turn command at 0 the robot slowly rotated, -36 degrees over 20 seconds on the -1 diagonal. The turn command is a rate, not a heading, the reward barely notices a tiny rate, and the policy has no heading input. So it is expected, not broken. A fix would be a controller outside the policy, not retraining.

**Tools.** I put the scripts in `Scripts/`: train, rollout (it now saves the motion data), reward plot, motion graphs (distance, velocity against the command, rotation) and a bird's-eye trail map. I also looked for a chase mode. MuJoCo Playground has none, and the gait choices (bound, pace, pronk) exist for the Spot robot, not the Go1.

**Test 8 (rough terrain).** The first different ground: a bumpy rock patch. I asked for 110M steps, but the GPU ran out of memory at about 24 minutes, after 91.75M steps (82% of the plan), so the run stopped early. The checkpoints were saved, so I used the last one. The reward reached 24.1 and was still rising. The robot followed all four commands I tried: straight (`track` 0.93), the -1 diagonal (0.82), the diagonal while turning (0.91) and a slow diagonal with a fast spin of 3 (0.85). With a turn of 1 it walked three laps of a circle about 2.8 m wide. With a turn of 3, far beyond the 1.2 it trained on, it spun about nine laps of a circle about 0.5 m wide.

**Things that broke.** My first rollouts for the two circling commands crashed, because saving every step for the video filled the GPU. Moving each step to the CPU fixed it, at a cost of about 11 minutes per video. The rough patch is only 20 m wide, so a robot walking straight reaches the edge after 7 to 13 seconds. I stop the rollout there. My time guess for the training (1.5 hours) was also too high, because it ran at about 66k steps per second.

**How the robot is actually built.** I wondered what the 3D model is. It is five mesh files (STL: trunk, hip, thigh, mirrored thigh, calf) that the simulator draws, and it reuses them: one body, four hips, four thighs and four calves, so 13 pieces on screen. The pieces are joined by hinges, like door hinges: each piece can only swing around one axis, within limits, relative to the piece before it. The body itself floats free, and the 12 hinges (3 per leg) are the motors. What the simulator tracks is just 18 numbers: the body's position and tilt, and 12 joint angles. The meshes are only for the picture. The physics uses invisible simple shapes instead, like hitboxes in a game. The model has a box, cylinders and capsules for the body and legs, but they are switched off for collisions, so only the four foot spheres are live hitboxes and only they touch the ground. I also tested that a visual-only hat can be drawn on the robot without touching the physics.

**Is it like a game?** Mostly yes: rigid pieces, hinges, contacts and springs are the same ingredients as a ragdoll game, and each motor works like an elastic band at the hinge (a spring that pulls the joint towards the target angle the policy picks, with some damping). The difference is the goal. Game physics aims to look right and stay stable. MuJoCo aims to be accurate for one real robot (total mass 12.7 kg) and to run thousands of copies at once on the GPU. It still takes shortcuts: the solver does a single pass per step, which is why the screen fills with "solver iterations limit reached". MuJoCo was not made by Fan Shi. It came from Emo Todorov's group at the University of Washington, and DeepMind bought it in 2021 and made it free in 2022.

**Where walking starts.** I rolled out all 16 checkpoints of the 120M run. The robot stood still up to 57.3M steps and was walking at 65.5M, so the switch is somewhere in that gap and it is sudden. Walking was a bit slow and drifting at 65.5M, and got polished by about 80M. Reward alone had been misleading: it was already 17 at 16M steps while the robot stood still.

</details>

---

<details>
<summary><h2>Oct 22 — More seeds, and one run that collapsed</h2></summary>

**Test 9 (seeds).** I trained five more runs on flat ground (seeds 1 to 5) to see whether walking shows up reliably. Training was faster than I guessed, about 20 minutes each. Four of them (seeds 2 to 5) had reward curves like the Test 7 run: about 19 to 20 for a long stretch, then a climb that starts around 52 to 59M steps and ends at 26 to 27. I have not rolled them out yet, so I have not confirmed that they walk. Seed 1 broke. Its reward rose to 18.7 by 18M steps, dropped to about 12, and then fell to 0.2 by 85M. No checkpoint walked, and its speed stayed near zero. But seed 1 was also the only run with 60 evaluations (the others had 31), so I cannot tell whether it was the seed or the setup. In my Colab Test 4 the reward also fell late in training, and that may have been the same kind of collapse, but I do not know.

**Test 9.1 (seed or setup?).** To split the two, I trained seed 1 again with 31 evaluations, and seeds 2 and 3 with 60. Seed 1 with 31 evaluations reached 27.06, and seeds 2 and 3 with 60 stayed below 21 (they ended at 20.05 and 12.78). I rolled them out on Oct 25 (see below).

**What a seed is.** The starting number for all the randomness in training (starting network weights, random commands, noise). Same seed and settings give nearly the same run, and a different seed is different luck. Every run before Test 9 used seed 0.

**The paper.** I looked at the vision part of the paper from the lab (arXiv 2410.03076, Fan Shi is a co-author). I read the PDF text (8 pages): the vision is a very small **simulated depth image** (16 by 12 pixels, with 84 by 84 for one CNN comparison), not RGB and not a real camera. It is rendered depth rays, flattened and fed to a single dense layer, and the paper says no separate image network may be needed. The text speaks of a "field of view" but not where the camera is mounted, and it describes no real-robot experiment. The quadruped is an ANYmal C simulated in MuJoCo MJX, the same framework I use, and the navigation part uses a simple point-mass robot. So their "perception" is a low-resolution depth input to a policy trained in simulation. My Go1 policy is blind, and a small depth or height-map input would be the cheap version of the same idea.

**Accessories.** Next I want to make three weightless accessories for the robot: a pumpkin head, a cone birthday hat and a backpack. They are visual only, so they cannot change how it moves.

</details>

---

<details>
<summary><h2>Oct 25 — Seed or setup? The 60-evaluation runs stand still</h2></summary>

**Test 9.1 rollouts.** I rolled out the final policy of the three Test 9.1 runs on 1 / 0 / 0 (`--no-video`, one 20 s episode each). Seed 1 with 31 evaluations **walks** (`track` 0.968, about 0.95 m/s, 18.9 m in 20 s). Seeds 2 and 3 with 60 evaluations both **stand still** (`track` 0.0187, moved 2 to 3 cm). So seed 1 is not a broken seed. In my runs, the 60-evaluation setup is what stops the walking: all three runs with 60 evaluations failed, and the runs with 16 or 31 walked or reached 26 to 27 in reward.

**Standing, not fallen.** A fallen robot would also read near-zero speed, so I checked the saved motion data. Both 60-evaluation robots stayed at about 0.3 m body height and upright to the end. I did not watch a video.

**Reward misled me again.** Seed 2 ended at 20.05 and seed 3 at 12.78, but both give the same `track`. A low reward does not mean the robot fell, and I cannot read what the robot does from it.

**`num_evals` is not only a plotting setting.** I had assumed it only sets how often the reward is measured. Here the same seed 1 walked at 31 and collapsed at 60, so it changed the training. My guess (not tested) is that each evaluation interval is cut into chunks and all environments reset at the start of each chunk, which would make a reset every 40 steps at 60 evaluations against 80 at 31. The episode is 1000 steps, so at 60 the robot would only ever see very short pieces. The test is 60 evaluations with `num_resets_per_eval` 5. Until then, I keep `num_evals` the same for runs I compare.

**Limits.** Three runs at 60 against six at 16 or 31, one rollout each, one command, and `num_evals` also changes the random path. I have not tried values in between (such as 45).

</details>

---

<details open>
<summary><h2>Oct 26 — Test 9.2: it is the reset interval, and my first overnight run went badly</h2></summary>

**What I tried.** Test 9.1 left me with "the 60-evaluation setup stops the walking". My reading of the Brax code (not checked line by line) is that all environments reset every so many steps, and that this interval depends on `num_evals` and `num_resets_per_eval`: every 40 steps at 60 evaluations, 80 at 31, 200 at 16. So I set the interval directly. I trained seed 2 and seed 3 at 60 evaluations with `num_resets_per_eval` 5 (interval 80), and at 31 evaluations with `num_resets_per_eval` 20 (interval 40). I also trained seed 4 at 60 evaluations with the default.

**What came out.** All four pairs follow the interval, not the number of evaluations. With a reset every 80 steps the robot **walks** (`track` 0.955 for seed 2, 0.971 for seed 3), and with a reset every 40 steps it **stands still** (0.018 to 0.019), whichever `num_evals` I used. Seed 4 at 60 evaluations also stands. The failing runs end at a reward of about 18.5, which is the standing level. Each seed keeps its own switch time: seed 2 switches at about 50M steps in both of its walking runs, and seed 3 switches late (after 68M) in both of its. Two seeds, both directions, one rollout each, so it is "in my runs", and I do not know why short intervals stop the walking.

**The sweeps of the old runs.** All five 31-evaluation runs walk at the end (`track` 0.952 to 0.968). The switch is sudden in each, between about 49M and 65M steps for four of them, and after 68.8M for seed 3. Seed 3's reward passed 21 at 55.7M while it still stood still on 1 / 0 / 0, so reward misled me again.

**The first overnight run went badly.** I queued about 3.8 hours of work in one chain: sweeps, six training runs with a sweep after each, then the plots and the copy into the repo at the very end. The laptop was shut off in the morning while run 6 was training (seed 4, 60 evaluations, resets 5, stopped at 52.4M of 96.7M steps). That killed the chain, and because the plots and the copy came last, nothing from the night was in the repo. Five of six runs did finish, so the science is fine, but the plan was poor. Everything lived in WSL until the end, and one bad moment cost the lot. I also had to drop my planned 45-evaluation run, because every setting from 31 to 59 evaluations gives the same 80-step interval at 96M steps, so it would not have shown where the line is.

**New rule: small pieces.** One unit is one training run plus its own sweep (about 30 minutes). After each unit I copy its small files into the repo and plot it before the next one starts, and I check that `go1_final` exists. No more than two or three units chained. Before leaving I check `tmux ls` and the progress file, and the laptop stays plugged in and awake while a job runs.

**Next.** Plot and copy the five finished runs, rerun seed 4 with resets 5 on its own, and try one more interval (for example a reset about every 120 steps) to see where the line is. Details are in [`Tests/Test9.2/Test9.2.md`](Tests/Test9.2/Test9.2.md).

</details>

---

<details open>
<summary><h2>What I know now</h2></summary>


- Reward isn't enough. Standing still can score about 19, and walking scored 26. I have to measure speed and watch the video.
- At about 29.5M steps the robot stands still. At about 100M it walks. (One run each, so I say "in my runs".)
- The Colab notebook wasn't broken. It was just undertrained.
- The step count in the log is the real number, not the setting. Asking for 18M gave 29.5M.
- Walking sideways and diagonally works when the command is inside the trained range (both policies).
- A command outside the range is not blocked. One policy followed it and one did not (two runs, so not a rule).
- The robot drifts in heading a little, because the turn command is a rate. The tilt is a limit of this kind of policy, not a bug.
- On rough ground the robot still followed every command I tried, even a turn of 3 (one run, stopped at 82%).
- A video rollout can run out of GPU memory. Moving each saved step to the CPU fixes it.
- The robot model is 13 mesh pieces from 5 STL files, joined by 12 hinges, with an invisible simpler shape set used for the physics. Looks and physics are separate.
- The motors act like springs at the hinges, and the solver is set to one pass per step on purpose, so the "solver iterations" warnings are normal.
- In the one run I checked with rollouts, walking started between 57.3M and 65.5M steps, suddenly. Four other seeds have reward curves that suggest about 52 to 59M, but I have not rolled them out. About 80M to 100M steps is a sensible budget, and more barely helps (the reward was flat from 98M to 123M).
- Training can collapse: seed 1's reward fell to 0.2. Test 9.1 (rolled out Oct 25) suggests it was the 60-evaluation setup and not the seed: seed 1 with 31 evaluations walks (`track` 0.968), and two other seeds with 60 evaluations stand still. Three runs against six, one rollout each, and the reason is not tested.
- `num_evals` changes the training and not only the plot, so it must stay the same for runs I compare. Test 9.2 suggests the reason is the reset interval: a reset every 80 steps walks and every 40 steps stands, whatever `num_evals` is (two seeds, both directions; my reading of the Brax code, not checked line by line).
- Do not chain a whole night's work into one job. One unit (a run and its sweep) at a time, copied to the repo as it finishes.
- A reward of 12.8 and a reward of 20.0 can both be a robot that stands still (`track` 0.0187 for both).
- The vision in the lab's paper is a tiny simulated depth image (read from the PDF text), with no real-robot experiment described.

</details>

<details>
<summary><h2>Still don't know</h2></summary>


- Whether seeds 2 to 5 actually walk, and whether the switch lands in the same window for each (needs rollouts).
- Why a reset every 40 steps stops the walking, and where the line is between 80 and 200 steps (only 40 and 80 are tried). The seed 4 replicate of the fix did not finish.
- Whether it walks backward and turns in place (the diagonals work).
- How much a different random seed changes things, and whether the drift following the sideways command is real.
- Whether more `num_envs` (8192) changes the result.
- Whether the full 111M steps on rough ground would change the result (the reward was still rising), and why the GPU ran out of memory at epoch 14.

</details>

## Next

See [`todo.md`](todo.md) for the running list.
