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

<details open>
<summary><h2>Oct 22 — More seeds, and one run that collapsed</h2></summary>


**Test 9 (seeds).** I trained five more runs on flat ground (seeds 1 to 5) to see whether walking shows up reliably. Training was faster than I guessed, about 20 minutes each. Four of them (seeds 2 to 5) had reward curves like the Test 7 run: about 19 to 20 for a long stretch, then a climb that starts around 52 to 59M steps and ends at 26 to 27. I have not rolled them out yet, so I have not confirmed that they walk. Seed 1 broke. Its reward rose to 18.7 by 18M steps, dropped to about 12, and then fell to 0.2 by 85M. No checkpoint walked, and its speed stayed near zero. But seed 1 was also the only run with 60 evaluations (the others had 31), so I cannot tell whether it was the seed or the setup. In my Colab Test 4 the reward also fell late in training, and that may have been the same kind of collapse, but I do not know.

**Test 9.1 (seed or setup?).** To split the two, I trained seed 1 again with 31 evaluations, and seeds 2 and 3 with 60. Seed 1 with 31 evaluations reached 27.06, and seeds 2 and 3 with 60 stayed below 21 (they ended at 20.05 and 12.78). On Oct 25 I rolled them out on 1 / 0 / 0: seed 1 (31 evaluations) **walks** (`track` 0.968, about 0.95 m/s), and both 60-evaluation runs **stand still** (`track` 0.0187, moved 2 to 3 cm in 20 s, still upright). So seed 1 is not a broken seed, and in my runs the 60-evaluation setup is what stops the walking. Reward misled me again: the run that ended at 12.78 and the one that ended at 20.05 behave the same. I haven't proven why (my guess is that the environments reset every 40 steps instead of 80, but I haven't tested that), and it is three runs against six.

**What a seed is.** The starting number for all the randomness in training (starting network weights, random commands, noise). Same seed and settings give nearly the same run, and a different seed is different luck. Every run before Test 9 used seed 0.

**The paper.** I looked at the vision part of the paper from the lab (arXiv 2410.03076, Fan Shi is a co-author). I read the PDF text (8 pages): the vision is a very small **simulated depth image** (16 by 12 pixels, with 84 by 84 for one CNN comparison), not RGB and not a real camera. It is rendered depth rays, flattened and fed to a single dense layer, and the paper says no separate image network may be needed. The text speaks of a "field of view" but not where the camera is mounted, and it describes no real-robot experiment. The quadruped is an ANYmal C simulated in MuJoCo MJX, the same framework I use, and the navigation part uses a simple point-mass robot. So their "perception" is a low-resolution depth input to a policy trained in simulation. My Go1 policy is blind, and a small depth or height-map input would be the cheap version of the same idea.

**Accessories.** Next I want to make three weightless accessories for the robot: a pumpkin head, a cone birthday hat and a backpack. They are visual only, so they cannot change how it moves.

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
- Training can collapse: seed 1's reward fell to 0.2. Test 9.1 suggests it was the 60-evaluation setup and not the seed: seed 1 with 31 evaluations walks (`track` 0.968), and two other seeds with 60 evaluations stand still. Three runs against six, one rollout each, and the reason is not tested.
- A reward of 12.8 and a reward of 20.0 can both be a robot that stands still (`track` 0.0187 for both).
- The vision in the lab's paper is a tiny simulated depth image (read from the PDF text), with no real-robot experiment described.

</details>

<details>
<summary><h2>Still don't know</h2></summary>


- Whether seeds 2 to 5 actually walk, and whether the switch lands in the same window for each (needs rollouts).
- Why the 60-evaluation schedule stops the walking (my guess is the reset every 40 steps, not tested), and where between 31 and 60 the line is.
- Whether it walks backward and turns in place (the diagonals work).
- How much a different random seed changes things, and whether the drift following the sideways command is real.
- Whether more `num_envs` (8192) changes the result.
- Whether the full 111M steps on rough ground would change the result (the reward was still rising), and why the GPU ran out of memory at epoch 14.

</details>

## Next

See [`todo.md`](todo.md) for the running list.
