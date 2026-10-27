#!/bin/bash
# Test 9.2 overnight queue (one GPU job at a time, in this order):
#   1. sweep the 40M to 70M window of the Test 9 / 9.1 runs (does the switch land in the same window?)
#   2. six new training runs that test "is it the reset interval?", each followed by its own sweep
#   3. plots, and copy the small result files into the repo
#
#   bash run_queue.sh smoke    # about 3 minutes: tiny training run to check the setup works
#   bash run_queue.sh          # the full queue, about 3.5 hours
# Progress: tail -f ~/test92_progress.txt

source ~/go1env/bin/activate
REPO=/mnt/c/Users/Admin/Documents/HaiDuong
SCRIPTS=$REPO/Scripts
DEST=$REPO/Tests/Test9.2
PROGRESS=~/test92_progress.txt
filter() { grep --line-buffered -vE "solver iterations|To disable|linesearch iterations"; }
say() { echo "[$(date +%H:%M)] $*" | tee -a $PROGRESS; }

# sweep <folder> [every]: command 1 / 0 / 0, checkpoints from 40M to 70M, plus go1_final
sweep() {
  say "sweep $1"
  (cd ~/$1 && python $SCRIPTS/sweep.py 1 0 0 --min-step 40e6 --max-step 70e6 --every ${2:-1} 2>&1 | filter > sweep_out.txt)
  tail -n 3 ~/$1/sweep_out.txt | tee -a $PROGRESS
}

# train <name> <seed> <num_evals> <num_resets_per_eval or "-"> [num_timesteps]
train() {
  local name=$1 seed=$2 evals=$3 resets=$4 steps=${5:-96_000_000}
  say "train $name (seed $seed, evals $evals, resets $resets, steps $steps)"
  mkdir -p ~/$name && cd ~/$name || return
  tr -d '\r' < $SCRIPTS/train_test.py | sed \
    -e "s/^ppo_params.num_timesteps = .*/ppo_params.num_timesteps = $steps/" \
    -e "s/^ppo_params.num_evals = .*/ppo_params.num_evals = $evals/" \
    -e "s/^    policy_params_fn=save_ckpt,\$/&\n    seed=$seed,/" > train_test.py
  if [ "$resets" != "-" ]; then
    sed -i "s/^ppo_params.num_envs = .*/&\nppo_params.num_resets_per_eval = $resets/" train_test.py
  fi
  grep -nE "num_timesteps|num_evals|num_resets|seed=" train_test.py | tee -a $PROGRESS
  python train_test.py 2>&1 | filter > run_$name.log
  tail -n 4 run_$name.log | tee -a $PROGRESS
  python $SCRIPTS/plot_reward.py run_$name.log -o reward_plot.png > /dev/null
  cd ~
}

# copy_small <folder> [with_log]: sweep csv (and the filtered log and script for new runs) into the repo
copy_small() {
  mkdir -p $DEST/$1
  cp ~/$1/sweep_x1.0_y0.0_yaw0.0.csv ~/$1/sweep_out.txt $DEST/$1/ 2>/dev/null
  if [ "$2" = "with_log" ]; then cp ~/$1/run_$1.log ~/$1/reward_plot.png ~/$1/train_test.py $DEST/$1/ 2>/dev/null; fi
}

if [ "$1" = "smoke" ]; then
  train go1_t92_smoke 2 3 20 3_000_000
  ls ~/go1_t92_smoke | head -20 | tee -a $PROGRESS
  (cd ~/go1_t92_smoke && python $SCRIPTS/sweep.py 1 0 0 2>&1 | filter | tail -n 6 | tee -a $PROGRESS)
  say "smoke test finished"
  exit 0
fi

say "queue started"

# 1. the existing runs: Test 9 (seeds 2 to 5 and seed 1 at 31 evaluations) and Test 9.1 (60 evaluations)
for d in go1_seed1_e31 go1_seed2 go1_seed3 go1_seed4 go1_seed5; do sweep $d; done
for d in go1_seed2_e60 go1_seed3_e60; do sweep $d 2; done

# 2. new runs, same seed as an existing pair so they can be compared directly:
#    seed 2 at 31 evaluations walks, at 60 stands still (Test 9, 9.1); seed 3 at 31 walks, at 60 stands still
train go1_t92_s2_e60_r5  2 60 5     # 60 evaluations but a reset every 80 steps, like the walking e31 runs
sweep go1_t92_s2_e60_r5 2
train go1_t92_s2_e31_r20 2 31 20    # 31 evaluations but a reset every 40 steps, like the failing e60 runs
sweep go1_t92_s2_e31_r20
train go1_t92_s3_e60_r5  3 60 5
sweep go1_t92_s3_e60_r5 2
train go1_t92_s3_e31_r20 3 31 20
sweep go1_t92_s3_e31_r20
train go1_t92_s4_e60     4 60 -     # a fourth seed at 60 evaluations, nothing else changed (expected: stands still)
sweep go1_t92_s4_e60 2
train go1_t92_s4_e60_r5  4 60 5     # the same seed 4 with the reset interval doubled (expected: walks, if the idea holds)
sweep go1_t92_s4_e60_r5 2

# 3. plots and copy the small files into the repo
say "plots"
cd ~
python $SCRIPTS/plot_sweep.py go1_seed1_e31 go1_seed2 go1_seed3 go1_seed4 go1_seed5 -o $DEST/sweep_e31_seeds.png
python $SCRIPTS/plot_sweep.py go1_seed2 go1_seed2_e60 go1_t92_s2_e60_r5 go1_t92_s2_e31_r20 -o $DEST/sweep_seed2.png
python $SCRIPTS/plot_sweep.py go1_seed3 go1_seed3_e60 go1_t92_s3_e60_r5 go1_t92_s3_e31_r20 -o $DEST/sweep_seed3.png
python $SCRIPTS/plot_sweep.py go1_t92_s4_e60 go1_t92_s4_e60_r5 -o $DEST/sweep_seed4.png
for d in go1_seed1_e31 go1_seed2 go1_seed3 go1_seed4 go1_seed5 go1_seed2_e60 go1_seed3_e60; do copy_small $d; done
for d in go1_t92_s2_e60_r5 go1_t92_s2_e31_r20 go1_t92_s3_e60_r5 go1_t92_s3_e31_r20 go1_t92_s4_e60 go1_t92_s4_e60_r5; do copy_small $d with_log; done
cp $PROGRESS $DEST/progress.txt
say "queue finished"
