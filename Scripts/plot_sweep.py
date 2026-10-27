"""Plot `track` against steps for several runs, with the eval reward below on the same steps axis.

Reads sweep_x1.0_y0.0_yaw0.0.csv (from sweep.py) and the training log of each run folder.

    python plot_sweep.py ~/go1_seed2 ~/go1_seed3 ~/go1_seed4 -o sweep_seeds.png
    python plot_sweep.py ~/go1_seed1_e31 ~/go1_seed2_e60 --cmd 1.0 0.0 0.0
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in WSL
import matplotlib.pyplot as plt

from plot_reward import parse

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument("folders", nargs="+", help="run folders (each with a sweep CSV and a *.log)")
ap.add_argument("--cmd", nargs=3, default=["1.0", "0.0", "0.0"], metavar=("X", "Y", "YAW"),
                help="command of the sweep to plot (default: 1.0 0.0 0.0)")
ap.add_argument("-o", "--out", default="sweep_plot.png")
args = ap.parse_args()

csv_name = f"sweep_x{args.cmd[0]}_y{args.cmd[1]}_yaw{args.cmd[2]}.csv"
fig, (ax_t, ax_r) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
for i, folder in enumerate(args.folders):
    folder = Path(folder).expanduser()
    color = f"C{i % 10}"
    label = folder.name
    sweep = folder / csv_name
    if sweep.exists():
        rows = [r for r in csv.DictReader(open(sweep)) if int(r["step"]) >= 0]
        ax_t.plot([int(r["step"]) / 1e6 for r in rows], [float(r["track"]) for r in rows],
                  marker="o", ms=3, color=color, label=label)
    else:
        print(f"{label}: no {csv_name}")
    for log in sorted(folder.glob("*.log")):
        rewards = parse(log)
        if rewards:
            ax_r.plot([r[0] / 1e6 for r in rewards], [r[1] for r in rewards], color=color, label=label)
            break

ax_t.axhline(0.0183, color="gray", ls=":", lw=1)  # exp(-4): what standing still scores at x = 1
ax_t.set_ylabel(f"track at command {' / '.join(args.cmd)}")
ax_t.set_title("Go1 joystick: rollout track and eval reward vs steps")
ax_t.grid(alpha=0.3)
ax_t.legend()
ax_r.set_xlabel("environment steps (millions)")
ax_r.set_ylabel("eval episode reward")
ax_r.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(args.out, dpi=150)
print(f"saved {args.out}")
