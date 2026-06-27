"""Closed-loop goal-reaching DASHBOARD figure (single PNG, no env needed).

Reads the two stored closed-loop evaluations (full_view + masked) and renders one
figure that puts the headline comparison the advisor asked for in plain sight:

  [panel 1] success_by_H  vs H (entered the 8px ball at ANY step 1..H -> "passed through")
  [panel 2] success_at_H  vs H (still inside the ball AT the deadline H -> "stopped there")
  [panel 3] summary table: aggregate by_H / at_H for masked & full-view critic vs B3b / random.

success_by_H is the "can the agent reach a designated reachable coordinate within a small
ball" metric the user wanted (goals = teacher-visited => known-reachable; 8px tolerance).
This is NOT the in-training eval/success_rate red curve (uniform-random goals, n=30) -- that
one is a different, harder, noisy protocol and its absolute height is not comparable.

Read-only over aggregate_metrics.json; touches no model.
"""
import os, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FV = "artifacts/seaquest/goal_control/full_view/evaluation"
MK = "artifacts/seaquest/goal_control/masked/evaluation"
HS = (16, 32, 64)
# (label, eval-dir, policy-key, color, linestyle, marker)
SERIES = [
    ("B3b expert (teacher)", "FV", "B3b",    "#2ca02c", "-",  "o"),
    ("Critic — full view",   "FV", "critic", "#1f77b4", "-",  "s"),
    ("Critic — masked",      "MK", "critic", "#d62728", "--", "D"),
    ("Random (B1)",          "FV", "B1",     "#888888", ":",  "^"),
]


def load(d):
    return json.load(open(os.path.join(d, "aggregate_metrics.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-view-dir", default=FV)
    ap.add_argument("--masked-dir", default=MK)
    ap.add_argument("--out", default="artifacts/seaquest/goal_control/closed_loop_dashboard.png")
    args = ap.parse_args()
    dirs = {"FV": load(args.full_view_dir), "MK": load(args.masked_dir)}

    fig = plt.figure(figsize=(15, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.28)
    ax0, ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])

    def per_h(src, pol, field):
        ph = dirs[src]["per_horizon"][pol]
        if field == "success_by_H":
            return [ph[str(H)]["success_by_H"]["mean"] for H in HS]
        return [ph[str(H)]["success_at_H"] for H in HS]

    for ax, field, title in [
        (ax0, "success_by_H", "success_by_H  (entered 8px ball at any step ≤ H)\n“passed through the goal”"),
        (ax1, "success_at_H", "success_at_H  (inside the ball AT deadline H)\n“stopped at the goal”"),
    ]:
        for label, src, pol, col, ls, mk in SERIES:
            ax.plot(HS, per_h(src, pol, field), color=col, ls=ls, marker=mk, lw=2, ms=7, label=label)
        ax.set_xticks(HS); ax.set_xlabel("horizon H (steps)"); ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("success rate"); ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    ax0.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # panel 3: aggregate summary table
    ax2.axis("off")
    def agg(src, pol, k):
        a = dirs[src]["aggregate"][pol]
        return a["mean"] if k == "by_H" else a["success_at_H"]
    rows = [
        ("Critic — full view", f"{agg('FV','critic','by_H'):.3f}", f"{agg('FV','critic','at_H'):.3f}"),
        ("Critic — masked",    f"{agg('MK','critic','by_H'):.3f}", f"{agg('MK','critic','at_H'):.3f}"),
        ("B3b expert (teacher)",    f"{agg('FV','B3b','by_H'):.3f}",    f"{agg('FV','B3b','at_H'):.3f}"),
        ("Random (B1)",             f"{agg('FV','B1','by_H'):.3f}",     f"{agg('FV','B1','at_H'):.3f}"),
    ]
    col_lab = ["policy", "by_H\n(passed thru)", "at_H\n(stopped)"]
    tbl = ax2.table(cellText=rows, colLabels=col_lab, loc="center", cellLoc="center",
                    colWidths=[0.50, 0.25, 0.25])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.5); tbl.scale(1, 2.4)
    cmap = {0: "#1f77b4", 1: "#d62728", 2: "#2ca02c", 3: "#888888"}
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#222"); cell.set_text_props(color="w", weight="bold")
            cell.set_fontsize(8.5)
        elif c == 0:
            cell.get_text().set_color(cmap[r - 1]); cell.get_text().set_weight("bold")
            cell.get_text().set_ha("left"); cell.PAD = 0.04
    ax2.set_title("Aggregate (8640 rollouts, H16/32/64 pooled)", fontsize=10.5, pad=14)
    ax2.text(0.5, 0.04,
             "Goals = teacher-visited coords (known-reachable), 8px tolerance.\n"
             "NOT the in-training uniform-random eval/success_rate curve.",
             transform=ax2.transAxes, ha="center", va="top", fontsize=8, color="#666")

    fig.suptitle("Seaquest goal-reaching: can the policy reach a designated reachable coordinate?",
                 fontsize=12.5, weight="bold", y=1.0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
