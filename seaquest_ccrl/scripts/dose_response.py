"""Dose-response curve: sweep THETA (confounding strength) x measure the
naive-vs-oracle gap.

The confounder U = oxygen drives the behavior policy's surfacing action (U->A). A
NAIVE learner sees only the masked frame -> oxygen is hidden -> it cannot tell when
the policy will surface beyond the marginal base rate. An ORACLE that observes oxygen
can. The "naive-vs-oracle gap" is how much predictability of the action the naive
learner LOSES by not seeing the confounder. THETA sets how wide an oxygen range the
policy conditions on, so the gap is the dose-response to confounding strength.

This is statistics on rollouts only (no training, no critic, no world model). For each
THETA we record (oxygen, is_surface_action) pairs -- NO frames needed, so it is fast.

Outputs (under seaquest_ccrl/figure/):
  dose_response.png   -- the curve
  dose_response.csv   -- the raw numbers

Run: python -m seaquest_ccrl.scripts.dose_response
"""
import os
import io
import csv
import contextlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ocatari.core import OCAtari

from seaquest_ccrl import config as C
from seaquest_ccrl.envs.seaquest_gc import _player_pos, _oxygen
from seaquest_ccrl.policies.scripted_behavior import ScriptedBehaviorPolicy

UP_ACTIONS = {2, 6, 7, 10, 14, 15}          # surfacing-containing actions (full 18-set)
THETA_SWEEP = [5, 10, 15, 20, 25, 30, 35, 40, 45]
STEPS_PER_THETA = 6000
OXY_BIN_W = 8                                # bin oxygen 0..63 into 8-wide bins
REGION_CELL = 16                             # px side of a spatial "region" cell
FIG_DIR = "seaquest_ccrl/figure"


def count_regions(positions):
    """Number of distinct REGION_CELL x REGION_CELL grid cells the sub visited.
    A coverage/diversity measure of achieved goals (submarine positions)."""
    cells = set()
    for x, y in positions:
        if x is None or y is None or (x != x):   # skip None / NaN
            continue
        cells.add((int(x) // REGION_CELL, int(y) // REGION_CELL))
    return len(cells)


def _state_from(env):
    objs = [o for o in env.objects if o.category != "NoObject"]
    return {"player_pos": _player_pos(objs), "oxygen": _oxygen(objs)}


def rollout_oxygen_action(theta, n_steps, seed):
    """Fast rollout (no render): returns (oxygen, is_surface, positions) over n_steps."""
    cfg = C.DEFAULT
    env = OCAtari(cfg.game_id, mode=cfg.mode, hud=cfg.hud, render_mode=cfg.render_mode,
                  frameskip=cfg.frameskip, repeat_action_probability=cfg.repeat_action_probability)
    env.reset(seed=seed)
    pol = ScriptedBehaviorPolicy(cfg, theta=theta)
    rng = np.random.RandomState(seed)

    def sample_target():
        return (rng.randint(*cfg.target_x_range), rng.randint(*cfg.target_y_range))

    oxy, surf, positions = [], [], []
    # OCAtari's Seaquest RAM extractor prints "Orientation.E/W" on direction
    # changes; silence stdout during stepping so the sweep logs cleanly.
    with contextlib.redirect_stdout(io.StringIO()):
        state = _state_from(env)
        target = sample_target()
        for _ in range(n_steps):
            a = pol.act(state, target)
            o = state["oxygen"]
            if o is not None and o >= 0:
                oxy.append(o)
                surf.append(1 if a in UP_ACTIONS else 0)
            if state["player_pos"] is not None:
                positions.append(state["player_pos"])   # achieved-goal coverage
            _, _, term, trunc, _ = env.step(a)
            state = _state_from(env)
            if pol.reached(state, target):
                target = sample_target()
            if term or trunc:
                env.reset(seed=seed)
                state = _state_from(env)
                target = sample_target()
    env.close()
    return np.asarray(oxy, dtype=float), np.asarray(surf, dtype=float), positions


def gap_metrics(oxy, surf):
    """Compute naive-vs-oracle gap + supporting confounding-strength stats."""
    n = len(surf)
    if n == 0:
        return dict(surface_rate=np.nan, dP=np.nan, oracle_acc=np.nan,
                    naive_acc=np.nan, gap=np.nan, mi_bits=np.nan)

    surface_rate = surf.mean()

    # --- coupling strength: P(surface|low oxy) - P(surface|high oxy) ---
    q_lo, q_hi = np.percentile(oxy, 25), np.percentile(oxy, 75)
    lo, hi = oxy <= q_lo, oxy >= q_hi
    p_lo = surf[lo].mean() if lo.any() else np.nan
    p_hi = surf[hi].mean() if hi.any() else np.nan
    dP = p_lo - p_hi

    # --- naive vs oracle predictor of the surfacing action ---
    # Oracle SEES oxygen: best predictor = per-oxygen-bin majority vote.
    # Naive does NOT (oxygen masked): best predictor = global marginal majority.
    bins = (oxy // OXY_BIN_W).astype(int)
    oracle_correct = 0
    mi = 0.0
    p_y1 = surface_rate
    p_y = np.array([1 - p_y1, p_y1])
    for b in np.unique(bins):
        m = bins == b
        nb = m.sum()
        y1 = surf[m].sum(); y0 = nb - y1
        oracle_correct += max(y0, y1)                 # bin-majority prediction
        # mutual information contribution I(A_surface ; oxygen-bin)
        p_x = nb / n
        for yc, cnt in ((1, y1), (0, y0)):
            if cnt > 0 and p_y[yc] > 0:
                p_xy = cnt / n
                mi += p_xy * np.log2(p_xy / (p_x * p_y[yc]))
    oracle_acc = oracle_correct / n
    naive_acc = max(p_y1, 1 - p_y1)                   # marginal-majority accuracy
    gap = oracle_acc - naive_acc                      # the naive-vs-oracle gap

    return dict(surface_rate=surface_rate, dP=dP, oracle_acc=oracle_acc,
                naive_acc=naive_acc, gap=gap, mi_bits=mi)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    rows = []
    print(f"Sweeping THETA={THETA_SWEEP}  ({STEPS_PER_THETA} steps each)")
    for i, theta in enumerate(THETA_SWEEP):
        oxy, surf, positions = rollout_oxygen_action(theta, STEPS_PER_THETA, seed=100 + i)
        m = gap_metrics(oxy, surf)
        m["theta"] = theta
        m["n_regions"] = count_regions(positions)        # distinct visited regions
        rows.append(m)
        print(f"  THETA={theta:2d} | surf_rate={m['surface_rate']:.2f} "
              f"dP={m['dP']:.2f} oracle={m['oracle_acc']:.3f} naive={m['naive_acc']:.3f} "
              f"gap={m['gap']:.3f} MI={m['mi_bits']:.3f} bits | "
              f"regions={m['n_regions']:3d}")

    th = [r["theta"] for r in rows]
    gap = [r["gap"] for r in rows]
    dP = [r["dP"] for r in rows]
    mi = [r["mi_bits"] for r in rows]
    sr = [r["surface_rate"] for r in rows]
    nreg = [r["n_regions"] for r in rows]

    # --- CSV ---
    csv_path = os.path.join(FIG_DIR, "dose_response.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["theta", "surface_rate", "dP_low_minus_high",
                    "oracle_acc", "naive_acc", "naive_vs_oracle_gap", "mi_bits",
                    f"n_regions_visited_cell{REGION_CELL}px"])
        for r in rows:
            w.writerow([r["theta"], f"{r['surface_rate']:.4f}", f"{r['dP']:.4f}",
                        f"{r['oracle_acc']:.4f}", f"{r['naive_acc']:.4f}",
                        f"{r['gap']:.4f}", f"{r['mi_bits']:.4f}", r["n_regions"]])

    # --- figure ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4.6))

    ax1.plot(th, gap, "o-", color="#c0392b", lw=2, ms=7,
             label="naive-vs-oracle gap\n(oracle_acc - naive_acc)")
    ax1.plot(th, dP, "s--", color="#2980b9", lw=1.6, ms=6,
             label=r"$P(\mathrm{surface}\,|\,\mathrm{low\ O_2}) - P(\cdot\,|\,\mathrm{high\ O_2})$")
    ax1.set_xlabel(r"$\Theta$  (surfacing threshold = confounding strength)")
    ax1.set_ylabel("action-prediction gap")
    ax1.set_title("Dose-response: confounding strength -> naive-vs-oracle gap")
    ax1.grid(alpha=0.3); ax1.legend(fontsize=9, loc="best")
    ax1.set_ylim(bottom=min(0, min(gap + dP) - 0.02))

    ax2.plot(th, mi, "o-", color="#8e44ad", lw=2, ms=7, label="I(action ; oxygen)")
    ax2.set_xlabel(r"$\Theta$"); ax2.set_ylabel("mutual information (bits)", color="#8e44ad")
    ax2.tick_params(axis="y", labelcolor="#8e44ad")
    ax2.set_title("Oxygen->action information & surfacing rate")
    ax2.grid(alpha=0.3)
    ax2b = ax2.twinx()
    ax2b.plot(th, sr, "^--", color="#16a085", lw=1.6, ms=6, label="surfacing rate")
    ax2b.set_ylabel("P(surface)", color="#16a085")
    ax2b.tick_params(axis="y", labelcolor="#16a085")
    l1, lb1 = ax2.get_legend_handles_labels()
    l2, lb2 = ax2b.get_legend_handles_labels()
    ax2.legend(l1 + l2, lb1 + lb2, fontsize=9, loc="best")

    ax3.plot(th, nreg, "D-", color="#d35400", lw=2, ms=7,
             label=f"distinct regions ({REGION_CELL}x{REGION_CELL}px cells)")
    ax3.set_xlabel(r"$\Theta$"); ax3.set_ylabel("# distinct position regions visited")
    ax3.set_title("State coverage vs confounding strength")
    ax3.grid(alpha=0.3); ax3.legend(fontsize=9, loc="best")

    fig.suptitle("Seaquest Level-1: dose-response of the oxygen confounder (U->A)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    png_path = os.path.join(FIG_DIR, "dose_response.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {png_path}\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
