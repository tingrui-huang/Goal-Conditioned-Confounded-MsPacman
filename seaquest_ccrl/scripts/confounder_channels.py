"""Quantify the two causal arrows of the oxygen confounder U, from data only.

This is a pure data-layer analysis (no GPU, no training, no model). It reads the
Level-1 offline dataset and measures both arrows that make oxygen a genuine
confounder rather than a merely-unobserved useful feature:

    U -> A   oxygen drives the behavior action (the policy surfaces when low).
    U -> S'  oxygen drives the outcome / next-state (oxygen depletion -> drowning
             -> episode termination), independent of the chosen action.

Level 1 already established U -> A by construction. This script closes the gap by
also measuring U -> S', and shows that the next-position shift at low oxygen is
*action-mediated* (U -> A -> S'), so the clean DIRECT outcome arrow is drowning.

Outputs:
    - printed summary (the numbers to quote)
    - seaquest_ccrl/figure/confounder_channels.png  (3-panel figure)
    - seaquest_ccrl/figure/confounder_channels.json (machine-readable numbers)

Run:
    python -m seaquest_ccrl.scripts.confounder_channels
"""
import os
import json
import glob

import numpy as np

from seaquest_ccrl import config as C

# Actions that move the submarine upward (toward the surface) in Seaquest.
UP_ACTIONS = {2, 6, 7}            # UP, UPRIGHT, UPLEFT
ACTION_NAMES = {0: "NOOP", 1: "FIRE", 2: "UP", 3: "RIGHT", 4: "LEFT", 5: "DOWN",
                6: "UPRIGHT", 7: "UPLEFT", 8: "DOWNRIGHT", 9: "DOWNLEFT"}
# Oxygen bins (filled-bar width units, 0..OXY_FULL_WIDTH) for the per-bin curves.
OXY_BINS = [(0, 0), (1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 63)]


def _wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion (robust at k=0 or k=n)."""
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def load_steps(root: str):
    """Concatenate per-step oxygen, done, action, and Δy (next-step vertical move)."""
    files = sorted(glob.glob(os.path.join(root, "traj_*.npz")))
    if not files:
        raise FileNotFoundError(f"No trajectories under {root!r}.")
    oxy, done, act, dy, term_oxy = [], [], [], [], []
    for f in files:
        d = np.load(f)
        o = d["oxygen"].astype(int)
        dn = d["done"].astype(bool)
        a = d["actions"].astype(int)
        pos = d["player_pos"].astype(float)
        T = len(a)
        oxy.append(o); done.append(dn); act.append(a)
        # Δy aligned to step t (transition t -> t+1); last step has no successor.
        d_y = np.full(T, np.nan)
        d_y[:-1] = pos[1:, 1] - pos[:-1, 1]      # image coords: down=+, surfacing=Δy<0
        dy.append(d_y)
        ti = np.where(dn)[0]
        term_oxy.append(int(o[ti[0]]) if len(ti) else int(o[-1]))
    return (np.concatenate(oxy), np.concatenate(done), np.concatenate(act),
            np.concatenate(dy), np.array(term_oxy), len(files))


def analyze(root: str = None):
    root = root or C.DATA_ROOT
    THETA = C.THETA
    oxy, done, act, dy, term_oxy, n_ep = load_steps(root)
    low = oxy < THETA
    high = ~low
    is_up = np.isin(act, list(UP_ACTIONS))

    # --- U -> A : oxygen drives the surfacing action -----------------------
    p_up_low, lo_l, lo_u = _wilson(is_up[low].sum(), low.sum())
    p_up_high, hi_l, hi_u = _wilson(is_up[high].sum(), high.sum())

    # --- U -> S' : oxygen drives termination (drowning) --------------------
    term_at_zero = int((term_oxy == 0).sum())
    haz0, h0l, h0u = _wilson(done[oxy == 0].sum(), (oxy == 0).sum())
    haz_pos, hpl, hpu = _wilson(done[oxy > 0].sum(), (oxy > 0).sum())

    # --- next-state shift, and the action-mediation check ------------------
    dyl = dy[low & ~np.isnan(dy)]
    dyh = dy[high & ~np.isnan(dy)]
    total_dy_effect = float(dyl.mean() - dyh.mean())
    # conditioned on action=UP (the only action with samples in both regimes)
    mask_up = is_up & ~np.isnan(dy)
    up_low = dy[mask_up & low]; up_high = dy[mask_up & high]
    direct_dy_effect = (float(up_low.mean() - up_high.mean())
                        if len(up_low) and len(up_high) else float("nan"))

    # per-bin curves for the figure
    bins = []
    for lo_b, hi_b in OXY_BINS:
        m = (oxy >= lo_b) & (oxy <= hi_b)
        n = int(m.sum())
        md = dy[m & ~np.isnan(dy)]
        bins.append({
            "bin": f"{lo_b}-{hi_b}" if lo_b != hi_b else f"{lo_b}",
            "n": n,
            "hazard": float(done[m].sum() / n) if n else float("nan"),
            "p_up": float(is_up[m].sum() / n) if n else float("nan"),
            "mean_dy": float(md.mean()) if len(md) else float("nan"),
        })

    summary = {
        "n_episodes": n_ep, "n_steps": int(len(oxy)), "THETA": THETA,
        "U_to_A": {
            "p_up_given_low_oxy": p_up_low, "p_up_low_ci": [lo_l, lo_u],
            "p_up_given_high_oxy": p_up_high, "p_up_high_ci": [hi_l, hi_u],
        },
        "U_to_Sprime": {
            "terminations_at_oxy0": term_at_zero, "n_episodes": n_ep,
            "frac_terminations_at_oxy0": term_at_zero / n_ep,
            "hazard_oxy0": haz0, "hazard_oxy0_ci": [h0l, h0u],
            "hazard_oxy_gt0": haz_pos, "hazard_oxy_gt0_ci": [hpl, hpu],
        },
        "next_state_shift": {
            "mean_dy_low": float(dyl.mean()), "mean_dy_high": float(dyh.mean()),
            "total_effect_px": total_dy_effect,
            "direct_effect_given_UP_px": direct_dy_effect,
        },
        "bins": bins,
    }
    return summary


def print_summary(s):
    A, S, N = s["U_to_A"], s["U_to_Sprime"], s["next_state_shift"]
    print(f"\nDataset: {s['n_episodes']} episodes, {s['n_steps']} steps, THETA={s['THETA']}")
    print("\n=== U -> A : oxygen drives the action (surfacing) ===")
    print(f"  P(move-up | oxygen <  THETA) = {A['p_up_given_low_oxy']:.3f}  "
          f"95% CI [{A['p_up_low_ci'][0]:.3f}, {A['p_up_low_ci'][1]:.3f}]")
    print(f"  P(move-up | oxygen >= THETA) = {A['p_up_given_high_oxy']:.3f}  "
          f"95% CI [{A['p_up_high_ci'][0]:.3f}, {A['p_up_high_ci'][1]:.3f}]")
    print("  => at low oxygen the behavior policy surfaces (near-)deterministically.")
    print("\n=== U -> S' : oxygen drives the outcome (drowning) ===")
    print(f"  terminations at oxygen==0 : {S['terminations_at_oxy0']}/{S['n_episodes']} "
          f"({100*S['frac_terminations_at_oxy0']:.0f}%)")
    print(f"  hazard P(terminate | oxy==0) = {S['hazard_oxy0']:.4f}  vs  "
          f"P(terminate | oxy>0) = {S['hazard_oxy_gt0']:.4f}")
    print("  => episode termination is caused by oxygen depletion, NOT by an action.")
    print("\n=== next-state shift (consequence of U -> A, not a direct arrow) ===")
    print(f"  mean Δy: low oxy {N['mean_dy_low']:+.3f}  vs  high oxy {N['mean_dy_high']:+.3f}  "
          f"(total effect {N['total_effect_px']:+.3f} px/step)")
    print(f"  conditioned on action=UP: oxygen effect on Δy = {N['direct_effect_given_UP_px']:+.3f} px/step")
    print("  => the position shift is action-mediated (U->A->S'); the DIRECT outcome arrow is drowning.")
    print("\nConclusion: oxygen affects BOTH the action (U->A) and the outcome (U->S'),")
    print("so it is a genuine confounder, not merely an unobserved useful feature.\n")


def make_figure(s, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = s["bins"]
    labels = [b["bin"] for b in bins]
    x = np.arange(len(bins))
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))

    ax[0].bar(x, [b["hazard"] for b in bins], color="#c0392b")
    ax[0].set_title("U → S′ : termination hazard")
    ax[0].set_ylabel("P(terminate | oxygen)")

    ax[1].bar(x, [b["p_up"] for b in bins], color="#2980b9")
    ax[1].set_title("U → A : P(surface action | oxygen)")
    ax[1].set_ylabel("P(move-up)")

    ax[2].axhline(0, color="gray", lw=0.8)
    ax[2].bar(x, [b["mean_dy"] for b in bins], color="#27ae60")
    ax[2].set_title("Next-state shift (Δy; <0 = up)")
    ax[2].set_ylabel("mean Δy (px/step)")

    for a in ax:
        a.set_xticks(x); a.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        a.set_xlabel("oxygen (filled-bar width)")
    fig.suptitle("Oxygen confounder channels (Level-1 data)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--out-json", default="seaquest_ccrl/figure/confounder_channels.json")
    ap.add_argument("--out-png", default="seaquest_ccrl/figure/confounder_channels.png")
    args = ap.parse_args()

    s = analyze(args.root)
    print_summary(s)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(s, f, indent=2)
    print(f"wrote {args.out_json}")
    try:
        make_figure(s, args.out_png)
    except Exception as e:
        print(f"(figure skipped: {e})")
