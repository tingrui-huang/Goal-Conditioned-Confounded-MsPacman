"""Corrected acceptance checks for the Seaquest Level-1 confounded dataset.

- check_B: the REAL mask-completeness / unidentifiability test (containment sweep).
- check_C: the 2-minute "does the behavior policy actually use oxygen" test (U->A),
           computed on the COLLECTED dataset.

Run:  python -m seaquest_ccrl.scripts.checks_bc
"""

import numpy as np
from ocatari.core import OCAtari


# --------------------------------------------------------------------------
# Check B - mask actually hides oxygen
# --------------------------------------------------------------------------
# WRONG (old, vacuous version): "after masking, is the rect region constant?"
#   -> it's zeroed by construction, so it ALWAYS passes and tests nothing.
# RIGHT: the oxygen indicator lives only in the bar strip (OxygenBar U
#   OxygenBarDepleted). So masking hides oxygen iff that strip is fully INSIDE
#   the mask rect at EVERY oxygen level. We sweep full->low and check containment.

def check_B(rect, n_steps=600, verbose=True):
    x, y, w, h = rect
    rx0, rx1, ry0, ry1 = x, x + w, y, y + h

    env = OCAtari("ALE/Seaquest-v5", mode="ram", hud=True, render_mode="rgb_array",
                  frameskip=4, repeat_action_probability=0.0)
    env.reset(seed=0)

    levels_seen, leaks = set(), []
    for step in range(n_steps):
        _, _, term, trunc, _ = env.step(5)  # DOWN: stay submerged so oxygen sweeps down
        if term or trunc:
            env.reset(seed=0)
            continue
        strip = [o for o in env.objects
                 if o.category in ("OxygenBar", "OxygenBarDepleted")]
        if not strip:
            continue
        bx0 = min(o.x for o in strip); bx1 = max(o.x + o.w for o in strip)
        by0 = min(o.y for o in strip); by1 = max(o.y + o.h for o in strip)
        ob = [o for o in env.objects if o.category == "OxygenBar"]
        level = ob[0].w if ob else -1
        levels_seen.add(level)

        if not (bx0 >= rx0 and bx1 <= rx1 and by0 >= ry0 and by1 <= ry1):
            leaks.append((level, (bx0, bx1, by0, by1)))
    env.close()

    if verbose:
        print(f"[B] mask rect x[{rx0},{rx1}] y[{ry0},{ry1}] | "
              f"swept {len(levels_seen)} oxygen levels")
        if leaks:
            for lvl, ext in leaks[:5]:
                print(f"    LEAK @oxygen={lvl}: strip {ext} pokes outside rect")
            print(f"[B] FAIL - oxygen leaks at {len(leaks)} frames. Enlarge OXY_MASK_RECT.")
        else:
            print("[B] PASS - full oxygen-bar strip is inside the mask at every level.")
    return len(leaks) == 0


# --------------------------------------------------------------------------
# Check C - does the behavior policy condition on oxygen?  (U -> A)
# --------------------------------------------------------------------------
# Run on the COLLECTED dataset. If the policy surfaces when oxygen is low, then
# "go up" actions should be much more frequent at low oxygen than at high oxygen.
# If the two rates are ~equal, the policy ignores oxygen -> NO confounding.

# Seaquest full 18-action set; "up"/surfacing-containing actions:
#   2 UP, 6 UPRIGHT, 7 UPLEFT, 10 UPFIRE, 14 UPRIGHTFIRE, 15 UPLEFTFIRE
# (confirm against your env's action_meanings if you reduced the set)
UP_ACTIONS = {2, 6, 7, 10, 14, 15}


def check_C(dataset_dir, verbose=True):
    from seaquest_ccrl.data.dataset import SeaquestOfflineDataset
    ds = SeaquestOfflineDataset(dataset_dir, oracle=True)

    oxy, up = [], []
    for traj in ds.trajectories():
        a = np.asarray(traj["action"])           # loader exposes "action" (singular)
        o = np.asarray(traj["oxygen"], dtype=float)
        valid = o >= 0                            # drop transient -1 (unknown) steps
        oxy.append(o[valid])
        up.append(np.isin(a, list(UP_ACTIONS))[valid])
    oxy = np.concatenate(oxy)
    up = np.concatenate(up).astype(float)

    lo = oxy <= np.percentile(oxy, 25)   # low oxygen quartile
    hi = oxy >= np.percentile(oxy, 75)   # high oxygen quartile
    p_lo, p_hi = up[lo].mean(), up[hi].mean()

    if verbose:
        print(f"[C] P(surface | LOW oxygen)  = {p_lo:.3f}")
        print(f"[C] P(surface | HIGH oxygen) = {p_hi:.3f}")
        ratio = p_lo / max(p_hi, 1e-6)
        if p_lo > p_hi + 0.05:
            print(f"[C] PASS - surfacing is {ratio:.1f}x more likely at low oxygen "
                  f"=> behavior conditions on oxygen (U->A is real).")
        else:
            print("[C] WEAK/FAIL - surfacing barely depends on oxygen. "
                  "The confounder may be dead; raise THETA / make surfacing more oxygen-driven.")
    return p_lo, p_hi


if __name__ == "__main__":
    from seaquest_ccrl.config import OXY_MASK_RECT
    check_B(OXY_MASK_RECT)
    check_C("seaquest_ccrl/data/raw")   # dataset already collected
