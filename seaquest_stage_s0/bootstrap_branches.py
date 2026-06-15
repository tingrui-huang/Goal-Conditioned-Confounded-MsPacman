"""Bootstrap CIs over anchors for forced-first-action divergence (Section 17.6).
Host-side, reads branches/raw_branch_outputs.npz. No container needed.
"""
import json, numpy as np
BASE = "artifacts/seaquest/stage_s0"
HOR = [1, 2, 4, 8, 12, 16, 24, 32]
d = np.load(f"{BASE}/branches/raw_branch_outputs.npz", allow_pickle=True)
rows = d["rows"]; cols = list(d["columns"])
ci = {c: cols.index(c) for c in cols}
# views (exclude player position; world-only also excludes own missile)
NOPLAYER = ["cum_reward", "terminated", "n_shark", "n_submarine", "n_diver",
            "n_player_missile", "n_enemy_missile", "score", "lives"]
WORLDONLY = ["cum_reward", "terminated", "n_shark", "n_submarine", "n_diver",
             "n_enemy_missile", "score", "lives"]
anchors = np.unique(rows[:, ci["anchor"]]).astype(int)


def diverge_indicator(view_cols, h):
    idx = [ci[c] for c in view_cols]
    out = {}
    for anc in anchors:
        sub = rows[(rows[:, ci["anchor"]] == anc) & (rows[:, ci["horizon"]] == h)]
        if sub.shape[0] < 2:
            continue
        vecs = sub[:, idx]
        vecs = np.nan_to_num(vecs, nan=-12345.0)
        base = vecs[0]
        out[anc] = int(np.any(np.any(vecs != base, axis=1)))
    return out


def bootstrap(ind, B=2000, seed=1):
    keys = np.array(sorted(ind.keys()))
    vals = np.array([ind[k] for k in keys], dtype=float)
    if len(vals) == 0:
        return None
    rng = np.random.RandomState(seed)
    stats = []
    for _ in range(B):
        s = rng.choice(len(vals), size=len(vals), replace=True)
        stats.append(vals[s].mean())
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"point": float(vals.mean()), "ci95_low": float(lo), "ci95_high": float(hi),
            "n_anchors": int(len(vals))}


res = {"horizons": {}, "n_bootstrap": 2000, "n_anchors": int(len(anchors))}
for h in HOR:
    res["horizons"][h] = {
        "noplayer": bootstrap(diverge_indicator(NOPLAYER, h)),
        "worldonly": bootstrap(diverge_indicator(WORLDONLY, h)),
    }
json.dump(res, open(f"{BASE}/branches/bootstrap_ci.json", "w"), indent=2)
print("wrote branches/bootstrap_ci.json")
for h in HOR:
    npv = res["horizons"][h]["noplayer"]; wo = res["horizons"][h]["worldonly"]
    print(f"  H={h:2d}: noplayer {npv['point']:.2f} [{npv['ci95_low']:.2f},{npv['ci95_high']:.2f}]  "
          f"worldonly {wo['point']:.2f} [{wo['ci95_low']:.2f},{wo['ci95_high']:.2f}]")
