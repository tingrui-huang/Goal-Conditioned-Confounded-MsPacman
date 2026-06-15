"""Stage-S0.5 figures (ocatari image; matplotlib). Native object-conditioned heatmaps
are infeasible (EnvPool has no objects) -> documented in a JSON + listed in SUMMARY."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/work/artifacts/seaquest/stage_s05"
FIG = f"{BASE}/figures"
os.makedirs(FIG, exist_ok=True)
CFG = json.load(open(f"{BASE}/config/resolved_config.json"))
MEAN = CFG["ALE_MEANINGS"]


def load(p, d=None):
    try:
        return json.load(open(p))
    except Exception:
        return d


norm = np.load(f"{BASE}/policy_heatmaps/normalized_probabilities.npz", allow_pickle=True)
cnt = np.load(f"{BASE}/policy_heatmaps/raw_counts.npz", allow_pickle=True)
bins = json.load(open(f"{BASE}/policy_heatmaps/bin_definitions.json"))


def heat(ax, M, title, xlab, ylab, xticks=None, yticks=None, vmin=None, vmax=None):
    im = ax.imshow(M, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9); ax.set_xlabel(xlab, fontsize=8); ax.set_ylabel(ylab, fontsize=8)
    if xticks is not None:
        ax.set_xticks(range(len(xticks))); ax.set_xticklabels(xticks, fontsize=6, rotation=45)
    if yticks is not None:
        ax.set_yticks(range(len(yticks))); ax.set_yticklabels(yticks, fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046)


# A: player_y x oxygen P(UP) ported (O-Sampled). Always save support-count beside it.
for mode, fn in [("O-Sampled", "ported"), ("O-Greedy", "ported_greedy")]:
    k = f"{mode}__A__P_UP_related"
    if k in norm.files:
        M = norm[k]; Cn = cnt[f"{mode}__A__P_UP_related__count"]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        oxb = CFG["bins"]["oxygen"]; pyb = CFG["bins"]["player_y"]
        heat(ax[0], M, f"P(UP-related) | {mode}", "oxygen bin", "player_y bin",
             [f"{oxb[i]:.0f}" for i in range(len(oxb)-1)], [f"{pyb[i]:.0f}" for i in range(len(pyb)-1)], 0, 1)
        heat(ax[1], Cn, f"support count | {mode}", "oxygen bin", "player_y bin")
        plt.tight_layout(); plt.savefig(f"{FIG}/player_y_x_oxygen_up_probability_{fn}.png", dpi=110); plt.close()

# B: enemy relative x movement (ported)
k = "O-Sampled__B__P_LEFT_related"
if k in norm.files:
    fig, ax = plt.subplots(2, 3, figsize=(13, 8))
    exb = CFG["bins"]["enemy_dx"]; eyb = CFG["bins"]["enemy_dy"]
    xt = [f"{exb[i]:.0f}" for i in range(len(exb)-1)]; yt = [f"{eyb[i]:.0f}" for i in range(len(eyb)-1)]
    for i, role in enumerate(["LEFT_related", "RIGHT_related", "UP_related", "DOWN_related", "FIRE_related"]):
        r, c = divmod(i, 3)
        heat(ax[r][c], norm[f"O-Sampled__B__P_{role}"], f"P({role})", "enemy_dx", "enemy_dy", xt, yt, 0, 1)
    heat(ax[1][2], cnt["O-Sampled__B__P_LEFT_related__count"], "support count", "enemy_dx", "enemy_dy", xt, yt)
    plt.tight_layout(); plt.savefig(f"{FIG}/enemy_relative_x_movement_ported.png", dpi=110); plt.close()

# E: state strata x 18 actions (mean prob) ported
E = bins["heatmap_meta"]["O-Sampled"].get("heatmap_E", {})
if E.get("mean_prob"):
    strata = E["strata"]; M = np.array([E["mean_prob"][s] for s in strata])
    fig, ax = plt.subplots(figsize=(11, 5))
    heat(ax, M, "Mean teacher P(action) by state stratum (O-Sampled)", "action", "stratum",
         MEAN, strata, 0, M.max())
    plt.tight_layout(); plt.savefig(f"{FIG}/state_strata_x_action_probability_ported.png", dpi=110); plt.close()

# F: native minus ported marginal action prob
npm = load(f"{BASE}/closed_loop/native_ported_marginal.json", {})
nps = npm.get("N-Sampled_vs_O-Sampled", {})
if nps:
    pn = np.array(nps["native_dist"]); pp = np.array(nps["ported_dist"])
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(18)
    ax.bar(x - 0.2, pn, 0.4, label="native (EnvPool)"); ax.bar(x + 0.2, pp, 0.4, label="ported (OCAtari)")
    ax.set_xticks(x); ax.set_xticklabels(MEAN, rotation=60, fontsize=7); ax.legend()
    ax.set_title(f"Native vs ported MARGINAL action dist (TV={nps['total_variation']:.3f}, JS={nps['jensen_shannon']:.4f})")
    plt.tight_layout(); plt.savefig(f"{FIG}/native_minus_ported_action_probability.png", dpi=110); plt.close()

# action support counts
gs = load(f"{BASE}/action_support/global_action_counts.json", {})
if gs:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(range(18), gs["action_counts"]); ax.set_xticks(range(18)); ax.set_xticklabels(MEAN, rotation=60, fontsize=7)
    ax.set_title(f"O-Sampled action support counts (eff {gs['effective_num_actions']:.1f}/18)")
    plt.tight_layout(); plt.savefig(f"{FIG}/action_support_counts.png", dpi=110); plt.close()

# local entropy + dominant fraction distributions
lp = f"{BASE}/action_support/local_propensity.npz"
if os.path.exists(lp):
    d = np.load(lp)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(d["local_entropy"], bins=40, color="teal"); ax.set_title("Local action entropy (k=50) distribution")
    ax.set_xlabel("nats"); plt.tight_layout(); plt.savefig(f"{FIG}/local_action_entropy_distribution.png", dpi=110); plt.close()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(d["dominant_proportion"], bins=40, color="indianred"); ax.axvline(0.90, ls="--", c="k", label="0.90 gate")
    ax.set_title("Dominant category proportion (k=50)"); ax.legend(); ax.set_xlabel("proportion")
    plt.tight_layout(); plt.savefig(f"{FIG}/dominant_action_fraction_distribution.png", dpi=110); plt.close()

# confusion matrix (state-only exact)
clf = load(f"{BASE}/action_support/state_only_classifier.json", {})
if clf:
    cm = np.array(clf["exact_action"]["confusion_matrix"], dtype=float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    heat(ax, cmn, f"State-only action confusion (acc {clf['exact_action']['test_accuracy']:.2f})",
         "predicted", "true", MEAN, MEAN, 0, 1)
    plt.tight_layout(); plt.savefig(f"{FIG}/state_only_action_confusion_matrix.png", dpi=110); plt.close()

# action-added future prediction
inc = load(f"{BASE}/future_prediction/incremental_action_metrics.json", {})
if inc:
    rows = []
    for h, comps in inc["incremental"].items():
        for k, v in comps.items():
            if isinstance(v, dict) and "dR2" in v:
                rows.append((f"{k}@H{h}", v["dR2"]))
    if rows:
        labels, vals = zip(*rows)
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.bar(range(len(vals)), vals, color=["green" if v > 0 else "red" for v in vals])
        ax.axhline(0, c="k", lw=.8); ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=70, fontsize=6)
        ax.set_title("Incremental held-out future ΔR² from adding action (O-Sampled)")
        plt.tight_layout(); plt.savefig(f"{FIG}/action_added_future_prediction.png", dpi=110); plt.close()

# branch action-pair heatmaps: shared color scale across horizons (directly
# comparable), support-count panel beside each distance matrix, key+sha in title.
bp = f"{BASE}/branches/raw_outputs.npz"
integ = load(f"{BASE}/branches/heatmap_integrity.json", {})
man = (integ or {}).get("plotted_matrix_manifest", {})
if os.path.exists(bp):
    import hashlib
    d = np.load(bp)
    for view, key in [("no_player", "np"), ("world_only", "wo")]:
        vmax = max([np.nanmax(d[f"{key}_H{h}"]) for h in [4, 8, 16]
                    if f"{key}_H{h}" in d.files and np.isfinite(d[f"{key}_H{h}"]).any()] + [1e-6])
        for h in [4, 8, 16]:
            kk = f"{key}_H{h}"; ck = f"count_H{h}"
            if kk in d.files:
                sha = man.get(kk, {}).get("sha256", "")[:12]
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                heat(ax[0], d[kk], f"{view} z-dist H={h} (shared scale)\nkey={kk} sha={sha}",
                     "action j", "action i", MEAN, MEAN, 0, vmax)
                heat(ax[1], d[ck] if ck in d.files else np.zeros((18, 18)),
                     f"support count (pairs) H={h}", "action j", "action i", MEAN, MEAN)
                plt.tight_layout(); plt.savefig(f"{FIG}/branch_{view}_action_pair_H{h}.png", dpi=110); plt.close()

# document infeasible native object heatmaps
json.dump({"infeasible_figures": [
    "player_y_x_oxygen_up_probability_native.png",
    "enemy_relative_x_movement_native.png",
    "state_strata_x_action_probability_native.png"],
    "reason": ("Native EnvPool exposes no RAM/object state (verified: info has only "
               "lives/reward/terminated/truncated/elapsed_step). Object-conditioned heatmaps "
               "require object features only available in the OCAtari port. The native side is "
               "instead summarized by the MARGINAL action distribution (native_minus_ported_*.png) "
               "plus exact tensor parity + identical preprocessing ops.")},
    open(f"{FIG}/NATIVE_OBJECT_HEATMAPS_INFEASIBLE.json", "w"), indent=2)

print("figures written:", sorted(os.listdir(FIG)))
