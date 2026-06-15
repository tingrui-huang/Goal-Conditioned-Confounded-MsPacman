"""Generate Stage-S0 figures (runs in ocatari image; matplotlib present)."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/work/artifacts/seaquest/stage_s0"
FIG = f"{BASE}/figures"
os.makedirs(FIG, exist_ok=True)
H = [1, 2, 4, 8, 12, 16, 24, 32]


def load(p, d=None):
    try:
        return json.load(open(p))
    except Exception:
        return d


# 1. horizon divergence curves
hz = load(f"{BASE}/branches/horizon_metrics.json", {})
div = hz.get("divergence_by_view_horizon", {})
if div:
    plt.figure(figsize=(7, 4.5))
    for view, mk in [("full", "o-"), ("noplayer", "s-"), ("worldonly", "^-")]:
        per = div.get(view, {})
        ys = [(per.get(str(h)) or per.get(h) or {}).get("frac_diverged", np.nan) for h in H]
        plt.plot(H, ys, mk, label=view)
    plt.axhline(0.20, ls="--", c="grey", label="Gate C 20%")
    plt.xlabel("horizon (agent steps)"); plt.ylabel("fraction of anchors diverged")
    plt.title("Forced-first-action divergence vs horizon"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{FIG}/horizon_divergence.png", dpi=120); plt.close()

# 2. component change-rate heatmap
comp = load(f"{BASE}/branches/component_metrics.json", {})
cc = comp.get("component_change_rates_by_horizon", {})
if cc:
    comps = list(next(iter(cc.values())).keys())
    M = np.array([[cc.get(str(h), cc.get(h, {})).get(c, {}).get("rate", np.nan) for h in H] for c in comps])
    plt.figure(figsize=(8, 6))
    im = plt.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, label="cross-action change rate")
    plt.yticks(range(len(comps)), comps); plt.xticks(range(len(H)), H)
    plt.xlabel("horizon"); plt.title("Component change rate across forced first actions")
    plt.tight_layout(); plt.savefig(f"{FIG}/component_heatmap.png", dpi=120); plt.close()

# 3. action histograms native vs port (candidate A)
na = load(f"{BASE}/teacher/native_eval_A.json", {})
po = load(f"{BASE}/teacher/ocatari_eval_A.json", {})
hn = (na.get("eval_teacher") or {}).get("action_histogram")
hp = (po.get("port_eval") or {}).get("action_histogram")
if hn and hp:
    MEAN = ['NOOP','FIRE','UP','RIGHT','LEFT','DOWN','UPRT','UPLT','DNRT','DNLT',
            'UPF','RTF','LTF','DNF','URF','ULF','DRF','DLF']
    x = np.arange(18)
    plt.figure(figsize=(10, 4))
    hn = np.array(hn)/max(sum(hn),1); hp = np.array(hp)/max(sum(hp),1)
    plt.bar(x-0.2, hn, 0.4, label="native EnvPool")
    plt.bar(x+0.2, hp, 0.4, label="OCAtari port")
    plt.xticks(x, MEAN, rotation=60, fontsize=7); plt.ylabel("freq")
    plt.title("Candidate A action distribution: native vs OCAtari port"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{FIG}/action_dist_native_vs_port.png", dpi=120); plt.close()

# 4. oxygen-action association uplift bar
aa = load(f"{BASE}/oxygen_screen/action_association.json", {})
oa = load(f"{BASE}/oxygen_screen/outcome_association.json", {})
labels, vals = [], []
if aa.get("oxygen_logloss_improvement") is not None:
    labels.append("action"); vals.append(aa["oxygen_logloss_improvement"])
for hh, row in (oa.get("horizons") or {}).items():
    for tgt, r in row.items():
        if isinstance(r, dict) and r.get("oxygen_logloss_improvement") is not None:
            labels.append(f"{tgt}@{hh}"); vals.append(r["oxygen_logloss_improvement"])
if labels:
    plt.figure(figsize=(9, 4))
    colors = ["green" if v > 0 else "red" for v in vals]
    plt.bar(range(len(labels)), vals, color=colors)
    plt.axhline(0, c="k", lw=.8)
    plt.xticks(range(len(labels)), labels, rotation=60, fontsize=7)
    plt.ylabel("held-out log-loss improvement from +oxygen")
    plt.title("Oxygen incremental predictive value (episode-split)")
    plt.tight_layout(); plt.savefig(f"{FIG}/oxygen_uplift.png", dpi=120); plt.close()

print("figures written:", sorted(os.listdir(FIG)))
