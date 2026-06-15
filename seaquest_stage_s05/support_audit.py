"""P4 — dataset action-support audit on O-Sampled (Stage-S0.5 section 11, Gate P4).
Runs in the ocatari image (scikit-learn). Episode-level splits; standardize on train.
"""
import json, os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (accuracy_score, top_k_accuracy_score, log_loss,
                             f1_score, confusion_matrix)

BASE = "/work/artifacts/seaquest/stage_s05"
CFG = json.load(open(f"{BASE}/config/resolved_config.json"))
SF = CFG["state_features"]
SEMANTIC = CFG["semantic_categories"]; CAT_NAMES = list(SEMANTIC.keys())
A2CAT = {a: i for i, (k, v) in enumerate(SEMANTIC.items()) for a in v}
K = CFG["knn_k"]; TH = CFG["support_thresholds"]
MEAN = CFG["ALE_MEANINGS"]


def load(mode="O-Sampled"):
    d = np.load(f"{BASE}/closed_loop/rows_{mode}.npz", allow_pickle=True)
    cols = list(d["columns"]); R = d["rows"]; ci = {c: i for i, c in enumerate(cols)}
    return R, ci


def main():
    R, ci = load("O-Sampled")
    X = R[:, [ci[f] for f in SF]].astype(np.float64)
    # impute missing with column train-median later; track validity
    a = R[:, ci["sampled_action"]].astype(int)
    cat = np.array([A2CAT[x] for x in a])
    ep = R[:, ci["episode"]].astype(int)
    eps = np.unique(ep)
    rng = np.random.RandomState(CFG["seeds"]["splits"])
    order = rng.permutation(eps)
    ntest = max(1, len(eps) // 5); nval = max(1, len(eps) // 5)
    test_e = set(order[:ntest]); val_e = set(order[ntest:ntest + nval]); train_e = set(order[ntest + nval:])
    tr = np.isin(ep, list(train_e)); va = np.isin(ep, list(val_e)); te = np.isin(ep, list(test_e))

    # impute by train median (fallback 0 for all-NaN cols), standardize on train
    with np.errstate(all="ignore"):
        med = np.nanmedian(X[tr], axis=0)
    med = np.nan_to_num(med, nan=0.0)
    Xi = np.where(np.isnan(X), med, X)
    Xi = np.nan_to_num(Xi, nan=0.0)
    sc = StandardScaler().fit(Xi[tr]); Xs = np.nan_to_num(sc.transform(Xi), nan=0.0)

    # ---- 11.1 global support
    h = np.bincount(a, minlength=18).astype(float); p = h / h.sum()
    eff_n = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    hc = np.bincount(cat, minlength=12).astype(float); pc = hc / hc.sum()
    glob = {"action_counts": h.astype(int).tolist(), "action_freq": p.tolist(),
            "action_entropy_nats": float(-(p[p > 0] * np.log(p[p > 0])).sum()),
            "effective_num_actions": eff_n, "min_action_freq": float(p.min()),
            "frac_actions_lt_30_samples": float((h < 30).mean()),
            "category_counts": hc.astype(int).tolist(), "category_names": CAT_NAMES,
            "category_freq": pc.tolist(),
            "effective_num_categories": float(np.exp(-(pc[pc > 0] * np.log(pc[pc > 0])).sum())),
            "n_transitions": int(len(R))}
    json.dump(glob, open(f"{BASE}/action_support/global_action_counts.json", "w"), indent=2)

    # ---- 11.2 state-only classifier (exact action + category)
    def fit_report(y, nclass, labels_names):
        clf = LogisticRegression(max_iter=400, multi_class="multinomial", C=1.0)
        clf.fit(Xs[tr], y[tr])
        proba = clf.predict_proba(Xs[te]); classes = clf.classes_
        pred = clf.predict(Xs[te])
        acc = float(accuracy_score(y[te], pred))
        try:
            t3 = float(top_k_accuracy_score(y[te], proba, k=3, labels=classes))
        except Exception:
            t3 = None
        ll = float(log_loss(y[te], proba, labels=classes))
        f1 = float(f1_score(y[te], pred, average="macro"))
        cm = confusion_matrix(y[te], pred, labels=list(range(nclass))).tolist()
        # majority baseline
        maj = float(max(np.bincount(y[tr], minlength=nclass)) / tr.sum())
        return {"test_accuracy": acc, "test_top3_accuracy": t3, "test_log_loss": ll,
                "macro_f1": f1, "majority_class_baseline_acc": maj,
                "confusion_matrix": cm, "labels": labels_names}
    clf_exact = fit_report(a, 18, MEAN)
    clf_cat = fit_report(cat, 12, CAT_NAMES)
    json.dump({"split": {"train": sorted(int(x) for x in train_e),
                         "val": sorted(int(x) for x in val_e),
                         "test": sorted(int(x) for x in test_e)},
               "exact_action": clf_exact, "semantic_category": clf_cat,
               "interpretation": ("high state-only accuracy => action nearly determined by state "
                                  "=> weak local overlap; this is descriptive, not proof.")},
              open(f"{BASE}/action_support/state_only_classifier.json", "w"), indent=2)

    # ---- 11.3 local kNN support (cross-episode neighbors)
    nn = NearestNeighbors(n_neighbors=max(K) + 60).fit(Xs)
    dist, idx = nn.kneighbors(Xs)
    knn_res = {}; local_prop = {}
    N = len(R)
    for k in K:
        n_distinct = np.zeros(N); loc_ent = np.zeros(N); dom = np.zeros(N)
        min_prop = np.zeros(N); cat_distinct = np.zeros(N); allmajor = np.zeros(N)
        d_same = np.full(N, np.nan); d_diff = np.full(N, np.nan)
        prop_mat = np.zeros((N, 12))
        for i in range(N):
            # neighbors from OTHER episodes
            cand = idx[i][1:]; cand = cand[ep[cand] != ep[i]][:k]
            if len(cand) == 0:
                continue
            na = a[cand]; nc = cat[cand]; nd = dist[i][1:][ep[idx[i][1:]] != ep[i]][:k]
            ph = np.bincount(na, minlength=18).astype(float); pp = ph / ph.sum()
            n_distinct[i] = (ph > 0).sum()
            loc_ent[i] = -(pp[pp > 0] * np.log(pp[pp > 0])).sum()
            pch = np.bincount(nc, minlength=12).astype(float); ppc = pch / pch.sum()
            prop_mat[i] = ppc
            dom[i] = ppc.max(); cat_distinct[i] = (pch > 0).sum()
            min_prop[i] = ppc[ppc > 0].min() if (ppc > 0).any() else 0
            allmajor[i] = int((pch[[A2CAT.get(x) for x in [1, 2, 5, 4, 3]]] > 0).all())  # FIRE,UP,DOWN,LEFT,RIGHT present
            same = nd[na == a[i]]; diff = nd[na != a[i]]
            if len(same): d_same[i] = same.min()
            if len(diff): d_diff[i] = diff.min()
        ratio = d_diff / d_same
        knn_res[f"k={k}"] = {
            "mean_distinct_actions": float(np.nanmean(n_distinct[n_distinct > 0])),
            "mean_local_action_entropy": float(np.nanmean(loc_ent[n_distinct > 0])),
            "median_dominant_category_proportion": float(np.nanmedian(dom[n_distinct > 0])),
            "frac_transitions_ge2_categories": float((cat_distinct >= 2).mean()),
            "frac_transitions_min_alt_propensity_ge_0.05": float(
                ((np.sort(prop_mat, axis=1)[:, -2] >= 0.05)).mean()),
            "frac_all_major_categories_present": float(allmajor.mean()),
            "median_cross_over_same_action_distance_ratio": float(np.nanmedian(ratio)),
            "mean_min_alt_category_propensity": float(np.nanmean(np.sort(prop_mat, axis=1)[:, -2]))}
        if k == 50:
            local_prop = {"dominant_proportion": dom, "local_entropy": loc_ent,
                          "cat_distinct": cat_distinct, "propensity_matrix": prop_mat,
                          "second_category_propensity": np.sort(prop_mat, axis=1)[:, -2]}
    json.dump(knn_res, open(f"{BASE}/action_support/nearest_neighbor_support.json", "w"), indent=2)
    np.savez_compressed(f"{BASE}/action_support/local_propensity.npz", **local_prop)
    json.dump({"train": sorted(int(x) for x in train_e), "val": sorted(int(x) for x in val_e),
               "test": sorted(int(x) for x in test_e), "n_episodes": int(len(eps)),
               "n_transitions": int(len(R))},
              open(f"{BASE}/action_support/split_manifest.json", "w"), indent=2)
    print("P4 support done.")
    print(f"  global: entropy={glob['action_entropy_nats']:.3f} eff_actions={eff_n:.2f} "
          f"min_freq={glob['min_action_freq']:.4f} cat_entropy={glob['effective_num_categories']:.2f}")
    print(f"  state-only exact acc={clf_exact['test_accuracy']:.3f} (maj {clf_exact['majority_class_baseline_acc']:.3f}) "
          f"top3={clf_exact['test_top3_accuracy']}; cat acc={clf_cat['test_accuracy']:.3f}")
    for k in K:
        r = knn_res[f"k={k}"]
        print(f"  k={k}: ge2cat={r['frac_transitions_ge2_categories']:.2f} "
              f"med_dom={r['median_dominant_category_proportion']:.2f} "
              f"min_alt>=.05={r['frac_transitions_min_alt_propensity_ge_0.05']:.2f}")


if __name__ == "__main__":
    main()
