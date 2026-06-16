"""Phase 2.1b duplicate-robustness — steps 5-7 (metric recompute, NO retraining).

Consumes the ALREADY-SAVED V1/V4 source-audit test predictions + the dup_cache hashes
from oxy4_dup_inventory, and recomputes metrics on four test subsets:
  S0 full test | S1 duplicate-free (drop test rows whose input hash is in train) |
  S2 unique-hash-weighted | S3 one representative per unique test hash.
Then applies the PREDECLARED >=0.15 absolute-R2-drop rule and emits one outcome + SUMMARY.md.

NOTE: only argmax class labels were persisted by the source audit (not probabilities), so
classification log-loss cannot be recomputed per subset; accuracy / balanced-acc / macro-F1
and all regression metrics ARE recomputed. The decision is R2-based and fully available.
"""
import os, json, argparse
import numpy as np

from seaquest_ccrl.scripts.oxy4_audit_probes import boot_scalar, clf_extra

DR = "artifacts/seaquest/oxygen_4frame/leakage/source_audit/duplicate_robustness"
PRED = "artifacts/seaquest/oxygen_4frame/leakage/source_audit/predictions"
SUBST_DROP = 0.15      # predeclared (unchanged from the source audit)
MIN_KEEP = 0.50
MIN_R2 = 0.50
VAR = {"V1": "V1_newest_oxybar_masked", "V4": "V4_four_gameplay_crop"}


def reg_metrics(y, p, w=None):
    if w is None: w = np.ones(len(y))
    wm = (w * y).sum() / w.sum()
    ss = (w * (p - y) ** 2).sum(); tot = (w * (y - wm) ** 2).sum() + 1e-9
    mae = (w * np.abs(p - y)).sum() / w.sum(); rmse = np.sqrt((w * (p - y) ** 2).sum() / w.sum())
    return {"mae": float(mae), "rmse": float(rmse), "r2": float(1 - ss / tot)}


def clf_metrics(y, lab, w=None):
    if w is None: w = np.ones(len(y))
    acc = float((w * (lab == y)).sum() / w.sum())
    ex = clf_extra(y, lab, np.zeros(len(y)))           # balanced/macro-f1/confusion (unweighted)
    return {"accuracy": acc, "balanced_accuracy": ex["balanced_accuracy"],
            "macro_f1": ex["macro_f1"], "confusion_matrix": ex["confusion_matrix"],
            "log_loss": None}                          # probs not persisted -> unavailable per subset


def subset_metrics(yox, preg, ycl, plab, ep, mask=None, w=None):
    idx = np.arange(len(yox)) if mask is None else np.where(mask)[0]
    yo, pr, yc, pl, e = yox[idx], preg[idx], ycl[idx], plab[idx], ep[idx]
    ww = None if w is None else w[idx]
    out = {"n": int(len(idx)), "n_episodes": int(len(np.unique(e))),
           "regression": reg_metrics(yo, pr, ww), "classification": clf_metrics(yc, pl, ww)}
    out["regression"]["r2_ci"] = boot_scalar(yo, pr, e, "r2")
    out["classification"]["acc_ci"] = boot_scalar(yc, pl, e, "acc")
    out["classification"]["balanced_acc_ci"] = boot_scalar(yc, pl, e, "balanced_acc")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dr", default=DR)
    ap.add_argument("--pred", default=PRED)
    args = ap.parse_args()
    cache = np.load(f"{args.dr}/dup_cache.npz", allow_pickle=True)
    ep_te = cache["episode_te"]; t_te = cache["timestep_te"]; ox_te = cache["oxygen_te"].astype(float)
    published = {}
    pubpath = "artifacts/seaquest/oxygen_4frame/leakage/source_audit/audit_probe_metrics.json"
    if os.path.exists(pubpath):
        published = json.load(open(pubpath)).get("visual_probes", {})

    results, align = {}, {"outcome": "ALIGNED", "checks": {}}
    for V, predname in VAR.items():
        pp = f"{args.pred}/{predname}.npz"
        if not os.path.exists(pp):
            results[V] = {"status": "PREDICTIONS_UNAVAILABLE", "expected": pp}
            continue
        P = np.load(pp, allow_pickle=True)
        preg = P["pred_reg"].astype(float).ravel(); plab = P["pred_clf"].astype(int).ravel()
        ycl = P["y_class"].astype(int).ravel(); yoxP = P["y_oxygen"].astype(float).ravel()
        # alignment: predictions must line up with the reconstructed test rows
        aligned = (len(preg) == len(ox_te) and np.allclose(yoxP, ox_te) and
                   np.array_equal(P["episode"].ravel(), ep_te) and np.array_equal(P["timestep"].ravel(), t_te))
        align["checks"][V] = {"aligned": bool(aligned), "n_pred": int(len(preg)), "n_recon": int(len(ox_te))}
        if not aligned:
            align["outcome"] = "DUPLICATE_AUDIT_ALIGNMENT_FAILURE"
            results[V] = {"status": "ALIGNMENT_FAILURE"}
            continue
        in_train = cache[f"{V}_in_train"]; hte = cache[f"{V}_test_hash"]

        # S0 full, S1 duplicate-free
        S = {}
        S["S0_full"] = subset_metrics(ox_te, preg, ycl, plab, ep_te)
        S["S1_duplicate_free"] = subset_metrics(ox_te, preg, ycl, plab, ep_te, mask=~in_train)
        # S2 unique-hash-weighted (weight 1/count_in_test per sample)
        cnt = {}
        for h in hte: cnt[h] = cnt.get(h, 0) + 1
        w = np.array([1.0 / cnt[h] for h in hte])
        S["S2_unique_hash_weighted"] = subset_metrics(ox_te, preg, ycl, plab, ep_te, w=w)
        # S3 one representative per unique test hash: smallest (episode,timestep)
        best = {}
        for i, h in enumerate(hte):
            key = (int(ep_te[i]), int(t_te[i]))
            if h not in best or key < best[h][1]: best[h] = (i, key)
        rep = np.zeros(len(hte), bool); rep[[v[0] for v in best.values()]] = True
        S["S3_one_per_unique_hash"] = subset_metrics(ox_te, preg, ycl, plab, ep_te, mask=rep)

        r2_0, r2_1 = S["S0_full"]["regression"]["r2"], S["S1_duplicate_free"]["regression"]["r2"]
        acc0, acc1 = S["S0_full"]["classification"]["accuracy"], S["S1_duplicate_free"]["classification"]["accuracy"]
        keep_frac = S["S1_duplicate_free"]["n"] / S["S0_full"]["n"]
        pub_r2 = published.get(predname, {}).get("regression", {}).get("r2")
        results[V] = {"status": "OK", "subsets": S,
                      "published_r2": pub_r2, "recomputed_S0_r2": r2_0,
                      "reproduces_published": (None if pub_r2 is None else bool(abs(pub_r2 - r2_0) < 1e-3)),
                      "primary": {"r2_drop_S0_minus_S1": r2_0 - r2_1, "acc_drop_S0_minus_S1": acc0 - acc1,
                                  "duplicate_free_keep_fraction": keep_frac,
                                  "duplicate_free_r2": r2_1}}
        np.savez_compressed(f"{args.dr}/filtered_predictions_{V}.npz",
                            pred_reg=preg[~in_train], pred_clf=plab[~in_train],
                            y_oxygen=ox_te[~in_train], y_class=ycl[~in_train],
                            episode=ep_te[~in_train], timestep=t_te[~in_train])
        json.dump(results[V], open(f"{args.dr}/filtered_metrics_{V}.json", "w"), indent=2)
        print(f"[{V}] S0 R2={r2_0:.4f} (pub {pub_r2}) S1 R2={r2_1:.4f} drop={r2_0-r2_1:+.4f} "
              f"keep={keep_frac*100:.1f}% acc {acc0:.3f}->{acc1:.3f}")

    json.dump(align, open(f"{args.dr}/alignment.json", "w"), indent=2)

    # ---- decision (step 7) ----
    ok = {V: r for V, r in results.items() if r.get("status") == "OK"}
    if align["outcome"] == "DUPLICATE_AUDIT_ALIGNMENT_FAILURE":
        outcome = "DUPLICATE_AUDIT_ALIGNMENT_FAILURE"
    elif len(ok) < 2:
        outcome = "PENDING_PREDICTIONS"   # need authoritative V1 & V4 predictions
    else:
        insufficient = any(r["primary"]["duplicate_free_keep_fraction"] < MIN_KEEP for r in ok.values())
        material = any(r["primary"]["r2_drop_S0_minus_S1"] >= SUBST_DROP or
                       r["primary"]["duplicate_free_r2"] < MIN_R2 for r in ok.values())
        if insufficient:
            outcome = "DUPLICATE_FREE_SUBSET_INSUFFICIENT"
        elif material:
            outcome = "DUPLICATES_MATERIALLY_INFLATE_RESULTS"
        else:
            outcome = "DUPLICATES_NOT_MATERIAL"

    write_summary(args.dr, results, outcome, align)
    json.dump({"outcome": outcome, "alignment": align["outcome"],
               "per_variant": {V: r.get("primary") for V, r in results.items() if r.get("status") == "OK"}},
              open(f"{args.dr}/decision.json", "w"), indent=2)
    print("OUTCOME:", outcome)


def write_summary(dr, results, outcome, align):
    inv = {V: json.load(open(f"{dr}/duplicate_inventory_{V}.json")) for V in VAR if os.path.exists(f"{dr}/duplicate_inventory_{V}.json")}
    L = ["# Duplicate-Robustness Audit — SUMMARY", "",
         f"**Final outcome: `{outcome}`**  (alignment: `{align['outcome']}`)",
         f"Predeclared rule: material if absolute R² drop ≥ {SUBST_DROP} or duplicate-free R² < {MIN_R2}; "
         f"insufficient if duplicate-free subset keeps < {MIN_KEEP*100:.0f}% of test.", ""]
    for V in VAR:
        iv = inv.get(V, {}); r = results.get(V, {})
        L.append(f"## {V} — {VAR[V]}")
        if iv:
            L += [f"- actual model-input shape {iv['input_spec']['shape']} {iv['input_spec']['dtype']}",
                  f"- test samples shared with train: **{iv['n_test_samples_in_train']} "
                  f"({iv['frac_test_samples_in_train']*100:.1f}%)**; shared hashes {iv['n_hashes_shared_train_test']}; "
                  f"max multiplicity {iv['max_multiplicity_shared_hash']}",
                  f"- shared frames: surface {iv['shared_sample_category']['static_surface_frac']*100:.0f}% / "
                  f"opening {iv['shared_sample_category']['opening_reset_frac']*100:.0f}% / "
                  f"general {iv['shared_sample_category']['general_gameplay_frac']*100:.0f}%",
                  f"- same image → same EXACT oxygen {iv.get('label_consistency',{}).get('sample_weighted_frac_identical_oxygen')}; "
                  f"→ same BIN {iv.get('label_consistency',{}).get('sample_weighted_frac_identical_bin')}"]
        if r.get("status") == "OK":
            p = r["primary"]; S = r["subsets"]
            L += [f"- **S0 R² {S['S0_full']['regression']['r2']:.4f} → S1 (dup-free) R² "
                  f"{S['S1_duplicate_free']['regression']['r2']:.4f} (drop {p['r2_drop_S0_minus_S1']:+.4f})**; "
                  f"S1 keeps {p['duplicate_free_keep_fraction']*100:.1f}% of test",
                  f"- S2 weighted R² {S['S2_unique_hash_weighted']['regression']['r2']:.4f}; "
                  f"S3 one-per-hash R² {S['S3_one_per_unique_hash']['regression']['r2']:.4f}",
                  f"- bin accuracy S0 {S['S0_full']['classification']['accuracy']:.3f} → S1 "
                  f"{S['S1_duplicate_free']['classification']['accuracy']:.3f}",
                  f"- reproduces published R²: {r['reproduces_published']} (pub {r['published_r2']})"]
        else:
            L.append(f"- metrics: **{r.get('status','PENDING')}** (needs authoritative source-audit predictions)")
        L.append("")
    L += ["## Answers",
          f"1. V1 model inputs shared train/test: {inv.get('V1',{}).get('n_test_samples_in_train','?')}",
          f"2. V4 stacks shared train/test: {inv.get('V4',{}).get('n_test_samples_in_train','?')}",
          f"3. Fraction of test affected: V1 {inv.get('V1',{}).get('frac_test_samples_in_train','?')}, "
          f"V4 {inv.get('V4',{}).get('frac_test_samples_in_train','?')}",
          "4. Shared inputs ↔ identical oxygen: EXACT often differs (see frac_identical_oxygen); "
          "BIN nearly always identical (~0.97).",
          "5/6. Original vs duplicate-free V1/V4 metrics: see per-variant block above.",
          f"7/8. Did duplicates materially inflate? → **{outcome}**."]
    open(f"{dr}/SUMMARY.md", "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
