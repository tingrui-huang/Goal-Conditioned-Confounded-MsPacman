"""Paired masked-vs-full-view statistics from the per-anchor closed-loop outcomes.

Reads the two critics' raw_rollouts.npz (same anchor set), matches by (anchor_id, horizon),
and computes — from the 1440 per-anchor PAIRED binary outcomes, not an aggregated scalar:
  * discordant-pair counts (full-success/masked-fail; full-fail/masked-success);
  * McNemar test (continuity-corrected chi2 + exact binomial);
  * clustered bootstrap CIs (resampling the 24 episodes/seeds = the independent unit) for the
    aggregate success-rate difference and per horizon H16/H32/H64, reporting the between-cluster
    SD so a narrow interval is transparent rather than hidden.
No models are changed or rerun; this is a statistics/reporting correction only.
"""
import os, json, argparse
import numpy as np
from scipy import stats

FV = "artifacts/seaquest/goal_control/full_view/evaluation"
MK = "artifacts/seaquest/goal_control/masked/evaluation"


def load_critic(d):
    r = np.load(os.path.join(d, "raw_rollouts.npz"), allow_pickle=True)
    m = r["policy"].astype(str) == "critic"
    return {"anchor": r["anchor_id"][m], "H": r["H"][m], "seed": r["seed"][m],
            "succ": r["success_by_H"][m].astype(int),
            "min": r["min_dist"][m].astype(float), "fin": r["final_dist"][m].astype(float)}


def mcnemar(fs, ms):
    b = int(((fs == 1) & (ms == 0)).sum())   # full success, masked failure
    c = int(((fs == 0) & (ms == 1)).sum())   # full failure, masked success
    n = b + c
    chi2 = ((abs(b - c) - 1) ** 2 / n) if n > 0 else 0.0
    return {"full_succ_masked_fail_b": b, "full_fail_masked_succ_c": c, "discordant": n,
            "concordant_both_succ": int(((fs == 1) & (ms == 1)).sum()),
            "concordant_both_fail": int(((fs == 0) & (ms == 0)).sum()),
            "mcnemar_chi2_cc": float(chi2),
            "p_chi2_cc": float(stats.chi2.sf(chi2, 1)) if n > 0 else 1.0,
            "p_exact_binomial": float(stats.binomtest(min(b, c), n, 0.5).pvalue) if n > 0 else 1.0}


def clustered_boot(diff, seed, nboot=5000, seed0=0):
    uniq = np.unique(seed); by = {e: diff[seed == e] for e in uniq}
    rng = np.random.RandomState(seed0)
    means = [np.concatenate([by[e] for e in rng.choice(uniq, len(uniq), True)]).mean() for _ in range(nboot)]
    return {"mean_diff": float(diff.mean()),
            "ci95_clustered": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "between_cluster_sd": float(np.std([by[e].mean() for e in uniq])), "n_clusters": int(len(uniq))}


def fmean(a):
    a = a[np.isfinite(a)]; return float(a.mean()) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-view-dir", default=FV)
    ap.add_argument("--masked-dir", default=MK)
    ap.add_argument("--out", default="artifacts/seaquest/goal_control/paired_stats.json")
    args = ap.parse_args()
    fv, mk = load_critic(args.full_view_dir), load_critic(args.masked_dir)
    key = {(int(a), int(h)): i for i, (a, h) in enumerate(zip(fv["anchor"], fv["H"]))}
    idx = np.array([key[(int(a), int(h))] for a, h in zip(mk["anchor"], mk["H"])])
    assert (fv["anchor"][idx] == mk["anchor"]).all() and (fv["H"][idx] == mk["H"]).all(), "anchor mismatch"
    fs, ms, seed, H = fv["succ"][idx], mk["succ"], mk["seed"], mk["H"]
    diff = ms - fs                                        # per-anchor paired difference in {-1,0,+1}

    out = {"n_matched_anchors": int(len(diff)),
           "success_rate": {"full_view": float(fs.mean()), "masked": float(ms.mean()),
                            "diff_masked_minus_full": float(diff.mean())},
           "per_anchor_diff_value_counts": {str(int(v)): int((diff == v).sum()) for v in (-1, 0, 1)},
           "mcnemar": {"aggregate": mcnemar(fs, ms)},
           "clustered_bootstrap_by_seed": {"aggregate": clustered_boot(diff, seed)},
           "distances_px": {"mean_min": {"full_view": fmean(fv["min"][idx]), "masked": fmean(mk["min"])},
                            "mean_final_finite": {"full_view": fmean(fv["fin"][idx]), "masked": fmean(mk["fin"])}}}
    for h in (16, 32, 64):
        hm = H == h
        out["success_rate"][f"H{h}"] = {"full_view": float(fs[hm].mean()), "masked": float(ms[hm].mean())}
        out["mcnemar"][f"H{h}"] = mcnemar(fs[hm], ms[hm])
        out["clustered_bootstrap_by_seed"][f"H{h}"] = clustered_boot(diff[hm], seed[hm])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)

    a = out["mcnemar"]["aggregate"]; cb = out["clustered_bootstrap_by_seed"]
    print(f"matched={out['n_matched_anchors']}  diff value counts (-1/0/+1): "
          f"{out['per_anchor_diff_value_counts']}")
    print(f"AGG success full={fs.mean():.3f} masked={ms.mean():.3f} diff={diff.mean():+.4f}  "
          f"clustered CI{cb['aggregate']['ci95_clustered']} (between-seed SD {cb['aggregate']['between_cluster_sd']:.4f})")
    print(f"AGG McNemar  b(full>masked)={a['full_succ_masked_fail_b']}  c(masked>full)={a['full_fail_masked_succ_c']}  "
          f"chi2cc={a['mcnemar_chi2_cc']:.2f} p_cc={a['p_chi2_cc']:.4f}  p_exact={a['p_exact_binomial']:.4f}")
    for h in (16, 32, 64):
        m = out["mcnemar"][f"H{h}"]; c = cb[f"H{h}"]
        print(f"H{h}: full={out['success_rate'][f'H{h}']['full_view']:.3f} "
              f"masked={out['success_rate'][f'H{h}']['masked']:.3f} diff={c['mean_diff']:+.3f} "
              f"CI{c['ci95_clustered']} | b={m['full_succ_masked_fail_b']} c={m['full_fail_masked_succ_c']} "
              f"p_exact={m['p_exact_binomial']:.4f}")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
