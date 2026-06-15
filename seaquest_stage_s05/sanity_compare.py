"""Requirement 9: sanity-compare corrected (frozen-adapter) vs superseded
(inline-Gumbel) aggregate results. Sanity check ONLY — final gates/SUMMARY use the
corrected data exclusively. Host-side (stdlib)."""
import json, os
BASE = "artifacts/seaquest/stage_s05"
SUP = f"{BASE}/superseded_pre_adapter_rerun"


def load(p, d=None):
    try:
        return json.load(open(p))
    except Exception:
        return d


def grab(root):
    out = {"modes": {}}
    for m in ["N-Greedy", "N-Sampled", "O-Greedy", "O-Sampled"]:
        s = load(f"{root}/closed_loop/summary_{m}.json")
        if s:
            out["modes"][m] = {k: s.get(k) for k in ["return_mean", "action_entropy_nats",
                                                     "n_distinct_actions", "n_transitions"]}
    g = load(f"{root}/action_support/global_action_counts.json", {})
    out["global"] = {"entropy": g.get("action_entropy_nats"),
                     "eff_actions": g.get("effective_num_actions")}
    clf = load(f"{root}/action_support/state_only_classifier.json", {})
    out["state_only_exact_acc"] = (clf.get("exact_action") or {}).get("test_accuracy")
    knn = load(f"{root}/action_support/nearest_neighbor_support.json", {})
    out["knn_k50_ge2cat"] = (knn.get("k=50") or {}).get("frac_transitions_ge2_categories")
    inc = load(f"{root}/future_prediction/incremental_action_metrics.json", {})
    out["action_adds_info"] = inc.get("action_adds_info_any_horizon_component")
    p6 = load(f"{root}/branches/action_pair_metrics.json", {})
    g6 = p6.get("gate_p6", {})
    out["p6_pass"] = g6.get("pass")
    out["p6_horizons"] = {h: {"np": v.get("noplayer_frac"), "wo": v.get("worldonly_frac")}
                          for h, v in (g6.get("horizons") or {}).items()}
    return out


def main():
    cor = grab(BASE); sup = grab(SUP)
    cmp = {"note": ("Sanity check only; final gates use CORRECTED (frozen-adapter) data. "
                    "Differences expected because the sampled action RNG path changed from inline "
                    "jax-PRNG Gumbel to the frozen adapter (numpy uniform -> teacher.gumbel_from_uniform "
                    "-> teacher.sample_action). Distributions should be qualitatively the same."),
           "corrected": cor, "superseded": sup,
           "qualitative_agreement": {
               "global_entropy_close": _close(cor["global"]["entropy"], sup["global"]["entropy"], 0.2),
               "eff_actions_close": _close(cor["global"]["eff_actions"], sup["global"]["eff_actions"], 2.0),
               "state_only_acc_close": _close(cor["state_only_exact_acc"], sup["state_only_exact_acc"], 0.1),
               "both_action_adds_info": bool(cor["action_adds_info"]) and bool(sup["action_adds_info"]),
               "both_p6_pass": bool(cor["p6_pass"]) and bool(sup["p6_pass"])}}
    json.dump(cmp, open(f"{BASE}/sanity_corrected_vs_superseded.json", "w"), indent=2)
    print("sanity comparison written.")
    for m in cor["modes"]:
        c = cor["modes"][m]; s = sup["modes"].get(m, {})
        print(f"  {m}: ret {c['return_mean']:.0f} vs {s.get('return_mean')}; "
              f"ent {c['action_entropy_nats']:.2f} vs {s.get('action_entropy_nats')}")
    print(f"  P4 state-only acc: {cor['state_only_exact_acc']} vs {sup['state_only_exact_acc']}")
    print(f"  P6 pass: {cor['p6_pass']} vs {sup['p6_pass']}")


def _close(a, b, tol):
    try:
        return bool(abs(float(a) - float(b)) <= tol)
    except Exception:
        return None


if __name__ == "__main__":
    main()
