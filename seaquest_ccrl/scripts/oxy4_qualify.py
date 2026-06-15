"""Oxygen qualification report — aggregates Phase 2.1/2.2/2.3 into exactly one outcome.
Thresholds are PREDECLARED (do not change after viewing results).

predict_action : U->A exact-action oxygen log-loss improvement, CI excludes 0 and mean>0.
predict_future : >=1 CONTINUOUS non-oxygen future target at some H with oxygen MSE-reduction
                 CI excludes 0 and mean>0.
recoverable    : leakage recoverable_fraction (masked oxygen info / visible oxygen info).
                 essentially_all >= 0.90 ; substantial >= 0.50.
Outcome: DOES_NOT_PREDICT_ACTION / DOES_NOT_PREDICT_FUTURE / NOT_HIDDEN_ENOUGH /
         PARTIALLY_OBSERVED_BUT_QUALIFIED / QUALIFIED.
"""
import os, json, glob, argparse

ESS_ALL, SUBSTANTIAL = 0.90, 0.50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="artifacts/seaquest/oxygen_4frame")
    args = ap.parse_args()
    leak = json.load(open(f"{args.base}/leakage/metrics.json"))
    act = json.load(open(f"{args.base}/oxygen_to_action/metrics.json"))
    fut = {os.path.basename(p): json.load(open(p)) for p in glob.glob(f"{args.base}/oxygen_to_future/metrics_H*.json")}

    # U -> A
    ua = act["exact_action"]["oxygen_logloss_improvement"]
    predict_action = bool(ua["mean"] > 0 and ua["ci_excludes_0"])

    # U -> future (continuous, non-oxygen)
    future_hits = []
    for hk, m in fut.items():
        for tname, d in (m.get("continuous") or {}).items():
            imp = d["oxygen_improvement_mse_reduction"]
            if imp["mean"] > 0 and imp["ci_excludes_0"]:
                future_hits.append({"horizon": m["H"], "target": tname,
                                    "mse_reduction": imp["mean"], "ci95": imp["ci95"]})
    predict_future = len(future_hits) > 0

    # leakage recoverable fraction (prefer R2; fall back to accuracy)
    rec_r2 = leak["interpretation"].get("recoverable_fraction_r2")
    rec_acc = leak["interpretation"].get("recoverable_fraction_acc")
    recoverable = rec_r2 if rec_r2 is not None else rec_acc

    if not predict_action:
        outcome = "OXYGEN_DOES_NOT_PREDICT_ACTION"
    elif not predict_future:
        outcome = "OXYGEN_DOES_NOT_PREDICT_FUTURE"
    elif recoverable is not None and recoverable >= ESS_ALL:
        outcome = "OXYGEN_NOT_HIDDEN_ENOUGH"
    elif recoverable is not None and recoverable >= SUBSTANTIAL:
        outcome = "OXYGEN_PARTIALLY_OBSERVED_BUT_QUALIFIED"
    else:
        outcome = "OXYGEN_QUALIFIED"

    qualifies_for_oracle = outcome in ("OXYGEN_QUALIFIED", "OXYGEN_PARTIALLY_OBSERVED_BUT_QUALIFIED")
    report = {
        "outcome": outcome, "qualifies_for_oracle": qualifies_for_oracle,
        "predeclared_thresholds": {"essentially_all": ESS_ALL, "substantial": SUBSTANTIAL},
        "leakage": {"masked_r2": leak["masked"]["regression"]["r2"],
                    "visible_r2": leak["visible"]["regression"]["r2"],
                    "masked_acc": leak["masked"]["classification"]["accuracy"],
                    "visible_acc": leak["visible"]["classification"]["accuracy"],
                    "baseline_acc": leak["trivial_baseline"]["clf_acc"],
                    "recoverable_fraction_r2": rec_r2, "recoverable_fraction_acc": rec_acc},
        "U_to_A": {"predict_action": predict_action, "exact_action_logloss_improvement": ua},
        "U_to_future": {"predict_future": predict_future, "significant_targets": future_hits,
                        "note": "termination_before_H and refill_before_H are degenerate in raw_hf "
                                "(done only at episode end; oxygen never jumps >5/step) -> binary "
                                "future events unavailable; evidence rests on continuous targets."},
    }
    json.dump(report, open(f"{args.base}/oxygen_qualification.json", "w"), indent=2)
    print(json.dumps({"outcome": outcome, "predict_action": predict_action,
                      "predict_future": predict_future, "recoverable_fraction": recoverable,
                      "qualifies_for_oracle": qualifies_for_oracle}, indent=2))
    print(f"WROTE {args.base}/oxygen_qualification.json")


if __name__ == "__main__":
    main()
