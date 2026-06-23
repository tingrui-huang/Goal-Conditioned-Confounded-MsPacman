"""Log the ACTUAL closed-loop comparison metrics to TensorBoard, separately from training.

train/diag_acc shows whether the contrastive objective is learning; these eval/* scalars show
whether that learning translates into goal-reaching CONTROL. Reads each critic's closed-loop
evaluation dir (aggregate_metrics.json + raw_rollouts.npz) and logs:
  eval/full_view_success_rate, eval/masked_success_rate,
  eval/full_view_mean_distance, eval/masked_mean_distance   (mean final distance-to-goal).
"""
import os, json, argparse
import numpy as np

FV = "artifacts/seaquest/goal_control/full_view/evaluation"
MK = "artifacts/seaquest/goal_control/masked/evaluation"


def critic_metrics(d):
    """(success_rate aggregate_by_H, mean final distance) for the critic policy, or None."""
    agg_p = os.path.join(d, "aggregate_metrics.json"); raw_p = os.path.join(d, "raw_rollouts.npz")
    if not (os.path.exists(agg_p) and os.path.exists(raw_p)):
        return None
    succ = json.load(open(agg_p))["aggregate"]["critic"]["mean"]
    raw = np.load(raw_p, allow_pickle=True)
    pol = raw["policy"].astype(str); m = pol == "critic"
    return {"success_rate": float(succ),
            "mean_final_distance": float(raw["final_dist"][m].mean()),
            "mean_min_distance": float(raw["min_dist"][m].mean()), "n": int(m.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-view-dir", default=FV)
    ap.add_argument("--masked-dir", default=MK)
    ap.add_argument("--tb-logdir", default="artifacts/seaquest/goal_control/eval_tb")
    ap.add_argument("--out", default="artifacts/seaquest/goal_control/eval_comparison.json")
    args = ap.parse_args()
    os.makedirs(args.tb_logdir, exist_ok=True); os.makedirs(os.path.dirname(args.out), exist_ok=True)

    fv = critic_metrics(args.full_view_dir); mk = critic_metrics(args.masked_dir)
    from torch.utils.tensorboard import SummaryWriter
    w = SummaryWriter(args.tb_logdir)
    rows = []
    for name, m in [("full_view", fv), ("masked", mk)]:
        if m is None:
            print(f"[{name}] evaluation not found -> skipped"); continue
        w.add_scalar(f"eval/{name}_success_rate", m["success_rate"], 0)
        w.add_scalar(f"eval/{name}_mean_distance", m["mean_final_distance"], 0)
        rows.append((name, m))
    txt = "| critic | success_rate | mean_final_dist | mean_min_dist | n |\n|---|---|---|---|---|\n"
    txt += "\n".join(f"| {n} | {m['success_rate']:.3f} | {m['mean_final_distance']:.2f} | "
                     f"{m['mean_min_distance']:.2f} | {m['n']} |" for n, m in rows)
    w.add_text("eval/closed_loop_comparison", txt, 0)
    w.flush(); w.close()
    cmp = {"full_view": fv, "masked": mk,
           "note": "eval/* are goal-reaching control metrics; NOT the same as train/diag_acc."}
    if fv and mk:
        cmp["masked_minus_full_view_success"] = mk["success_rate"] - fv["success_rate"]
    json.dump(cmp, open(args.out, "w"), indent=2)
    print(json.dumps(cmp, indent=2))
    print(f"WROTE {args.out} + TB scalars at {args.tb_logdir}")


if __name__ == "__main__":
    main()
