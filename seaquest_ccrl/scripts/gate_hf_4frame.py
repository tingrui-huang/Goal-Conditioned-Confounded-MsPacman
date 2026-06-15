"""Phase-1 four-frame action gate + persistence (committed; no notebook inline logic).

Reads the action-use diagnostic + training history, applies the gate, refuses any
MsPacman/ambiguous checkpoint, collects checkpoint + resolved config + history + diag +
stack audit + hashes into the output dir, and makes a downloadable ZIP.

Gate (PASS = all): shuffle delta>0, 95% CI lower bound>0, trained action spread > random
spread, training optimizes normally. Outcome FOUR_FRAME_CRITIC_USES_ACTION else
STOP_4FRAME_CRITIC_DOES_NOT_USE_ACTION.
"""
import os, argparse, json, glob, hashlib, shutil
import numpy as np
import torch


def _losses(h):
    """Extract the scalar training-loss series. train() writes loss as a list of
    [step, loss, acc] triples; also tolerate dict-rows or bare scalars."""
    raw = None
    if isinstance(h, dict):
        for k in ("loss", "train_loss", "losses"):
            if isinstance(h.get(k), list):
                raw = h[k]; break
    elif isinstance(h, list):
        raw = h
    if raw is None:
        return []
    out = []
    for x in raw:
        if isinstance(x, (list, tuple)) and len(x) >= 2:
            out.append(float(x[1]))          # [step, loss, acc] -> loss
        elif isinstance(x, dict) and x.get("loss") is not None:
            out.append(float(x["loss"]))
        elif isinstance(x, (int, float)):
            out.append(float(x))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="seaquest_ccrl/checkpoints/hf_4frame_seed0/critic_naive.pt")
    ap.add_argument("--diag", default="artifacts/seaquest/oxygen_4frame/naive_critic/action_use_diag.json")
    ap.add_argument("--ckpt-dir", default="seaquest_ccrl/checkpoints/hf_4frame_seed0")
    ap.add_argument("--out-dir", default="artifacts/seaquest/oxygen_4frame/naive_critic")
    ap.add_argument("--zip", default="seaquest_oxygen_4frame_phase1")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # checkpoint hard guards
    low = args.ckpt.lower()
    assert "mspacman" not in low and "pacman" not in low, f"refusing MsPacman checkpoint path: {args.ckpt}"
    sd = torch.load(args.ckpt, map_location="cpu")
    rc = sd.get("cfg", {})
    in_ch = sd["state_dict"]["sa_encoder.conv.0.weight"].shape[1]
    assert in_ch == 12, f"first-conv in_channels must be 12, got {in_ch}"
    assert rc.get("frame_stack") == 4, f"cfg.frame_stack must be 4, got {rc.get('frame_stack')}"
    assert rc.get("nb_actions") == 18, f"cfg.nb_actions must be 18, got {rc.get('nb_actions')}"
    assert sd.get("oracle") is False, "naive checkpoint must have oracle=False"
    print("[guard] checkpoint OK: in_channels=12 frame_stack=4 nb_actions=18 oracle=False")

    diag = json.load(open(args.diag))
    d = diag["trained"]["delta_shuffle"]
    tr_spread = diag["trained"]["all_action_spread"]["mean_score_std_across_actions"]
    ct_spread = diag["random_init_control"]["all_action_spread"]["mean_score_std_across_actions"]
    hist_path = os.path.join(args.ckpt_dir, "history_naive.json")
    hist = json.load(open(hist_path)) if os.path.exists(hist_path) else {}
    ls = [x for x in _losses(hist) if isinstance(x, (int, float))]
    optimizes = bool(ls) and bool(np.isfinite(ls).all()) and (ls[-1] < ls[0])

    gate = {
        "shuffle_delta_gt_0": bool(d["mean"] > 0),
        "ci_lower_bound_gt_0": bool(d["ci_lower_above_0"]),
        "trained_spread_gt_control": bool(tr_spread > ct_spread),
        "optimizes_normally": optimizes,
    }
    PASS = all(gate.values())
    outcome = "FOUR_FRAME_CRITIC_USES_ACTION" if PASS else "STOP_4FRAME_CRITIC_DOES_NOT_USE_ACTION"
    result = {"gate": gate, "pass": PASS, "outcome": outcome,
              "delta_shuffle": d, "trained_spread": tr_spread, "control_spread": ct_spread,
              "optimizes_normally": optimizes,
              "loss_first": ls[0] if ls else None, "loss_last": ls[-1] if ls else None,
              "demonstrated_action_top1": diag["trained"]["all_action_spread"]["demonstrated_action_top1_rate"],
              "demonstrated_action_top1_chance": diag["trained"]["all_action_spread"]["top1_chance"]}
    json.dump(result, open(f"{args.out_dir}/action_use_gate.json", "w"), indent=2)
    json.dump(rc, open(f"{args.out_dir}/resolved_config.json", "w"), indent=2)

    # collect artifacts
    for f in [args.ckpt, hist_path]:
        if os.path.exists(f):
            shutil.copy(f, args.out_dir)
    if os.path.exists(f"{args.out_dir}/history_naive.json"):
        shutil.copy(f"{args.out_dir}/history_naive.json", f"{args.out_dir}/history.json")

    def sha(p):
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    hashes = {os.path.basename(p): sha(p) for p in glob.glob(args.out_dir + "/*") if os.path.isfile(p)}
    json.dump({"frame_stack": 4, "nb_actions": 18, "first_conv_in_channels": 12, "oracle": False,
               "outcome": outcome, "files_sha256": hashes},
              open(f"{args.out_dir}/checkpoint_hash.json", "w"), indent=2)
    shutil.make_archive(args.zip, "zip", args.out_dir)

    print(json.dumps(gate, indent=2))
    print(f"GATE: {'PASS' if PASS else 'FAIL'} -> {outcome}")
    print(f"  delta_shuffle={d['mean']:+.4f} CI{[round(x,4) for x in d['ci95']]} | "
          f"trained_spread={tr_spread:.4f} > control_spread={ct_spread:.4f} = {tr_spread>ct_spread} | "
          f"optimizes={optimizes}")
    print(f"WROTE {args.out_dir}/action_use_gate.json + {args.zip}.zip")


if __name__ == "__main__":
    main()
