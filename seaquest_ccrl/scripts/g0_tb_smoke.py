"""TensorBoard smoke gate — run a short (1k-2k step) masked training with TB logging and
verify, BEFORE the full 50k run, that:
  1. TensorBoard events update (all 5 expected scalar tags present);
  2. all logged values are finite;
  3. checkpoint save + reload works;
  4. diag/action_shuffle_delta runs (logged + finite);
  5. output files persist locally (checkpoint, history, event file, provenance).

Outcome TB_SMOKE_PASS / TB_SMOKE_FAIL. The notebook adds the Drive-persistence check on top.
"""
import os, json, glob, argparse
import numpy as np

from seaquest_ccrl.games import get_game
from seaquest_ccrl.training.config import TrainConfig
from seaquest_ccrl.training.train_critic import train, load_critic
from seaquest_ccrl.scripts.run_hf_4frame import _write_provenance

EXPECTED = ["train/loss", "train/diag_acc", "train/logit_gap", "train/grad_norm",
            "diag/action_shuffle_delta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="seaquest_ccrl/data/raw_hf")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--shuffle-every", type=int, default=500)
    ap.add_argument("--ckpt-dir", default="artifacts/_tb_smoke/ckpt")
    ap.add_argument("--tb-logdir", default="artifacts/_tb_smoke/tb")
    ap.add_argument("--out", default="artifacts/_tb_smoke/tb_smoke_gate.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.ckpt_dir, exist_ok=True); os.makedirs(args.tb_logdir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    game = get_game("seaquest")
    gx0, gx1, gy0, gy1 = game.goal_box
    cfg = TrainConfig(steps=args.steps, seed=0, nb_actions=game.nb_actions, frame_stack=4,
                      goal_x_lo=gx0, goal_x_hi=gx1, goal_y_lo=gy0, goal_y_hi=gy1,
                      goal_radius=game.eps, ckpt_dir=args.ckpt_dir)
    path = train(oracle=False, cfg=cfg, game=game, root=args.root, device=device, verbose=True,
                 tb_logdir=args.tb_logdir, tb_shuffle_every=args.shuffle_every)
    # provenance (same writer as the full run)
    class _A: pass
    a = _A(); a.root = args.root; a.oracle = False; a.ckpt_dir = args.ckpt_dir; a.tb_logdir = args.tb_logdir
    _write_provenance(a, cfg, path)

    checks, fails = {}, []
    def rec(k, ok, d):
        checks[k] = {"pass": bool(ok), "detail": d}
        if not ok: fails.append(k)

    # 1+2+4: read events
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(args.tb_logdir); ea.Reload()
    tags = ea.Tags().get("scalars", [])
    tag_stats = {}
    for t in EXPECTED:
        vals = [s.value for s in ea.Scalars(t)] if t in tags else []
        tag_stats[t] = {"n": len(vals), "finite": bool(vals and np.all(np.isfinite(vals))),
                        "last": (float(vals[-1]) if vals else None)}
    rec("1_events_update_all_tags", all(tag_stats[t]["n"] > 0 for t in EXPECTED),
        {t: tag_stats[t]["n"] for t in EXPECTED})
    rec("2_all_values_finite", all(tag_stats[t]["finite"] for t in EXPECTED if tag_stats[t]["n"]),
        {t: tag_stats[t]["finite"] for t in EXPECTED})
    rec("4_action_shuffle_delta_runs", tag_stats["diag/action_shuffle_delta"]["n"] > 0
        and tag_stats["diag/action_shuffle_delta"]["finite"],
        tag_stats["diag/action_shuffle_delta"])

    # 3: checkpoint save + reload
    try:
        critic, ccfg, oracle = load_critic(path, device)
        reload_ok = (oracle is False and getattr(ccfg, "frame_stack", 1) == 4)
    except Exception as e:
        reload_ok = False; print("reload error:", e)
    rec("3_checkpoint_save_reload", reload_ok, {"path": path, "oracle_false_fs4": reload_ok})

    # 5: output files exist locally
    ev = glob.glob(f"{args.tb_logdir}/events.out.tfevents.*")
    files = {"checkpoint": os.path.exists(path),
             "history": os.path.exists(os.path.join(args.ckpt_dir, "history_naive.json")),
             "event_file": len(ev) > 0,
             "provenance": os.path.exists(os.path.join(args.ckpt_dir, "run_provenance.json"))}
    rec("5_output_files_persist_local", all(files.values()), files)

    outcome = "TB_SMOKE_PASS" if not fails else "TB_SMOKE_FAIL"
    report = {"outcome": outcome, "failed": fails, "expected_tags": EXPECTED,
              "tag_stats": tag_stats, "checks": checks, "device": device, "steps": args.steps}
    json.dump(report, open(args.out, "w"), indent=2)
    print(json.dumps({"outcome": outcome, "failed": fails,
                      "tag_counts": {t: tag_stats[t]["n"] for t in EXPECTED}}, indent=2))
    print(f"WROTE {args.out}")
    if fails:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
