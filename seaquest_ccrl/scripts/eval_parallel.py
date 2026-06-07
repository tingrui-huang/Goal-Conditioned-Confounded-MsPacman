"""Parallel goal-reaching eval of ALREADY-TRAINED checkpoints (no retraining).

Eval is CPU-bound (the OCAtari env steps serially), so a high-episode-count run
-- e.g. 500 episodes to push the per-rate binomial SE down to ~0.02 -- is split
across CPU worker processes. Loads existing critic_{naive,oracle}.pt; training is
NOT repeated.

Reports, per seed: naive/oracle success rate with binomial SE, and the gap. Then
across seeds: mean +/- standard error (the figure-grade number). 500-ep removes
WITHIN-seed measurement noise; the across-seed SE still reflects training
variance, so keep >= 3 seeds.

Run:
  python -m seaquest_ccrl.scripts.eval_parallel \
      --ckpt-base "$WORK/ckpt" --seeds 0 1 2 \
      --episodes 500 --max-steps 600 --workers 0   # 0 => all CPU cores
"""
import os
import json
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def _worker(payload):
    """One chunk of episodes in its own process (CPU-only to avoid CUDA+fork)."""
    ckpt, oracle, n_eps, max_steps, seed, game_name = payload
    import torch
    torch.set_num_threads(1)                       # avoid thread oversubscription
    from seaquest_ccrl.training.train_critic import load_critic
    from seaquest_ccrl.evaluation.evaluate import evaluate
    from seaquest_ccrl.games import get_game
    critic, cfg, _ = load_critic(ckpt, device="cpu")
    res = evaluate(critic, cfg, get_game(game_name), oracle, n_episodes=n_eps,
                   max_steps=max_steps, device="cpu", seed=seed, verbose=False)
    # return raw counts so chunks can be summed exactly
    return res["successes"], res["n_episodes"], res["mean_min_dist"] * res["n_episodes"]


def eval_checkpoint(ckpt, oracle, episodes, max_steps, workers, game_name="seaquest"):
    # split episodes across workers; give each a distinct seed so targets differ
    chunks = [episodes // workers + (1 if i < episodes % workers else 0)
              for i in range(workers)]
    payloads = [(ckpt, oracle, c, max_steps, 1000 + i, game_name)
                for i, c in enumerate(chunks) if c > 0]
    succ = tot = dist_w = 0
    if workers == 1:
        for p in payloads:
            s, n, dw = _worker(p); succ += s; tot += n; dist_w += dw
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for s, n, dw in ex.map(_worker, payloads):
                succ += s; tot += n; dist_w += dw
    p = succ / tot
    return {"success_rate": p, "successes": succ, "n_episodes": tot,
            "binom_se": float(np.sqrt(p * (1 - p) / tot)),
            "mean_min_dist": dist_w / tot}


def _mean_se(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float("nan")
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-base", required=True,
                    help="dir holding seed{S}/critic_{naive,oracle}.pt (e.g. $WORK/ckpt)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--workers", type=int, default=0, help="0 => os.cpu_count()")
    ap.add_argument("--game", default="seaquest", help="seaquest | mspacman")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    W = args.workers or (os.cpu_count() or 2)
    print(f"parallel eval: {args.episodes} eps/critic, max_steps={args.max_steps}, workers={W}")

    per_seed = []
    for s in args.seeds:
        row = {"seed": s}
        for oracle in (False, True):
            tag = "oracle" if oracle else "naive"
            ckpt = os.path.join(args.ckpt_base, f"seed{s}", f"critic_{tag}.pt")
            if not os.path.exists(ckpt):
                print(f"  seed{s} {tag}: MISSING ({ckpt})")
                row[tag] = None
                continue
            r = eval_checkpoint(ckpt, oracle, args.episodes, args.max_steps, W, args.game)
            row[tag] = r
            print(f"  seed{s} {tag:6s}: {r['success_rate']:.3f} +/- {r['binom_se']:.3f} "
                  f"({r['successes']}/{r['n_episodes']})")
        if row.get("naive") and row.get("oracle"):
            row["gap"] = row["oracle"]["success_rate"] - row["naive"]["success_rate"]
            print(f"  seed{s} GAP = {row['gap']:+.3f}")
        per_seed.append(row)

    naive = [r["naive"]["success_rate"] for r in per_seed if r.get("naive")]
    oracle = [r["oracle"]["success_rate"] for r in per_seed if r.get("oracle")]
    gaps = [r["gap"] for r in per_seed if "gap" in r]
    nm, nse = _mean_se(naive); om, ose = _mean_se(oracle); gm, gse = _mean_se(gaps)

    print("\n" + "=" * 56)
    print(f"  ACROSS {len(gaps)} SEEDS ({args.episodes} eps each)")
    print("=" * 56)
    print(f"  naive  : {nm:.3f} +/- {nse:.3f}")
    print(f"  oracle : {om:.3f} +/- {ose:.3f}")
    print(f"  gap    : {gm:.3f} +/- {gse:.3f}" +
          (f"   ({gm/gse:.1f} sigma)" if gse and gse > 0 else ""))
    print("=" * 56)

    summary = {"episodes": args.episodes, "max_steps": args.max_steps,
               "per_seed": per_seed,
               "naive_mean_se": [nm, nse], "oracle_mean_se": [om, ose],
               "gap_mean_se": [gm, gse]}
    out = args.out or os.path.join(args.ckpt_base, os.pardir, "results", "eval500_summary.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
