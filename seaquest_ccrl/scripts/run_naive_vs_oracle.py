"""End-to-end Level-2 deliverable: train naive + oracle critics, evaluate both,
print the confounding gap.

    success_rate(oracle) - success_rate(naive) = confounding gap

gap > 0  => the hidden oxygen confounder is hurting the naive learner.
gap ~ 0  => confounding too weak / critic insensitive (check THETA, Level-1
            dose-response).

Both critics use IDENTICAL architecture + hyperparameters; the ONLY difference
is masked (naive) vs unmasked (oracle) frames (acceptance check E).

CPU note: torch here is CPU-only, so the skill's 100K-step target is slow.
--steps defaults to a runnable value; raise it (and re-run) for the faithful
budget. Training is the bottleneck, not eval.
"""
import argparse
import json
import os

import torch

from seaquest_ccrl.training.config import TrainConfig
from seaquest_ccrl.training.train_critic import train, load_critic
from seaquest_ccrl.evaluation.evaluate import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50000, help="training steps per critic")
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=600, help="env steps per eval episode")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 => greedy argmax")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0=auto)")
    ap.add_argument("--seed", type=int, default=0,
                    help="experiment seed; varies net init, batch sampling, and eval env/targets")
    ap.add_argument("--skip-train", action="store_true",
                    help="reuse existing checkpoints (still re-runs eval)")
    ap.add_argument("--ckpt-dir", default=None,
                    help="base dir for checkpoints (per-seed subdir appended); "
                         "point at Google Drive to persist across Colab sessions")
    ap.add_argument("--out", default=None,
                    help="results JSON (default: seaquest_ccrl/figure/level2_seed{seed}.json)")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="if >0, run eval every N steps to build a success-vs-step curve "
                         "(saved as history_{tag}.json next to the checkpoint)")
    ap.add_argument("--curve-eval-episodes", type=int, default=30,
                    help="episodes per in-training eval point (keep small; the final "
                         "headline eval still uses --eval-episodes)")
    ap.add_argument("--game", default="seaquest", help="seaquest | mspacman")
    ap.add_argument("--goal-radius", type=float, default=None,
                    help="in-batch goals within this px radius count as positives "
                         "(goal-collision fix). Default: game eval radius (eps); 0 disables.")
    args = ap.parse_args()

    from seaquest_ccrl.games import get_game
    game = get_game(args.game)
    goal_radius = game.eps if args.goal_radius is None else args.goal_radius

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Per-seed config: distinct checkpoint dir so seeds don't clobber each other.
    # Goal-normalisation box + action count come from the game spec.
    ckpt_base = args.ckpt_dir or f"{args.game}_ccrl/checkpoints"
    gx0, gx1, gy0, gy1 = game.goal_box
    cfg = TrainConfig(steps=args.steps, seed=args.seed, nb_actions=game.nb_actions,
                      goal_x_lo=gx0, goal_x_hi=gx1, goal_y_lo=gy0, goal_y_hi=gy1,
                      goal_radius=goal_radius,
                      ckpt_dir=os.path.join(ckpt_base, f"seed{args.seed}"))
    out_path = args.out or f"{args.game}_ccrl/figure/level2_seed{args.seed}.json"

    results = {}
    for oracle in (False, True):
        tag = "oracle" if oracle else "naive"
        ckpt = os.path.join(cfg.ckpt_dir, f"critic_{tag}.pt")
        if args.skip_train and os.path.exists(ckpt):
            print(f"[{tag}] reusing checkpoint {ckpt}")
            critic, ccfg, _ = load_critic(ckpt, device)
        else:
            train(oracle=oracle, cfg=cfg, game=game, device=device,
                  eval_every=args.eval_every,
                  eval_episodes=args.curve_eval_episodes,
                  eval_max_steps=args.max_steps)
            critic, ccfg, _ = load_critic(ckpt, device)
        res = evaluate(critic, ccfg, game, oracle, n_episodes=args.eval_episodes,
                       max_steps=args.max_steps, device=device,
                       temperature=args.temperature, seed=args.seed)
        results[tag] = res

    gap = results["oracle"]["success_rate"] - results["naive"]["success_rate"]
    summary = {
        "seed": args.seed,
        "steps": args.steps,
        "eval_episodes": args.eval_episodes,
        "naive_success_rate": results["naive"]["success_rate"],
        "oracle_success_rate": results["oracle"]["success_rate"],
        "confounding_gap": gap,
        "details": results,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 56)
    print(f"  LEVEL-2 CONTRASTIVE RL  —  naive vs oracle  (seed {args.seed})")
    print("=" * 56)
    print(f"  naive  (masked)   success rate : {results['naive']['success_rate']:.3f}")
    print(f"  oracle (unmasked) success rate : {results['oracle']['success_rate']:.3f}")
    print(f"  confounding gap (oracle-naive) : {gap:+.3f}")
    print("=" * 56)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
