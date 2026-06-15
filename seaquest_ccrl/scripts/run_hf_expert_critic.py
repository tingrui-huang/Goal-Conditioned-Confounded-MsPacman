"""Seaquest HF-expert retry — thin runner around the FROZEN original critic training.

The ONLY scientific change vs the original action-blind run is the dataset source: the contrastive
critic is trained on `seaquest_ccrl/data/raw_hf` (HF CleanRL expert, O-Sampled) instead of the old
scripted-policy `data/raw`. Everything else (critic, sampler, loss, encoders, goal rule, gamma,
frame size, repr dim, batch, optimizer) is the unchanged committed pipeline.

This duplicates NO training logic — it builds the exact `TrainConfig` that the committed
`run_naive_vs_oracle.py` builds for Seaquest and calls the frozen `train()`.

Config recovery: the seed-0 checkpoint on disk (seaquest_ccrl/checkpoints/seed0) predates the
current committed pipeline — it was trained with nb_actions=9 (the scripted policy's reduced action
set) and an older goal box, which is INCOMPATIBLE with the 18-action HF expert. Per the task's
fallback clause we therefore use the committed defaults wired exactly as run_naive_vs_oracle does:
nb_actions=18, goal_radius=game.eps, goal box=game.goal_box, frame_stack=game.frame_stack, and the
shared steps/seed/batch/lr/gamma/frame_size/repr_dim (which DO match the old run). Seed 0 only.
"""
import os
import argparse

import torch

from seaquest_ccrl.games import get_game
from seaquest_ccrl.training.config import TrainConfig
from seaquest_ccrl.training.train_critic import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="seaquest_ccrl/data/raw_hf", help="HF dataset root")
    ap.add_argument("--oracle", action="store_true",
                    help="unmasked view; default is the masked (naive) learner per the original definition")
    ap.add_argument("--steps", type=int, default=50000, help="recovered previous action-blind budget")
    ap.add_argument("--seed", type=int, default=0, help="seed 0 only")
    ap.add_argument("--ckpt-dir", default="seaquest_ccrl/checkpoints/hf_seed0")
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    game = get_game("seaquest")
    gx0, gx1, gy0, gy1 = game.goal_box
    cfg = TrainConfig(steps=args.steps, seed=args.seed, nb_actions=game.nb_actions,
                      goal_x_lo=gx0, goal_x_hi=gx1, goal_y_lo=gy0, goal_y_hi=gy1,
                      goal_radius=game.eps, frame_stack=game.frame_stack,
                      ckpt_dir=args.ckpt_dir)
    tag = "oracle" if args.oracle else "naive"
    print(f"[HF-expert critic] view={tag} root={args.root} device={device}")
    print(f"  cfg: steps={cfg.steps} seed={cfg.seed} nb_actions={cfg.nb_actions} batch={cfg.batch_size} "
          f"lr={cfg.lr} gamma={cfg.gamma} frame_size={cfg.frame_size} frame_stack={cfg.frame_stack} "
          f"repr_dim={cfg.repr_dim} goal_radius={cfg.goal_radius} goal_box=({gx0},{gx1},{gy0},{gy1})")
    path = train(oracle=args.oracle, cfg=cfg, game=game, root=args.root, device=device, verbose=True)
    print(f"[HF-expert critic] DONE -> {path}")


if __name__ == "__main__":
    main()
