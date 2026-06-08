"""Discriminate "in-batch goal collision" vs "critic can't tell goals apart".

Low hard diag-acc (exact argmax==i) can mean two very different things:
  (a) goal COLLISION: many in-batch negative goals sit within a few px of the
      positive (Pac-Man positions cluster), so even a perfect critic can't hit the
      exact index -> BCE has irreducible residual, loss plateaus high, diag-acc caps
      low. The critic is actually fine (picks a near-neighbour goal). => metric artifact.
  (b) the critic genuinely cannot separate goals => deeper representation/training issue.

Test on an EXISTING checkpoint (no retrain). Reports:
  - hard diag-acc      : exact argmax_j logits[i,j] == i
  - soft diag-acc      : picked goal within `radius` px of the true goal
  - collision count    : mean # of OTHER in-batch goals within `radius` px of the positive
  - ceiling            : mean 1/(1+collisions) = best possible hard diag-acc given collisions

soft >> hard  AND  hard ~ ceiling  =>  collision confirmed (case a): undertrain + metric,
   fix by discretizing goals to ~radius-px cells (+ train longer / more data).
soft also low =>  case (b): the critic really can't distinguish goals.

Run:
  python -m seaquest_ccrl.scripts.soft_diag_acc --ckpt PATH --game mspacman
"""
import argparse

import numpy as np
import torch

from seaquest_ccrl.training.train_critic import load_critic
from seaquest_ccrl.training.dataset_sampler import HindsightSampler
from seaquest_ccrl.games import get_game


def run(ckpt, game_name, batches=30, B=256, radius=8.0, device="cpu"):
    g = get_game(game_name)
    critic, cfg, oracle = load_critic(ckpt, device)
    critic.eval()
    rng = np.random.default_rng(0)
    samp = HindsightSampler(g, oracle=oracle, cfg=cfg, device=device, rng=rng)

    lo = np.array([cfg.goal_x_lo, cfg.goal_y_lo], dtype=np.float64)
    span = np.array([cfg.goal_x_hi - cfg.goal_x_lo, cfg.goal_y_hi - cfg.goal_y_lo],
                    dtype=np.float64)

    hard = soft = n = 0
    collisions = []      # per-anchor: # other in-batch goals within radius of positive
    pick_dists = []      # per-anchor: distance from picked goal to true goal
    with torch.no_grad():
        for _ in range(batches):
            frames, actions, goals = samp.sample(B)
            logits = critic(frames, actions, goals)              # (B,B)
            am = logits.argmax(dim=1).cpu().numpy()
            graw = goals.cpu().numpy().astype(np.float64) * span + lo   # (B,2) px
            # pairwise distances between all goals in the batch
            diff = graw[:, None, :] - graw[None, :, :]
            D = np.sqrt((diff ** 2).sum(-1))                     # (B,B)
            np.fill_diagonal(D, np.inf)
            for i in range(B):
                if am[i] == i:
                    hard += 1
                dpick = float(np.hypot(*(graw[am[i]] - graw[i])))
                pick_dists.append(dpick)
                if dpick <= radius:
                    soft += 1
                collisions.append(int((D[i] <= radius).sum()))
                n += 1

    collisions = np.array(collisions)
    ceiling = float(np.mean(1.0 / (1.0 + collisions)))
    print(f"ckpt = {ckpt}")
    print(f"game = {game_name}  oracle = {oracle}  B = {B}  batches = {batches}")
    print(f"  hard diag-acc (exact argmax==i)            : {hard/n:.3f}")
    print(f"  soft diag-acc (picked goal <= {radius:.0f}px of true): {soft/n:.3f}")
    print(f"  mean in-batch collisions (<= {radius:.0f}px of positive): {collisions.mean():.2f}")
    print(f"  collision ceiling (best possible hard acc)  : {ceiling:.3f}")
    print(f"  median dist picked->true goal: {np.median(pick_dists):.1f}px  "
          f"(p25 {np.percentile(pick_dists,25):.1f}, p75 {np.percentile(pick_dists,75):.1f})")
    print("  --")
    if soft / n >= 2.5 * max(hard / n, 1e-6) and hard / n <= 1.8 * ceiling:
        print("  => COLLISION CONFIRMED (case a): critic picks near-neighbour goals; low hard")
        print("     diag-acc is a metric artifact. Fix: discretize goals to ~radius-px cells,")
        print("     train longer + more data. Objective is mildly ill-posed, not broken.")
    elif soft / n < 0.15:
        print("  => case (b): critic genuinely cannot separate goals -> deeper repr/training issue.")
    else:
        print("  => mixed/inconclusive; inspect numbers above.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--game", default="mspacman")
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--radius", type=float, default=8.0)
    args = ap.parse_args()
    run(args.ckpt, args.game, args.batches, args.batch_size, args.radius)
