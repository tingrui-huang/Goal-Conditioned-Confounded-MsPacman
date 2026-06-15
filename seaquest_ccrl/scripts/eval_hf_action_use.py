"""Stage HF-expert retry — minimal ACTION-USE diagnostic (evaluation only; trains nothing).

Question: does the original Seaquest contrastive critic, trained on HF-expert data, actually USE the
action input? Loads the trained critic + the HF dataset, builds fixed reproducible evaluation tuples
(s_t, a_t, g_future) with the SAME geometric-hindsight rule as the frozen HindsightSampler, and:

  Test A (empirical marginal action shuffle): shuffle demonstrated actions across tuples keeping
    (s_t, g_future) fixed; delta_shuffle = mean(score_true - score_shuffled), trajectory bootstrap CI.
  Test B (all-action score spread): score all 18 actions for each (s_t, g_future); report score std
    across actions, top-minus-bottom, demonstrated-action mean rank, top-1 / top-3 rates.
  Negative control: the SAME diagnostic on an untrained randomly-initialized critic (same arch).

f(s,a,g) = phi(s,a) . psi(g)  (sa_encoder . g_encoder). The critic/model is never modified.
"""
import os
import json
import argparse

import numpy as np
import torch

from seaquest_ccrl.games import get_game
from seaquest_ccrl.training.train_critic import load_critic
from seaquest_ccrl.training.dataset_sampler import HindsightSampler
from seaquest_ccrl.models.contrastive_critic import ContrastiveCritic


def build_fixed_tuples(sampler, n_tuples, seed):
    """Pre-generate reproducible (anchor_global, future_global, episode) indices via the frozen rule."""
    rng = np.random.default_rng(seed)
    ep = rng.integers(0, sampler.n_ep, size=n_tuples)
    t = rng.integers(0, sampler.lengths[ep])
    k = rng.geometric(sampler.p_geom, size=n_tuples)
    fut = np.minimum(t + k, sampler.lengths[ep] - 1)
    anchor = (sampler.offsets[ep] + t).astype(np.int64)
    future = (sampler.offsets[ep] + fut).astype(np.int64)
    return anchor, future, ep.astype(np.int64)


def _anchor_frames(sampler, anchor):
    gt = torch.as_tensor(anchor, device=sampler.device)
    if sampler.k_stack == 1:
        return sampler.frames.index_select(0, gt)                     # (N,H,W,3)
    idx = sampler.stack_idx.index_select(0, gt)
    st = sampler.frames[idx]; B, k, H, W, Cc = st.shape
    return st.permute(0, 2, 3, 1, 4).reshape(B, H, W, k * Cc)


@torch.no_grad()
def score_pairs(critic, frames, actions, goals_norm, device, bs=512):
    """f(s_i, a_i, g_i) for matched tuples -> (N,)."""
    out = []
    for s in range(0, frames.shape[0], bs):
        sl = slice(s, s + bs)
        sa = critic.sa_encoder(frames[sl], actions[sl])
        g = critic.g_encoder(goals_norm[sl])
        out.append((sa * g).sum(1).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def score_all_actions(critic, frames, goals_norm, nb_actions, device, bs=256):
    """f(s_i, a, g_i) for every action a -> (N, nb_actions)."""
    out = []
    for s in range(0, frames.shape[0], bs):
        sl = slice(s, s + bs); fr = frames[sl]; gg = goals_norm[sl]; B = fr.shape[0]
        g = critic.g_encoder(gg)                                       # (B,d)
        cols = []
        for a in range(nb_actions):
            acts = torch.full((B,), a, dtype=torch.long, device=device)
            sa = critic.sa_encoder(fr, acts)                          # (B,d)
            cols.append((sa * g).sum(1, keepdim=True))                # (B,1)
        out.append(torch.cat(cols, 1).cpu().numpy())                  # (B,A)
    return np.concatenate(out, 0)


def _boot_ci_by_traj(values, episodes, seed=0, n_boot=2000):
    """Trajectory-level bootstrap of mean(values): resample episodes, average within."""
    uniq = np.unique(episodes)
    by = {e: values[episodes == e] for e in uniq}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        means.append(np.concatenate([by[e] for e in pick]).mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(values.mean()), "ci95": [float(lo), float(hi)],
            "ci_lower_above_0": bool(lo > 0), "n_traj": int(len(uniq))}


def run_diagnostic(critic, sampler, anchor, future, ep, cfg, device, label):
    frames = _anchor_frames(sampler, anchor)
    actions = sampler.actions.index_select(0, torch.as_tensor(anchor, device=device))
    goals_norm = (sampler.goals.index_select(0, torch.as_tensor(future, device=device))
                  - sampler._goal_lo) / sampler._goal_span
    a_np = actions.cpu().numpy()
    nb = int(cfg.nb_actions)

    score_true = score_pairs(critic, frames, actions, goals_norm, device)
    # Test A: shuffle demonstrated actions across tuples (fixed permutation), (s,g) fixed
    perm = np.random.default_rng(12345).permutation(len(a_np))
    a_shuf = actions.index_select(0, torch.as_tensor(perm, device=device))
    score_shuf = score_pairs(critic, frames, a_shuf, goals_norm, device)
    delta = score_true - score_shuf
    shuffle = _boot_ci_by_traj(delta, ep, seed=0)

    # Test B: all-action spread
    sa_all = score_all_actions(critic, frames, goals_norm, nb, device)         # (N,nb)
    std_across = sa_all.std(1)
    top_minus_bottom = sa_all.max(1) - sa_all.min(1)
    order = (-sa_all).argsort(1)                                                # best..worst action ids
    rank = np.array([int(np.where(order[i] == a_np[i])[0][0]) + 1 for i in range(len(a_np))])  # 1=best
    top1 = (sa_all.argmax(1) == a_np).astype(float)
    top3 = np.array([a_np[i] in order[i, :3] for i in range(len(a_np))], float)
    return {
        "label": label, "n_tuples": int(len(a_np)),
        "delta_shuffle": shuffle,
        "all_action_spread": {
            "mean_score_std_across_actions": float(std_across.mean()),
            "mean_top_minus_bottom": float(top_minus_bottom.mean()),
            "demonstrated_action_mean_rank": float(rank.mean()),
            "demonstrated_action_rank_baseline_chance": (nb + 1) / 2.0,
            "demonstrated_action_top1_rate": float(top1.mean()),
            "demonstrated_action_top3_rate": float(top3.mean()),
            "top1_chance": 1.0 / nb, "top3_chance": 3.0 / nb,
        },
        "score_true_mean": float(score_true.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="seaquest_ccrl/checkpoints/hf_seed0/critic_naive.pt")
    ap.add_argument("--root", default="seaquest_ccrl/data/raw_hf")
    ap.add_argument("--n-tuples", type=int, default=4096)
    ap.add_argument("--tuple-seed", type=int, default=777)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="artifacts/seaquest/hf_original_critic/action_use_diag.json")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    critic, cfg, oracle = load_critic(args.ckpt, device)
    game = get_game("seaquest")
    sampler = HindsightSampler(game, oracle=oracle, cfg=cfg, device=device,
                               rng=np.random.default_rng(cfg.seed), root=args.root)
    anchor, future, ep = build_fixed_tuples(sampler, args.n_tuples, args.tuple_seed)

    trained = run_diagnostic(critic, sampler, anchor, future, ep, cfg, device, "trained")
    # negative control: untrained randomly-initialized critic, same arch
    torch.manual_seed(0)
    rand = ContrastiveCritic(cfg.repr_dim, cfg.frame_size, cfg.nb_actions,
                             getattr(cfg, "frame_stack", 1)).to(device).eval()
    control = run_diagnostic(rand, sampler, anchor, future, ep, cfg, device, "random_init_control")

    out = {"ckpt": args.ckpt, "oracle_view": bool(oracle), "root": args.root,
           "n_tuples": args.n_tuples, "tuple_seed": args.tuple_seed,
           "nb_actions": int(cfg.nb_actions), "trained": trained, "random_init_control": control}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    np.savez(os.path.splitext(args.out)[0] + "_tuples.npz", anchor=anchor, future=future, episode=ep)
    t, c = trained, control
    print(f"== ACTION-USE DIAGNOSTIC ({'oracle' if oracle else 'naive'} critic, n={args.n_tuples}) ==")
    print(f"TRAINED  delta_shuffle={t['delta_shuffle']['mean']:+.4f} CI{[round(x,4) for x in t['delta_shuffle']['ci95']]} "
          f"lower>0={t['delta_shuffle']['ci_lower_above_0']}")
    print(f"         action score std={t['all_action_spread']['mean_score_std_across_actions']:.4f} "
          f"demo rank={t['all_action_spread']['demonstrated_action_mean_rank']:.2f}/"
          f"{t['all_action_spread']['demonstrated_action_rank_baseline_chance']:.1f} "
          f"top1={t['all_action_spread']['demonstrated_action_top1_rate']:.3f} "
          f"top3={t['all_action_spread']['demonstrated_action_top3_rate']:.3f}")
    print(f"CONTROL  delta_shuffle={c['delta_shuffle']['mean']:+.4f} CI{[round(x,4) for x in c['delta_shuffle']['ci95']]} "
          f"  action score std={c['all_action_spread']['mean_score_std_across_actions']:.4f} "
          f"demo rank={c['all_action_spread']['demonstrated_action_mean_rank']:.2f}")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
