"""Stage-1O — OFFLINE actor lambda-sweep + on-trajectory preview for the Pong critics.

The Pong goal-conditioned critics have only ever been deployed as a per-step argmax
(eval/critic_local_control.critic_control; Stage-1N: worse than NOOP). This trains the
ORIGINAL Contrastive-RL learned actor pi(a|s,g) that maximises the critic (actor.train_actor,
objective (1-lam)*E_pi[f] + lam*log pi(a_teacher) + ent_coef*H(pi), critic frozen) and sweeps
    lam=0.0 pure-critic / 0.5 critic+BC / 1.0 pure-BC (critic-free control baseline).

Env-free half: rebuild the critic's own training tuples (state, a_teacher, goal) via the
UNCHANGED pipeline (Stage-1L reconstruct_episodes -> Stage-1G build_mh), train the actors, and
report on-trajectory agreement (actor argmax vs teacher / vs argmax-critic), actor entropy, and
the critic value E_pi[f] attained vs the argmax ceiling. This PREVIEWS whether an actor can do
better than the argmax; the closed-loop verdict is a separate env step (critic_local_control with
the actor swapped in for argmax_action / critic_control).

State-based critic => CPU-fast; everything runs locally (no GPU/Colab needed).
Default critic = Stage-1L E01_full seed4101 (the support-improved observational critic).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import dataset as D
from .actor import on_trajectory_agreement, train_actor
from .stage1d_rich_goal import RichStateCritic
from .stage1g_multihorizon_critic import build_mh
from .stage1l_support_controlled_critic import N_EPISODES, VAL_FRAC, reconstruct_episodes

N_ACTIONS = 6
DEFAULT_CKPT = "artifacts/pong_action_gate/stage1l/critic_E01_full_seed4101.pt"


def load_critic(ckpt: str, goal_dim: int):
    critic = RichStateCritic(D.STATE_DIM, N_ACTIONS, goal_dim=goal_dim)
    critic.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    critic.eval()
    return critic


def build_tuples(epsilon: str, seed: int):
    """Reconstruct the critic's TRAIN tuples (state, a_teacher, goal) exactly as it was trained.

    build_mh recomputes train-only normalization stats from the same (epsilon, seed-split)
    episodes the critic used -> goals are normalized identically to training."""
    eps = reconstruct_episodes(epsilon)
    tr_ids, _ = D.split_episodes(N_EPISODES, VAL_FRAC, seed)
    A = build_mh(eps, tr_ids)
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic-ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--epsilon", default="0.1", choices=["0.0", "0.1", "0.2"],
                    help="data collection the critic was trained on (E0=0.0, E01=0.1)")
    ap.add_argument("--seed", type=int, default=4101, help="episode-split seed (must match the critic)")
    ap.add_argument("--goal-dim", type=int, default=9)
    ap.add_argument("--lams", default="0.0,0.5,1.0")
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out-dir", default="pong_action_gate/results/stage1o_actor_lambda_sweep")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    critic = load_critic(args.critic_ckpt, args.goal_dim)
    print(f"[1O] critic={args.critic_ckpt} goal_dim={args.goal_dim} eps={args.epsilon} seed={args.seed}")
    A = build_tuples(args.epsilon, args.seed)
    assert A["goal"].shape[1] == args.goal_dim, f"goal dim {A['goal'].shape[1]} != critic {args.goal_dim}"
    print(f"[1O] train tuples: {A['state'].shape[0]} (state {A['state'].shape[1]}d, goal {A['goal'].shape[1]}d)")

    results = {"critic_ckpt": args.critic_ckpt, "epsilon": args.epsilon, "seed": args.seed,
               "goal_dim": args.goal_dim, "steps": args.steps, "ent_coef": args.ent_coef,
               "n_train_tuples": int(A["state"].shape[0]), "lambda_sweep": {}}
    ckpt_dir = out / "actors"; ckpt_dir.mkdir(exist_ok=True)
    for lam in [float(x) for x in args.lams.split(",")]:
        kind = "pure-critic" if lam == 0 else "pure-BC" if lam == 1 else "critic+BC"
        print(f"\n=== actor lam={lam} ({kind}) ===")
        actor = train_actor(critic, A, N_ACTIONS, lam=lam, steps=args.steps, batch=args.batch,
                            lr=args.lr, device="cpu", ent_coef=args.ent_coef, seed=args.seed)
        torch.save(actor.state_dict(), ckpt_dir / f"actor_lam{lam:.2f}.pt")
        agree = on_trajectory_agreement(actor, critic, A, device="cpu", seed=args.seed + 4242)
        results["lambda_sweep"][f"{lam:.2f}"] = agree
        print(f"  [lam={lam}] {json.dumps(agree)}")

    (out / "on_trajectory_preview.json").write_text(json.dumps(results, indent=2))
    print(f"\nWROTE {out / 'on_trajectory_preview.json'}")
    print("NOTE: env-free preview. Closed-loop verdict = critic_local_control with the actor "
          "swapped in for the argmax policy (separate env step).")


if __name__ == "__main__":
    main()
