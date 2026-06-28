"""Goal-conditioned LEARNED actor for the Pong state critics (Stage-1D/1G/1L).

So far the Pong critics are only ever deployed as a per-step argmax over the critic:
    a_t = argmax_a  f(s_t, a, g)              (eval/critic_local_control.critic_control)
which Stage-1N found controls WORSE than NOOP on paddle error. The original Contrastive RL
(Eysenbach et al. 2022) instead deploys a LEARNED actor pi(a|s,g) that MAXIMISES the critic,
with an optional behavioural-cloning term. This module adds that actor so we can ask the same
question for Pong: does the (action-weak) critic, once parameterised through an actor, support
better closed-loop control than the greedy argmax?

FAITHFUL objective (same as the Seaquest actor):
    max_pi   (1-lam) * E_{a~pi}[ f(s,a,g) ]  +  lam * log pi(a_teacher | s,g)  +  ent_coef * H(pi)
with the critic FROZEN. f(s,a,g) over all actions = critic.scores_all_actions(state, goal)
(the SAME hook the argmax policy uses), so no critic logic is duplicated. State-based MLP =>
CPU-fast, fp32 throughout (no AMP); lam=0 pure-critic needs the entropy bonus or it collapses
the softmax to a constant action (vanishing gradient on the near-flat critic landscape).
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalConditionedStateActor(nn.Module):
    """pi(a | s, g): concat(state, normalized-goal) -> MLP -> action logits."""

    def __init__(self, state_dim: int, goal_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + goal_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, state: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state.float(), goal.float()], dim=1))


@torch.no_grad()
def on_trajectory_agreement(actor, critic, A: Dict[str, Any], device: str = "cpu",
                            n: int = 4096, seed: int = 0) -> Dict[str, Any]:
    """Env-free preview on the critic's own training tuples (state, a_teacher, goal)."""
    N = A["state"].shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=min(n, N), replace=False)
    s = torch.as_tensor(A["state"][idx], device=device)
    g = torch.as_tensor(A["goal"][idx], device=device)
    a_teacher = torch.as_tensor(A["action"][idx], device=device)
    logits = actor(s, g)
    pi = F.softmax(logits, dim=1)
    a_actor = logits.argmax(1)
    ent = float(-(pi * torch.log(pi.clamp_min(1e-9))).sum(1).mean())
    f_all = critic.scores_all_actions(s, g)                 # (n, n_actions)
    a_crit = f_all.argmax(1)
    return {
        "top1_agree_teacher": float((a_actor == a_teacher).float().mean()),
        "top1_agree_argmax_critic": float((a_actor == a_crit).float().mean()),
        "actor_entropy_nats": ent,
        "entropy_collapsed": bool(ent < 0.05),
        "critic_value_actor_Epi_f": float((pi * f_all).sum(1).mean()),
        "critic_value_argmax_f": float(f_all.max(1).values.mean()),
        "n": int(len(idx)),
    }


def train_actor(critic, A: Dict[str, Any], n_actions: int, lam: float = 0.5,
                steps: int = 4000, batch: int = 256, lr: float = 3e-4,
                device: str = "cpu", ent_coef: float = 0.01, seed: int = 0,
                verbose: bool = True):
    """Train one goal-conditioned actor maximising the FROZEN critic on the tuples A.

    lam=0.0 pure-critic / 0.5 critic+BC / 1.0 pure-BC (critic-free). ent_coef = entropy bonus
    (the discrete analog of the original continuous Gaussian policy's entropy; lam=0 collapses
    without it). Everything is fp32 (state MLP is tiny)."""
    critic.eval()
    for p in critic.parameters():
        p.requires_grad_(False)
    rng = np.random.default_rng(seed + 777)
    state_dim = A["state"].shape[1]
    goal_dim = A["goal"].shape[1]
    actor = GoalConditionedStateActor(state_dim, goal_dim, n_actions).to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    St = torch.as_tensor(A["state"], device=device)
    Gt = torch.as_tensor(A["goal"], device=device)
    At = torch.as_tensor(A["action"], device=device).long()
    N = St.shape[0]
    log_every = max(1, steps // 5)
    for step in range(1, steps + 1):
        idx = torch.as_tensor(rng.integers(N, size=batch), device=device)
        s, g, a0 = St[idx], Gt[idx], At[idx]
        with torch.no_grad():
            f_all = critic.scores_all_actions(s, g)         # (B, n_actions) fp32
        logits = actor(s, g)
        logp = F.log_softmax(logits, dim=1)
        pi = logp.exp()
        critic_term = (pi * f_all).sum(1).mean()            # E_pi[f]
        bc_term = logp.gather(1, a0.view(-1, 1)).squeeze(1).mean()  # log pi(a_teacher)
        ent_term = -(pi * logp).sum(1).mean()               # H(pi)
        loss = -((1.0 - lam) * critic_term + lam * bc_term + ent_coef * ent_term)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if verbose and step % log_every == 0:
            with torch.no_grad():
                bc_acc = float((logits.argmax(1) == a0).float().mean())
            print(f"  [actor lam={lam}] step {step:5d}/{steps}  loss {float(loss):+.4f}  "
                  f"BC-acc {bc_acc:.3f}  H {float(ent_term):.3f}")
    return actor
