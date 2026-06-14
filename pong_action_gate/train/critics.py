"""Dot-product contrastive critics: f(o,a,g) = phi(o,a) . psi(g).

The action enters phi through a dedicated linear `action_embed` so its gradient and
its zero/replace ablations are cleanly isolable. Pure architecture: no TD, AWR,
decoder, reward head, or auxiliary task.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalEncoder(nn.Module):
    def __init__(self, repr_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(), nn.Linear(hidden, repr_dim))

    def forward(self, g_norm: torch.Tensor) -> torch.Tensor:
        return self.net(g_norm.view(-1, 1))


class _SACriticBase(nn.Module):
    """Shares the action pathway + goal encoder + dot-product interface."""

    def __init__(self, n_actions: int, repr_dim: int, action_dim: int = 32):
        super().__init__()
        self.n_actions = n_actions
        self.repr_dim = repr_dim
        self.action_embed = nn.Linear(n_actions, action_dim, bias=False)
        self.psi = GoalEncoder(repr_dim)

    def _obs_feat(self, obs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def sa_repr(self, obs: torch.Tensor, action: torch.Tensor, zero_action: bool = False) -> torch.Tensor:
        h = self._obs_feat(obs)
        a = self.action_embed(F.one_hot(action.long(), self.n_actions).float())
        if zero_action:
            a = torch.zeros_like(a)
        return self.head(torch.cat([h, a], dim=1))

    def g_repr(self, g_norm: torch.Tensor) -> torch.Tensor:
        return self.psi(g_norm)

    def logits_matrix(self, obs, action, g_norm, zero_action: bool = False) -> torch.Tensor:
        sa = self.sa_repr(obs, action, zero_action)     # (B,d)
        g = self.g_repr(g_norm)                          # (B,d)
        return sa @ g.t()                                # (B,B)

    def scores_all_actions(self, obs, g_norm) -> torch.Tensor:
        """For each row, score every action against its own goal. Returns (B, n_actions)."""
        g = self.g_repr(g_norm)                          # (B,d)
        out = []
        for a in range(self.n_actions):
            act = torch.full((obs.shape[0],), a, dtype=torch.long, device=obs.device)
            sa = self.sa_repr(obs, act)
            out.append((sa * g).sum(1, keepdim=True))
        return torch.cat(out, dim=1)


class StateSACritic(_SACriticBase):
    def __init__(self, state_dim: int, n_actions: int, repr_dim: int = 128, action_dim: int = 32):
        super().__init__(n_actions, repr_dim, action_dim)
        self.trunk = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(128 + action_dim, 128), nn.ReLU(), nn.Linear(128, repr_dim))

    def _obs_feat(self, obs):
        return self.trunk(obs)


class PixelSACritic(_SACriticBase):
    def __init__(self, k: int, n_actions: int, frame: int = 84, repr_dim: int = 128, action_dim: int = 32):
        super().__init__(n_actions, repr_dim, action_dim)
        self.conv = nn.Sequential(
            nn.Conv2d(k, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, k, frame, frame)).flatten(1).shape[1]
        self.head = nn.Sequential(nn.Linear(flat + action_dim, 256), nn.ReLU(), nn.Linear(256, repr_dim))

    def _obs_feat(self, obs):
        return self.conv(obs).flatten(1)


def nce_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Sigmoid BCE on the B×B matrix; sum over candidates, mean over anchors."""
    return F.binary_cross_entropy_with_logits(logits, targets, reduction="none").sum(1).mean()
