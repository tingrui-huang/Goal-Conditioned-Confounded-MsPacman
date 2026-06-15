"""Stage-S1 vanilla action-conditioned contrastive state critic (PyTorch only).

f(s,a,g) = phi(s,a)^T psi(g) / sqrt(d).  Eysenbach-style. NO pixels, NO RNN/attention/
residual, NO weight sharing between phi and psi. A matched-capacity no-action baseline
f(s,g) uses the SAME encoders minus the action input.
"""
import torch
import torch.nn as nn

EMB_DIM = 128
ACTION_DIM = 18


def _mlp(in_dim, out_dim=EMB_DIM):
    return nn.Sequential(
        nn.Linear(in_dim, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, out_dim))


class StateActionCritic(nn.Module):
    """phi(state ⊕ one_hot(action)) and psi(goal); score = phi·psi / sqrt(d)."""
    def __init__(self, state_dim, goal_dim, action_dim=ACTION_DIM, emb_dim=EMB_DIM):
        super().__init__()
        self.state_dim = state_dim; self.goal_dim = goal_dim
        self.action_dim = action_dim; self.emb_dim = emb_dim
        self.use_action = True
        self.phi = _mlp(state_dim + action_dim, emb_dim)
        self.psi = _mlp(goal_dim, emb_dim)
        self._scale = emb_dim ** 0.5

    def encode_sa(self, states, actions_onehot):
        x = torch.cat([states, actions_onehot], dim=-1)
        return self.phi(x)

    def encode_g(self, goals):
        return self.psi(goals)

    def score_matrix(self, sa_repr, g_repr):
        return sa_repr @ g_repr.t() / self._scale

    def forward(self, states, actions_onehot, goals):
        sa = self.encode_sa(states, actions_onehot)
        g = self.encode_g(goals)
        return (sa * g).sum(-1) / self._scale  # diagonal scores


class NoActionCritic(nn.Module):
    """Matched-capacity baseline: phi(state) only (action absent), psi(goal)."""
    def __init__(self, state_dim, goal_dim, emb_dim=EMB_DIM, action_dim=ACTION_DIM):
        super().__init__()
        self.state_dim = state_dim; self.goal_dim = goal_dim
        self.action_dim = action_dim; self.emb_dim = emb_dim
        self.use_action = False
        self.phi = _mlp(state_dim, emb_dim)
        self.psi = _mlp(goal_dim, emb_dim)
        self._scale = emb_dim ** 0.5

    def encode_sa(self, states, actions_onehot=None):
        return self.phi(states)

    def encode_g(self, goals):
        return self.psi(goals)

    def score_matrix(self, sa_repr, g_repr):
        return sa_repr @ g_repr.t() / self._scale

    def forward(self, states, actions_onehot, goals):
        sa = self.phi(states); g = self.psi(goals)
        return (sa * g).sum(-1) / self._scale


def one_hot(actions, n=ACTION_DIM, device=None):
    actions = torch.as_tensor(actions, dtype=torch.long, device=device)
    return torch.nn.functional.one_hot(actions, n).float()


def build_critic(kind, state_dim, goal_dim):
    if kind == "action":
        return StateActionCritic(state_dim, goal_dim)
    if kind == "no_action":
        return NoActionCritic(state_dim, goal_dim)
    raise ValueError(kind)
