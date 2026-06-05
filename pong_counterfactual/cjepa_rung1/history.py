"""Rung 1.1 — history-enriched model input for the partial-observability diagnostic.

Hypothesis (from the Rung-1 README): the real-Pong p=0 abduction gap is a HIDDEN
LATENT — the ALE paddle's momentum/charge, whose response to an action depends on
recent ACTION HISTORY, not just current position + one-step velocity. That latent is
a FUNCTION of observed history, so feeding more history should let the model OBSERVE
it directly, collapsing the p=0 gap. The injected wind is NOT a function of any input,
so the p>0 gap must survive. This module builds the `history(N)` input variant.

The feature vector for the transition at step k (window ending at k, steps <= k only):
  - current 4 object positions            (ball/enemy context for their heads)
  - one-step velocities (4)                (kept identical to the Rung-1 model)
  - last N player_y positions              (paddle trajectory)
  - last N intended actions, one-hot       (THE charge signal: how long held)
Only the CURRENT action (the last in the window) is swapped for the counterfactual;
all past history is shared between factual and CF (the intervention is at k).

INVARIANT: history is OBSERVATION ONLY — past player_y + past intended actions. Never
the seed, the wind, the executed action, or the next state. The wind stays un-feedable;
that is the entire point of the dissociation.
"""
import numpy as np
from sklearn.neural_network import MLPClassifier

from pong_counterfactual.cjepa_rung1.noise import state_vec
from pong_counterfactual.cjepa_rung1.model import POS_NORM, VEL_NORM, _Constant

ACTIONS_H = [0, 2, 3]           # NOOP, UP, DOWN (model-facing intended-action vocab)
UP, DOWN = 2, 3


def _onehot(a):
    return [1.0 if a == x else 0.0 for x in ACTIONS_H]


def hist_features(states, intended, k, N, override_action=None):
    """Build the history(N) feature vector for the transition at step k.

    override_action replaces ONLY the current (step-k) intended action -- used to form
    the counterfactual input. Past actions/positions are untouched (shared history)."""
    pos_k = state_vec(states[k])
    vel = pos_k - state_vec(states[k - 1])
    feat = list(pos_k / POS_NORM) + list(vel / VEL_NORM)
    # paddle position history y_{k-N+1..k}
    feat += [state_vec(states[j])[2] / POS_NORM for j in range(k - N + 1, k + 1)]
    # intended-action history a_{k-N+1..k}, current possibly overridden for the CF
    acts = list(intended[k - N + 1:k + 1])
    if override_action is not None:
        acts[-1] = override_action
    for a in acts:
        feat += _onehot(a)
    return np.array(feat, dtype=np.float64)


def valid_ks(ep, N, move_only=False):
    """Steps k usable as a transition with an N-window of valid observations."""
    out = []
    for k in range(N - 1, len(ep.intended)):
        a = ep.intended[k]
        if a not in ACTIONS_H:
            continue
        if move_only and a not in (UP, DOWN):
            continue
        if any(state_vec(ep.states[j]) is None for j in range(k - N + 1, k + 2)):
            continue
        out.append(k)
    return out


class HistModel:
    """4 independent per-dim MLP categoricals over delta-bins, on history(N) features."""

    def __init__(self, discretizer):
        self.disc = discretizer
        self.clfs = [None] * 4

    def fit(self, X, tokens):
        X = np.asarray(X)
        tokens = np.asarray(tokens)
        for d in range(4):
            y = tokens[:, d]
            if len(np.unique(y)) < 2:
                self.clfs[d] = _Constant(int(y[0]))
            else:
                self.clfs[d] = MLPClassifier((64,), max_iter=400,
                                             random_state=0).fit(X, y)
        return self

    def logits(self, x):
        x = np.asarray(x).reshape(1, -1)
        out = []
        for d in range(4):
            pr = self.clfs[d].predict_proba(x)[0]
            full = np.full(self.disc.nbins(d), 1e-9)
            for c, q in zip(self.clfs[d].classes_, pr):
                full[int(c)] = max(float(q), 1e-9)
            out.append(np.log(full))
        return out

    def top1(self, X, tokens):
        X = np.asarray(X)
        tokens = np.asarray(tokens)
        acc = np.zeros(4)
        for i in range(len(X)):
            lg = self.logits(X[i])
            for d in range(4):
                acc[d] += (int(np.argmax(lg[d])) == tokens[i, d])
        return acc / len(X)
