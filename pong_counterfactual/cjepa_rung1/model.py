"""The ONLY trained component: P(next-delta | state, intended_action).

A small MLP per state-dimension (4 independent categoricals, v1). Input = normalized
state (4) + one-hot intended action (3). Output = logits over that dim's delta-bins.
Trained by cross-entropy (sklearn MLPClassifier == softmax + CE).

The model NEVER sees the seed or the wind (Invariant 1). The wind shows up only
implicitly: at p>0 the executed action is sometimes NOOP, so P(next-delta | state,
intended a) becomes a genuine MIXTURE the model must represent -- and that mixture is
exactly what abduction later disentangles.
"""
import numpy as np
from sklearn.neural_network import MLPClassifier

from pong_counterfactual.cjepa_rung1.collect import ACTIONS

NORM = 210.0  # screen scale for feature normalization


class _Constant:
    """Fallback 'classifier' for a dim whose delta never varies (1 bin)."""
    def __init__(self, cls):
        self.classes_ = np.array([cls])

    def predict_proba(self, X):
        return np.ones((len(X), 1))


class TransitionModel:
    def __init__(self, discretizer, actions=ACTIONS):
        self.disc = discretizer
        self.actions = list(actions)
        self.clfs = [None] * 4

    def _feat(self, pos, vel, intended):
        P = np.asarray(pos, dtype=np.float64).reshape(-1, 4) / NORM
        V = np.asarray(vel, dtype=np.float64).reshape(-1, 4) / NORM
        A = np.asarray(intended).reshape(-1)
        oh = np.zeros((len(A), len(self.actions)))
        for j, a in enumerate(self.actions):
            oh[:, j] = (A == a)
        return np.hstack([P, V, oh])

    def fit(self, pos, vel, intended, tokens):
        """tokens: (N,4) int delta-bin indices."""
        X = self._feat(pos, vel, intended)
        for d in range(4):
            y = np.asarray(tokens)[:, d]
            if len(np.unique(y)) < 2:
                self.clfs[d] = _Constant(int(y[0]))
            else:
                clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=400,
                                    random_state=0)
                clf.fit(X, y)
                self.clfs[d] = clf
        return self

    def logits(self, pos, vel, intended):
        """Return a list of 4 logit arrays, each of length discretizer.nbins(d),
        i.e. log P(delta-bin | pos, vel, intended) padded over the FULL vocabulary."""
        X = self._feat([pos], [vel], [intended])
        out = []
        for d in range(4):
            clf = self.clfs[d]
            proba = clf.predict_proba(X)[0]
            full = np.full(self.disc.nbins(d), 1e-9)
            for c, pr in zip(clf.classes_, proba):
                full[int(c)] = max(float(pr), 1e-9)
            out.append(np.log(full))
        return out

    def top1_accuracy(self, pos, vel, intended, tokens):
        """Per-dim fraction where argmax-logit == observed token (calibration check)."""
        tokens = np.asarray(tokens)
        acc = np.zeros(4)
        for i in range(len(pos)):
            lg = self.logits(pos[i], vel[i], intended[i])
            for d in range(4):
                acc[d] += (int(np.argmax(lg[d])) == tokens[i, d])
        return acc / len(pos)
