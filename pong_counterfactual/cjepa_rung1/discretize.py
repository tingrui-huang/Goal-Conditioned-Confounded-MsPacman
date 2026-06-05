"""Delta discretization: Gumbel-Max needs categorical variables.

We predict per-dimension DELTAS  d = next_state - state  (not absolute positions),
and bin each dim's delta into a small fixed vocabulary. Deltas are integers and
small (ball moves a few px/step; the paddle moves a fixed step or 0), so the set of
distinct observed deltas per dim is tiny -- we just use that set as the vocabulary.

The 4 dims (ball_x, ball_y, player_y, enemy_y) are treated as 4 INDEPENDENT
categoricals for v1 (skill default).
"""
import numpy as np

DIMS = ("ball_x", "ball_y", "player_y", "enemy_y")


class Discretizer:
    def __init__(self):
        self.vocabs = None  # list of 4 int arrays: the distinct delta values per dim

    def fit(self, states, next_states):
        deltas = np.rint(np.asarray(next_states) - np.asarray(states)).astype(int)
        self.vocabs = [np.array(sorted(set(deltas[:, d].tolist()))) for d in range(4)]
        return self

    def nbins(self, d):
        return len(self.vocabs[d])

    def to_tokens(self, state, next_state):
        """(state, next_state) -> list of 4 bin indices (nearest bin per dim)."""
        d = np.rint(np.asarray(next_state) - np.asarray(state)).astype(int)
        return [int(np.argmin(np.abs(self.vocabs[i] - d[i]))) for i in range(4)]

    def decode(self, state, tokens):
        """(state, 4 tokens) -> next_state = state + delta(tokens)."""
        delta = np.array([self.vocabs[i][tokens[i]] for i in range(4)], dtype=np.float64)
        return np.asarray(state, dtype=np.float64) + delta

    def coverage_report(self, states, next_states):
        """Fraction of deltas that land EXACTLY on a vocab value (1.0 = full coverage)."""
        deltas = np.rint(np.asarray(next_states) - np.asarray(states)).astype(int)
        exact = []
        for i in range(4):
            inset = np.isin(deltas[:, i], self.vocabs[i])
            exact.append(float(inset.mean()))
        return dict(zip(DIMS, exact))
