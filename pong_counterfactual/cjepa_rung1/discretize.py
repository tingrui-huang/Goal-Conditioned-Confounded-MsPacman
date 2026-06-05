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


# Per-dim coarsening WIDTHS (skill: "a handful of bins per dim suffices"). A delta is
# binned to the nearest multiple of width; the bin's representative delta is that
# multiple. Coarse bins keep the learned categorical SHARP (few classes -> confident
# softmax), so a near-deterministic dim is predicted near-deterministically and the
# no-abduction baseline does not "win" on model softness. Ball moves in clean small
# integers (width 1); the paddle has ~1px sub-pixel jitter, so width 2 absorbs it.
DEFAULT_WIDTHS = (1, 1, 2, 2)  # ball_x, ball_y, player_y, enemy_y


class Discretizer:
    def __init__(self, widths=DEFAULT_WIDTHS):
        self.widths = tuple(widths)
        self.vocabs = None  # list of 4 int arrays: the distinct quantized deltas per dim

    def _quant(self, deltas):
        return np.rint(np.asarray(deltas) / np.array(self.widths)).astype(int)

    def fit(self, states, next_states):
        q = self._quant(np.asarray(next_states) - np.asarray(states))
        self.vocabs = [np.array(sorted(set(q[:, d].tolist()))) for d in range(4)]
        return self

    def nbins(self, d):
        return len(self.vocabs[d])

    def to_tokens(self, state, next_state):
        """(state, next_state) -> list of 4 bin indices (nearest quantized bin per dim)."""
        q = self._quant(np.asarray(next_state) - np.asarray(state)).reshape(-1)
        return [int(np.argmin(np.abs(self.vocabs[i] - q[i]))) for i in range(4)]

    def decode(self, state, tokens):
        """(state, 4 tokens) -> next_state = state + representative_delta(tokens)."""
        delta = np.array([self.vocabs[i][tokens[i]] * self.widths[i] for i in range(4)],
                         dtype=np.float64)
        return np.asarray(state, dtype=np.float64) + delta

    def coverage_report(self, states, next_states):
        """Fraction of deltas whose quantized bin exists in the vocab (1.0 = full)."""
        q = self._quant(np.asarray(next_states) - np.asarray(states))
        exact = []
        for i in range(4):
            exact.append(float(np.isin(q[:, i], self.vocabs[i]).mean()))
        return dict(zip(DIMS, exact))
