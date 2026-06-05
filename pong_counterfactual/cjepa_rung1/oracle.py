"""Seed-replay oracle: the ground-truth single-step counterfactual.

The oracle has god-mode access the model is denied: it OWNS the seed and the wind
bits, so it can replay the exact factual history up to step k and then branch on the
intervened action -- while HOLDING THE SAME WIND BIT FIXED (Invariant 5):

    executed'_k = NOOP   if wind[k] fired
                = a'      otherwise

This is the counterfactual ground truth precisely because it keeps the noise the
model must infer. We use replay-from-seed (deterministic env), the skill's simple
fallback, rather than ALE clone/restore.
"""
from pong_counterfactual.env import PongEnv
from pong_counterfactual.cjepa_rung1.noise import NOOP, state_vec


class Oracle:
    def __init__(self):
        self.env = PongEnv()

    def cf_next_state(self, seed, executed, wind, k, a_prime):
        """Replay executed[0:k] from `seed`, then take the counterfactual action at k
        (NOOP if the wind fired at k, else a_prime). Returns the 4-vector next-state."""
        s = self.env.reset(seed=seed)
        for t in range(k):
            s, _, _ = self.env.step(int(executed[t]))
        exec_k = NOOP if wind[k] else int(a_prime)
        s2, _, _ = self.env.step(exec_k)
        return state_vec(s2)

    def factual_next_state(self, seed, executed, k):
        """Sanity helper: replay reproduces the recorded factual next-state."""
        s = self.env.reset(seed=seed)
        for t in range(k + 1):
            s, _, _ = self.env.step(int(executed[t]))
        return state_vec(s)

    def close(self):
        self.env.close()
