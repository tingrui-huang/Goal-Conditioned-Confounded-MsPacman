"""Collect offline Pong transitions with injected NOOP-override wind.

We log, per step, everything the ORACLE needs to replay (seed, full executed-action
sequence, wind bits) AND everything the MODEL is allowed to see (state,
intended_action, next_state). The model-facing training set is built from the
post-serve steps only; the seed/wind are kept strictly for the oracle (Invariant 1:
they never reach the model).

Behavior policy: a noisy ball-tracking paddle controller using only {NOOP, UP, DOWN}.
FIRE is used only to serve when the ball is absent (those steps are pre-serve and are
skipped from the training set).
"""
from dataclasses import dataclass, field
from typing import List

import numpy as np

from pong_counterfactual.env import PongEnv
from pong_counterfactual.cjepa_rung1.noise import make_wind, apply_wind_noop, state_vec

NOOP, FIRE, UP, DOWN = 0, 1, 2, 3
ACTIONS = [NOOP, UP, DOWN]          # the intended-action vocabulary the model sees

# Rung-1 env config differs from Rung-0 on ONE thing: frameskip.
# Rung 0 used frameskip=4. At frameskip=4 the paddle accelerates across the 4 hidden
# frames, so its net integer delta depends on sub-step momentum we cannot observe ->
# the 4-number state is NOT Markov even with velocity, and that hidden latent leaks
# as a fake p=0 abduction gap. frameskip=1 exposes one frame of physics per step and
# makes the paddle ~deterministic given (position, velocity, action). (Verified: 35/38
# (vel,action) buckets become single-valued at fs=1 vs 8/52 at fs=4.)
FRAMESKIP = 1


def behavior_policy(s, rng, last_exe=NOOP, tol=3, eps=0.2):
    """Noisy ball-tracker with a TAP discipline: never issue a move two steps in a
    row (if the paddle just moved, release with NOOP). This keeps the paddle slow so
    it does not build momentum that the integer state cannot represent. Returns an
    INTENDED action in {NOOP, UP, DOWN}, or FIRE to serve when the ball is absent."""
    ball, py = s["ball"], s["player_y"]
    if ball is None or py is None:
        return FIRE
    if last_exe in (UP, DOWN):
        return NOOP                  # tap-release: no two consecutive moves
    if rng.random() < eps:
        return int(rng.choice(ACTIONS))
    target = ball[1]                 # ball y; move paddle to match it
    if target > py + tol:
        return DOWN                  # ball below paddle center -> increase y
    if target < py - tol:
        return UP                    # ball above -> decrease y
    return NOOP


@dataclass
class Episode:
    seed: int
    intended: List[int]             # full per-step intended actions
    executed: List[int]             # full per-step executed actions (wind-applied)
    wind: np.ndarray                # full per-step bool array
    states: List                    # state DICTS, length len(executed)+1
    valid_idx: List[int] = field(default_factory=list)  # steps usable as transitions


def collect(p, n_episodes, T=250, base_seed=0, policy_seed=0):
    """Roll n_episodes with NOOP-override wind at probability p. Returns list[Episode]."""
    env = PongEnv(frameskip=FRAMESKIP)
    prng = np.random.default_rng(policy_seed)
    episodes = []
    for ep in range(n_episodes):
        seed = base_seed + ep
        wind = make_wind(T, p, wind_seed=10_000 + ep)
        s = env.reset(seed=seed)
        states = [s]
        intended, executed = [], []
        last_exe = NOOP
        for t in range(T):
            a_int = behavior_policy(s, prng, last_exe)
            a_exe = NOOP if wind[t] else a_int     # NOOP-override wind
            s2, _, done = env.step(a_exe)
            intended.append(int(a_int))
            executed.append(int(a_exe))
            states.append(s2)
            s = s2
            last_exe = a_exe
            if done:
                # pad wind so indexing stays valid if an episode ends early
                wind = wind[: t + 1]
                break
        # A step t is a usable transition iff intended is in the model vocab AND the
        # THREE frames t-1, t, t+1 all have a full 4-vector. We need t-1 so the model
        # can see the one-step VELOCITY (pos[t]-pos[t-1]); positions alone are not
        # Markov (hidden ball direction + paddle momentum), which would otherwise
        # leak as a fake p=0 gap. Velocity is still object-only -- no seed/wind.
        valid = []
        for t in range(1, len(executed)):
            if intended[t] not in ACTIONS:
                continue
            if (state_vec(states[t - 1]) is None or state_vec(states[t]) is None
                    or state_vec(states[t + 1]) is None):
                continue
            valid.append(t)
        episodes.append(Episode(seed, intended, executed, wind, states, valid))
    env.close()
    return episodes


def build_training_arrays(episodes):
    """Flatten episodes -> (pos[N,4], vel[N,4], intended[N], next_pos[N,4]).

    vel[t] = pos[t] - pos[t-1]  (one-step velocity; makes the state ~Markov).
    """
    P, V, A, P2 = [], [], [], []
    for ep in episodes:
        for t in ep.valid_idx:
            p_tm1 = state_vec(ep.states[t - 1])
            p_t = state_vec(ep.states[t])
            P.append(p_t)
            V.append(p_t - p_tm1)
            A.append(ep.intended[t])
            P2.append(state_vec(ep.states[t + 1]))
    return np.asarray(P), np.asarray(V), np.asarray(A), np.asarray(P2)
