"""Rung-1 noise model + state-vector helper.

REUSES `make_wind` from the Rung-0 demo (a recordable/replayable bool array). The
ONLY thing that differs from Rung 0 is how the wind is APPLIED:

  Rung 0 (wind_counterfactual.apply_wind):  interrupted -> repeat last action (sticky)
  Rung 1 (apply_wind_noop, here):           interrupted -> NOOP (action 0)

Why NOOP for Rung 1 (locked by the skill): with a NOOP override, the executed action
-- and hence the next-state distribution given (state, intended_action) -- is fully
determined by the model's inputs. Sticky-repeat would make the outcome depend on the
PREVIOUS action too, which the model does not see. NOOP keeps the SCM clean.
"""
from typing import Optional

import numpy as np

# reuse the Rung-0 generator unchanged
from pong_counterfactual.wind_counterfactual import make_wind  # noqa: F401  (re-exported)

NOOP = 0
SENTINEL = -1.0  # ball absent (pre-serve / between points)


def apply_wind_noop(intended, wind, noop: int = NOOP):
    """Map intended actions -> executed actions under a NOOP-override wind.

    History-independent: at EVERY step t, if wind[t] fired the executed action is
    NOOP, else it equals the intended action. (Contrast apply_wind, which repeats
    the previous action.)
    """
    executed = list(intended)
    for t in range(len(executed)):
        if wind[t]:
            executed[t] = noop
    return executed


def state_vec(s) -> Optional[np.ndarray]:
    """Pong object state -> the 4-number vector (ball_x, ball_y, player_y, enemy_y).

    Returns None when any component is missing (ball not yet served, or a transient
    death/respawn frame). Callers SKIP those steps so the sentinel never pollutes
    training (skill: 'represent absent ball with a sentinel and skip those steps').
    """
    ball, py, ey = s["ball"], s["player_y"], s["enemy_y"]
    if ball is None or py is None or ey is None:
        return None
    return np.array([ball[0], ball[1], py, ey], dtype=np.float64)
