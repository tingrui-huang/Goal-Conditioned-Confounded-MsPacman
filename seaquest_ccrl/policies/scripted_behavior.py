"""Scripted oxygen-aware behavior policy (the U->A confounding channel).

The policy sees env-side state (player_pos + oxygen) and a per-episode target. It
greedily moves toward the target in open water (NO A* / maze planning) and OVERRIDES
to SURFACE (move UP) whenever oxygen < THETA. THETA is the confounding-strength knob:
it is the ONLY place oxygen is allowed to influence the action (Invariant 7).
"""
from typing import Optional, Tuple, Dict, Any

from seaquest_ccrl import config as C

# Seaquest action ids (full 18-action set; we only emit movement actions).
NOOP = 0
UP, RIGHT, LEFT, DOWN = 2, 3, 4, 5
UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT = 6, 7, 8, 9


def _dir_action(dx: float, dy: float, tol: int) -> int:
    """Map a desired displacement (dx=target.x-px, dy=target.y-py) to a move action.
    Screen coords: +x right, +y DOWN. So dy<0 means target is ABOVE -> move UP.
    """
    want_right = dx > tol
    want_left = dx < -tol
    want_down = dy > tol
    want_up = dy < -tol
    if want_up and want_right:
        return UPRIGHT
    if want_up and want_left:
        return UPLEFT
    if want_down and want_right:
        return DOWNRIGHT
    if want_down and want_left:
        return DOWNLEFT
    if want_up:
        return UP
    if want_down:
        return DOWN
    if want_right:
        return RIGHT
    if want_left:
        return LEFT
    return NOOP


class ScriptedBehaviorPolicy:
    def __init__(self, cfg: C.Config = C.DEFAULT, theta: Optional[int] = None):
        self.cfg = cfg
        # allow per-instance THETA override for the strength-knob sweep (check E)
        self.theta = cfg.theta if theta is None else theta

    def act(self, state: Dict[str, Any], target: Tuple[float, float]) -> int:
        pos = state.get("player_pos")
        oxy = state.get("oxygen")

        # Surfacing override: U -> A. Low oxygen forces an ascent regardless of target.
        if oxy is not None and oxy < self.theta:
            return UP

        if pos is None:
            return NOOP
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        return _dir_action(dx, dy, self.cfg.move_tol)

    def reached(self, state: Dict[str, Any], target: Tuple[float, float]) -> bool:
        pos = state.get("player_pos")
        if pos is None:
            return False
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        return (dx * dx + dy * dy) ** 0.5 <= self.cfg.eps
