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


# === ENEMY-CONFOUNDER demonstrator (Level-1 v2) ============================
# Goal-seeking + enemy-avoiding navigator. The DETOUR around a nearby enemy is the
# confounded action: its direction depends on the hidden enemy position (U->A).
# Oxygen is now a VISIBLE feature -- surfacing for oxygen is NOT confounded; we
# surface generously (OXY_SAFE) so the sub never drowns and deaths are enemy-only.

import math

# unit displacement per movement action (screen coords: +x right, +y DOWN)
_S = 1.0 / math.sqrt(2.0)
_ACTION_VECS = {
    NOOP: (0.0, 0.0), UP: (0.0, -1.0), DOWN: (0.0, 1.0),
    LEFT: (-1.0, 0.0), RIGHT: (1.0, 0.0),
    UPRIGHT: (_S, -_S), UPLEFT: (-_S, -_S),
    DOWNRIGHT: (_S, _S), DOWNLEFT: (-_S, _S),
}


class EnemyAvoidingPolicy:
    """Move toward the target, but detour around enemies within radius rho.

    Action is chosen by scoring each candidate move by (progress to goal) plus a
    penalty for ending up inside rho of any enemy. With no enemy in range the
    penalty is zero for all actions, so it reduces to greedy goal-seeking.
    `last_info` exposes the per-step confounding flags for the channels analysis.
    """

    def __init__(self, cfg: C.Config = C.DEFAULT, rho: Optional[float] = None,
                 step_px: float = 6.0, lam: float = 3.0):
        self.cfg = cfg
        self.rho = float(C.RHO if rho is None else rho)
        self.step_px = step_px
        self.lam = lam
        self._surfacing = False        # hysteresis: commit to a full refill once low
        self.last_info: Dict[str, Any] = {}

    def reset(self):
        self._surfacing = False

    @staticmethod
    def _centers(enemies):
        return [(x + w / 2.0, y + h / 2.0) for (x, y, w, h) in (enemies or [])]

    def act(self, state: Dict[str, Any], target: Tuple[float, float]) -> int:
        pos = state.get("player_pos")
        oxy = state.get("oxygen")
        centers = self._centers(state.get("enemies"))

        # nearest-enemy distance (for U->S' analysis + enemy_near flag)
        min_d = math.inf
        for (ex, ey) in centers:
            if pos is not None:
                d = math.hypot(ex - pos[0], ey - pos[1])
                min_d = min(min_d, d)
        enemy_near = min_d <= self.rho

        # Oxygen safety (VISIBLE feature, NOT confounded): surface EARLY with
        # hysteresis -- once oxygen dips below the trigger, commit to going straight
        # UP until refilled, else the accelerated low-oxygen depletion drowns the sub
        # mid-ascent. This is driven by visible oxygen, so it does not confound.
        if oxy is not None:
            if oxy < C.OXY_SURFACE_TRIGGER:
                self._surfacing = True
            elif oxy >= C.OXY_REFILLED:
                self._surfacing = False
        if self._surfacing:
            self.last_info = {"detour": False, "enemy_near": enemy_near,
                              "min_enemy_dist": float(min_d), "surfacing": True}
            return UP
        if pos is None:
            self.last_info = {"detour": False, "enemy_near": enemy_near,
                              "min_enemy_dist": float(min_d), "surfacing": False}
            return NOOP

        dx, dy = target[0] - pos[0], target[1] - pos[1]
        greedy = _dir_action(dx, dy, self.cfg.move_tol)

        # score each candidate move: progress to goal + enemy-proximity penalty
        best_a, best_score = NOOP, -math.inf
        for a, (vx, vy) in _ACTION_VECS.items():
            nx, ny = pos[0] + self.step_px * vx, pos[1] + self.step_px * vy
            goal_term = -math.hypot(target[0] - nx, target[1] - ny)
            safe_term = 0.0
            for (ex, ey) in centers:
                de = math.hypot(ex - nx, ey - ny)
                if de < self.rho:
                    safe_term -= (self.rho - de)        # closer than rho => penalty
            score = goal_term + self.lam * safe_term
            if score > best_score:
                best_score, best_a = score, a

        detour = enemy_near and (best_a != greedy)
        self.last_info = {"detour": bool(detour), "enemy_near": bool(enemy_near),
                          "min_enemy_dist": float(min_d), "surfacing": False}
        return best_a

    def reached(self, state: Dict[str, Any], target: Tuple[float, float]) -> bool:
        pos = state.get("player_pos")
        if pos is None:
            return False
        dx, dy = target[0] - pos[0], target[1] - pos[1]
        return (dx * dx + dy * dy) ** 0.5 <= self.cfg.eps
