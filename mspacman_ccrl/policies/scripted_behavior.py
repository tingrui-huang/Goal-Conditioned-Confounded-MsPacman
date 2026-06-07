"""Maze-aware, ghost-avoiding goal-seeking demonstrator (the U->A channel).

Ms. Pac-Man has no wall objects in OCAtari, so legality is read from PIXELS: a
cardinal direction is blocked if the pixel a few px ahead of Pac-Man is a pink
wall. Among legal moves the demonstrator scores each by progress-to-goal minus a
penalty for ending up within rho of a ghost. The DETOUR (choosing a non-greedy
legal move because of a nearby ghost) is the confounded action -- its direction
depends on the hidden ghost positions.
"""
import math
from typing import Optional, Dict, Any, Tuple

import numpy as np

from mspacman_ccrl import config as C

NOOP, UP, RIGHT, LEFT, DOWN = 0, 1, 2, 3, 4
CARDINAL = {UP: (0.0, -1.0), RIGHT: (1.0, 0.0), LEFT: (-1.0, 0.0), DOWN: (0.0, 1.0)}


class GhostAvoidingPolicy:
    def __init__(self, cfg: C.Config = C.DEFAULT, rho: Optional[float] = None,
                 step_px: float = 6.0, lam: float = 4.0):
        self.cfg = cfg
        self.rho = float(C.RHO if rho is None else rho)
        self.step_px = step_px
        self.lam = lam
        self._wall = np.asarray(C.WALL_COLOR, dtype=np.int16)
        self.last_info: Dict[str, Any] = {}

    def reset(self):
        pass

    def _blocked(self, frame, pos, d) -> bool:
        H, W = frame.shape[:2]
        px = int(round(pos[0] + self.cfg.wall_probe_px * d[0]))
        py = int(round(pos[1] + self.cfg.wall_probe_px * d[1]))
        if px < 0 or px >= W or py < 0 or py >= H:
            return True
        return bool(np.abs(frame[py, px].astype(np.int16) - self._wall).sum() < 60)

    def act(self, state: Dict[str, Any], target: Tuple[float, float], frame) -> int:
        pos = state.get("player_pos")
        centers = state.get("ghost_centers") or []
        min_d = math.inf
        if pos is not None:
            for (gx, gy) in centers:
                min_d = min(min_d, math.hypot(gx - pos[0], gy - pos[1]))
        ghost_near = min_d <= self.rho

        if pos is None:
            self.last_info = {"detour": False, "ghost_near": ghost_near,
                              "min_ghost_dist": float(min_d)}
            return NOOP

        legal = [a for a, d in CARDINAL.items() if not self._blocked(frame, pos, d)]
        if not legal:
            self.last_info = {"detour": False, "ghost_near": ghost_near,
                              "min_ghost_dist": float(min_d)}
            return NOOP

        def goal_term(a):
            dx, dy = CARDINAL[a]
            nx, ny = pos[0] + self.step_px * dx, pos[1] + self.step_px * dy
            return -math.hypot(target[0] - nx, target[1] - ny)

        def score(a):
            dx, dy = CARDINAL[a]
            nx, ny = pos[0] + self.step_px * dx, pos[1] + self.step_px * dy
            safe = 0.0
            for (gx, gy) in centers:
                de = math.hypot(gx - nx, gy - ny)
                if de < self.rho:
                    safe -= (self.rho - de)
            return goal_term(a) + self.lam * safe

        best = max(legal, key=score)
        greedy = max(legal, key=goal_term)
        detour = ghost_near and (best != greedy)
        self.last_info = {"detour": bool(detour), "ghost_near": bool(ghost_near),
                          "min_ghost_dist": float(min_d)}
        return best

    def reached(self, state: Dict[str, Any], target: Tuple[float, float]) -> bool:
        pos = state.get("player_pos")
        if pos is None:
            return False
        return math.hypot(target[0] - pos[0], target[1] - pos[1]) <= self.cfg.eps
