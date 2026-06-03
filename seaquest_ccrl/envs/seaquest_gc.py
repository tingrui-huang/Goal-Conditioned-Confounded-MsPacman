"""OCAtari-backed Seaquest wrapper for the confounded goal-conditioned task.

reset()/step() return the UNMASKED pixel frame (210,160,3) plus an env-side `state`
dict {player_pos, oxygen, done}. OCAtari positions/oxygen are ENV-SIDE METADATA used
to build goals + the mask + the scripted policy — they MUST NOT be fed to a learner
as an observation (Invariant 1). The learner-facing observation is produced later, at
load time, by masking this frame.
"""
from typing import Optional, Tuple, Dict, Any

import numpy as np
from ocatari.core import OCAtari

from seaquest_ccrl import config as C


def _player_pos(objects) -> Optional[Tuple[float, float]]:
    for o in objects:
        if o.category == "Player":
            return (o.x + o.w / 2.0, o.y + o.h / 2.0)
    return None


def _oxygen(objects) -> Optional[int]:
    """Oxygen level U = filled OxygenBar width (0..OXY_FULL_WIDTH).

    When the bar is empty the OxygenBar object disappears (only OxygenBarDepleted
    remains) -> oxygen = 0.
    """
    for o in objects:
        if o.category == "OxygenBar":
            return int(o.w)
    # bar absent: either empty (depleted present) -> 0
    for o in objects:
        if o.category == "OxygenBarDepleted":
            return 0
    return None  # transient (death/respawn): unknown this step


class SeaquestGCEnv:
    """Thin wrapper. Frame is the observation substrate; state is env-side metadata."""

    def __init__(self, cfg: C.Config = C.DEFAULT):
        self.cfg = cfg
        self.env = OCAtari(
            cfg.game_id,
            mode=cfg.mode,
            hud=cfg.hud,
            render_mode=cfg.render_mode,
            frameskip=cfg.frameskip,
            repeat_action_probability=cfg.repeat_action_probability,
        )
        self.nb_actions = self.env.nb_actions
        self._last_pos: Optional[Tuple[float, float]] = None
        self._last_oxy: Optional[int] = None

    # -- helpers ------------------------------------------------------------
    def _frame(self) -> np.ndarray:
        f = self.env.render()
        return np.asarray(f, dtype=np.uint8)

    def _state(self, done: bool) -> Dict[str, Any]:
        objs = [o for o in self.env.objects if o.category != "NoObject"]
        pos = _player_pos(objs)
        oxy = _oxygen(objs)
        # carry forward last-known through transient missing frames (death/respawn)
        if pos is None:
            pos = self._last_pos
        else:
            self._last_pos = pos
        if oxy is None:
            oxy = self._last_oxy
        else:
            self._last_oxy = oxy
        return {"player_pos": pos, "oxygen": oxy, "done": bool(done)}

    # -- gym-ish API --------------------------------------------------------
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self.env.reset(seed=seed)
        self._last_pos = None
        self._last_oxy = None
        # step a NOOP-free render: OCAtari populates objects after reset
        frame = self._frame()
        state = self._state(done=False)
        return frame, state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        _, _, term, trunc, _ = self.env.step(int(action))
        frame = self._frame()                 # UNMASKED frame (stored as-is)
        state = self._state(done=bool(term or trunc))
        reward = 0.0                           # game score DROPPED (Level 1)
        return frame, reward, bool(term), bool(trunc), state

    def close(self):
        self.env.close()
