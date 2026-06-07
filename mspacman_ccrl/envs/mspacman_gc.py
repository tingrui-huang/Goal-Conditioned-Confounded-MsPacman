"""OCAtari-backed Ms. Pac-Man wrapper for the confounded goal-conditioned task.

reset()/step() return the UNMASKED frame (210,160,3) plus env-side `state`:
{player_pos, ghosts (bboxes + centers), lives, done}. Ghost positions are env-side
metadata used to build the mask + drive the demonstrator -- never fed to a learner
(the learner sees the ghost-inpainted frame, produced at load time).
"""
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
from ocatari.core import OCAtari

from mspacman_ccrl import config as C
from mspacman_ccrl.envs._compat import patch_ocatari
patch_ocatari()


def _player(objects):
    for o in objects:
        if o.category == "Player":
            return (o.x + o.w / 2.0, o.y + o.h / 2.0)
    return None


def _ghosts(objects):
    boxes = []
    for o in objects:
        if o.category == C.GHOST_CATEGORY:
            boxes.append((int(o.x), int(o.y), int(o.w), int(o.h)))
    return boxes


class MsPacmanGCEnv:
    def __init__(self, cfg: C.Config = C.DEFAULT):
        self.cfg = cfg
        self.env = OCAtari(cfg.game_id, mode=cfg.mode, hud=cfg.hud,
                           render_mode=cfg.render_mode, frameskip=cfg.frameskip,
                           repeat_action_probability=cfg.repeat_action_probability)
        self.nb_actions = self.env.nb_actions
        self._last_pos: Optional[Tuple[float, float]] = None
        self._lives: Optional[int] = None

    def _frame(self) -> np.ndarray:
        return np.asarray(self.env.render(), dtype=np.uint8)

    def _state(self, done: bool) -> Dict[str, Any]:
        objs = [o for o in self.env.objects if o.category != "NoObject"]
        pos = _player(objs)
        ghosts = _ghosts(objs)
        if pos is None:
            pos = self._last_pos
        else:
            self._last_pos = pos
        centers = [(x + w / 2.0, y + h / 2.0) for (x, y, w, h) in ghosts]
        return {"player_pos": pos, "ghosts": ghosts, "ghost_centers": centers,
                "lives": self._lives, "done": bool(done)}

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        _, info = self.env.reset(seed=seed)
        self._last_pos = None
        self._lives = info.get("lives") if isinstance(info, dict) else None
        return self._frame(), self._state(done=False)

    def step(self, action: int):
        _, _, term, trunc, info = self.env.step(int(action))
        self._lives = info.get("lives") if isinstance(info, dict) else self._lives
        frame = self._frame()
        return frame, 0.0, bool(term), bool(trunc), self._state(done=bool(term or trunc))

    def close(self):
        self.env.close()
