"""Thin OCAtari-backed Pong loader.

This is the WHOLE point of step 1: an Atari "state" is not pixels, it is a few
numbers. OCAtari reads the emulator RAM and hands us objects with (x, y, w, h).
For Pong the only moving objects are:

    Player  -> the RIGHT paddle  (x ~ 140)  -> we track its y
    Enemy   -> the LEFT  paddle  (x ~ 16)   -> we track its y
    Ball    -> the ball          -> we track its (x, y)

So the entire object state is 4 numbers: ball_x, ball_y, player_y, enemy_y.
That tiny vector is what makes a clean counterfactual possible: we can run the
exact same dynamics, change one action, and watch these 4 numbers diverge.
"""
from typing import Optional, Tuple, Dict, Any

from ocatari.core import OCAtari

GAME_ID = "ALE/Pong-v5"


def _center_y(o) -> Optional[float]:
    # OCAtari leaves placeholder objects at the origin when an object is absent;
    # treat a paddle pinned at y<=0 as "not visible this frame".
    if o is None or o.y <= 0:
        return None
    return o.y + o.h / 2.0


def _ball_xy(o) -> Optional[Tuple[float, float]]:
    # Between points the ball disappears (RAM clears it) and OCAtari parks the
    # placeholder at the origin -> report None rather than a fake (0,0).
    if o is None or (o.x <= 0 and o.y <= 0):
        return None
    return (o.x + o.w / 2.0, o.y + o.h / 2.0)


class PongEnv:
    """Deterministic Pong wrapper. step()/reset() return the 4-number object state.

    Determinism is enforced at construction:
      - repeat_action_probability=0.0  -> no sticky-action noise
      - frameskip=4 fixed              -> no random frameskip
      - reset(seed=...)                -> fixes the ALE RNG
    Same seed + same action sequence  ==>  byte-identical trajectory.
    """

    def __init__(self, frameskip: int = 4):
        self.env = OCAtari(
            GAME_ID,
            mode="ram",
            hud=False,
            render_mode="rgb_array",
            frameskip=frameskip,
            repeat_action_probability=0.0,
        )
        self.nb_actions = self.env.nb_actions

    def _state(self, done: bool) -> Dict[str, Any]:
        player = enemy = ball = None
        for o in self.env.objects:
            if o is None:
                continue
            if o.category == "Player":
                player = o
            elif o.category == "Enemy":
                enemy = o
            elif o.category == "Ball":
                ball = o
        return {
            "ball": _ball_xy(ball),       # (x, y) or None between points
            "player_y": _center_y(player),  # right paddle (agent)
            "enemy_y": _center_y(enemy),    # left paddle (built-in AI)
            "done": bool(done),
        }

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        self.env.reset(seed=seed)
        return self._state(done=False)

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool]:
        _, reward, term, trunc, _ = self.env.step(int(action))
        return self._state(done=bool(term or trunc)), float(reward), bool(term or trunc)

    def close(self):
        self.env.close()


# Pong action meanings (full ALE action set is 6 for Pong):
#   0 NOOP   1 FIRE   2 RIGHT(=UP)   3 LEFT(=DOWN)   4 RIGHTFIRE   5 LEFTFIRE
ACTIONS = {0: "NOOP", 1: "FIRE", 2: "UP", 3: "DOWN", 4: "UPFIRE", 5: "DOWNFIRE"}
