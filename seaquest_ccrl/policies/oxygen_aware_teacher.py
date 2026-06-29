"""Oxygen-aware wrapper around the frozen HF teacher (the ONLY behavior change = the oxygen
response). Normal gameplay returns the teacher's action unchanged; when oxygen falls below
SURFACE_TRIGGER the wrapper enters a surfacing mode (ascend) and stays in it -- hysteresis --
until oxygen is refilled to REFILLED (which can only happen at the surface), then hands control
back to the teacher. No other teacher behavior is touched: on every non-surfacing step the action
is byte-identical to the teacher's.

This makes oxygen materially drive vertical actions (the U->A confounding channel) WITHOUT a
scripted goal-seeker -- the realistic HF teacher still hunts/fires/collects divers exactly as
before between surfacings.
"""
from typing import Optional

UP = 2        # ALE Seaquest 'UP' (pure ascent)
UPFIRE = 10   # ascend AND fire -- clears enemies in the ascent path so surfacing survives


class OxygenAwareTeacher:
    def __init__(self, teacher, surface_trigger: int = 20, refilled: int = 58,
                 surface_y: float = 52.0, surface_action: int = UPFIRE):
        self.teacher = teacher
        self.surface_trigger = surface_trigger     # enter surfacing when oxygen < this
        self.refilled = refilled                   # exit surfacing when oxygen >= this
        self.surface_y = surface_y                 # player_y at/above this == at the surface
        self.surface_action = surface_action       # vertical action during surfacing (UPFIRE survives enemies)
        self._surfacing = False

    def reset(self):
        self._surfacing = False

    def _teacher_action(self, teacher_obs, mode, rng):
        if mode == "greedy":
            return int(self.teacher.greedy_action(teacher_obs)[0])
        noise = self.teacher.gumbel_from_uniform(rng.uniform(size=18))
        return int(self.teacher.sample_action(teacher_obs, noise)[0])

    def act(self, teacher_obs, oxygen: Optional[float], player_y: Optional[float] = None,
            mode: str = "greedy", rng=None):
        """Return (action, is_surfacing). `oxygen`/`player_y` come from the env features."""
        if oxygen is not None and oxygen >= 0:
            if oxygen < self.surface_trigger:
                self._surfacing = True
            elif oxygen >= self.refilled:
                self._surfacing = False
        if self._surfacing:
            return self.surface_action, True        # ascend (refills at surface); fire to clear the path
        return self._teacher_action(teacher_obs, mode, rng), False
