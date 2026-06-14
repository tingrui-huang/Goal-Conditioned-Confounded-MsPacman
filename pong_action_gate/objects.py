"""Authoritative Pong object extraction from ALE RAM, aligned with the teacher's env.

RAM offsets are taken from OCAtari (`OC_Atari/ocatari/ram/pong.py`), read from the
SAME ALE instance the teacher steps (`env.env.ale.getRAM()`), so object positions
are exactly aligned with the teacher's observations — no second env, no desync.

Native ALE frame is 210 (H) x 160 (W). The teacher frame is a full-frame resize to
64x64 (AtariPreprocessing: cv2.resize 210x160 -> 64x64, no crop), so:
    x64 = x_native * 64/160 ,  y64 = y_native * 64/210 .
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

NATIVE_W, NATIVE_H = 160, 210
IMG = 64
SX, SY = IMG / NATIVE_W, IMG / NATIVE_H

# Own (right) paddle and opponent (left) paddle fixed x columns + size (OCAtari).
PLAYER_X, OPP_X, PADDLE_W, PADDLE_H = 140, 16, 4, 15
BALL_W, BALL_H = 2, 4


@dataclass
class PongObjects:
    ball_present: bool
    ball_x: Optional[float]      # native, field-relative (ram[49]-49)
    ball_y: Optional[float]      # native (ram[54]-14)
    player_y: Optional[float]    # native top-y of own (right) paddle (ram[51]-13)
    opp_y: Optional[float]       # native top-y of opponent (left) paddle (ram[50]-15)
    player_score: int            # ram[14]
    enemy_score: int             # ram[13]

    def player_cy(self) -> Optional[float]:
        return None if self.player_y is None else self.player_y + PADDLE_H / 2

    def opp_cy(self) -> Optional[float]:
        return None if self.opp_y is None else self.opp_y + PADDLE_H / 2


def extract_pong_objects(ram: np.ndarray) -> PongObjects:
    ram = np.asarray(ram).astype(np.int64)
    # ball
    if ram[54] != 0 and ram[49] > 49:
        ball_present = True
        ball_x = float(ram[49] - 49)
        ball_y = float(ram[54] - 14)
    else:
        ball_present, ball_x, ball_y = False, None, None
    # opponent (enemy, left paddle)
    opp_y = float(ram[50] - 15) if ram[50] > 33 else None
    # player (own, right paddle)
    player_y = float(ram[51] - 13) if ram[51] > 13 else None
    return PongObjects(
        ball_present=ball_present, ball_x=ball_x, ball_y=ball_y,
        player_y=player_y, opp_y=opp_y,
        player_score=int(ram[14]), enemy_score=int(ram[13]),
    )


def opp_paddle_box64(obj: PongObjects, margin: int = 1):
    """Opponent-paddle bounding box in 64x64 coords (x0,y0,x1,y1), or None if absent."""
    if obj.opp_y is None:
        return None
    x0 = int(np.floor(OPP_X * SX)) - margin
    x1 = int(np.ceil((OPP_X + PADDLE_W) * SX)) + margin
    y0 = int(np.floor(obj.opp_y * SY)) - margin
    y1 = int(np.ceil((obj.opp_y + PADDLE_H) * SY)) + margin
    return (max(0, x0), max(0, y0), min(IMG, x1), min(IMG, y1))


def region_boxes64(obj: PongObjects):
    """Named (x0,y0,x1,y1) regions in 64x64 for saliency attribution; None if absent."""
    boxes = {}

    def box(cx_native, cy_native, w, h):
        x0 = int(np.floor((cx_native) * SX))
        x1 = int(np.ceil((cx_native + w) * SX))
        y0 = int(np.floor((cy_native) * SY))
        y1 = int(np.ceil((cy_native + h) * SY))
        return (max(0, x0), max(0, y0), min(IMG, x1), min(IMG, y1))

    if obj.ball_present:
        # ball_x/ball_y are already native 160x210 screen coords (OCAtari convention).
        boxes["ball"] = box(obj.ball_x, obj.ball_y, BALL_W + 2, BALL_H + 2)
    if obj.player_y is not None:
        boxes["own_paddle"] = box(PLAYER_X, obj.player_y, PADDLE_W, PADDLE_H)
    if obj.opp_y is not None:
        boxes["opp_paddle"] = box(OPP_X, obj.opp_y, PADDLE_W, PADDLE_H)
    # scoreboard occupies the top band (native y in ~[1,21] across the width)
    boxes["scoreboard"] = (0, 0, IMG, int(np.ceil(22 * SY)))
    return boxes
