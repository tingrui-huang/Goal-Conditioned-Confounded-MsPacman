"""Deterministic tests for the frozen Stage-H0 hostile metadata schema."""
import numpy as np
import pytest

from seaquest_ccrl.hostile import schema as S
from seaquest_ccrl.hostile import extraction as EX


class _Obj:
    def __init__(self, cat, x, y, w, h, ori=None):
        self.category, self.x, self.y, self.w, self.h = cat, x, y, w, h
        if ori is not None:
            self.orientation = type("O", (), {"name": ori})()


def test_class_id_maps_frozen():
    assert S.HOSTILE_ID == {"Shark": 1, "Submarine": 2, "SurfaceSubmarine": 3, "EnemyMissile": 4}
    assert S.PROTECTED_ID == {"Player": 1, "Diver": 2, "PlayerMissile": 3}
    assert S.ENEMY_IDS == (1, 2, 3) and S.MISSILE_IDS == (4,)


def test_encode_decode_roundtrip():
    recs = [{"class_id": 1, "bbox": (10, 20, 8, 7), "orientation": 0},
            {"class_id": 4, "bbox": (5, 6, 6, 4), "orientation": 2}]
    bbox, cls, valid, ori = S.encode_objects(recs, S.MAX_HOSTILES, "hostile")
    assert valid.sum() == 2 and cls[0] == 1 and cls[1] == 4
    assert tuple(bbox[1]) == (5, 6, 6, 4) and ori[1] == 2
    out = S.decode_valid(bbox, cls, valid, ori)
    assert out == recs


def test_encode_overflow_raises():
    recs = [{"class_id": 1, "bbox": (0, 0, 1, 1)}] * (S.MAX_HOSTILES + 1)
    with pytest.raises(ValueError):
        S.encode_objects(recs, S.MAX_HOSTILES, "hostile")


def test_clip_bbox_bounds():
    # off the left/top
    assert S.clip_bbox(-3, -2, 8, 7) == (0, 0, 5, 5)
    # off the right edge
    x0, y0, cw, ch = S.clip_bbox(156, 10, 8, 7, W=160, H=210)
    assert x0 == 156 and x0 + cw <= 160
    # fully off-screen -> zero area
    assert S.clip_bbox(200, 10, 8, 7, W=160, H=210)[2] == 0
    # clipped fraction
    assert S.bbox_clipped_fraction(-4, 0, 8, 8) == pytest.approx(0.5)


def test_validate_hostile_arrays_consistency():
    objs = [_Obj("Shark", 40, 100, 8, 7), _Obj("Submarine", 120, 75, 8, 11, ori="W"),
            _Obj("EnemyMissile", 50, 141, 6, 4), _Obj("Player", 76, 46, 16, 11)]
    harr = S.empty_hostile_arrays(1)
    parr = S.empty_protected_arrays(1)
    ex = EX.extract_objects(objs)
    EX.fill_step(harr, parr, 0, ex)
    S.validate_hostile_arrays(harr)
    assert harr["enemy_count"][0] == 2 and harr["enemy_missile_count"][0] == 1
    assert harr["hostile_count"][0] == 3


def test_validate_rejects_padding_with_class():
    harr = S.empty_hostile_arrays(1)
    harr["hostile_class"][0, 5] = 2          # class set but not valid
    with pytest.raises(AssertionError):
        S.validate_hostile_arrays(harr)


def test_extraction_drops_offscreen_and_keeps_partial():
    objs = [_Obj("Shark", -100, 100, 8, 7),   # fully off-screen -> dropped
            _Obj("Shark", -3, 100, 8, 7)]     # partially on -> kept clipped
    ex = EX.extract_objects(objs)
    assert len(ex["hostile"]) == 1
    assert any(d["reason"] == "off_screen_after_clip" for d in ex["dropped"])
    assert ex["hostile"][0]["bbox"][0] == 0   # clipped to left edge


def test_orientation_codes():
    objs = [_Obj("Submarine", 10, 10, 8, 11, ori="W"),
            _Obj("Submarine", 30, 10, 8, 11, ori="E")]
    ex = EX.extract_objects(objs)
    oris = {tuple(r["bbox"]): r["orientation"] for r in ex["hostile"]}
    assert oris[(10, 10, 8, 11)] == S.ORI_LEFT
    assert oris[(30, 10, 8, 11)] == S.ORI_RIGHT
