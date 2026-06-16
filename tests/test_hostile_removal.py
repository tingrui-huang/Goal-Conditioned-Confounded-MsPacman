"""Deterministic tests for class-aware pixel removal + invariants."""
import numpy as np
import pytest

from seaquest_ccrl.hostile import schema as S
from seaquest_ccrl.hostile import removal as R


def _frame():
    fr = np.zeros((210, 160, 3), np.uint8)
    fr[:] = S.WATER_COLOR
    return fr


def _arrs(hostiles, protecteds):
    harr = S.empty_hostile_arrays(1)
    parr = S.empty_protected_arrays(1)
    hb, hc, hv, ho = S.encode_objects(
        [{"class_id": c, "bbox": b, "orientation": 0} for c, b in hostiles], S.MAX_HOSTILES, "h")
    harr["hostile_bbox"][0], harr["hostile_class"][0], harr["hostile_valid"][0] = hb, hc, hv
    harr["hostile_count"][0] = len(hostiles)
    harr["enemy_count"][0] = sum(1 for c, _ in hostiles if c in S.ENEMY_IDS)
    harr["enemy_missile_count"][0] = sum(1 for c, _ in hostiles if c in S.MISSILE_IDS)
    pb, pc, pv, _ = S.encode_objects(
        [{"class_id": c, "bbox": b, "orientation": 0} for c, b in protecteds], S.MAX_PROTECTED, "p")
    parr["protected_bbox"][0], parr["protected_class"][0], parr["protected_valid"][0] = pb, pc, pv
    return {k: v[0] for k, v in harr.items()}, {k: v[0] for k, v in parr.items()}


def test_only_class_pixels_changed_and_water_outside_identical():
    fr = _frame()
    fr[100:107, 40:48] = S.PALETTE_RGB["Shark"]
    ht, pt = _arrs([(S.HOSTILE_ID["Shark"], (40, 100, 8, 7))], [])
    rem, changed, stats = R.remove_frame(fr, ht, pt)
    R.verify_removal(fr, rem, changed, ht, pt)
    # shark pixels -> water; everything else byte-identical
    assert changed.sum() == 8 * 7
    assert np.array_equal(rem[100:107, 40:48],
                          np.broadcast_to(np.array(S.WATER_COLOR, np.uint8), (7, 8, 3)))
    outside = changed.copy(); outside[100:107, 40:48] = False
    assert outside.sum() == 0


def test_changed_subset_of_bbox_union():
    fr = _frame()
    fr[100:107, 40:48] = S.PALETTE_RGB["Shark"]
    # paint a stray shark-coloured pixel OUTSIDE any bbox: must NOT be removed
    fr[5, 5] = S.PALETTE_RGB["Shark"]
    ht, pt = _arrs([(S.HOSTILE_ID["Shark"], (40, 100, 8, 7))], [])
    rem, changed, _ = R.remove_frame(fr, ht, pt)
    R.verify_removal(fr, rem, changed, ht, pt)
    assert not changed[5, 5]
    assert np.array_equal(rem[5, 5], fr[5, 5])


def test_protected_diver_pixels_preserved_under_blue_missile():
    """EnemyMissile and Diver share blue; a missile bbox overlapping a diver must NOT
    erase diver pixels."""
    fr = _frame()
    # diver region (protected) and an overlapping enemy-missile bbox, both blue
    fr[140:151, 30:38] = S.PALETTE_RGB["Diver"]
    ht, pt = _arrs([(S.HOSTILE_ID["EnemyMissile"], (28, 142, 12, 6))],
                   [(S.PROTECTED_ID["Diver"], (30, 140, 8, 11))])
    rem, changed, stats = R.remove_frame(fr, ht, pt)
    R.verify_removal(fr, rem, changed, ht, pt)
    # diver pixels (inside its protected bbox) unchanged
    assert np.array_equal(fr[140:151, 30:38], rem[140:151, 30:38])
    assert stats["ambiguous"] is True


def test_zero_compatible_flagged():
    fr = _frame()  # no shark sprite painted, but a shark bbox is declared
    ht, pt = _arrs([(S.HOSTILE_ID["Shark"], (40, 100, 8, 7))], [])
    rep = R.compatible_pixel_report(fr, ht)
    assert rep[0]["zero_compatible"] is True and rep[0]["compatible_px"] == 0


def test_player_pixels_never_removed():
    fr = _frame()
    fr[46:57, 76:92] = S.PALETTE_RGB["Player"]
    fr[100:107, 40:48] = S.PALETTE_RGB["Shark"]
    ht, pt = _arrs([(S.HOSTILE_ID["Shark"], (40, 100, 8, 7))],
                   [(S.PROTECTED_ID["Player"], (76, 46, 16, 11))])
    rem, changed, _ = R.remove_frame(fr, ht, pt)
    R.verify_removal(fr, rem, changed, ht, pt)
    assert np.array_equal(fr[46:57, 76:92], rem[46:57, 76:92])


def test_resolved_palette_dict_and_list_forms():
    fr = _frame()
    obs = (90, 184, 90)                      # observed shark colour (slightly off RAM default)
    fr[100:107, 40:48] = obs
    ht, pt = _arrs([(S.HOSTILE_ID["Shark"], (40, 100, 8, 7))], [])
    # dict form with per-class tol
    pal_dict = {"Shark": {"rgb": [list(obs)], "tol": 4}}
    rem, changed, _ = R.remove_frame(fr, ht, pt, palettes=pal_dict, tol=0)
    R.verify_removal(fr, rem, changed, ht, pt, palettes=pal_dict, tol=0)
    assert changed.sum() == 8 * 7
    # list form (match any of several observed colours)
    pal_list = {"Shark": [list(obs), [92, 186, 92]]}
    m = R.class_compatible_mask(fr[100:107, 40:48], "Shark", pal_list, tol=2)
    assert m.sum() == 8 * 7


def test_remove_is_pure():
    fr = _frame(); fr[100:107, 40:48] = S.PALETTE_RGB["Shark"]
    before = fr.copy()
    ht, pt = _arrs([(S.HOSTILE_ID["Shark"], (40, 100, 8, 7))], [])
    R.remove_frame(fr, ht, pt)
    assert np.array_equal(fr, before)   # input not mutated
