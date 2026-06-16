"""Deterministic tests for the fixed four-frame U feature representation."""
import numpy as np
import pytest

from seaquest_ccrl.hostile import schema as S
from seaquest_ccrl.hostile import features as F


def _harr(hostiles):
    harr = S.empty_hostile_arrays(1)
    hb, hc, hv, ho = S.encode_objects(
        [{"class_id": c, "bbox": b, "orientation": 0} for c, b in hostiles], S.MAX_HOSTILES, "h")
    harr["hostile_bbox"][0], harr["hostile_class"][0], harr["hostile_valid"][0] = hb, hc, hv
    return {k: v[0] for k, v in harr.items()}


def test_per_frame_dim_and_finite():
    ht = _harr([(S.HOSTILE_ID["Shark"], (40, 100, 8, 7))])
    ff = F.per_frame_features(ht, (84.0, 51.0))
    assert ff.shape == (F.PER_FRAME,) and np.isfinite(ff).all()


def test_missing_flags_when_no_object():
    ht = _harr([])
    ff = F.per_frame_features(ht, (84.0, 51.0))
    # nearest_enemy_missing index in enemy block = 7 ; nearest_missile_missing = enemy_block+4
    assert ff[7] == 1.0
    assert ff[F.ENEMY_BLOCK + 4] == 1.0
    assert np.isfinite(ff).all()


def test_non_finite_player_is_handled():
    ht = _harr([(S.HOSTILE_ID["Shark"], (40, 100, 8, 7))])
    ff = F.per_frame_features(ht, (np.nan, np.nan))
    assert np.isfinite(ff).all()
    assert ff[7] == 1.0   # nearest enemy missing because player unknown


def test_grid_cell_assignment():
    # enemy to the upper-left of the player -> cell row0,col0 = 0
    ht = _harr([(S.HOSTILE_ID["Shark"], (0, 0, 8, 7))])
    g = F.presence_grid(ht, (120.0, 120.0), "enemy")
    assert g[0] == 1 and g.sum() == 1
    # enemy at player centre -> centre cell 4
    ht2 = _harr([(S.HOSTILE_ID["Shark"], (116, 116, 8, 7))])
    g2 = F.presence_grid(ht2, (120.0, 120.0), "enemy")
    assert g2[4] == 1


def test_stack_dims():
    ht = _harr([(S.HOSTILE_ID["Shark"], (40, 100, 8, 7)),
                (S.HOSTILE_ID["EnemyMissile"], (50, 60, 6, 4))])
    ff = F.per_frame_features(ht, (84.0, 51.0))
    st = np.stack([ff, ff, ff, ff])
    U = F.stack_features(st)
    assert U["U_enemy_stack"].shape == (F.ENEMY_BLOCK * 4,)
    assert U["U_missile_stack"].shape == (F.MISSILE_BLOCK * 4,)
    assert U["U_joint_stack"].shape == (F.PER_FRAME * 4,)


def test_clipped_count():
    ht = _harr([(S.HOSTILE_ID["Shark"], (i * 10, 100, 8, 7)) for i in range(5)])
    assert F.clipped_count(ht, "enemy") == F.CLIP_COUNT   # 5 clipped to 3


def test_normalizer_train_only():
    rng = np.random.RandomState(0)
    Xtr = rng.randn(100, 5).astype(np.float32) * 3 + 2
    Xte = rng.randn(20, 5).astype(np.float32)
    nz = F.Normalizer().fit(Xtr)
    out = nz.apply(Xte)
    # stats come from train: train mean ~0 after apply, not test
    assert np.allclose(((Xtr - nz.mu) / nz.sd).mean(0), 0, atol=1e-5)
    assert out.shape == Xte.shape
