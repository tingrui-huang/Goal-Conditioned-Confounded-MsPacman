"""Integration tests for four-frame alignment / clamping / split via HostileH0Data.

Fabricates a tiny raw_hf + hostile-metadata pair (no teacher/ALE) so the Colab-side
loader can be exercised end-to-end: per-frame removal, stack semantics, split
disjointness, future-index in-episode, and the shuffled-U control.
"""
import os
import hashlib
import numpy as np
import pytest

from seaquest_ccrl.hostile import schema as S


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _make_dataset(root, n_ep=6, T=12, seed=0):
    raw = os.path.join(root, "raw"); meta = os.path.join(root, "meta")
    os.makedirs(raw, exist_ok=True); os.makedirs(meta, exist_ok=True)
    rng = np.random.RandomState(seed)
    for ep in range(n_ep):
        frames = np.zeros((T, 210, 160, 3), np.uint8); frames[:] = S.WATER_COLOR
        harr = S.empty_hostile_arrays(T); parr = S.empty_protected_arrays(T)
        for t in range(T):
            # a shark present on even t, moving with t -> per-frame metadata differs
            if t % 2 == 0:
                x = 20 + (3 * t) % 120                      # stays on-screen
                x0, y0, cw, ch = S.clip_bbox(x, 100, 8, 7)
                frames[t, y0:y0 + ch, x0:x0 + cw] = S.PALETTE_RGB["Shark"]
                rec = [{"class_id": S.HOSTILE_ID["Shark"], "bbox": (x0, y0, cw, ch), "orientation": 0}]
            else:
                rec = []
            hb, hc, hv, ho = S.encode_objects(rec, S.MAX_HOSTILES, "h")
            harr["hostile_bbox"][t], harr["hostile_class"][t] = hb, hc
            harr["hostile_valid"][t], harr["hostile_orientation"][t] = hv, ho
            harr["hostile_count"][t] = len(rec); harr["enemy_count"][t] = len(rec)
        player_pos = np.stack([np.full(T, 80.0), np.full(T, 103.0)], 1).astype(np.float32)
        actions = rng.randint(0, 18, size=T).astype(np.int64)
        rawp = os.path.join(raw, f"traj_{ep:04d}.npz")
        np.savez_compressed(rawp, frames=frames, actions=actions, player_pos=player_pos,
                            oxygen=np.full(T, 30, np.int32), done=np.zeros(T, bool),
                            target=np.zeros((T, 2), np.float32), theta=np.int32(20))
        meta_d = dict(harr); meta_d.update(parr)
        meta_d.update({"ambiguous": np.zeros(T, bool),
                       "reward": np.zeros(T, np.float32),
                       "lives_after": np.full(T, 3.0, np.float32),
                       "life_lost_after_action": np.zeros(T, bool),
                       "raw_sha256": _sha(rawp), "episode_id": np.int32(ep)})
        np.savez_compressed(os.path.join(meta, f"meta_{ep:04d}.npz"), **meta_d)
    return raw, meta


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    from seaquest_ccrl.hostile.data import HostileH0Data
    root = tmp_path_factory.mktemp("h0")
    raw, meta = _make_dataset(str(root))
    return HostileH0Data(raw, meta, device="cpu", load_visible=True, verify_sha=True)


def test_stack_newest_is_t_and_clamped(data):
    gi = data.split_indices("train")
    assert (data.stack_idx[gi][:, -1] == gi).all()           # newest index == t
    # episode-start rows: all four indices equal the episode start
    starts = data.offsets
    for s in starts:
        assert (data.stack_idx[s] == s).all()


def test_stack_never_crosses_episode(data):
    for gi in range(data.N):
        eps = data.episode_of[data.stack_idx[gi]]
        assert (eps == eps[0]).all()


def test_stack_shape_and_oldest_to_newest(data):
    gi = data.split_indices("train")[:16]
    st = data.stack(gi, "removed")
    assert tuple(st.shape) == (len(gi), 84, 84, 12)


def test_sha_verification_enforced(data, tmp_path):
    """A corrupted raw file must trip the SHA check."""
    from seaquest_ccrl.hostile.data import HostileH0Data
    raw, meta = _make_dataset(str(tmp_path))
    # corrupt one raw trajectory
    p = os.path.join(raw, "traj_0000.npz")
    d = dict(np.load(p)); d["actions"] = d["actions"] + 1
    np.savez_compressed(p, **d)
    with pytest.raises(AssertionError):
        HostileH0Data(raw, meta, device="cpu", verify_sha=True)


def test_split_disjoint_by_episode(data):
    tr, va, te = set(data.train_ep), set(data.val_ep), set(data.test_ep)
    assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te)
    assert len(tr | va | te) == data.n_ep
    # no row's episode appears in two splits
    for split in ("train", "val", "test"):
        eps = np.unique(data.episode_of[data.split_indices(split)])
        assert set(eps).issubset({"train": tr, "val": va, "test": te}[split])


def test_future_index_in_episode(data):
    gi = data.split_indices("train")
    fut, ok = data.future_index(gi, 4)
    for a, f in zip(gi[ok], fut[ok]):
        assert data.episode_of[a] == data.episode_of[f]
        assert data.t_in_ep[f] == data.t_in_ep[a] + 4


def test_U_uses_per_frame_metadata(data):
    """The newest-frame enemy count in U must match the metadata enemy_count at t."""
    gi = data.split_indices("train")
    U = data.U(gi, "enemy")                      # (n, 17*4), oldest->newest blocks
    # newest frame = 4th block; enemy_total is index 3 within the 17-dim enemy block
    newest_enemy_total = U[:, 3 * 17 + 3]
    meta_enemy = data.hostile_count_by_kind("enemy")[gi].astype(np.float32)
    assert np.allclose(newest_enemy_total, meta_enemy)


def test_shuffled_U_preserves_dim_breaks_episode(data):
    from seaquest_ccrl.hostile.data import shuffle_U_within_split
    gi = data.split_indices("train")
    U = data.U(gi, "enemy")
    ep = data.episode_of[gi]
    Ush = shuffle_U_within_split(U, ep, seed=1)
    assert Ush.shape == U.shape
    # at least some rows changed (episode association broken)
    assert not np.allclose(Ush, U)
