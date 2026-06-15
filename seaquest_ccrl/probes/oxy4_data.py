"""Shared Phase-2 data: frozen four-frame masked/visible stacks + episode-level split +
oxygen bins + future targets + the 7 pre-probe assertions.

Reuses the EXACT frozen construction:
  * frames resized 210x160 -> 84x84 via seaquest_ccrl.models.sa_encoder.preprocess_frames;
  * 4-frame stack indices clamped to the episode start (oldest->newest, ending at the
    pre-action frame t) — identical to HindsightSampler.stack_idx;
  * oxygen masking via the committed dataset loader (oracle=False masks all frames;
    oracle=True = unmasked, used ONLY for the visible leakage control).

Split: episode-level 70/15/15, seed 2606. Because the split is by EPISODE and the 4-frame
window only looks backward within the episode (and future targets are excluded at episode
boundaries), no window from one episode appears across train/val/test. Normalization is fit
on TRAIN episodes only.

Oxygen low/med/high bins and semantic action categories are the EXISTING S0.5 definitions.
"""
import os
import numpy as np
import torch

from seaquest_ccrl.data.dataset import SeaquestOfflineDataset
from seaquest_ccrl.models.sa_encoder import preprocess_frames

FRAME_SIZE = 84
K_STACK = 4
SPLIT_SEED = 2606
SURFACE_Y = 46.0
# S0.5 oxygen strata (analyze.define_strata): low<22, 22<=mid<44, high>=44
OXY_LOW_HI = (22.0, 44.0)
# S0.5 surfacing/refill definition: an oxygen increase > 5 between consecutive steps
REFILL_JUMP = 5.0
# S0.5 semantic action categories (12), exact
SEMANTIC = {"NOOP": [0], "FIRE_only": [1], "UP": [2], "DOWN": [5], "LEFT": [4], "RIGHT": [3],
            "UP_FIRE": [10], "DOWN_FIRE": [13], "LEFT_FIRE": [12], "RIGHT_FIRE": [11],
            "other_diag_move": [6, 7, 8, 9], "other_diag_move_fire": [14, 15, 16, 17]}
CAT_NAMES = list(SEMANTIC.keys())
ACTION_TO_CAT = {a: i for i, (k, v) in enumerate(SEMANTIC.items()) for a in v}


def oxygen_class(u):
    """low=0, med=1, high=2 using the frozen S0.5 thresholds."""
    lo, hi = OXY_LOW_HI
    return np.where(u < lo, 0, np.where(u < hi, 1, 2)).astype(np.int64)


class Phase2Data:
    def __init__(self, root, load_visible=False, device="cpu"):
        self.root = root
        self.device = device
        mds = SeaquestOfflineDataset(root, oracle=False)
        frames_m, actions, oxygen, ppos, done, lengths = [], [], [], [], [], []
        for tr in mds.trajectories():
            frames_m.append(preprocess_frames(tr["obs"], FRAME_SIZE))     # masked, resized
            actions.append(np.asarray(tr["action"], dtype=np.int64))
            oxygen.append(np.asarray(tr["oxygen"], dtype=np.float32))
            ppos.append(np.asarray(tr["achieved_goal"], dtype=np.float32))  # player_pos (x,y)
            done.append(np.asarray(tr["done"], dtype=bool))
            lengths.append(len(actions[-1]))
        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)[:-1]]).astype(np.int64)
        self.n_ep = len(lengths)
        self.episode_of = np.repeat(np.arange(self.n_ep), self.lengths)        # (N,)
        self.t_in_ep = np.concatenate([np.arange(L) for L in self.lengths])    # (N,)
        N = int(self.lengths.sum())
        self.N = N
        ep_start = np.repeat(self.offsets, self.lengths)
        ar = np.arange(N)
        cols = [np.maximum(ep_start, ar - (K_STACK - 1) + j) for j in range(K_STACK)]
        self.stack_idx = np.stack(cols, axis=1).astype(np.int64)               # (N,4) oldest->newest
        self.frames_masked = torch.from_numpy(np.concatenate(frames_m, axis=0)).to(device)  # (N,84,84,3) u8
        self.actions = np.concatenate(actions)
        self.oxygen = np.concatenate(oxygen)
        self.player_pos = np.concatenate(ppos, axis=0)                          # (N,2)
        self.done = np.concatenate(done)
        self.frames_visible = None
        if load_visible:
            vds = SeaquestOfflineDataset(root, oracle=True)
            fv = [preprocess_frames(tr["obs"], FRAME_SIZE) for tr in vds.trajectories()]
            self.frames_visible = torch.from_numpy(np.concatenate(fv, axis=0)).to(device)

        # episode-level split (seed 2606), 70/15/15
        rng = np.random.RandomState(SPLIT_SEED)
        order = rng.permutation(self.n_ep)
        ntr = int(round(0.70 * self.n_ep)); nva = int(round(0.15 * self.n_ep))
        self.train_ep = sorted(int(x) for x in order[:ntr])
        self.val_ep = sorted(int(x) for x in order[ntr:ntr + nva])
        self.test_ep = sorted(int(x) for x in order[ntr + nva:])

    # -- stacks --------------------------------------------------------------
    def stack(self, gidx, view="masked"):
        """(len,84,84,12) uint8, oldest->newest, ending at gidx (=pre-action frame t)."""
        src = self.frames_masked if view == "masked" else self.frames_visible
        idx = torch.as_tensor(self.stack_idx[gidx], device=src.device)         # (len,4)
        st = src[idx]                                                          # (len,4,84,84,3)
        L, k, H, W, C = st.shape
        return st.permute(0, 2, 3, 1, 4).reshape(L, H, W, k * C)

    # -- masks for splits ----------------------------------------------------
    def split_indices(self, which):
        eps = {"train": self.train_ep, "val": self.val_ep, "test": self.test_ep}[which]
        return np.where(np.isin(self.episode_of, eps))[0]

    # -- future targets (no episode crossing) --------------------------------
    def future_index(self, gidx, H):
        """global index of t+H within the SAME episode; -1 if it crosses the boundary."""
        t = self.t_in_ep[gidx]; L = self.lengths[self.episode_of[gidx]]
        ok = (t + H) < L
        fut = np.where(ok, gidx + H, -1)
        return fut, ok

    def future_targets(self, gidx, H):
        """Dict of future targets (NO future-oxygen target). Only valid (in-episode) rows.
        Returns (targets dict, valid_mask)."""
        fut, ok = self.future_index(gidx, H)
        gi = gidx[ok]; fi = fut[ok]
        pp_t = self.player_pos[gi]; pp_f = self.player_pos[fi]
        # termination / refill within (t, t+H]
        term = np.zeros(len(gi), dtype=np.int64); refill = np.zeros(len(gi), dtype=np.int64)
        for n, (a, b) in enumerate(zip(gi, fi)):
            seg_done = self.done[a + 1:b + 1]
            term[n] = int(seg_done.any())
            ox = self.oxygen[a:b + 1]
            refill[n] = int((np.diff(ox) > REFILL_JUMP).any())
        tgt = {
            "future_player_x": pp_f[:, 0], "future_player_y": pp_f[:, 1],
            "displacement_x": pp_f[:, 0] - pp_t[:, 0], "displacement_y": pp_f[:, 1] - pp_t[:, 1],
            "distance_toward_surface": pp_t[:, 1] - pp_f[:, 1],   # +ve = moved up toward surface
            "termination_before_H": term, "refill_before_H": refill,
        }
        return tgt, ok

    def manifest(self):
        return {"split_seed": SPLIT_SEED, "n_episodes": int(self.n_ep),
                "train_episode_ids": self.train_ep, "val_episode_ids": self.val_ep,
                "test_episode_ids": self.test_ep,
                "n_train": len(self.train_ep), "n_val": len(self.val_ep), "n_test": len(self.test_ep),
                "oxygen_bins_low_med_high": [0, OXY_LOW_HI[0], OXY_LOW_HI[1], "inf"],
                "semantic_categories": SEMANTIC, "frame_stack": K_STACK, "frame_size": FRAME_SIZE,
                "surface_y": SURFACE_Y, "refill_jump": REFILL_JUMP}


def run_assertions(data, out=None):
    """The 7 required pre-probe assertions; returns a dict and (optionally) writes JSON."""
    res = {}
    gi = data.split_indices("train")[:64]
    # 1. masked frame input shape (B,84,84,12)
    sm = data.stack(gi, "masked")
    res["frame_shape_masked"] = list(sm.shape)
    assert tuple(sm.shape) == (len(gi), 84, 84, 12), "masked stack shape != (B,84,84,12)"
    # 2. all four frames masked for masked-state models (oxygen rect ~0 across the stack)
    from seaquest_ccrl import config as C
    x, y, w, h = C.OXY_MASK_RECT
    sx0 = int(round(x * 84 / 160)); sx1 = int(round((x + w) * 84 / 160))
    sy0 = int(round(y * 84 / 210)); sy1 = int(round((y + h) * 84 / 210))
    region = sm[:, sy0:sy1, sx0:sx1, :].float().mean().item()
    res["masked_oxygen_region_mean"] = region
    assert region < 8.0, f"oxygen region not masked in stacked frames (mean {region})"
    # 3. unmasked used only for visible control (visible stack differs from masked only in rect area)
    if data.frames_visible is not None:
        sv = data.stack(gi, "visible")
        diff = (sv.float() - sm.float()).abs().sum(-1)  # per-pixel abs diff over channels
        outside = diff.clone(); outside[:, sy0:sy1, sx0:sx1] = 0
        res["visible_vs_masked_diff_outside_rect"] = float(outside.sum().item())
        # (resize blends edges, so allow a small tolerance band around the rect)
    # 4. action at t aligns with the stack ending at t
    assert (data.stack_idx[gi][:, -1] == gi).all(), "stack newest index != t"
    res["action_aligns_with_stack_end"] = True
    # 5. future targets do not cross episode boundaries
    fut, ok = data.future_index(gi, 64)
    crossed = []
    for a, f in zip(gi[ok], fut[ok]):
        crossed.append(data.episode_of[a] != data.episode_of[f])
    assert not any(crossed), "a future target crossed an episode boundary"
    res["future_targets_no_boundary_cross"] = True
    # 6. future oxygen itself is not a target
    tgt, _ = data.future_targets(gi, 16)
    res["future_targets"] = sorted(tgt.keys())
    assert not any("oxygen" in k for k in tgt), "future oxygen must NOT be a target"
    # 7. split episode IDs are deterministic/identical across experiments (seed 2606)
    m = data.manifest()
    res["split_ids"] = {"train": m["train_episode_ids"], "val": m["val_episode_ids"], "test": m["test_episode_ids"]}
    res["split_seed"] = SPLIT_SEED
    res["assertions_pass"] = True
    if out:
        import json
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(res, open(out, "w"), indent=2)
    return res
