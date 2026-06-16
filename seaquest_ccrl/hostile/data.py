"""Colab-side loader: raw_hf frames + hostile metadata -> removed four-frame states.

This module performs NO environment work (no ALE/OCAtari/jax). It:
  * loads the FROZEN raw_hf trajectories and the separately stored hostile metadata;
  * verifies every raw trajectory SHA256 against the value recorded in the metadata;
  * applies class-aware PIXEL removal to every frame using THAT frame's own metadata;
  * resizes to 84x84 and exposes four-frame stacks with the EXACT frozen semantics
    (oldest->newest, clamped to episode start, newest = pre-action frame t);
  * builds the fixed per-frame hostile feature blocks and the U_* stack vectors;
  * provides the episode-level 70/15/15 split (seed 2606) and probe targets.

Memory model mirrors Phase2Data: store per-frame removed (and optionally visible)
84x84x3 frames flat (N,84,84,3) and build 12-channel stacks on the fly via stack_idx.
"""
import os
import glob
import json
import hashlib

import numpy as np
import torch

from seaquest_ccrl.models.sa_encoder import preprocess_frames
from seaquest_ccrl.hostile import schema as S
from seaquest_ccrl.hostile import removal as R
from seaquest_ccrl.hostile import features as F

FRAME_SIZE = 84
K_STACK = 4
SPLIT_SEED = 2606

# Frozen S0.5 semantic action categories (identical to seaquest_stage_s05.common.SEMANTIC).
SEMANTIC = {"NOOP": [0], "FIRE_only": [1], "UP": [2], "DOWN": [5], "LEFT": [4], "RIGHT": [3],
            "UP_FIRE": [10], "DOWN_FIRE": [13], "LEFT_FIRE": [12], "RIGHT_FIRE": [11],
            "other_diag_move": [6, 7, 8, 9], "other_diag_move_fire": [14, 15, 16, 17]}
CAT_NAMES = list(SEMANTIC.keys())
ACTION_TO_CAT = {a: i for i, (k, v) in enumerate(SEMANTIC.items()) for a in v}
N_ACTIONS = 18


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class HostileH0Data:
    def __init__(self, raw_root, meta_root, palettes=None, tol=R.DEFAULT_TOL,
                 device="cpu", load_visible=False, verify_sha=True, limit_episodes=None):
        self.raw_root = raw_root
        self.meta_root = meta_root
        self.device = device
        self.palettes = palettes
        self.tol = tol
        raw_files = sorted(glob.glob(os.path.join(raw_root, "traj_*.npz")))
        meta_files = sorted(glob.glob(os.path.join(meta_root, "meta_*.npz")))
        assert raw_files, f"no raw_hf trajectories under {raw_root}"
        assert len(raw_files) == len(meta_files), (
            f"raw/meta count mismatch: {len(raw_files)} raw vs {len(meta_files)} meta")
        if limit_episodes:
            raw_files = raw_files[:limit_episodes]
            meta_files = meta_files[:limit_episodes]

        rem_frames, vis_frames = [], []
        actions, oxygen, ppos, done = [], [], [], []
        reward, lives_after, life_lost = [], [], []
        hb, hc, hv, ho = [], [], [], []
        pb, pc, pv = [], [], []
        ambiguous = []
        lengths = []
        self.sha_checked = []

        for rf, mf in zip(raw_files, meta_files):
            d = np.load(rf)
            m = np.load(mf, allow_pickle=False)
            raw_sha = str(m["raw_sha256"]) if "raw_sha256" in m.files else None
            if verify_sha:
                got = _sha256_file(rf)
                if raw_sha is None or got != raw_sha:
                    raise AssertionError(
                        f"RAW SHA mismatch for {os.path.basename(rf)}: "
                        f"meta={raw_sha} file={got}")
                self.sha_checked.append({"file": os.path.basename(rf), "sha256": got})
            frames = d["frames"]                                  # (T,210,160,3) u8
            T = len(frames)
            harr_t = {k: m[k] for k in ("hostile_bbox", "hostile_class", "hostile_valid",
                                        "hostile_orientation", "hostile_count",
                                        "enemy_count", "enemy_missile_count")}
            parr_t = {k: m[k] for k in ("protected_bbox", "protected_class", "protected_valid")}
            S.validate_hostile_arrays(harr_t)
            # remove hostiles per frame with that frame's metadata; resize; verify invariants
            removed_full = np.empty_like(frames)
            for t in range(T):
                ht = {k: v[t] for k, v in harr_t.items() if v.ndim >= 1 and v.shape[0] == T}
                pt = {k: v[t] for k, v in parr_t.items()}
                rem, changed, _ = R.remove_frame(frames[t], ht, pt, self.palettes, self.tol)
                R.verify_removal(frames[t], rem, changed, ht, pt, self.palettes, self.tol)
                removed_full[t] = rem
            rem_frames.append(preprocess_frames(removed_full, FRAME_SIZE))
            if load_visible:
                vis_frames.append(preprocess_frames(frames, FRAME_SIZE))

            actions.append(np.asarray(d["actions"], dtype=np.int64))
            oxygen.append(np.asarray(d["oxygen"], dtype=np.float32))
            ppos.append(np.asarray(d["player_pos"], dtype=np.float32))
            done.append(np.asarray(d["done"], dtype=bool))
            reward.append(np.asarray(m["reward"], dtype=np.float32) if "reward" in m.files
                          else np.full(T, np.nan, np.float32))
            lives_after.append(np.asarray(m["lives_after"], dtype=np.float32) if "lives_after" in m.files
                               else np.full(T, np.nan, np.float32))
            life_lost.append(np.asarray(m["life_lost_after_action"], dtype=bool) if "life_lost_after_action" in m.files
                             else np.zeros(T, bool))
            hb.append(harr_t["hostile_bbox"]); hc.append(harr_t["hostile_class"])
            hv.append(harr_t["hostile_valid"]); ho.append(harr_t["hostile_orientation"])
            pb.append(parr_t["protected_bbox"]); pc.append(parr_t["protected_class"])
            pv.append(parr_t["protected_valid"])
            amb = (m["ambiguous"].astype(bool) if "ambiguous" in m.files
                   else np.array([R.ambiguous_row({k: v[t] for k, v in harr_t.items()
                                                   if v.shape and v.shape[0] == T},
                                                  {k: v[t] for k, v in parr_t.items()})
                                  for t in range(T)], dtype=bool))
            ambiguous.append(amb)
            lengths.append(T)

        self.lengths = np.asarray(lengths, dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)[:-1]]).astype(np.int64)
        self.n_ep = len(lengths)
        self.episode_of = np.repeat(np.arange(self.n_ep), self.lengths)
        self.t_in_ep = np.concatenate([np.arange(L) for L in self.lengths])
        N = int(self.lengths.sum())
        self.N = N
        ep_start = np.repeat(self.offsets, self.lengths)
        ar = np.arange(N)
        cols = [np.maximum(ep_start, ar - (K_STACK - 1) + j) for j in range(K_STACK)]
        self.stack_idx = np.stack(cols, axis=1).astype(np.int64)        # (N,4) oldest->newest

        self.frames_removed = torch.from_numpy(np.concatenate(rem_frames, axis=0)).to(device)
        self.frames_visible = (torch.from_numpy(np.concatenate(vis_frames, axis=0)).to(device)
                               if load_visible else None)
        self.actions = np.concatenate(actions)
        self.oxygen = np.concatenate(oxygen)
        self.player_pos = np.concatenate(ppos, axis=0)
        self.done = np.concatenate(done)
        self.reward = np.concatenate(reward)
        self.lives_after = np.concatenate(lives_after)
        self.life_lost = np.concatenate(life_lost)
        self.hostile_bbox = np.concatenate(hb, axis=0)
        self.hostile_class = np.concatenate(hc, axis=0)
        self.hostile_valid = np.concatenate(hv, axis=0)
        self.hostile_orientation = np.concatenate(ho, axis=0)
        self.protected_bbox = np.concatenate(pb, axis=0)
        self.protected_class = np.concatenate(pc, axis=0)
        self.protected_valid = np.concatenate(pv, axis=0)
        self.ambiguous = np.concatenate(ambiguous)

        # episode-level split (seed 2606), 70/15/15 — identical recipe to Phase2Data
        rng = np.random.RandomState(SPLIT_SEED)
        order = rng.permutation(self.n_ep)
        ntr = int(round(0.70 * self.n_ep)); nva = int(round(0.15 * self.n_ep))
        self.train_ep = sorted(int(x) for x in order[:ntr])
        self.val_ep = sorted(int(x) for x in order[ntr:ntr + nva])
        self.test_ep = sorted(int(x) for x in order[ntr + nva:])

        self._per_frame_feats = None     # lazily built (N,33)

    # -- stacks --------------------------------------------------------------
    def stack(self, gidx, view="removed"):
        """(len,84,84,12) uint8, oldest->newest, ending at gidx (=pre-action frame t)."""
        src = self.frames_removed if view == "removed" else self.frames_visible
        assert src is not None, f"view {view} not loaded"
        idx = torch.as_tensor(self.stack_idx[gidx], device=src.device)
        st = src[idx]                                                  # (len,4,84,84,3)
        L, k, H, W, C = st.shape
        return st.permute(0, 2, 3, 1, 4).reshape(L, H, W, k * C)

    def split_indices(self, which):
        eps = {"train": self.train_ep, "val": self.val_ep, "test": self.test_ep}[which]
        return np.where(np.isin(self.episode_of, eps))[0]

    # -- per-frame features + U stacks --------------------------------------
    def _harr_row(self, j):
        return {"hostile_bbox": self.hostile_bbox[j], "hostile_class": self.hostile_class[j],
                "hostile_valid": self.hostile_valid[j]}

    def per_frame_feats(self):
        if self._per_frame_feats is None:
            feats = np.zeros((self.N, F.PER_FRAME), dtype=np.float32)
            for j in range(self.N):
                feats[j] = F.per_frame_features(self._harr_row(j), self.player_pos[j])
            self._per_frame_feats = feats
        return self._per_frame_feats

    def U(self, gidx, which):
        """Return U_{which}_stack rows (len, dim) for global indices gidx.

        which in {enemy, missile, joint}. Built by concatenating per-frame blocks over
        the four stacked frame indices (oldest->newest)."""
        pf = self.per_frame_feats()
        idx = self.stack_idx[gidx]                                     # (len,4)
        block = pf[idx]                                                # (len,4,33)
        if which == "enemy":
            sub = block[:, :, 0:F.ENEMY_BLOCK]
        elif which == "missile":
            sub = block[:, :, F.ENEMY_BLOCK:F.ENEMY_BLOCK + F.MISSILE_BLOCK]
        elif which == "joint":
            sub = block
        else:
            raise ValueError(which)
        return sub.reshape(len(gidx), -1).astype(np.float32)

    def U_dim(self, which):
        return {"enemy": F.ENEMY_BLOCK * 4, "missile": F.MISSILE_BLOCK * 4,
                "joint": F.PER_FRAME * 4}[which]

    # -- hiddenness targets (from PRE-REMOVAL metadata at frame t = gidx) ----
    def hidden_targets(self, gidx, which):
        grid = np.zeros((len(gidx), 9), dtype=np.int64)
        pres = np.zeros(len(gidx), dtype=np.int64)
        cnt = np.zeros(len(gidx), dtype=np.int64)
        ndx = np.zeros(len(gidx), dtype=np.float32)
        ndy = np.zeros(len(gidx), dtype=np.float32)
        nmiss = np.zeros(len(gidx), dtype=np.int64)
        for r, j in enumerate(gidx):
            ht = self._harr_row(j)
            grid[r] = F.presence_grid(ht, self.player_pos[j], which)
            pres[r] = F.presence(ht, which)
            cnt[r] = F.clipped_count(ht, which)
            dx, dy, ms = F.nearest_offset(ht, self.player_pos[j], which)
            ndx[r], ndy[r], nmiss[r] = dx, dy, int(ms)
        return {"grid": grid, "presence": pres, "count": cnt,
                "nearest_dx": ndx, "nearest_dy": ndy, "nearest_missing": nmiss}

    # -- action targets ------------------------------------------------------
    def action_targets(self, gidx):
        a = self.actions[gidx]
        cat = np.array([ACTION_TO_CAT[int(x)] for x in a], dtype=np.int64)
        return {"action18": a.astype(np.int64), "action_cat12": cat}

    # -- future targets (no episode crossing) --------------------------------
    def future_index(self, gidx, H):
        t = self.t_in_ep[gidx]; L = self.lengths[self.episode_of[gidx]]
        ok = (t + H) < L
        fut = np.where(ok, gidx + H, -1)
        return fut, ok

    def future_targets(self, gidx, H):
        """Primary player targets + secondary event targets. Returns (dict, valid_mask)."""
        fut, ok = self.future_index(gidx, H)
        gi = gidx[ok]; fi = fut[ok]
        pp_t = self.player_pos[gi]; pp_f = self.player_pos[fi]
        term = np.zeros(len(gi), dtype=np.int64)
        life = np.zeros(len(gi), dtype=np.int64)
        rew = np.zeros(len(gi), dtype=np.float32)
        for n, (a, b) in enumerate(zip(gi, fi)):
            term[n] = int(self.done[a + 1:b + 1].any())
            life[n] = int(self.life_lost[a + 1:b + 1].any()) if np.isfinite(self.life_lost[a:b+1]).all() else 0
            seg = self.reward[a + 1:b + 1]
            rew[n] = float(np.nansum(seg)) if seg.size else 0.0
        tgt = {
            "future_player_x": pp_f[:, 0], "future_player_y": pp_f[:, 1],
            "displacement_x": pp_f[:, 0] - pp_t[:, 0], "displacement_y": pp_f[:, 1] - pp_t[:, 1],
            "life_loss_before_H": life, "termination_before_H": term,
            "cumulative_reward_to_H": rew,
        }
        return tgt, ok

    # -- support / active masks ---------------------------------------------
    def active_mask(self, which):
        """Rows where the component is present (enemy_present / missile_present)."""
        if which == "enemy":
            return (self.hostile_count_by_kind("enemy") > 0)
        return (self.hostile_count_by_kind("missile") > 0)

    def hostile_count_by_kind(self, which):
        cls, valid = self.hostile_class, self.hostile_valid
        sel = np.isin(cls, S.ENEMY_IDS) if which == "enemy" else np.isin(cls, S.MISSILE_IDS)
        return (sel & valid).sum(1)

    # -- predeclared support (Section 12) -----------------------------------
    SUPPORT_MIN = {"enemy": {"transitions": 1000, "episodes": 20},
                   "missile": {"transitions": 300, "episodes": 10}}

    def support_summary(self):
        en = self.hostile_count_by_kind("enemy") > 0
        mi = self.hostile_count_by_kind("missile") > 0
        joint = en | mi
        out = {"N": int(self.N), "n_episodes": int(self.n_ep),
               "ambiguous_rows": int(self.ambiguous.sum()),
               "ambiguous_fraction": float(self.ambiguous.mean())}
        for which, mask in [("enemy", en), ("missile", mi), ("joint", joint)]:
            eps = np.unique(self.episode_of[mask])
            mn = self.SUPPORT_MIN.get(which)
            d = {"active_transitions": int(mask.sum()),
                 "active_fraction": float(mask.mean()),
                 "active_episodes": int(len(eps))}
            for split in ("train", "val", "test"):
                si = self.split_indices(split)
                d[f"active_{split}"] = int(mask[si].sum())
            if mn:
                d["pass"] = bool(d["active_transitions"] >= mn["transitions"]
                                 and d["active_episodes"] >= mn["episodes"])
                d["min_required"] = mn
                d["code"] = (None if d["pass"] else
                             ("ENEMY_INSUFFICIENT_SUPPORT" if which == "enemy"
                              else "MISSILE_INSUFFICIENT_SUPPORT"))
            else:  # joint inherits: passes if either enemy or missile passes
                d["pass"] = None
            out[which] = d
        # joint pass = enemy pass OR missile pass
        out["joint"]["pass"] = bool(out["enemy"]["pass"] or out["missile"]["pass"])
        # per-class counts
        cls, val = self.hostile_class, self.hostile_valid
        out["per_class_object_count"] = {
            name: int(((cls == cid) & val).sum()) for name, cid in S.HOSTILE_ID.items()}
        return out

    def manifest(self):
        return {"split_seed": SPLIT_SEED, "n_episodes": int(self.n_ep),
                "train_episode_ids": self.train_ep, "val_episode_ids": self.val_ep,
                "test_episode_ids": self.test_ep,
                "n_train_ep": len(self.train_ep), "n_val_ep": len(self.val_ep),
                "n_test_ep": len(self.test_ep),
                "frame_stack": K_STACK, "frame_size": FRAME_SIZE,
                "semantic_categories": SEMANTIC,
                "feature_schema": F.feature_schema(),
                "removal_tol": self.tol}


def shuffle_U_within_split(U, episodes, seed=0):
    """Episode-shuffled U control: permute U rows across DIFFERENT episodes only,
    preserving the U dimension but breaking the episode<->state association.

    Each row gets a U vector drawn (without replacement at episode granularity) from
    a row in a DIFFERENT episode. Same shape out.
    """
    rng = np.random.RandomState(seed)
    out = U.copy()
    uniq = np.unique(episodes)
    # map each episode to a different donor episode
    donors = uniq.copy()
    rng.shuffle(donors)
    for _ in range(8):
        if not np.any(donors == uniq):
            break
        rng.shuffle(donors)
    donor_of = {int(e): int(d) for e, d in zip(uniq, donors)}
    for e in uniq:
        rows = np.where(episodes == e)[0]
        drows = np.where(episodes == donor_of[int(e)])[0]
        pick = rng.choice(drows, size=len(rows), replace=True)
        out[rows] = U[pick]
    return out
