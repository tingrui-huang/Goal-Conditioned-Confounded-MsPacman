"""Per-episode schema, serialization, and the k=4 learner-stack builder for M4.

One compressed .npz per episode. Every per-step field is an array of length T.
Raw RGB frames are preserved so learner preprocessing / future masks can be
regenerated without re-collecting teacher trajectories.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Per-step fields (documented; validated in collect.validate_episode).
ARRAY_FIELDS = [
    "timestep",          # (T,) int32
    "raw_rgb",           # (T,210,160,3) uint8  -- raw teacher frame (source of truth)
    "teacher_obs",       # (T,7,64,64)  uint8   -- exact teacher input pre-normalisation (/255*2-1 at use)
    "gray_learner",      # (T,84,84)    uint8   -- learner grayscale frame (stacked k=4 at train time)
    "action",            # (T,) int64           -- sampled behaviour action
    "logits",            # (T,6) float32        -- original teacher logits
    "probs_T",           # (T,6) float32        -- softmax(logits / T)
    "behavior_prob",     # (T,) float32         -- probs_T[action]
    "reward",            # (T,) float32
    "terminated",        # (T,) bool
    "truncated",         # (T,) bool
    "score_diff",        # (T,) int32           -- agent-opp BEFORE this step's reward (obs-time score)
    "agent_score",       # (T,) int32           -- AFTER this step's reward
    "opp_score",         # (T,) int32           -- AFTER this step's reward
    "is_scoring_event",  # (T,) bool            -- reward != 0
    "ball_x", "ball_y",  # (T,) float32 (nan if absent)
    "player_y", "opp_y", # (T,) float32 (nan if absent)
    "ball_present",      # (T,) bool
    "player_valid",      # (T,) bool
    "opp_valid",         # (T,) bool
]
SCALAR_FIELDS = ["episode_id", "env_seed", "temperature", "initial_hx_norm", "initial_cx_norm"]


def save_episode(path: Path, ep: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **ep)
    return sha256(path)


def load_episode(path: Path) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_kstack(gray_learner: np.ndarray, idx: int, k: int = 4) -> np.ndarray:
    """k-frame stack ending at idx, oldest->newest, left-padded to the episode start.

    Indices are clamped to 0 (the episode's first frame) so a stack NEVER crosses the
    episode boundary; early frames are left-padded by repeating frame 0.
    """
    T = gray_learner.shape[0]
    assert 0 <= idx < T
    cols = [max(0, idx - (k - 1) + j) for j in range(k)]   # oldest -> newest
    return gray_learner[cols]                               # (k,84,84)
