"""M6 dataset plumbing: episode-level split, matched anchors, state features, pixel stacks.

State features use ONLY pre-action current/history information (ball & paddle
positions/motion, presence flags, current PRE score difference). They never use reward,
post-action scores, event direction, the sampled future goal, or any future information.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from ..data.goal_sampler import ScoringGoalSampler, build_episode_index, next_event_for
from ..data.schema import build_kstack

ART_ROOT = Path("artifacts/pong_action_gate/m4")

# state-feature layout (all PRE-action). Order documented; 11 dims.
STATE_FEATURE_NAMES = [
    "ball_x_n", "ball_y_n", "ball_present",
    "ball_dx_n", "ball_dy_n",
    "player_y_n", "player_dy_n",
    "opp_y_n", "opp_present", "opp_dy_n",
    "score_diff_pre_n",
]
STATE_DIM = len(STATE_FEATURE_NAMES)


def load_subset(tag: str, episode_ids: List[int], with_pixels: bool) -> List[Dict[str, Any]]:
    paths = sorted((ART_ROOT / tag / "episodes").glob("*.npz"))
    keys = ["is_scoring_event", "reward", "agent_score", "opp_score", "score_diff",
            "action", "ball_x", "ball_y", "ball_present", "player_y", "opp_y",
            "player_valid", "opp_valid"]
    if with_pixels:
        keys.append("gray_learner")
    eps = []
    for i in episode_ids:
        with np.load(paths[i], allow_pickle=False) as z:
            eps.append({k: z[k] for k in keys})
    return eps


def split_episodes(n: int, val_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    val = sorted(perm[:n_val].tolist())
    train = sorted(perm[n_val:].tolist())
    return train, val


def build_state_features(ep: Dict[str, Any]) -> np.ndarray:
    T = len(ep["action"])
    bx = np.nan_to_num(ep["ball_x"], nan=0.0)
    by = np.nan_to_num(ep["ball_y"], nan=0.0)
    py = np.nan_to_num(ep["player_y"], nan=0.0)
    oy = np.nan_to_num(ep["opp_y"], nan=0.0)
    bp = ep["ball_present"].astype(np.float32)
    op = ep["opp_valid"].astype(np.float32)
    pre = ep["score_diff"].astype(np.float32)

    def motion(v, present):
        d = np.zeros(T, np.float32)
        d[1:] = v[1:] - v[:-1]
        # zero motion across absence (no valid previous/current)
        valid = np.zeros(T, bool); valid[1:] = (present[1:] > 0) & (present[:-1] > 0)
        d[~valid] = 0.0
        return d

    bdx = motion(bx, bp); bdy = motion(by, bp)
    pdy = motion(py, np.ones(T)); ody = motion(oy, op)

    feats = np.stack([
        bx / 160.0, by / 210.0, bp,
        np.clip(bdx / 20.0, -1, 1), np.clip(bdy / 20.0, -1, 1),
        py / 210.0, np.clip(pdy / 20.0, -1, 1),
        oy / 210.0, op, np.clip(ody / 20.0, -1, 1),
        pre / 21.0,
    ], axis=1).astype(np.float32)
    return feats   # (T, STATE_DIM)


@dataclass
class Anchor:
    ep: int          # local index into the loaded subset
    t: int
    action: int
    goal: int        # achieved post-event score diff (next_score_event)


def build_anchor_pool(eps: List[Dict[str, Any]], ep_local_ids: List[int],
                      include_immediate: bool = True) -> List[Anchor]:
    """All valid next_score_event anchors (ep,t) -> deterministic (action, goal).

    Goal is the POST-event score diff at the next scoring event in the SAME episode.
    """
    anchors: List[Anchor] = []
    for li in ep_local_ids:
        ep = eps[li]
        eidx = build_episode_index(ep)
        if len(eidx.event_idx) == 0:
            continue
        last = int(eidx.event_idx[-1])
        hi = last + 1 if include_immediate else last
        for t in range(hi):
            res = next_event_for(eidx, t, include_immediate)
            if res is None:
                continue
            _, goal, _, _ = res
            anchors.append(Anchor(ep=li, t=t, action=int(ep["action"][t]), goal=int(goal)))
    return anchors


def state_batch(anchors: List[Anchor], idx: np.ndarray, state_feats: List[np.ndarray]):
    s = np.stack([state_feats[anchors[i].ep][anchors[i].t] for i in idx])
    a = np.array([anchors[i].action for i in idx], np.int64)
    g = np.array([anchors[i].goal for i in idx], np.float32)
    return s, a, g


def pixel_batch(anchors: List[Anchor], idx: np.ndarray, eps: List[Dict[str, Any]], k: int = 4):
    frames = np.stack([build_kstack(eps[anchors[i].ep]["gray_learner"], anchors[i].t, k) for i in idx])
    a = np.array([anchors[i].action for i in idx], np.int64)
    g = np.array([anchors[i].goal for i in idx], np.float32)
    return frames.astype(np.float32) / 255.0, a, g    # (B,k,84,84) in [0,1]


def norm_goal(g: np.ndarray) -> np.ndarray:
    return ((g + 2.0) / 23.0).astype(np.float32)        # [-2,21] -> [0,1]
