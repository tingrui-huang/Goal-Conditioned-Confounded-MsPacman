"""Shared helpers for Stage-S0.5 (reuses S0 teacher + port)."""
import sys, os, json, hashlib
import numpy as np
sys.path.insert(0, "/work/seaquest_stage_s0")

ALE_MEANINGS = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
                'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
                'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']
SURFACE_Y = 46
CKPT = "/work/artifacts/seaquest/stage_s0/teacher/_downloads/cand_A/sebulba_ppo_envpool_impala_atari_wrapper.cleanrl_model"
SRC = "/work/artifacts/seaquest/stage_s0/teacher/_downloads/cand_A/sebulba_ppo_envpool_impala_atari_wrapper.py"

# semantic categories (frozen)
SEMANTIC = {"NOOP": [0], "FIRE_only": [1], "UP": [2], "DOWN": [5], "LEFT": [4], "RIGHT": [3],
            "UP_FIRE": [10], "DOWN_FIRE": [13], "LEFT_FIRE": [12], "RIGHT_FIRE": [11],
            "other_diag_move": [6, 7, 8, 9], "other_diag_move_fire": [14, 15, 16, 17]}
CAT_NAMES = list(SEMANTIC.keys())
ACTION_TO_CAT = {a: i for i, (k, v) in enumerate(SEMANTIC.items()) for a in v}
ROLE = {"UP_related": [2, 6, 7, 10, 14, 15], "DOWN_related": [5, 8, 9, 13, 16, 17],
        "LEFT_related": [4, 7, 9, 12, 15, 17], "RIGHT_related": [3, 6, 8, 11, 14, 16],
        "FIRE_related": [1, 10, 11, 12, 13, 14, 15, 16, 17]}

STATE_FEATURES = ["player_x", "player_y", "player_vx", "player_vy", "oxygen", "distance_to_surface",
                  "nearest_enemy_dx", "nearest_enemy_dy", "enemy_count", "enemy_centroid_x",
                  "enemy_centroid_y", "nearest_diver_dx", "nearest_diver_dy", "diver_count",
                  "player_missile_count", "enemy_missile_count", "score", "lives"]


def load_teacher(tag="A"):
    from teacher_adapter import CleanRLSeaquestTeacher
    return CleanRLSeaquestTeacher(CKPT, SRC, mod_name=f"cleanrl_src_{tag}")


def softmax(logits, T=1.0):
    z = np.asarray(logits, dtype=np.float64) / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def obs_hash(obs):
    return hashlib.sha256(np.ascontiguousarray(obs, dtype=np.uint8).tobytes()).hexdigest()[:16]


def enrich_features(f, prev=None):
    """Add derived fields (velocity, distances, centroids, distance_to_surface)."""
    px, py = f.get("player_x"), f.get("player_y")
    f["distance_to_surface"] = None if py is None else float(py - SURFACE_Y)
    f["player_vx"] = None if (prev is None or px is None or prev.get("player_x") is None) else float(px - prev["player_x"])
    f["player_vy"] = None if (prev is None or py is None or prev.get("player_y") is None) else float(py - prev["player_y"])
    ex = [v for v in (f.get("enemy_xs") or []) if v is not None]
    ey = [v for v in (f.get("enemy_ys") or []) if v is not None]
    f["enemy_count"] = float(len(ex))
    if ex:
        f["enemy_centroid_x"] = float(np.mean(ex)); f["enemy_centroid_y"] = float(np.mean(ey))
        if px is not None and py is not None:
            d = [abs(x - px) + abs(y - py) for x, y in zip(ex, ey)]
            j = int(np.argmin(d))
            f["nearest_enemy_dx"] = float(ex[j] - px); f["nearest_enemy_dy"] = float(ey[j] - py)
    dx = [v for v in (f.get("diver_xs") or []) if v is not None]
    dy = [v for v in (f.get("diver_ys") or []) if v is not None]
    f["diver_count"] = float(len(dx))
    if dx and px is not None and py is not None:
        d = [abs(x - px) + abs(y - py) for x, y in zip(dx, dy)]
        j = int(np.argmin(d))
        f["nearest_diver_dx"] = float(dx[j] - px); f["nearest_diver_dy"] = float(dy[j] - py)
    for k in ("enemy_centroid_x", "enemy_centroid_y", "nearest_enemy_dx", "nearest_enemy_dy",
              "nearest_diver_dx", "nearest_diver_dy"):
        f.setdefault(k, None)
    return f


def feat_row(f):
    return [np.nan if f.get(k) is None else float(f.get(k)) for k in STATE_FEATURES]


def cat_of(action):
    return ACTION_TO_CAT[int(action)]
