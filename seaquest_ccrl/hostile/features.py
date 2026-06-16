"""Deterministic fixed hostile feature representation for the four-frame state.

All bin boundaries are FROZEN here and echoed in `feature_schema()` so a stored U
vector can never be re-binned silently. We deliberately AVOID cross-frame object
identity matching (high-risk in Stage-H0): every feature is a per-frame summary,
and the stack feature is just the concatenation of the four per-frame summaries.

Per-frame blocks (computed from PRE-REMOVAL metadata + player centre at that frame):

  enemy block  (17): shark, submarine, surface_sub, enemy_total,
                     enemy_present, nearest_enemy_dx, nearest_enemy_dy,
                     nearest_enemy_missing, enemy_grid_3x3(9)
  missile block(14): enemy_missile, missile_present,
                     nearest_missile_dx, nearest_missile_dy,
                     nearest_missile_missing, missile_grid_3x3(9)
  threat block  (2): threat_within_24px, threat_within_48px

Missing nearest-object distances are stored as 0 with an explicit *_missing flag
(=1), so the feature matrix never contains NaN/Inf.
"""
import numpy as np

from seaquest_ccrl.hostile import schema as S

# -- FROZEN bin boundaries ---------------------------------------------------
GRID_EDGE = 16.0          # |Δ| <= EDGE -> centre column/row; else left/right, up/down
CLIP_COUNT = 3            # count classification clipped to {0,1,2,3+}
THREAT_NEAR_PX = 24.0
THREAT_FAR_PX = 48.0

ENEMY_BLOCK = 17
MISSILE_BLOCK = 14
THREAT_BLOCK = 2
PER_FRAME = ENEMY_BLOCK + MISSILE_BLOCK + THREAT_BLOCK   # 33


def _centers(bbox):
    """(N,4) x,y,w,h -> (N,2) centre x,y."""
    if len(bbox) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    b = bbox.astype(np.float64)
    return np.stack([b[:, 0] + b[:, 2] / 2.0, b[:, 1] + b[:, 3] / 2.0], axis=1)


def _grid_cell(dx, dy, edge=GRID_EDGE):
    """Map a relative offset to a 3x3 cell index 0..8 (row-major; row=y, col=x)."""
    col = 0 if dx < -edge else (2 if dx > edge else 1)
    row = 0 if dy < -edge else (2 if dy > edge else 1)
    return row * 3 + col


def _split_hostiles(harr_t):
    """Return (enemy_centers (Ne,2), missile_centers (Nm,2), per-class counts)."""
    bbox, cls, valid = harr_t["hostile_bbox"], harr_t["hostile_class"], harr_t["hostile_valid"]
    idx = np.where(valid)[0]
    c = cls[idx]
    b = bbox[idx]
    enemy_sel = np.isin(c, S.ENEMY_IDS)
    miss_sel = np.isin(c, S.MISSILE_IDS)
    counts = {
        "shark": int((c == S.HOSTILE_ID["Shark"]).sum()),
        "submarine": int((c == S.HOSTILE_ID["Submarine"]).sum()),
        "surface_sub": int((c == S.HOSTILE_ID["SurfaceSubmarine"]).sum()),
    }
    counts["enemy_total"] = counts["shark"] + counts["submarine"] + counts["surface_sub"]
    counts["enemy_missile"] = int(miss_sel.sum())
    return _centers(b[enemy_sel]), _centers(b[miss_sel]), counts


def _nearest(centers, px, py):
    """Nearest centre relative (dx,dy) and missing flag (1 if none)."""
    if len(centers) == 0:
        return 0.0, 0.0, 1.0
    d = np.abs(centers[:, 0] - px) + np.abs(centers[:, 1] - py)
    j = int(np.argmin(d))
    return float(centers[j, 0] - px), float(centers[j, 1] - py), 0.0


def _grid_counts(centers, px, py, edge=GRID_EDGE):
    g = np.zeros(9, dtype=np.float64)
    for cx, cy in centers:
        g[_grid_cell(cx - px, cy - py, edge)] += 1.0
    return g


def per_frame_features(harr_t, player_xy):
    """One frame's 33-dim feature vector. player_xy = (x,y) player centre at this frame.

    If the player centre is non-finite (no Player object yet), counts are still valid
    but all player-relative fields (nearest offsets, grids, threat) are set to
    missing/zero so the vector is finite.
    """
    px, py = float(player_xy[0]), float(player_xy[1])
    en_c, mi_c, counts = _split_hostiles(harr_t)
    if not (np.isfinite(px) and np.isfinite(py)):
        enemy_block = np.array(
            [counts["shark"], counts["submarine"], counts["surface_sub"], counts["enemy_total"],
             1.0 if counts["enemy_total"] > 0 else 0.0, 0.0, 0.0, 1.0] + [0.0] * 9,
            dtype=np.float64)
        missile_block = np.array(
            [counts["enemy_missile"], 1.0 if counts["enemy_missile"] > 0 else 0.0,
             0.0, 0.0, 1.0] + [0.0] * 9, dtype=np.float64)
        threat_block = np.array([0.0, 0.0], dtype=np.float64)
        return np.concatenate([enemy_block, missile_block, threat_block])
    # enemy block
    ne_dx, ne_dy, ne_missing = _nearest(en_c, px, py)
    en_grid = _grid_counts(en_c, px, py)
    enemy_present = 1.0 if counts["enemy_total"] > 0 else 0.0
    enemy_block = np.array(
        [counts["shark"], counts["submarine"], counts["surface_sub"], counts["enemy_total"],
         enemy_present, ne_dx, ne_dy, ne_missing] + list(en_grid), dtype=np.float64)
    # missile block
    nm_dx, nm_dy, nm_missing = _nearest(mi_c, px, py)
    mi_grid = _grid_counts(mi_c, px, py)
    missile_present = 1.0 if counts["enemy_missile"] > 0 else 0.0
    missile_block = np.array(
        [counts["enemy_missile"], missile_present, nm_dx, nm_dy, nm_missing] + list(mi_grid),
        dtype=np.float64)
    # threat block (nearest of any hostile, euclidean)
    all_c = np.concatenate([en_c, mi_c], axis=0) if (len(en_c) or len(mi_c)) else np.zeros((0, 2))
    if len(all_c):
        dmin = float(np.min(np.sqrt((all_c[:, 0] - px) ** 2 + (all_c[:, 1] - py) ** 2)))
    else:
        dmin = np.inf
    threat_block = np.array([1.0 if dmin <= THREAT_NEAR_PX else 0.0,
                             1.0 if dmin <= THREAT_FAR_PX else 0.0], dtype=np.float64)
    return np.concatenate([enemy_block, missile_block, threat_block])


# -- stack feature builders --------------------------------------------------
def _slice_blocks(frame_feats):
    """frame_feats (4,33) -> dict of stacked enemy/missile/threat blocks."""
    en = frame_feats[:, 0:ENEMY_BLOCK].reshape(-1)
    mi = frame_feats[:, ENEMY_BLOCK:ENEMY_BLOCK + MISSILE_BLOCK].reshape(-1)
    th = frame_feats[:, ENEMY_BLOCK + MISSILE_BLOCK:].reshape(-1)
    return en, mi, th


def stack_features(frame_feats):
    """frame_feats: (4,33) oldest->newest. Returns dict of U_* stack vectors."""
    en, mi, th = _slice_blocks(frame_feats)
    u_enemy = en
    u_missile = mi
    u_joint = np.concatenate([frame_feats.reshape(-1)])   # all 4x33, includes threat
    return {"U_enemy_stack": u_enemy.astype(np.float32),
            "U_missile_stack": u_missile.astype(np.float32),
            "U_joint_stack": u_joint.astype(np.float32)}


# -- hiddenness targets (multilabel grid / presence / clipped count) ---------
def presence_grid(harr_t, player_xy, which):
    """3x3 binary occupancy (1 if >=1 object of `which` in cell). which in {enemy,missile}."""
    px, py = float(player_xy[0]), float(player_xy[1])
    en_c, mi_c, _ = _split_hostiles(harr_t)
    centers = en_c if which == "enemy" else mi_c
    g = _grid_counts(centers, px, py)
    return (g > 0).astype(np.int64)


def presence(harr_t, which):
    en_c, mi_c, counts = _split_hostiles(harr_t)
    n = counts["enemy_total"] if which == "enemy" else counts["enemy_missile"]
    return int(n > 0)


def clipped_count(harr_t, which, clip=CLIP_COUNT):
    en_c, mi_c, counts = _split_hostiles(harr_t)
    n = counts["enemy_total"] if which == "enemy" else counts["enemy_missile"]
    return int(min(n, clip))


def nearest_offset(harr_t, player_xy, which):
    """(dx,dy,missing) nearest object of `which` relative to player centre."""
    px, py = float(player_xy[0]), float(player_xy[1])
    en_c, mi_c, _ = _split_hostiles(harr_t)
    centers = en_c if which == "enemy" else mi_c
    return _nearest(centers, px, py)


# -- normalization fit on TRAIN only -----------------------------------------
class Normalizer:
    """z-score normalizer; fit() must be called on TRAIN rows only."""

    def __init__(self):
        self.mu = None
        self.sd = None

    def fit(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-6
        return self

    def apply(self, X):
        assert self.mu is not None, "Normalizer.fit must be called (on TRAIN) first"
        return ((X - self.mu) / self.sd).astype(np.float32)

    def state(self):
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist()}


def feature_schema():
    return {
        "per_frame_dim": PER_FRAME,
        "enemy_block_dim": ENEMY_BLOCK, "missile_block_dim": MISSILE_BLOCK,
        "threat_block_dim": THREAT_BLOCK,
        "frozen_bins": {"grid_edge_px": GRID_EDGE, "clip_count": CLIP_COUNT,
                        "threat_near_px": THREAT_NEAR_PX, "threat_far_px": THREAT_FAR_PX},
        "stack_dims": {"U_enemy_stack": ENEMY_BLOCK * 4,
                       "U_missile_stack": MISSILE_BLOCK * 4,
                       "U_joint_stack": PER_FRAME * 4},
        "enemy_block_layout": ["shark", "submarine", "surface_sub", "enemy_total",
                               "enemy_present", "nearest_enemy_dx", "nearest_enemy_dy",
                               "nearest_enemy_missing"] + [f"enemy_grid_{i}" for i in range(9)],
        "missile_block_layout": ["enemy_missile", "missile_present", "nearest_missile_dx",
                                 "nearest_missile_dy", "nearest_missile_missing"]
                                + [f"missile_grid_{i}" for i in range(9)],
        "threat_block_layout": ["threat_within_24px", "threat_within_48px"],
        "grid_cell_order": "row-major row=y(top->bottom) col=x(left->right); "
                           "cell=(row*3+col), centre cell index 4",
    }
