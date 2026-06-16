"""Frozen Stage-H0 hostile/protected metadata schema.

Everything in this module is a FROZEN convention. The class IDs, padding widths,
orientation codes and palette references are written here once and referenced by
every collector / probe so the meaning of a stored array can never drift.

Class IDs
---------
Hostile (the candidate hidden variable U):
    0 = padding (no object)
    1 = Shark
    2 = Submarine
    3 = SurfaceSubmarine
    4 = EnemyMissile

Protected (must stay visible to the learner):
    0 = padding
    1 = Player
    2 = Diver
    3 = PlayerMissile

Orientation codes (only Submarine/SurfaceSubmarine/Player expose it reliably):
    0 = unknown / not-applicable
    1 = facing right (Orientation.E)
    2 = facing left  (Orientation.W)

KNOWN AMBIGUITY (see repo_audit.json): EnemyMissile and Diver share the SAME RGB
(66,72,200) and the same RAM slots; PlayerMissile shares Player's RGB (187,187,53).
Rows where an EnemyMissile bbox overlaps a Diver bbox are flagged AMBIGUOUS and
excluded from the primary qualification probes (their support is reported
separately).
"""
import numpy as np

SCHEMA_VERSION = "h0-1"

# -- frozen padding widths (>= OCAtari MAX_NB_OBJECTS for Seaquest) ----------
# Shark 12 + Submarine 12 + SurfaceSubmarine 1 + EnemyMissile 4 = 29 -> pad to 32.
MAX_HOSTILES = 32
# Player 1 + Diver 4 + PlayerMissile 1 = 6 -> pad to 8.
MAX_PROTECTED = 8

# -- frozen class id maps ----------------------------------------------------
HOSTILE_ID = {"Shark": 1, "Submarine": 2, "SurfaceSubmarine": 3, "EnemyMissile": 4}
HOSTILE_NAME = {v: k for k, v in HOSTILE_ID.items()}
HOSTILE_NAME[0] = "padding"
ENEMY_IDS = (HOSTILE_ID["Shark"], HOSTILE_ID["Submarine"], HOSTILE_ID["SurfaceSubmarine"])
MISSILE_IDS = (HOSTILE_ID["EnemyMissile"],)

PROTECTED_ID = {"Player": 1, "Diver": 2, "PlayerMissile": 3}
PROTECTED_NAME = {v: k for k, v in PROTECTED_ID.items()}
PROTECTED_NAME[0] = "padding"

# Orientation codes
ORI_UNKNOWN, ORI_RIGHT, ORI_LEFT = 0, 1, 2

# -- reference palette (OCAtari RAM-mode class colours) ----------------------
# These are the STARTING reference colours; the object-identity audit measures
# the ACTUAL rendered colours and the removal step matches within a tolerance.
PALETTE_RGB = {
    "Shark": (92, 186, 92),
    "Submarine": (170, 170, 170),
    "SurfaceSubmarine": (170, 170, 170),
    "EnemyMissile": (66, 72, 200),
    # protected (for reference / protected-pixel masks)
    "Player": (187, 187, 53),
    "PlayerMissile": (187, 187, 53),
    "Diver": (66, 72, 200),
}
# Canonical background water RGB (config.WATER_COLOR), used as the removal fallback.
WATER_COLOR = (0, 28, 136)

FRAME_H, FRAME_W = 210, 160


# --------------------------------------------------------------------------
# padded-array encode / decode
# --------------------------------------------------------------------------
def empty_hostile_arrays(T):
    """Allocate the padded hostile metadata arrays for T timesteps."""
    return {
        "hostile_bbox": np.zeros((T, MAX_HOSTILES, 4), dtype=np.int16),
        "hostile_class": np.zeros((T, MAX_HOSTILES), dtype=np.uint8),
        "hostile_valid": np.zeros((T, MAX_HOSTILES), dtype=bool),
        "hostile_orientation": np.zeros((T, MAX_HOSTILES), dtype=np.uint8),
        "hostile_count": np.zeros((T,), dtype=np.int32),
        "enemy_count": np.zeros((T,), dtype=np.int32),
        "enemy_missile_count": np.zeros((T,), dtype=np.int32),
    }


def empty_protected_arrays(T):
    return {
        "protected_bbox": np.zeros((T, MAX_PROTECTED, 4), dtype=np.int16),
        "protected_class": np.zeros((T, MAX_PROTECTED), dtype=np.uint8),
        "protected_valid": np.zeros((T, MAX_PROTECTED), dtype=bool),
    }


def encode_objects(records, max_slots, kind):
    """records: list of dicts {class_id:int, bbox:(x,y,w,h), orientation:int}.

    Returns (bbox (max_slots,4) int16, class (max_slots,) u8, valid (max_slots,) bool,
             orientation (max_slots,) u8). Raises if more than max_slots objects.
    """
    if len(records) > max_slots:
        raise ValueError(f"{kind}: {len(records)} objects > MAX={max_slots} "
                         f"(schema padding width too small)")
    bbox = np.zeros((max_slots, 4), dtype=np.int16)
    cls = np.zeros((max_slots,), dtype=np.uint8)
    valid = np.zeros((max_slots,), dtype=bool)
    ori = np.zeros((max_slots,), dtype=np.uint8)
    for i, r in enumerate(records):
        x, y, w, h = r["bbox"]
        bbox[i] = (int(x), int(y), int(w), int(h))
        cls[i] = int(r["class_id"])
        valid[i] = True
        ori[i] = int(r.get("orientation", ORI_UNKNOWN))
    return bbox, cls, valid, ori


def decode_valid(bbox, cls, valid, ori=None):
    """Return a list of {class_id, bbox, orientation} for the valid slots only."""
    out = []
    idx = np.where(valid)[0]
    for i in idx:
        rec = {"class_id": int(cls[i]), "bbox": tuple(int(v) for v in bbox[i])}
        if ori is not None:
            rec["orientation"] = int(ori[i])
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# bbox helpers + validation
# --------------------------------------------------------------------------
def clip_bbox(x, y, w, h, W=FRAME_W, H=FRAME_H):
    """Clip a (x,y,w,h) box to the frame. Returns (x0,y0,cw,ch) with cw,ch >= 0.

    The clipped box is the INTERSECTION with [0,W)x[0,H). A fully off-screen box
    returns zero width/height.
    """
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(W, int(x) + int(w))
    y1 = min(H, int(y) + int(h))
    cw = max(0, x1 - x0)
    ch = max(0, y1 - y0)
    return x0, y0, cw, ch


def bbox_clipped_fraction(x, y, w, h, W=FRAME_W, H=FRAME_H):
    """Fraction of the original bbox area cut off by the frame border."""
    area = max(1, int(w) * int(h))
    _, _, cw, ch = clip_bbox(x, y, w, h, W, H)
    return 1.0 - (cw * ch) / area


def validate_hostile_arrays(arr, W=FRAME_W, H=FRAME_H):
    """Hard schema checks on the padded hostile arrays. Raises AssertionError."""
    bbox, cls, valid = arr["hostile_bbox"], arr["hostile_class"], arr["hostile_valid"]
    assert bbox.dtype == np.int16 and cls.dtype == np.uint8 and valid.dtype == bool
    assert bbox.shape[1] == MAX_HOSTILES and bbox.shape[2] == 4
    # valid slots must carry a known hostile id; padding slots must be class 0
    assert np.all(np.isin(cls[valid], list(HOSTILE_NAME))), "invalid hostile class id"
    assert np.all(cls[valid] != 0), "valid slot with padding class id 0"
    assert np.all(cls[~valid] == 0), "padding slot with non-zero class id"
    # bbox in-bounds after clipping is enforced at extraction; here check stored boxes
    vb = bbox[valid]
    if len(vb):
        assert np.all(vb[:, 0] >= 0) and np.all(vb[:, 1] >= 0), "negative bbox origin"
        assert np.all(vb[:, 0] + vb[:, 2] <= W), "bbox x overflow"
        assert np.all(vb[:, 1] + vb[:, 3] <= H), "bbox y overflow"
    # counts consistent
    n_enemy = np.isin(cls, ENEMY_IDS) & valid
    n_miss = np.isin(cls, MISSILE_IDS) & valid
    assert np.all(arr["hostile_count"] == valid.sum(1)), "hostile_count mismatch"
    assert np.all(arr["enemy_count"] == n_enemy.sum(1)), "enemy_count mismatch"
    assert np.all(arr["enemy_missile_count"] == n_miss.sum(1)), "enemy_missile_count mismatch"


def schema_dict():
    """Machine-readable description of the frozen schema (for artifacts)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "max_hostiles": MAX_HOSTILES, "max_protected": MAX_PROTECTED,
        "hostile_class_ids": {**{str(v): k for k, v in HOSTILE_ID.items()}, "0": "padding"},
        "protected_class_ids": {**{str(v): k for k, v in PROTECTED_ID.items()}, "0": "padding"},
        "enemy_ids": list(ENEMY_IDS), "missile_ids": list(MISSILE_IDS),
        "orientation_codes": {"0": "unknown", "1": "right(E)", "2": "left(W)"},
        "palette_rgb_reference": {k: list(v) for k, v in PALETTE_RGB.items()},
        "water_color": list(WATER_COLOR),
        "frame_hw": [FRAME_H, FRAME_W],
        "known_ambiguity": "EnemyMissile.rgb==Diver.rgb==(66,72,200); "
                           "PlayerMissile.rgb==Player.rgb==(187,187,53)",
    }
