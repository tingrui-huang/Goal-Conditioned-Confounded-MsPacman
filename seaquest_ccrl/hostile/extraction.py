"""OCAtari objects -> Stage-H0 padded metadata (runs Docker-side at collection).

Pure, duck-typed logic: `extract_objects` accepts any iterable of objects exposing
`.category`, `.x`, `.y`, `.w`, `.h` and (optionally) `.orientation`. It does NOT import
ocatari, so it is unit-testable with lightweight mock objects.

Targets for the qualification probes (hiddenness etc.) are derived from THIS metadata,
i.e. from the original PRE-REMOVAL RAM objects — never from the removed image.
"""
import numpy as np

from seaquest_ccrl.hostile import schema as S


def _orientation_code(o):
    ori = getattr(o, "orientation", None)
    if ori is None:
        return S.ORI_UNKNOWN
    name = getattr(ori, "name", None)
    if name is None:
        name = str(ori).split(".")[-1]
    name = str(name).upper()
    if name.startswith("E") or name in ("NE", "SE"):
        return S.ORI_RIGHT
    if name.startswith("W") or name in ("NW", "SW"):
        return S.ORI_LEFT
    return S.ORI_UNKNOWN


def _bbox_of(o, W=S.FRAME_W, H=S.FRAME_H):
    """Clipped (x,y,w,h) and the original (unclipped) (x,y,w,h)."""
    ox, oy, ow, oh = int(o.x), int(o.y), int(o.w), int(o.h)
    cx, cy, cw, ch = S.clip_bbox(ox, oy, ow, oh, W, H)
    return (cx, cy, cw, ch), (ox, oy, ow, oh)


def extract_objects(objects, W=S.FRAME_W, H=S.FRAME_H):
    """Split one timestep's objects into hostile / protected records.

    Returns dict with:
      hostile  : list of {class_id, bbox(clipped), orientation, bbox_raw, clipped_fraction}
      protected: list of {class_id, bbox(clipped), orientation, bbox_raw}
      dropped  : list of {category, reason}  (off-screen / zero-area after clip)
    Objects with category 'NoObject' / HUD-only objects are ignored silently.
    """
    hostile, protected, dropped = [], [], []
    for o in objects:
        cat = getattr(o, "category", "") or o.__class__.__name__
        if cat in S.HOSTILE_ID:
            (cx, cy, cw, ch), raw = _bbox_of(o, W, H)
            if cw <= 0 or ch <= 0:
                dropped.append({"category": cat, "reason": "off_screen_after_clip",
                                "bbox_raw": list(raw)})
                continue
            hostile.append({"class_id": S.HOSTILE_ID[cat], "bbox": (cx, cy, cw, ch),
                            "orientation": _orientation_code(o), "bbox_raw": raw,
                            "clipped_fraction": S.bbox_clipped_fraction(*raw, W, H)})
        elif cat in S.PROTECTED_ID:
            (cx, cy, cw, ch), raw = _bbox_of(o, W, H)
            if cw <= 0 or ch <= 0:
                dropped.append({"category": cat, "reason": "off_screen_after_clip",
                                "bbox_raw": list(raw)})
                continue
            protected.append({"class_id": S.PROTECTED_ID[cat], "bbox": (cx, cy, cw, ch),
                              "orientation": _orientation_code(o), "bbox_raw": raw})
        # everything else (OxygenBar, PlayerScore, Lives, CollectedDiver, NoObject,...) ignored
    return {"hostile": hostile, "protected": protected, "dropped": dropped}


def fill_step(harr, parr, t, extracted):
    """Write one timestep's extracted records into preallocated padded arrays."""
    h = extracted["hostile"]
    bbox, cls, valid, ori = S.encode_objects(h, S.MAX_HOSTILES, "hostile")
    harr["hostile_bbox"][t] = bbox
    harr["hostile_class"][t] = cls
    harr["hostile_valid"][t] = valid
    harr["hostile_orientation"][t] = ori
    harr["hostile_count"][t] = len(h)
    harr["enemy_count"][t] = sum(1 for r in h if r["class_id"] in S.ENEMY_IDS)
    harr["enemy_missile_count"][t] = sum(1 for r in h if r["class_id"] in S.MISSILE_IDS)

    p = extracted["protected"]
    pb, pc, pv, _ = S.encode_objects(p, S.MAX_PROTECTED, "protected")
    parr["protected_bbox"][t] = pb
    parr["protected_class"][t] = pc
    parr["protected_valid"][t] = pv


def build_metadata(objects_by_t, W=S.FRAME_W, H=S.FRAME_H):
    """objects_by_t: list (len T) of per-timestep object iterables.

    Returns (hostile_arrays, protected_arrays, extra) where extra carries per-step
    'dropped' records and per-class counts for the support / object audits.
    """
    T = len(objects_by_t)
    harr = S.empty_hostile_arrays(T)
    parr = S.empty_protected_arrays(T)
    dropped_by_t, perclass = [], {k: np.zeros(T, dtype=np.int32) for k in S.HOSTILE_ID}
    for t, objs in enumerate(objects_by_t):
        ex = extract_objects(objs, W, H)
        fill_step(harr, parr, t, ex)
        dropped_by_t.append(ex["dropped"])
        for r in ex["hostile"]:
            perclass[S.HOSTILE_NAME[r["class_id"]]][t] += 1
    S.validate_hostile_arrays(harr, W, H)
    extra = {"dropped_by_t": dropped_by_t,
             "perclass_count": {k: v for k, v in perclass.items()}}
    return harr, parr, extra
