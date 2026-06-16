"""Class-aware PIXEL removal of hostile sprites (NOT whole-bbox fill).

For each valid hostile bbox we select ONLY the pixels whose colour is compatible
with that hostile class's palette (within a per-channel tolerance) and repaint just
those pixels with the local water background. Water OUTSIDE the bbox is untouched,
non-hostile pixels INSIDE the bbox are untouched, and protected sprite pixels
(Player / PlayerMissile / Diver) are explicitly shielded.

This is the ONLY removal path for Stage-H0. It is deliberately NOT
seaquest_ccrl.data.masking.apply_enemy_mask (that fills the whole bbox with water,
which reveals object location and deletes overlapping objects).

EnemyMissile / Diver share a colour, so an EnemyMissile bbox overlapping a Diver
bbox would otherwise eat diver pixels: those rows are flagged AMBIGUOUS by
`ambiguous_row` and excluded from primary probes upstream.
"""
import json
import numpy as np

from seaquest_ccrl.hostile import schema as S


def load_resolved_palettes(path):
    """Load a resolved_palette.json -> {class_name: {"rgb":[[r,g,b],...], "tol":k}}.

    Returned dict is directly usable as the `palettes` arg to remove_frame /
    class_compatible_mask. Includes both hostile and protected classes so protected
    pixels are shielded with their OBSERVED colours, not just the RAM defaults."""
    d = json.load(open(path))
    return d["classes"] if "classes" in d else d

# default per-channel colour tolerance for "compatible with this class"
DEFAULT_TOL = 28
# ring width (px) sampled around a bbox to estimate the local water colour
WATER_RING = 3
# a sampled pixel counts as water if within this L1 distance of canonical water
WATER_L1 = 120


def _palette_entry(palettes, class_name):
    """Return (list_of_rgb, tol) for class_name.

    `palettes[class_name]` may be:
      * a single (r,g,b) tuple           -> one colour, global tol
      * a list/tuple of (r,g,b) tuples   -> match ANY, global tol
      * {"rgb": [[r,g,b],...], "tol": k}  -> resolved-palette form (per-class tol)
    """
    pal = (palettes or S.PALETTE_RGB)[class_name]
    if isinstance(pal, dict):
        cols = [tuple(c) for c in pal["rgb"]]
        return cols, pal.get("tol", None)
    if len(pal) == 3 and np.isscalar(pal[0]):
        return [tuple(pal)], None
    return [tuple(c) for c in pal], None


def class_compatible_mask(region_rgb, class_name, palettes=None, tol=DEFAULT_TOL):
    """Boolean mask (h,w) of pixels in `region_rgb` matching `class_name`'s palette.

    Supports a single reference colour, a list of observed colours (match ANY), or the
    resolved-palette dict form with a per-class tolerance (channel-wise box)."""
    cols, ptol = _palette_entry(palettes, class_name)
    use_tol = tol if ptol is None else ptol
    reg = region_rgb.astype(np.int16)
    out = np.zeros(region_rgb.shape[:2], dtype=bool)
    for c in cols:
        d = np.abs(reg - np.asarray(c, dtype=np.int16))
        out |= np.all(d <= use_tol, axis=-1)
    return out


def _protected_mask(frame, parr_t, palettes=None, tol=DEFAULT_TOL):
    """Pixels inside any protected bbox matching the protected class palette.

    These pixels are never repainted (shields Diver blue from EnemyMissile removal,
    Player yellow from PlayerMissile removal, etc.).
    """
    H, W = frame.shape[:2]
    mask = np.zeros((H, W), dtype=bool)
    bbox, cls, valid = parr_t["protected_bbox"], parr_t["protected_class"], parr_t["protected_valid"]
    for i in np.where(valid)[0]:
        x, y, w, h = (int(v) for v in bbox[i])
        if w <= 0 or h <= 0:
            continue
        name = S.PROTECTED_NAME[int(cls[i])]
        sub = frame[y:y + h, x:x + w]
        mask[y:y + h, x:x + w] |= class_compatible_mask(sub, name, palettes, tol)
    return mask


def _local_water(frame, x, y, w, h, ring=WATER_RING):
    """Median water-like colour sampled in a ring around the bbox; canonical fallback."""
    H, W = frame.shape[:2]
    x0, y0 = max(0, x - ring), max(0, y - ring)
    x1, y1 = min(W, x + w + ring), min(H, y + h + ring)
    patch = frame[y0:y1, x0:x1].reshape(-1, 3).astype(np.int16)
    ref = np.asarray(S.WATER_COLOR, dtype=np.int16)
    water_like = patch[np.abs(patch - ref).sum(1) <= WATER_L1]
    if len(water_like) >= 4:
        return np.median(water_like, axis=0).astype(np.uint8)
    return np.asarray(S.WATER_COLOR, dtype=np.uint8)


def _bbox_overlap(a, b):
    """True if two (x,y,w,h) boxes overlap (positive area intersection)."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return (ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah
            and aw > 0 and ah > 0 and bw > 0 and bh > 0)


def ambiguous_row(harr_t, parr_t):
    """True if an EnemyMissile bbox overlaps a Diver bbox (colour-indistinguishable)."""
    hb, hc, hv = harr_t["hostile_bbox"], harr_t["hostile_class"], harr_t["hostile_valid"]
    pb, pc, pv = parr_t["protected_bbox"], parr_t["protected_class"], parr_t["protected_valid"]
    miss = [tuple(int(v) for v in hb[i]) for i in np.where(hv)[0]
            if int(hc[i]) == S.HOSTILE_ID["EnemyMissile"]]
    divers = [tuple(int(v) for v in pb[i]) for i in np.where(pv)[0]
              if int(pc[i]) == S.PROTECTED_ID["Diver"]]
    return any(_bbox_overlap(m, d) for m in miss for d in divers)


def remove_frame(frame, harr_t, parr_t, palettes=None, tol=DEFAULT_TOL):
    """Remove all valid hostile sprites from ONE frame using that frame's metadata.

    Returns (removed_frame uint8, changed_mask (H,W) bool, stats dict).
    Pure: `frame` is not mutated.
    """
    assert frame.dtype == np.uint8 and frame.shape[2] == 3
    H, W = frame.shape[:2]
    removed = frame.copy()
    changed = np.zeros((H, W), dtype=bool)
    union_bbox = np.zeros((H, W), dtype=bool)
    prot = _protected_mask(frame, parr_t, palettes, tol)

    bbox, cls, valid = harr_t["hostile_bbox"], harr_t["hostile_class"], harr_t["hostile_valid"]
    per_obj = []
    for i in np.where(valid)[0]:
        x, y, w, h = (int(v) for v in bbox[i])
        name = S.HOSTILE_NAME[int(cls[i])]
        if w <= 0 or h <= 0:
            per_obj.append({"slot": int(i), "class": name, "compatible_px": 0,
                            "zero_compatible": True, "overlaps_protected": False})
            continue
        union_bbox[y:y + h, x:x + w] = True
        sub = removed[y:y + h, x:x + w]
        comp = class_compatible_mask(sub, name, palettes, tol)      # (h,w)
        # never touch protected sprite pixels
        sub_prot = prot[y:y + h, x:x + w]
        to_change = comp & ~sub_prot
        fill = _local_water(frame, x, y, w, h)
        sub[to_change] = fill
        gy, gx = np.where(to_change)
        changed[y + gy, x + gx] = True
        per_obj.append({
            "slot": int(i), "class": name, "bbox": [x, y, w, h],
            "compatible_px": int(comp.sum()),
            "changed_px": int(to_change.sum()),
            "zero_compatible": bool(comp.sum() == 0),
            "overlaps_protected": bool((comp & sub_prot).any()),
        })

    stats = {
        "changed_px": int(changed.sum()),
        "n_hostiles": int(valid.sum()),
        "ambiguous": bool(ambiguous_row(harr_t, parr_t)),
        "per_obj": per_obj,
        "n_zero_compatible": int(sum(1 for o in per_obj if o.get("zero_compatible"))),
    }
    return removed, changed, stats


def verify_removal(frame, removed, changed, harr_t, parr_t, palettes=None, tol=DEFAULT_TOL):
    """Hard invariants. Returns dict; raises AssertionError on violation.

    1. changed pixels are a subset of the union of hostile bboxes;
    2. pixels outside hostile bboxes are byte-identical;
    3. protected sprite pixels are byte-identical.
    """
    H, W = frame.shape[:2]
    union = np.zeros((H, W), dtype=bool)
    bbox, valid = harr_t["hostile_bbox"], harr_t["hostile_valid"]
    for i in np.where(valid)[0]:
        x, y, w, h = (int(v) for v in bbox[i])
        if w > 0 and h > 0:
            union[y:y + h, x:x + w] = True
    diff = np.any(frame != removed, axis=-1)
    assert np.array_equal(diff, changed), "changed_mask disagrees with actual pixel diff"
    assert not np.any(diff & ~union), "HOSTILE_REMOVAL_INVALID: change outside hostile bboxes"
    prot = _protected_mask(frame, parr_t, palettes, tol)
    assert not np.any(diff & prot), "HOSTILE_REMOVAL_INVALID: protected pixel changed"
    return {"changed_px": int(diff.sum()), "outside_bbox_changes": 0,
            "protected_changes": 0, "ok": True}


def compatible_pixel_report(frame, harr_t, palettes=None, tol=DEFAULT_TOL):
    """Per valid hostile object: compatible-pixel count, clipped fraction, zero flag.

    Used by the object-identity audit to flag on-screen hostiles with ZERO
    class-compatible pixels (a systematic schema/palette mismatch).
    """
    bbox, cls, valid = harr_t["hostile_bbox"], harr_t["hostile_class"], harr_t["hostile_valid"]
    out = []
    for i in np.where(valid)[0]:
        x, y, w, h = (int(v) for v in bbox[i])
        name = S.HOSTILE_NAME[int(cls[i])]
        if w <= 0 or h <= 0:
            out.append({"slot": int(i), "class": name, "compatible_px": 0,
                        "zero_compatible": True})
            continue
        comp = class_compatible_mask(frame[y:y + h, x:x + w], name, palettes, tol)
        out.append({"slot": int(i), "class": name, "bbox": [x, y, w, h],
                    "compatible_px": int(comp.sum()),
                    "area_px": int(w * h),
                    "zero_compatible": bool(comp.sum() == 0)})
    return out
