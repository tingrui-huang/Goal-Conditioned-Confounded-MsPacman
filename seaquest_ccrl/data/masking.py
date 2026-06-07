"""Load-time oxygen masking (Invariant 3: mask the FULL strip, not the filled width).

The oxygen level leaks ONLY through the filled/empty boundary inside the bar strip, so
we zero a FIXED rectangle (OXY_MASK_RECT) covering the entire strip + label every time
-- the masked region is therefore constant w.r.t. oxygen (verified by check B). The
mask is applied HERE, at load, never at collection, so the unmasked oracle frame stays
recoverable (Invariant 4) and the dataset is re-maskable.
"""
import numpy as np

from seaquest_ccrl import config as C


def apply_oxygen_mask(frame: np.ndarray, rect=None) -> np.ndarray:
    """Return a copy of `frame` with the oxygen-bar rectangle zeroed."""
    x, y, w, h = rect if rect is not None else C.OXY_MASK_RECT
    out = frame.copy()
    out[y:y + h, x:x + w, :] = 0
    return out


def oracle(frame: np.ndarray) -> np.ndarray:
    """Oracle view = unmasked frame (confounder observable). Identity."""
    return frame


# === ENEMY CONFOUNDER masking (Level-1 v2) =================================
# CRITICAL: inpaint each enemy bbox with the BACKGROUND WATER COLOR, never a black
# box. A black box would reveal the enemy's location ("something is here") and
# collapse the confounding -- the learner must see calm, empty water where an
# enemy is. Enemies roam, so the mask is per-frame (bboxes come from OCAtari and
# are stored alongside the unmasked frame, keeping the oracle recoverable).

def apply_enemy_mask(frame: np.ndarray, enemy_bboxes, water_color=None) -> np.ndarray:
    """Return a copy of `frame` with every hostile-enemy bbox inpainted to background.

    Local-background fill: each bbox is filled by copying the adjacent water COLUMN
    (left or right neighbour, whichever is closer to the reference water colour --
    this avoids copying the black screen border at the edges and tracks any vertical
    shading). Falls back to the flat reference colour if neither side is water-like.
    A flat fixed colour leaves visible seams near the edges (it reveals the enemy
    location and weakens the confounding); copying real neighbouring water does not.

    enemy_bboxes: iterable of (x, y, w, h) int boxes (already filtered to hostiles).
    """
    ref = np.asarray(water_color if water_color is not None else C.WATER_COLOR,
                     dtype=np.int16)
    out = frame.copy()
    H, W = out.shape[:2]
    for (x, y, w, h) in enemy_bboxes:
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(W, int(x) + int(w)), min(H, int(y) + int(h))
        if x1 <= x0 or y1 <= y0:
            continue
        cands = []
        if x0 - 1 >= 0:
            cands.append(out[y0:y1, x0 - 1, :])      # left neighbour column (h,3)
        if x1 < W:
            cands.append(out[y0:y1, x1, :])          # right neighbour column
        best, best_dist = None, 1e9
        for col in cands:
            dist = float(np.abs(col.astype(np.int16).mean(axis=0) - ref).sum())
            if dist < best_dist:
                best_dist, best = dist, col
        if best is not None and best_dist < 90:       # neighbour looks like water
            out[y0:y1, x0:x1, :] = best[:, None, :]   # broadcast column across width
        else:
            out[y0:y1, x0:x1, :] = ref.astype(np.uint8)
    return out
