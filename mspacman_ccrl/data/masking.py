"""Ghost masking for Ms. Pac-Man (inpaint to local corridor background).

CRITICAL (same rule as the Seaquest enemy version): do NOT reveal the ghost's
location. Inpaint each ghost bbox with the surrounding background so the masked
frame looks like an empty corridor. Corridors are black, but ghosts sit next to
blue walls and pills, so we fill from the nearest background neighbour rather than
a flat colour to avoid seams / wall holes.
"""
import numpy as np

from mspacman_ccrl import config as C


def _is_bg(col, wall, corridor, tol=40):
    """A neighbour column is 'background' if it's mostly wall or corridor."""
    m = col.astype(np.int16).mean(axis=0)
    return min(np.abs(m - wall).sum(), np.abs(m - corridor).sum()) < tol * 3


def apply_ghost_mask(frame: np.ndarray, ghost_bboxes, corridor_color=None) -> np.ndarray:
    """Return a copy of `frame` with every ghost bbox inpainted to background.

    Fill each bbox by copying the nearest background neighbour column (left/right,
    whichever is more background-like), falling back to the flat corridor colour.
    """
    corridor = np.asarray(corridor_color if corridor_color is not None
                          else C.CORRIDOR_COLOR, dtype=np.int16)
    wall = np.asarray(C.WALL_COLOR, dtype=np.int16)
    out = frame.copy()
    H, W = out.shape[:2]
    for (x, y, w, h) in ghost_bboxes:
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(W, int(x) + int(w)), min(H, int(y) + int(h))
        if x1 <= x0 or y1 <= y0:
            continue
        cands = []
        if x0 - 1 >= 0:
            cands.append(out[y0:y1, x0 - 1, :])
        if x1 < W:
            cands.append(out[y0:y1, x1, :])
        best, best_dist = None, 1e9
        for col in cands:
            m = col.astype(np.int16).mean(axis=0)
            dist = float(min(np.abs(m - wall).sum(), np.abs(m - corridor).sum()))
            if dist < best_dist:
                best_dist, best = dist, col
        if best is not None and best_dist < 120:
            out[y0:y1, x0:x1, :] = best[:, None, :]
        else:
            out[y0:y1, x0:x1, :] = corridor.astype(np.uint8)
    return out


def oracle(frame: np.ndarray) -> np.ndarray:
    """Oracle view = unmasked frame (ghosts observable). Identity."""
    return frame
