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
    """Oracle view = unmasked frame (oxygen observable). Identity."""
    return frame
