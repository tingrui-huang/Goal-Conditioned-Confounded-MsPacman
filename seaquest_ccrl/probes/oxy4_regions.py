"""Phase 2.1b — RAW-frame (210x160) visual regions for the oxygen leakage source audit.

Coordinates were defined by INSPECTING actual raw frames (scripts/oxy4_audit_regions.py
renders the overlays) cross-referenced with a per-row temporal-variance profile, NOT by
guessing from resized 84x84 images and NOT by tuning on probe performance.

Seaquest 210x160 layout (verified on raw_hf traj_0000, oldest->newest within episode):
  rows   0- 31 : TOP HUD     — score digits (rows ~8-18) + lives/divers icons (rows ~22-30)
  rows  32- 45 : surface stripes (static rainbow band where the sub refills)
  rows  46-151 : UNDERWATER GAMEPLAY — player sub, fish, enemy subs, divers, motion
  rows 152-209 : BOTTOM HUD   — gray sea-floor panel, "OXYGEN" label (cols ~23-43),
                                oxygen bar (rows 170-174, cols 48-111), divers-collected
                                row (rows ~178-186), ACTIVISION logo (rows ~192-198)

All rects are (x, y, w, h) on the raw 210x160 frame -> rows [y:y+h], cols [x:x+w].
"""
import numpy as np

from seaquest_ccrl import config as C

RAW_H, RAW_W = 210, 160

# Oxygen-bar-only mask = the EXISTING confounder mask (config.OXY_MASK_RECT).
OXY_BAR_RECT = tuple(C.OXY_MASK_RECT)            # (46,162,69,16): rows 162-177, cols 46-114

# Complete bottom status panel (everything from the sea-floor band downward): oxygen bar,
# OXYGEN label, divers-collected indicators, ACTIVISION logo.
BOTTOM_HUD_RECT = (0, 152, RAW_W, RAW_H - 152)   # rows 152-209, full width

# Complete top scoreboard / lives display.
TOP_HUD_RECT = (0, 0, RAW_W, 32)                 # rows 0-31, full width

# Underwater gameplay crop: from the swimmable surface line (SURFACE_Y=46) down to just
# above the bottom HUD panel. Top scoreboard and bottom HUD are both excluded by the crop.
GAMEPLAY_CROP = (0, 46, RAW_W, 152 - 46)         # rows 46-151, full width -> (106,160,3)


def _apply_rect_zero(frame, rect):
    """Zero a (x,y,w,h) rect on a raw (210,160,3) uint8 frame (copy)."""
    x, y, w, h = rect
    out = frame.copy()
    out[y:y + h, x:x + w, :] = 0
    return out


def transform_raw(frame, variant):
    """Apply a region transform to ONE raw (210,160,3) uint8 frame. Returns a raw-sized
    frame for the mask variants, or a cropped frame for 'gameplay_crop' (resized later).

    variant:
      'visible'            -> unmasked (oracle)
      'oxybar_masked'      -> zero OXY_BAR_RECT only  (== the confounder mask)
      'bottomhud_masked'   -> zero the entire BOTTOM_HUD_RECT
      'tophud_masked'      -> zero the entire TOP_HUD_RECT
      'top_and_bottom_masked' -> zero both HUD panels
      'gameplay_crop'      -> crop GAMEPLAY_CROP rows/cols (still 3-channel, smaller)
    """
    if variant == "visible":
        return frame
    if variant == "oxybar_masked":
        return _apply_rect_zero(frame, OXY_BAR_RECT)
    if variant == "bottomhud_masked":
        return _apply_rect_zero(frame, BOTTOM_HUD_RECT)
    if variant == "tophud_masked":
        return _apply_rect_zero(frame, TOP_HUD_RECT)
    if variant == "top_and_bottom_masked":
        return _apply_rect_zero(_apply_rect_zero(frame, BOTTOM_HUD_RECT), TOP_HUD_RECT)
    if variant == "gameplay_crop":
        x, y, w, h = GAMEPLAY_CROP
        return frame[y:y + h, x:x + w, :]
    raise ValueError(f"unknown variant {variant!r}")


def regions_manifest():
    """Raw-frame coordinates of every region, for the audit JSON."""
    return {
        "raw_shape": [RAW_H, RAW_W, 3],
        "format": "(x, y, w, h) -> rows [y:y+h], cols [x:x+w] on the raw 210x160 frame",
        "oxy_bar_rect": list(OXY_BAR_RECT),
        "bottom_hud_rect": list(BOTTOM_HUD_RECT),
        "top_hud_rect": list(TOP_HUD_RECT),
        "gameplay_crop": list(GAMEPLAY_CROP),
        "surface_y": float(C.OXY_MASK_RECT and 46.0),
        "derivation": "row temporal-variance profile + visual inspection of raw_hf frames; "
                      "not tuned on probe performance",
    }
