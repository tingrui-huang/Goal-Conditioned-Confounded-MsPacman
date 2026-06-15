"""OCAtari branch-environment helpers for Stage-S0 (runs in the unified container).

Provides a thin, audited wrapper over OCAtari/ALE for Seaquest with:
  * exact, documented construction (frameskip, sticky, full action space);
  * raw RGB frame access;
  * ALE clone/restore of the FULL emulator state;
  * object-record extraction with EXPLICIT missing handling (no carry-forward);
  * multiple oxygen signal probes (OCAtari object, RAM bytes, pixel bar).

Nothing here masks oxygen or modifies dynamics. Descriptive only.
"""
import numpy as np


SEAQUEST_ENV_ID = "ALE/Seaquest-v5"


def make_ocatari(frameskip=4, sticky=0.0, full_action_space=True, seed=0):
    from ocatari.core import OCAtari
    env = OCAtari(SEAQUEST_ENV_ID, mode="ram", hud=True, render_mode="rgb_array",
                  frameskip=frameskip, repeat_action_probability=sticky,
                  full_action_space=full_action_space)
    env.reset(seed=seed)
    return env


def ale_of(env):
    # OCAtari wraps a gymnasium env; reach the ALE interface.
    e = env._env
    while hasattr(e, "env") and not hasattr(e, "ale"):
        if hasattr(e.unwrapped, "ale"):
            return e.unwrapped.ale
        e = e.env
    return env._env.unwrapped.ale


def raw_rgb(env):
    return np.asarray(env._env.unwrapped.ale.getScreenRGB(), dtype=np.uint8)


def ram(env):
    return np.asarray(env._env.unwrapped.ale.getRAM(), dtype=np.uint8).copy()


def object_records(env):
    """Return list of dicts for currently detected, non-empty objects.

    Missing/absent objects are simply ABSENT from the list (explicit), never
    carried forward. Each record carries whatever fields the object exposes.
    """
    recs = []
    for o in env.objects:
        cat = getattr(o, "category", type(o).__name__)
        if cat in ("NoObject",):
            continue
        x = getattr(o, "x", None); y = getattr(o, "y", None)
        w = getattr(o, "w", None); h = getattr(o, "h", None)
        rec = {"category": cat,
               "x": None if x is None else float(x),
               "y": None if y is None else float(y),
               "w": None if w is None else float(w),
               "h": None if h is None else float(h)}
        val = getattr(o, "value", None)
        if val is not None:
            try:
                rec["value"] = float(val)
            except Exception:
                rec["value"] = None
        rec["is_hud"] = bool(getattr(o, "hud", False))
        recs.append(rec)
    return recs


def oxygen_signals(env):
    """Probe ALL candidate oxygen signals. Missing -> explicit None, never carried.

    Returns dict with:
      oc_oxygenbar_width : filled width of an OCAtari OxygenBar-like object (px) or None
      oc_oxygen_value    : a .value field on such object, if present, else None
      ram_oxygen_<addr>  : raw RAM bytes at candidate addresses (diagnostic, unlabeled)
      pixel_bar_width    : measured filled width of the oxygen bar from pixels or None
    """
    out = {}
    # 1) OCAtari object-based oxygen
    oc_w = None; oc_val = None
    for o in env.objects:
        cat = getattr(o, "category", "")
        if "oxygen" in cat.lower() or "Oxygen" in cat:
            if "deplet" not in cat.lower():  # prefer the FILLED bar object
                w = getattr(o, "w", None)
                if w is not None:
                    oc_w = float(w)
                v = getattr(o, "value", None)
                if v is not None:
                    try: oc_val = float(v)
                    except Exception: pass
    out["oc_oxygenbar_width"] = oc_w
    out["oc_oxygen_value"] = oc_val

    # 2) Raw RAM diagnostic dump of a few plausible addresses (unlabeled, descriptive)
    r = ram(env)
    for addr in (0x66, 0x67, 0x68, 0x5B, 0x70, 0x71, 0x72, 102, 103):
        try:
            out[f"ram_byte_{addr}"] = int(r[addr])
        except Exception:
            out[f"ram_byte_{addr}"] = None

    # 3) Pixel-bar measurement: the oxygen strip is a horizontal bar low on screen.
    #    Measure the filled (bright) run width within a band; descriptive, may be None.
    try:
        rgb = raw_rgb(env)
        # band rows around the known Seaquest oxygen strip area (audited separately).
        band = rgb[170:176, 40:120, :]
        # "filled" pixels: not background water (dark blue) and not black.
        nonbg = ~((band[..., 2] > 100) & (band[..., 0] < 60) & (band[..., 1] < 80))
        col_any = nonbg.any(axis=(0, 2)) if nonbg.ndim == 3 else nonbg.any(axis=0)
        out["pixel_bar_width"] = int(col_any.sum())
    except Exception:
        out["pixel_bar_width"] = None
    return out
