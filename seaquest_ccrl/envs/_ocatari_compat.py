"""Runtime compatibility shim for OCAtari under modern NumPy (NEP 50).

ALE `getRAM()` returns a uint8 array. OCAtari's per-game RAM extractors do wide
arithmetic on those bytes (e.g. Seaquest `score*10000`, `ram[30+i]+16*j`). Under
NumPy >= 2.0 (NEP 50) a uint8 scalar stays uint8 through arithmetic, so those
expressions overflow and raise `OverflowError`. Older NumPy silently promoted to
Python int, which is what the extractors assume.

We fix this PORTABLY (works with a pip-installed `ocatari`, not just the vendored
checkout) by wrapping `ocatari.core.detect_objects_ram` to cast `ram_state` to
int64 once, at the single chokepoint through which every game's extractor runs.
`core.py` does `from ...extract_ram_info import detect_objects_ram`, so the live
binding to patch is the name in `ocatari.core`.

Idempotent: safe to call more than once.
"""
import functools

import numpy as np


def patch_ocatari_nep50() -> bool:
    """Install the int64 ram_state cast. Returns True if a patch was applied."""
    import ocatari.core as core

    orig = core.detect_objects_ram
    if getattr(orig, "_nep50_patched", False):
        return False

    @functools.wraps(orig)
    def detect_objects_ram(objects, ram_state, *args, **kwargs):
        if hasattr(ram_state, "astype"):
            ram_state = ram_state.astype(np.int64)
        return orig(objects, ram_state, *args, **kwargs)

    detect_objects_ram._nep50_patched = True
    core.detect_objects_ram = detect_objects_ram
    return True
