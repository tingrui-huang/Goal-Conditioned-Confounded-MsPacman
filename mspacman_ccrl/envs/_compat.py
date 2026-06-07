"""Reuse the tested OCAtari compat shim (NumPy NEP50 int64 cast + stdout silencing)."""
from seaquest_ccrl.envs._ocatari_compat import patch_ocatari_nep50


def patch_ocatari() -> bool:
    return patch_ocatari_nep50()
