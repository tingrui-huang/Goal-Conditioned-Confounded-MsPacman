"""Controlled adapter that imports the GENUINE DIAMOND teacher from the private repo.

The private repository `Confounded-Agent-Distillation-main/` is a **read-only external
dependency**. We never copy, strip, rewrite, or commit its teacher implementation.
This adapter only:

  * resolves the repo path (env var `CONF_AGENT_TEACHER_PATH`, else a local default);
  * validates it exists and contains the expected teacher files;
  * adds it to `sys.path` in a controlled way;
  * registers `teacher` as a *namespace* package so the heavy/broken
    `teacher/__init__.py` (which imports sebulba -> jax/envpool) never runs;
  * stubs ONLY two transitive deps of `teacher.utils` that our inference path never
    exercises (`wandb`, `omegaconf`) and only if they are not already installed;
  * imports the original `ActorCritic`, `ActorCriticConfig`, `extract_state_dict`,
    `make_atari_env`, `TorchEnv`, and `AtariPreprocessing` directly from the repo.

It fails loudly if the path or any required module is missing and NEVER falls back to
the local DQN checkpoint or to any vendored copy.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

ENV_VAR = "CONF_AGENT_TEACHER_PATH"

# Fallback when the env var is unset: the repo-local private copy (inner nested dir).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TEACHER_PATH = (
    _REPO_ROOT / "Confounded-Agent-Distillation-main" / "Confounded-Agent-Distillation-main"
)

# Transitive imports of teacher.utils that are NOT used on our inference path.
# Real packages are preferred; a name here is stubbed ONLY if it is genuinely absent.
# Any stub is removed from sys.modules immediately after the teacher import (narrow
# scope) — see `_install_optional_stubs` / `_remove_stubs`.
_OPTIONAL_STUBS = ("wandb", "omegaconf")

# Stubs are kept here (even after sys.modules removal) so tests can prove that no
# stubbed attribute was ever accessed on the inference path.
_STUB_REGISTRY: Dict[str, "_TrackingStub"] = {}


class _TrackingStub(types.ModuleType):
    """A fake module that records every non-dunder attribute access.

    Returns a harmless MagicMock for any access so import never breaks, while the
    recorded `accesses` list lets a test assert the inference path never used it.
    """

    def __init__(self, name: str):
        super().__init__(name)
        object.__setattr__(self, "accesses", [])

    def __getattr__(self, item: str):
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        self.accesses.append(item)
        return MagicMock(name=f"{self.__name__}.{item}")


def _install_optional_stubs() -> List[str]:
    """Install tracking stubs for any genuinely-missing optional dep. Returns names added."""
    added: List[str] = []
    for name in _OPTIONAL_STUBS:
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)            # prefer the real package
        except ImportError:
            # reuse the same stub object across loads so its access log persists
            stub = _STUB_REGISTRY.get(name) or _TrackingStub(name)
            sys.modules[name] = stub
            _STUB_REGISTRY[name] = stub
            added.append(name)
    return added


def _remove_stubs(names: List[str]) -> None:
    """Remove the fake modules we added, so they do not stay globally registered."""
    for name in names:
        if isinstance(sys.modules.get(name), _TrackingStub):
            del sys.modules[name]


def stub_accesses() -> Dict[str, List[str]]:
    """{stub_name: [attributes accessed]} — empty lists mean the inference path never used it."""
    return {name: list(stub.accesses) for name, stub in _STUB_REGISTRY.items()}

_REQUIRED_FILES = (
    "teacher/actor_critic.py",
    "teacher/utils.py",
    "envs/env.py",
    "envs/atari_preprocessing.py",
)


@dataclass(frozen=True)
class ExternalTeacher:
    root: str
    ActorCritic: Any
    ActorCriticConfig: Any
    extract_state_dict: Any
    make_atari_env: Any
    TorchEnv: Any
    AtariPreprocessing: Any
    provenance: Dict[str, str]
    stubbed_modules: List[str]


_CACHE: ExternalTeacher | None = None


def resolve_teacher_root() -> Path:
    raw = os.environ.get(ENV_VAR)
    root = Path(raw).expanduser() if raw else _DEFAULT_TEACHER_PATH
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Teacher repo path does not exist: {root}\n"
            f"Set {ENV_VAR}=/absolute/path/to/Confounded-Agent-Distillation-main "
            f"(the inner dir that contains teacher/ and envs/)."
        )
    missing = [rel for rel in _REQUIRED_FILES if not (root / rel).is_file()]
    if missing:
        # be forgiving about an extra nesting level, then re-check
        nested = root / root.name
        if all((nested / rel).is_file() for rel in _REQUIRED_FILES):
            return nested.resolve()
        raise FileNotFoundError(
            f"Teacher repo at {root} is missing required files: {missing}. "
            f"Point {ENV_VAR} at the dir that directly contains teacher/ and envs/."
        )
    return root


def _register_namespace_pkg(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    mod = importlib.util.module_from_spec(spec)
    mod.__path__ = [str(path)]
    sys.modules[name] = mod


def load_external_teacher(force: bool = False) -> ExternalTeacher:
    """Import and return the genuine teacher symbols from the private repo."""
    global _CACHE
    if _CACHE is not None and not force:
        return _CACHE

    root = resolve_teacher_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Avoid executing teacher/__init__.py (sebulba -> jax). `envs` __init__ is clean.
    _register_namespace_pkg("teacher", root / "teacher")

    # Install stubs ONLY for genuinely-missing optional deps, and only for the
    # duration of the teacher import; restore global state in the finally block.
    stubbed = _install_optional_stubs()
    _prev_dont_write = sys.dont_write_bytecode      # do not write .pyc into the private repo
    sys.dont_write_bytecode = True
    try:
        ac = importlib.import_module("teacher.actor_critic")
        ut = importlib.import_module("teacher.utils")
        ev = importlib.import_module("envs.env")
        ap = importlib.import_module("envs.atari_preprocessing")
    except Exception as exc:  # fail loudly; never fall back
        raise ImportError(
            f"Failed to import the genuine teacher modules from {root}. "
            f"Do NOT fall back to a vendored copy or the local DQN. Original error: {exc!r}"
        ) from exc
    finally:
        sys.dont_write_bytecode = _prev_dont_write
        _remove_stubs(stubbed)                      # narrow scope: do not leave fakes registered

    provenance = {
        "ActorCritic": ac.ActorCritic.__module__ + " @ " + os.path.abspath(ac.__file__),
        "ActorCriticConfig": ac.ActorCriticConfig.__module__,
        "extract_state_dict": ut.extract_state_dict.__module__ + " @ " + os.path.abspath(ut.__file__),
        "make_atari_env": ev.make_atari_env.__module__ + " @ " + os.path.abspath(ev.__file__),
        "TorchEnv": ev.TorchEnv.__module__,
        "AtariPreprocessing": ap.AtariPreprocessing.__module__ + " @ " + os.path.abspath(ap.__file__),
    }

    _CACHE = ExternalTeacher(
        root=str(root),
        ActorCritic=ac.ActorCritic,
        ActorCriticConfig=ac.ActorCriticConfig,
        extract_state_dict=ut.extract_state_dict,
        make_atari_env=ev.make_atari_env,
        TorchEnv=ev.TorchEnv,
        AtariPreprocessing=ap.AtariPreprocessing,
        provenance=provenance,
        stubbed_modules=stubbed,
    )
    return _CACHE
