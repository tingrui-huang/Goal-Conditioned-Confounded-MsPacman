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
# Stubbed only if genuinely absent; documented so the masking is never silent.
_OPTIONAL_STUBS = ("wandb", "omegaconf")

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

    stubbed: List[str] = []
    for name in _OPTIONAL_STUBS:
        try:
            importlib.import_module(name)
        except ImportError:
            sys.modules[name] = MagicMock(name=f"{name}_stub_for_teacher_import")
            stubbed.append(name)

    # Do not write .pyc/__pycache__ into the private repo: keep it strictly read-only.
    _prev_dont_write = sys.dont_write_bytecode
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
