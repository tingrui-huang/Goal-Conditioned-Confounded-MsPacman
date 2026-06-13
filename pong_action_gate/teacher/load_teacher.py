"""Load the DIAMOND Pong teacher and build the matching env.

Design choices (locked in M0, approved):
  - The teacher classes (`ActorCritic`, `ActorCriticConfig`), checkpoint utility
    (`extract_state_dict`), and preprocessing (`AtariPreprocessing`) are imported
    DIRECTLY from the private teacher repo via `external_teacher` — no vendored copy.
  - The DIAMOND actor-critic checkpoint is fetched from HuggingFace
    (`eloialonso/diamond -> atari_100k/models/Pong.pt`), NOT the local DQN
    `teacher/ckpt/Pong.pt` (which is a different model: a Nature-CNN DQN).
  - `ActorCriticConfig` is constructed directly from `config.TeacherArch`,
    bypassing the missing `hydra` dependency. Every field is asserted against
    the live environment before use.
  - The checkpoint's `actor_critic` sub-state-dict is loaded with strict=True;
    zero missing / unexpected keys are required.

Nothing here makes any claim about teacher *competence* — that is M2.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import ale_py  # noqa: F401  (registers ALE envs in-process)
import gymnasium
import numpy as np
import torch
from torch import Tensor
from torch.distributions.categorical import Categorical

from .. import config as C
from .external_teacher import load_external_teacher


# --------------------------------------------------------------------------- #
# Checkpoint
# --------------------------------------------------------------------------- #
def download_diamond_ckpt(game: str = C.GAME) -> Path:
    """Download the DIAMOND Atari-100k checkpoint for `game` from HuggingFace."""
    from huggingface_hub import hf_hub_download

    filename = C.HF_CKPT_TEMPLATE.format(game=game)
    path = hf_hub_download(repo_id=C.HF_REPO_ID, filename=filename)
    return Path(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_actor_critic(arch: C.TeacherArch, device: str):
    ext = load_external_teacher()
    cfg = ext.ActorCriticConfig(
        lstm_dim=arch.lstm_dim,
        img_channels=arch.img_channels,
        img_size=arch.img_size,
        channels=list(arch.channels),
        down=list(arch.down),
        num_actions=arch.num_actions,
    )
    model = ext.ActorCritic(cfg).to(device)
    return model


def load_teacher(
    arch: Optional[C.TeacherArch] = None,
    device: str = "cpu",
    ckpt_path: Optional[Path] = None,
) -> Tuple[ActorCritic, Dict[str, Any]]:
    """Build the actor-critic, strict-load the DIAMOND weights, return (model, meta).

    Raises RuntimeError if the checkpoint does not load with strict=True.
    """
    arch = arch or C.TeacherArch()
    if ckpt_path is None:
        ckpt_path = download_diamond_ckpt(C.GAME)
    ckpt_path = Path(ckpt_path)

    ext = load_external_teacher()
    model = build_actor_critic(arch, device)

    full = torch.load(ckpt_path, map_location=device, weights_only=True)
    if not isinstance(full, dict):
        raise RuntimeError(f"Unexpected checkpoint top-level type: {type(full)}")
    ac_sd = ext.extract_state_dict(full, "actor_critic")
    if len(ac_sd) == 0:
        raise RuntimeError(
            "extract_state_dict(ckpt, 'actor_critic') is empty — checkpoint key "
            f"layout unexpected. Top-level prefixes: "
            f"{sorted({k.split('.', 1)[0] for k in full})}"
        )

    missing, unexpected = model.load_state_dict(ac_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Strict checkpoint load failed.\n"
            f"  missing keys ({len(missing)}): {missing}\n"
            f"  unexpected keys ({len(unexpected)}): {unexpected}"
        )
    model.eval()

    meta = {
        "ckpt_path": str(ckpt_path),
        "ckpt_sha256": _sha256(ckpt_path),
        "n_actor_critic_params": int(sum(p.numel() for p in model.parameters())),
        "arch": asdict(arch),
        "actor_linear_out": int(model.actor_linear.out_features),
        "device": device,
        "teacher_source": "external-private-repo",
        "teacher_root": ext.root,
        "teacher_provenance": ext.provenance,
        "stubbed_modules": ext.stubbed_modules,
    }
    return model, meta


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
class SingleAtariEnv:
    """In-process single-env wrapper around the GENUINE teacher AtariPreprocessing.

    The preprocessing wrapper itself is imported unmodified from the private repo
    (`envs/atari_preprocessing.py`) via `external_teacher`. Faithful to the
    teacher's TorchEnv:
      * same preprocessing wrapper (frame_skip=4, noop_max=30, screen_size=64,
        num_stack=4, full_action_space=False, base frameskip=1);
      * same normalisation: obs/255*2-1, permuted to (1,C,H,W) in [-1,1]
        (TorchEnv._to_tensor, envs/env.py:91-97).
    Deliberately avoids AsyncVectorEnv: for a single-env rollout the subprocess
    worker adds no value and is fragile under Windows `spawn` (ALE not registered
    in the worker; BrokenPipe on teardown). done_on_life_loss is irrelevant for
    Pong (no ALE lives) but is honoured for parity: when True, life loss -> end.
    """

    def __init__(self, cfg: C.M1Config):
        ext = load_external_teacher()
        mes = None if cfg.max_episode_steps in (0, None) else cfg.max_episode_steps
        base = gymnasium.make(
            cfg.env_id,
            full_action_space=C.FULL_ACTION_SPACE,
            frameskip=1,
            render_mode="rgb_array",
            max_episode_steps=mes,
        )
        self.env = ext.AtariPreprocessing(
            env=base, noop_max=C.NOOP_MAX, frame_skip=C.FRAME_SKIP,
            screen_size=C.IMG_SIZE, num_stack=C.NUM_STACK,
        )
        self.device = cfg.device
        self.done_on_life_loss = cfg.done_on_life_loss
        self.num_actions = self.env.action_space.n
        meanings = list(self.env.unwrapped.get_action_meanings())
        if self.num_actions != C.N_ACTIONS:
            raise RuntimeError(f"num_actions={self.num_actions} != expected {C.N_ACTIONS}")
        if meanings != C.ACTION_MEANINGS:
            raise RuntimeError(f"action meanings {meanings} != expected {C.ACTION_MEANINGS}")
        self.action_meanings = meanings

    def _to_tensor(self, obs_hwc: np.ndarray) -> Tensor:
        x = torch.tensor(obs_hwc[None], device=self.device)        # (1,H,W,C)
        return x.div(255).mul(2).sub(1).permute(0, 3, 1, 2).contiguous().float()

    def reset(self, seed):
        if isinstance(seed, (list, tuple)):
            seed = int(seed[0])
        obs, info = self.env.reset(seed=seed)
        return self._to_tensor(obs), info

    def step(self, action):
        a = int(action.item()) if torch.is_tensor(action) else int(action)
        obs, rew, term, trunc, info = self.env.step(a)
        if self.done_on_life_loss and info.get("life_loss", False):
            term = True
        rew_t = torch.tensor([float(rew)], device=self.device)
        end_t = torch.tensor([bool(term)], dtype=torch.uint8, device=self.device)
        trunc_t = torch.tensor([bool(trunc)], dtype=torch.uint8, device=self.device)
        return self._to_tensor(obs), rew_t, end_t, trunc_t, info

    def close(self):
        self.env.close()


def make_env(cfg: C.M1Config, num_envs: int = 1):
    """Build the exact teacher env (test config) and assert the action set."""
    if num_envs != 1:
        raise NotImplementedError("M1 uses a single env; multi-env collection is later.")
    return SingleAtariEnv(cfg)


# --------------------------------------------------------------------------- #
# Policy wrapper (hides obs slicing + recurrence)
# --------------------------------------------------------------------------- #
class TeacherPolicy:
    """Stochastic recurrent DIAMOND policy.

    `act` reproduces the env_loop inference step exactly:
      logits, val, (hx,cx) = model.predict_act_value(obs[:, 4:], (hx,cx))
      action = Categorical(logits=logits).sample()
    Sampling uses torch's global RNG; seed it (torch.manual_seed) for a
    reproducible stochastic rollout.
    """

    def __init__(self, model: ActorCritic, device: str = "cpu"):
        self.model = model
        self.device = device

    def initial_state(self, batch: int = 1) -> Tuple[Tensor, Tensor]:
        hx = torch.zeros(batch, self.model.lstm_dim, device=self.device)
        cx = torch.zeros(batch, self.model.lstm_dim, device=self.device)
        return hx, cx

    @torch.no_grad()
    def act(self, obs7: Tensor, hx_cx: Tuple[Tensor, Tensor]):
        """obs7: (B,7,H,W) full env obs. Returns (action, logits, value, hx_cx)."""
        obs_rgb = obs7[:, C.TEACHER_OBS_SLICE, :, :]          # (B,3,H,W) colored frame
        logits, val, hx_cx = self.model.predict_act_value(obs_rgb, hx_cx)
        action = Categorical(logits=logits).sample()
        return action, logits, val, hx_cx


def save_resolved_config(path: Path, cfg: C.M1Config, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "m1_config": {
            "game": cfg.game, "env_id": cfg.env_id, "seed": cfg.seed,
            "done_on_life_loss": cfg.done_on_life_loss,
            "max_episode_steps": cfg.max_episode_steps,
            "safety_step_cap": cfg.safety_step_cap, "device": cfg.device,
            "arch": asdict(cfg.arch),
        },
        "teacher_meta": meta,
        "constants": {
            "n_actions": C.N_ACTIONS, "action_meanings": C.ACTION_MEANINGS,
            "img_size": C.IMG_SIZE, "num_stack": C.NUM_STACK,
            "n_obs_channels": C.N_OBS_CHANNELS,
            "teacher_obs_slice": [C.TEACHER_OBS_SLICE.start, C.TEACHER_OBS_SLICE.stop],
            "hf_repo_id": C.HF_REPO_ID,
            "hf_ckpt": C.HF_CKPT_TEMPLATE.format(game=C.GAME),
        },
    }
    with open(path, "w") as f:
        json.dump(blob, f, indent=2)
