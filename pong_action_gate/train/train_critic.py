"""M7 Phase A.1-3 — resumable critic training with PREDECLARED val-loss checkpoint selection.

Checkpoint selection consults ONLY validation loss (fixed step budget -> pick the
checkpoint with minimum validation loss). It NEVER consults action diagnostics.
Checkpoints are resumable (model + optimizer + step + RNG states saved frequently).
"""
from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from ..data.goal_sampler import duplicate_corrected_targets
from . import dataset as D
from .critics import PixelSACritic, StateSACritic, nce_loss

CKPT_ROOT = Path("artifacts/pong_action_gate/m7")


def resolve_device(device: str) -> str:
    if device in ("cuda", "auto") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class TrainConfig:
    tag: str = "full"
    critic: str = "state"          # state | pixel
    seed: int = 0
    n_episodes: int = 30
    val_frac: float = 0.2
    steps: int = 2000
    batch: int = 256
    lr: float = 3e-4
    eval_every: int = 200
    ckpt_every: int = 200
    val_batches: int = 4
    device: str = "cpu"


def build_data(cfg: TrainConfig):
    with_pixels = (cfg.critic == "pixel")
    train_ids, val_ids = D.split_episodes(cfg.n_episodes, cfg.val_frac, cfg.seed)
    all_ids = sorted(train_ids + val_ids)
    eps = D.load_subset(cfg.tag, all_ids, with_pixels=with_pixels)
    id_to_local = {g: i for i, g in enumerate(all_ids)}
    train_local = [id_to_local[g] for g in train_ids]
    val_local = [id_to_local[g] for g in val_ids]
    train_anchors = D.build_anchor_pool(eps, train_local)
    val_anchors = D.build_anchor_pool(eps, val_local)
    feats = [D.build_state_features(e) for e in eps] if cfg.critic == "state" else None
    return {"eps": eps, "train_ids": train_ids, "val_ids": val_ids,
            "train_anchors": train_anchors, "val_anchors": val_anchors, "feats": feats}


def make_batch_fn(cfg: TrainConfig, data, device):
    if cfg.critic == "state":
        def fn(anchors, idx):
            s, a, g = D.state_batch(anchors, idx, data["feats"])
            return (torch.as_tensor(s, device=device),
                    torch.as_tensor(a.astype(np.int64), device=device),
                    torch.as_tensor(D.norm_goal(g), device=device), g.astype(int))
    else:
        def fn(anchors, idx):
            f, a, g = D.pixel_batch(anchors, idx, data["eps"])
            return (torch.as_tensor(f, device=device),
                    torch.as_tensor(a.astype(np.int64), device=device),
                    torch.as_tensor(D.norm_goal(g), device=device), g.astype(int))
    return fn


def make_critic(cfg: TrainConfig):
    return StateSACritic(D.STATE_DIM, 6) if cfg.critic == "state" else PixelSACritic(4, 6)


def _val_loss(critic, make_batch, val_anchors, B, n_batches, rng) -> float:
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            idx = rng.integers(len(val_anchors), size=min(B, len(val_anchors)))
            obs, action, goaln, goals = make_batch(val_anchors, idx)
            tgt = torch.as_tensor(duplicate_corrected_targets(goals), device=obs.device)
            losses.append(float(nce_loss(critic.logits_matrix(obs, action, goaln), tgt)))
    return float(np.mean(losses))


def run_dir(cfg: TrainConfig) -> Path:
    return CKPT_ROOT / f"{cfg.critic}_seed{cfg.seed}"


def save_ckpt(path: Path, critic, opt, step, val_loss, cfg, train_rng):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": critic.state_dict(), "opt": opt.state_dict(), "step": step,
        "val_loss": val_loss, "config": asdict(cfg),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": train_rng.bit_generator.state,
    }, path)


def load_ckpt(path: Path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


_CKPT_RE = re.compile(r"^ckpt_(\d+)\.pt$")


def latest_ckpt(d: Path) -> Optional[Path]:
    """Return the checkpoint with the largest valid integer step.

    Accepts ONLY filenames matching exactly `^ckpt_(\\d+)\\.pt$`. Malformed or
    duplicate ckpt-style names (e.g. Google Drive's `ckpt_1800 (1).pt`) are ignored
    with a clear warning; unrelated `.pt` files are skipped silently. Selection is by
    integer step value (not lexicographic), so `ckpt_1000.pt` beats `ckpt_50.pt`.
    """
    valid: List[Tuple[int, Path]] = []
    for p in Path(d).glob("*.pt"):
        m = _CKPT_RE.match(p.name)
        if m:
            valid.append((int(m.group(1)), p))
        elif p.name.startswith("ckpt_"):
            warnings.warn(f"Ignoring malformed/duplicate checkpoint filename: {p.name!r}")
    if not valid:
        return None
    return max(valid, key=lambda t: t[0])[1]


def train(cfg: TrainConfig, resume: bool = False) -> Dict[str, Any]:
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    data = build_data(cfg)
    make_batch = make_batch_fn(cfg, data, device)
    critic = make_critic(cfg).to(device)
    opt = torch.optim.Adam(critic.parameters(), lr=cfg.lr)
    d = run_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)

    train_rng = np.random.default_rng(cfg.seed)
    val_rng = np.random.default_rng(cfg.seed + 999)
    start_step = 0
    val_curve: List[Dict[str, Any]] = []

    if resume and latest_ckpt(d) is not None:
        ck = load_ckpt(latest_ckpt(d), device)
        critic.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        torch.set_rng_state(ck["torch_rng"].cpu() if hasattr(ck["torch_rng"], "cpu") else ck["torch_rng"])
        train_rng.bit_generator.state = ck["numpy_rng"]
        if (d / "val_curve.json").exists():
            val_curve = json.loads((d / "val_curve.json").read_text())

    n = len(data["train_anchors"])
    for step in range(start_step, cfg.steps + 1):
        if step % cfg.eval_every == 0:
            vl = _val_loss(critic, make_batch, data["val_anchors"], cfg.batch, cfg.val_batches, val_rng)
            val_curve.append({"step": step, "val_loss": vl})
            (d / "val_curve.json").write_text(json.dumps(val_curve, indent=2))
        if step % cfg.ckpt_every == 0:
            save_ckpt(d / f"ckpt_{step}.pt", critic, opt, step, val_curve[-1]["val_loss"], cfg, train_rng)
        if step < cfg.steps:
            idx = train_rng.integers(n, size=cfg.batch)
            obs, action, goaln, goals = make_batch(data["train_anchors"], idx)
            tgt = torch.as_tensor(duplicate_corrected_targets(goals), device=device)
            opt.zero_grad(set_to_none=True)
            loss = nce_loss(critic.logits_matrix(obs, action, goaln), tgt)
            loss.backward(); opt.step()

    # PREDECLARED selection: min validation loss over the fixed budget (no action diagnostics)
    best = min(val_curve, key=lambda r: r["val_loss"])
    selected = {"rule": "min validation loss over fixed step budget (action diagnostics NOT consulted)",
                "selected_step": best["step"], "val_loss": best["val_loss"],
                "ckpt": f"ckpt_{best['step']}.pt", "device": device,
                "split": {"train_ids": data["train_ids"], "val_ids": data["val_ids"]}}
    (d / "selected.json").write_text(json.dumps(selected, indent=2))
    return {"run_dir": str(d), "selected": selected, "val_curve": val_curve, "device": device}


def main() -> None:
    ap = argparse.ArgumentParser(description="M7 critic training (resumable, val-loss selection).")
    for k, v in asdict(TrainConfig()).items():
        ap.add_argument(f"--{k.replace('_','-')}", type=type(v), default=v)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = TrainConfig(**{k: getattr(args, k) for k in asdict(TrainConfig())})
    out = train(cfg, resume=args.resume)
    print(json.dumps(out["selected"], indent=2))


if __name__ == "__main__":
    main()
