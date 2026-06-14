"""M7 Phase A.4-5 — frozen-checkpoint held-out action diagnostics + episode-level bootstrap CI.

The checkpoint is selected by train_critic (validation loss) and FROZEN before any action
diagnostic is computed. Diagnostics are computed per validation episode so the confidence
intervals come from an episode-level bootstrap (resample whole episodes with replacement).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from ..data.goal_sampler import duplicate_corrected_targets
from . import dataset as D
from .critics import nce_loss
from .train_critic import (TrainConfig, build_data, load_ckpt, make_batch_fn,
                           make_critic, resolve_device, run_dir)


def _episode_metrics(critic, make_batch, anchors_idx, anchors, B, rng) -> Dict[str, float]:
    idx = rng.choice(anchors_idx, size=min(B, len(anchors_idx)), replace=False)
    obs, action, goaln, goals = make_batch(anchors, idx)
    tgt = torch.as_tensor(duplicate_corrected_targets(goals), device=obs.device)
    with torch.no_grad():
        correct = float(nce_loss(critic.logits_matrix(obs, action, goaln), tgt))
        perm = torch.as_tensor(rng.permutation(len(action)), device=obs.device)
        shuffled = float(nce_loss(critic.logits_matrix(obs, action[perm], goaln), tgt))
        randa = torch.as_tensor(rng.integers(critic.n_actions, size=len(action)).astype(np.int64), device=obs.device)
        replaced = float(nce_loss(critic.logits_matrix(obs, randa, goaln), tgt))
        zeroed = float(nce_loss(critic.logits_matrix(obs, action, goaln, zero_action=True), tgt))
        sa = critic.scores_all_actions(obs, goaln)
        same_state_std = float(sa.std(1).mean())
    return {"shuffle_minus_correct": shuffled - correct,
            "replace_minus_correct": replaced - correct,
            "zero_minus_correct": zeroed - correct,
            "same_state_all_action_std": same_state_std}


def _bootstrap_ci(per_ep: List[Dict[str, float]], key: str, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    vals = np.array([m[key] for m in per_ep])
    point = float(vals.mean())
    boots = [float(vals[rng.integers(len(vals), size=len(vals))].mean()) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"point": point, "ci95": [float(lo), float(hi)],
            "ci_excludes_zero": bool(lo > 0 or hi < 0)}


def run(cfg: TrainConfig, n_boot: int = 2000, per_ep_batch: int = 256, seed: int = 0) -> Dict[str, Any]:
    device = resolve_device(cfg.device)
    d = run_dir(cfg)
    selected = json.loads((d / "selected.json").read_text())
    critic = make_critic(cfg).to(device)
    ck = load_ckpt(d / selected["ckpt"], device)
    critic.load_state_dict(ck["model"]); critic.eval()      # FROZEN

    data = build_data(cfg)
    make_batch = make_batch_fn(cfg, data, device)
    # per-validation-episode anchors (group anchors by episode)
    val_local = sorted(set(a.ep for a in data["val_anchors"]))
    by_ep = {li: [i for i, a in enumerate(data["val_anchors"]) if a.ep == li] for li in val_local}

    rng = np.random.default_rng(seed)
    per_ep = []
    for li, aidx in by_ep.items():
        if len(aidx) >= 8:
            per_ep.append(_episode_metrics(critic, make_batch, np.array(aidx), data["val_anchors"],
                                           per_ep_batch, rng))

    keys = ["shuffle_minus_correct", "replace_minus_correct", "zero_minus_correct", "same_state_all_action_std"]
    cis = {k: _bootstrap_ci(per_ep, k, n_boot, seed + 1) for k in keys}
    out = {
        "milestone": "M7-gate-diagnostics",
        "frozen_checkpoint": selected["ckpt"], "selected_by": selected["rule"],
        "selected_step": selected["selected_step"], "device": device,
        "n_val_episodes_used": len(per_ep), "per_episode_batch": per_ep_batch,
        "n_bootstrap": n_boot,
        "episode_level_bootstrap_ci": cis,
        "per_episode": per_ep,
        "NOTE": "Action diagnostics computed ONCE on the frozen val-loss-selected checkpoint.",
    }
    (d / "gate_diagnostics.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="M7 frozen-checkpoint action diagnostics + episode bootstrap CI.")
    ap.add_argument("--critic", type=str, default="state")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-episodes", type=int, default=30)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--per-ep-batch", type=int, default=256)
    args = ap.parse_args()
    cfg = TrainConfig(critic=args.critic, seed=args.seed, n_episodes=args.n_episodes,
                      val_frac=args.val_frac, device=args.device)
    out = run(cfg, n_boot=args.n_boot, per_ep_batch=args.per_ep_batch, seed=args.seed)
    print(json.dumps({"frozen_checkpoint": out["frozen_checkpoint"],
                      "episode_level_bootstrap_ci": out["episode_level_bootstrap_ci"]}, indent=2))


if __name__ == "__main__":
    main()
