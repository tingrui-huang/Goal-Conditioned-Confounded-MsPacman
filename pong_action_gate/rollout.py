"""M1 — seeded reproducible *stochastic* recurrent rollout of the DIAMOND Pong teacher.

This validates teacher integration ONLY. It makes no claim about competence (M2)
or action support (M3). Actions are sampled with `Categorical(logits).sample()`
(NOT argmax); reproducibility comes from seeding torch's global RNG.

Recurrent-state reset semantics (inspected, not assumed):
  * Pong has no ALE lives (ale.lives()==0), so `info['life_loss']` is always False;
    in the TEST env (done_on_life_loss=False) life loss never ends an episode anyway.
  * An episode ends only on `terminated` (game over) or `truncated` (max_episode_steps).
  * `dead = terminated | truncated`. At a boundary we reproduce env_loop.py's gate
    exactly: hx, cx *= (1 - dead)  -> hidden state zeroed for the new episode.
  * Under gymnasium 1.3 the vector env does NOT emit `info['final_observation']` and
    auto-resets on the *next* step (verified empirically). We therefore run one
    episode and stop at the first `dead`, then separately demonstrate a full
    `env.reset()` producing a fresh zeroed hidden state. No burn-in obs is emitted
    by this env (logged).
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from . import config as C
from .teacher.load_teacher import (
    TeacherPolicy,
    load_teacher,
    make_env,
    save_resolved_config,
)

ART_DIR = Path("artifacts/pong_action_gate/m1")


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _orig_frame(info: Dict[str, Any]) -> np.ndarray:
    """Extract the single-env hi-res (210,160,3) uint8 RGB frame from vector info."""
    fo = info["original_obs"]
    arr = np.asarray(fo)
    if arr.ndim == 4:        # (num_envs, H, W, 3)
        arr = arr[0]
    return arr.astype(np.uint8)


def _write_video(frames: List[np.ndarray], path: Path, fps: int = 15) -> bool:
    if not frames:
        return False
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), isColor=True)
    for fr in frames:
        writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    writer.release()
    return True


def run_rollout(cfg: C.M1Config) -> Dict[str, Any]:
    _seed_everything(cfg.seed)

    model, meta = load_teacher(arch=cfg.arch, device=cfg.device)
    policy = TeacherPolicy(model, device=cfg.device)
    env = make_env(cfg, num_envs=1)

    ART_DIR.mkdir(parents=True, exist_ok=True)
    save_resolved_config(ART_DIR / "resolved_config.json", cfg, meta)

    obs, info = env.reset(seed=[cfg.seed])
    assert tuple(obs.shape) == (1, C.N_OBS_CHANNELS, C.IMG_SIZE, C.IMG_SIZE), obs.shape
    assert obs.dtype == torch.float32
    obs_min, obs_max = float(obs.min()), float(obs.max())

    hx, cx = policy.initial_state(1)
    hx0_norm, cx0_norm = float(hx.norm()), float(cx.norm())   # must be 0 at start

    frames: List[np.ndarray] = [_orig_frame(info)]
    steps: List[Dict[str, Any]] = []
    action_hist = [0] * C.N_ACTIONS
    cum_score_diff = 0.0
    max_score_diff = 0.0
    reset_applied_at = None
    burnin_seen = False

    t = 0
    while True:
        action, logits, value, (hx, cx) = policy.act(obs, (hx, cx))
        a = int(action.item())
        action_hist[a] += 1

        obs, rew, end, trunc, info = env.step(action)
        burnin_seen = burnin_seen or ("burnin_obs" in info)
        r = float(rew.item())
        cum_score_diff += r
        max_score_diff = max(max_score_diff, cum_score_diff)
        term_i, trunc_i = int(end.item()), int(trunc.item())
        life_loss = info.get("life_loss")
        life_loss_i = int(bool(np.asarray(life_loss).any())) if life_loss is not None else 0
        lives = info.get("lives")
        lives_i = int(np.asarray(lives).reshape(-1)[0]) if lives is not None else -1
        dead = bool((end | trunc).item())

        step_rec = {
            "t": t,
            "action": a,
            "action_meaning": C.ACTION_MEANINGS[a],
            "reward": r,
            "cum_score_diff": cum_score_diff,
            "terminated": term_i,
            "truncated": trunc_i,
            "life_loss": life_loss_i,
            "lives": lives_i,
            "dead": int(dead),
            "value": float(value.item()),
            "hx_norm": float(hx.norm()),
            "cx_norm": float(cx.norm()),
            "hx_cx_reset": 0,
        }

        frames.append(_orig_frame(info))

        if dead:
            # Reproduce env_loop.py reset gate exactly.
            dead_t = (end | trunc).float().unsqueeze(1)        # (1,1)
            hx = hx * (1.0 - dead_t)
            cx = cx * (1.0 - dead_t)
            step_rec["hx_cx_reset"] = 1
            step_rec["hx_norm_after_reset"] = float(hx.norm())
            step_rec["cx_norm_after_reset"] = float(cx.norm())
            reset_applied_at = t
            steps.append(step_rec)
            break

        steps.append(step_rec)
        t += 1
        if t >= cfg.safety_step_cap:
            step_rec["safety_cap_hit"] = 1
            steps.append(step_rec)
            break

    # --- demonstrate FULL episode reset: fresh env + fresh zeroed hidden state ---
    obs2, _ = env.reset(seed=[cfg.seed + 1])
    hx2, cx2 = policy.initial_state(1)
    a2, _, _, _ = policy.act(obs2, (hx2, cx2))
    full_reset_demo = {
        "new_seed": cfg.seed + 1,
        "fresh_hx_norm": float(hx2.norm()),
        "fresh_cx_norm": float(cx2.norm()),
        "first_action": int(a2.item()),
        "first_action_meaning": C.ACTION_MEANINGS[int(a2.item())],
    }
    env.close()

    # carry-vs-reset evidence: hidden-state norm strictly grew across the episode
    hx_norms = [s["hx_norm"] for s in steps]
    hidden_evolved = len(set(round(x, 4) for x in hx_norms[: min(len(hx_norms), 20)])) > 1

    summary = {
        "milestone": "M1",
        "seed": cfg.seed,
        "device": cfg.device,
        "episode_length": len(steps),
        "ended_by": ("truncated" if steps[-1]["truncated"] else
                     "terminated" if steps[-1]["terminated"] else
                     "safety_cap"),
        "reset_applied_at_step": reset_applied_at,
        "final_score_diff": cum_score_diff,
        "max_score_diff": max_score_diff,
        "total_reward": cum_score_diff,
        "action_histogram": {C.ACTION_MEANINGS[i]: action_hist[i] for i in range(C.N_ACTIONS)},
        "n_distinct_actions_used": sum(1 for c in action_hist if c > 0),
        "obs_shape": list(obs.shape),
        "obs_range": [obs_min, obs_max],
        "initial_hidden_norm": [hx0_norm, cx0_norm],
        "hidden_state_evolved_within_episode": bool(hidden_evolved),
        "hidden_reset_to_zero_at_boundary": (
            float(steps[-1].get("hx_norm_after_reset", -1.0)) == 0.0
            and float(steps[-1].get("cx_norm_after_reset", -1.0)) == 0.0
        ),
        "burnin_obs_emitted": bool(burnin_seen),
        "full_episode_reset_demo": full_reset_demo,
        "teacher_meta": meta,
        "NOTE": "Integration check only. No competence (M2) or action-support (M3) claim.",
    }

    wrote_video = _write_video(frames, ART_DIR / "rollout.mp4")
    summary["video_written"] = wrote_video
    summary["video_path"] = str(ART_DIR / "rollout.mp4") if wrote_video else None
    summary["n_frames"] = len(frames)

    with open(ART_DIR / "episode_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(ART_DIR / "episode_steps.jsonl", "w") as f:
        for s in steps:
            f.write(json.dumps(s) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="M1 seeded stochastic recurrent Pong teacher rollout.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--max-episode-steps", type=int, default=0,
                    help="0 -> natural termination; >0 -> truncate (for a quick smoke run).")
    args = ap.parse_args()

    cfg = replace(C.M1Config(), seed=args.seed, device=args.device,
                  max_episode_steps=args.max_episode_steps)
    summary = run_rollout(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
