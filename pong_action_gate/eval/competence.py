"""M2 — DIAMOND Pong teacher competence evaluation.

Competence is measured by GAMEPLAY OUTCOMES only. This module deliberately does
NOT run saliency, opponent ablation, action-support analysis, or any critic
training (those are M3+). It also does NOT impose a competence threshold — it
reports the empirical evidence for human review.

Action selection is unchanged from M1: `Categorical(logits).sample()` (seeded,
stochastic). Each episode uses a fixed, recorded seed so the run is reproducible.

Score convention (validated, not assumed):
  * ALE Pong reward is +1 when the agent scores, -1 when the opponent scores.
  * agent_score = count of +1 rewards; opp_score = count of -1 rewards.
  * final_score_diff = agent_score - opp_score = cumulative reward (episode return).
  * A natural `terminated` Pong episode ends when one side reaches 21, so
    max(agent_score, opp_score) must equal 21 — this is asserted/flagged.
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .. import config as C
from ..teacher.load_teacher import TeacherPolicy, load_teacher, make_env

ART_ROOT = Path("artifacts/pong_action_gate/m2")


def _seed(s: int) -> None:
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)


def _orig_frame(info: Dict[str, Any]) -> np.ndarray:
    arr = np.asarray(info["original_obs"])
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.uint8)


def _write_video(frames: List[np.ndarray], path: Path, fps: int = 15) -> bool:
    if not frames:
        return False
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), isColor=True)
    for fr in frames:
        vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    vw.release()
    return True


def run_one_episode(policy: TeacherPolicy, cfg: C.M1Config, seed: int,
                    collect_frames: bool = False) -> Tuple[Dict[str, Any], Optional[List[np.ndarray]]]:
    _seed(seed)
    env = make_env(replace(cfg, seed=seed), num_envs=1)
    try:
        obs, info = env.reset(seed=[seed])
        hx, cx = policy.initial_state(1)
        agent_score = 0
        opp_score = 0
        cum = 0.0
        max_diff = -21
        first_plus15: Optional[int] = None
        action_hist = [0] * C.N_ACTIONS
        frames = [_orig_frame(info)] if collect_frames else None

        t = 0
        ended_by = "safety_cap"
        while True:
            action, _, _, (hx, cx) = policy.act(obs, (hx, cx))
            a = int(action.item())
            action_hist[a] += 1
            obs, rew, end, trunc, info = env.step(action)
            r = float(rew.item())
            if r > 0:
                agent_score += int(round(r))
            elif r < 0:
                opp_score += int(round(-r))
            cum += r
            diff = agent_score - opp_score
            max_diff = max(max_diff, diff)
            if first_plus15 is None and diff >= C.GOAL_STAR:
                first_plus15 = t
            if collect_frames:
                frames.append(_orig_frame(info))
            t += 1
            if bool((end | trunc).item()):
                ended_by = "terminated" if bool(end.item()) else "truncated"
                break
            if t >= cfg.safety_step_cap:
                break

        rec = {
            "seed": seed,
            "episode_length": t,
            "return": cum,
            "final_score_diff": agent_score - opp_score,
            "agent_score": agent_score,
            "opp_score": opp_score,
            "max_score_diff": max_diff,
            "win": int((agent_score - opp_score) > 0),
            "reached_plus15": int(first_plus15 is not None),
            "first_step_plus15": first_plus15,
            "ended_by": ended_by,
            "action_histogram": {C.ACTION_MEANINGS[i]: action_hist[i] for i in range(C.N_ACTIONS)},
        }
        return rec, frames
    finally:
        env.close()


def _validate_score_convention(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    violations = []
    terminated = 0
    for r in records:
        # return must equal agent-opp (no fractional/lost reward)
        if abs(r["return"] - r["final_score_diff"]) > 1e-9:
            violations.append({"seed": r["seed"], "kind": "return!=score_diff",
                               "return": r["return"], "score_diff": r["final_score_diff"]})
        if r["ended_by"] == "terminated":
            terminated += 1
            if max(r["agent_score"], r["opp_score"]) != 21:
                violations.append({"seed": r["seed"], "kind": "terminated_not_at_21",
                                   "agent": r["agent_score"], "opp": r["opp_score"]})
    return {
        "checked_episodes": len(records),
        "terminated_episodes": terminated,
        "all_terminated_end_at_21": all(
            v["kind"] != "terminated_not_at_21" for v in violations
        ),
        "violations": violations,
        "ok": len(violations) == 0,
    }


def _aggregate(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    def stats(key):
        xs = np.array([r[key] for r in records], dtype=np.float64)
        return {"mean": float(xs.mean()), "std": float(xs.std(ddof=0)),
                "min": float(xs.min()), "max": float(xs.max())}

    reached = [r for r in records if r["reached_plus15"]]
    first_steps = [r["first_step_plus15"] for r in reached]
    agg_hist = {m: 0 for m in C.ACTION_MEANINGS}
    for r in records:
        for m, c in r["action_histogram"].items():
            agg_hist[m] += c
    total_actions = sum(agg_hist.values()) or 1
    ended = {}
    for r in records:
        ended[r["ended_by"]] = ended.get(r["ended_by"], 0) + 1

    return {
        "n_episodes": len(records),
        "seeds": [r["seed"] for r in records],
        "return": stats("return"),
        "final_score_diff": stats("final_score_diff"),
        "episode_length": stats("episode_length"),
        "win_rate": float(np.mean([r["win"] for r in records])),
        "fraction_reached_plus15": float(np.mean([r["reached_plus15"] for r in records])),
        "first_step_plus15_among_reached": {
            "count": len(reached),
            "mean": float(np.mean(first_steps)) if first_steps else None,
            "min": int(np.min(first_steps)) if first_steps else None,
            "max": int(np.max(first_steps)) if first_steps else None,
        },
        "action_histogram_total": agg_hist,
        "action_fraction": {m: agg_hist[m] / total_actions for m in agg_hist},
        "ended_by_counts": ended,
    }


def _plots(records: List[Dict[str, Any]], agg: Dict[str, Any], outdir: Path) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    idx = list(range(len(records)))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(idx, [r["final_score_diff"] for r in records], color="#3a7")
    ax.axhline(C.GOAL_STAR, ls="--", c="k", lw=1, label=f"+{C.GOAL_STAR}")
    ax.axhline(0, c="gray", lw=0.8)
    ax.set(xlabel="episode index", ylabel="final score diff (agent-opp)",
           title="M2 per-episode final score difference")
    ax.legend()
    p = outdir / "final_score_diff.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(idx, [r["episode_length"] for r in records], color="#79c")
    ax.set(xlabel="episode index", ylabel="length (agent steps)", title="M2 episode length")
    p = outdir / "episode_length.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(agg["action_histogram_total"].keys())
    ax.bar(names, [agg["action_fraction"][m] for m in names], color="#c86")
    ax.set(ylabel="fraction of actions", title="M2 aggregate action histogram")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    p = outdir / "action_histogram.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
    paths.append(str(p))
    return paths


def run_competence(n_episodes: int, base_seed: int, tag: str,
                   n_videos: int = 2, device: str = "cpu") -> Dict[str, Any]:
    model, meta = load_teacher(device=device)
    policy = TeacherPolicy(model, device=device)
    cfg = replace(C.M1Config(), device=device)

    outdir = ART_ROOT / tag
    outdir.mkdir(parents=True, exist_ok=True)
    seeds = [base_seed + i for i in range(n_episodes)]

    records: List[Dict[str, Any]] = []
    for i, s in enumerate(seeds):
        collect = i < n_videos
        rec, frames = run_one_episode(policy, cfg, s, collect_frames=collect)
        records.append(rec)
        if collect and frames:
            _write_video(frames, outdir / f"ep_seed{s}.mp4")
        print(f"[{tag}] ep {i+1}/{n_episodes} seed={s} "
              f"len={rec['episode_length']} score={rec['agent_score']}-{rec['opp_score']} "
              f"diff={rec['final_score_diff']} win={rec['win']} "
              f"reach+15@={rec['first_step_plus15']} ended={rec['ended_by']}")

    validation = _validate_score_convention(records)
    agg = _aggregate(records)
    plot_paths = _plots(records, agg, outdir)

    # per-episode JSONL
    with open(outdir / "per_episode.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    # per-episode CSV
    import pandas as pd
    flat = []
    for r in records:
        row = {k: v for k, v in r.items() if k != "action_histogram"}
        for m, c in r["action_histogram"].items():
            row[f"act_{m}"] = c
        flat.append(row)
    pd.DataFrame(flat).to_csv(outdir / "per_episode.csv", index=False)
    # aggregate JSON
    out = {
        "milestone": "M2",
        "tag": tag,
        "base_seed": base_seed,
        "n_episodes": n_episodes,
        "n_videos_saved": min(n_videos, n_episodes),
        "device": device,
        "score_convention_validation": validation,
        "aggregate": agg,
        "plots": plot_paths,
        "teacher_meta": meta,
        "NOTE": "Competence evidence only. No threshold imposed; no saliency/ablation/critic.",
    }
    with open(outdir / "aggregate.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="M2 Pong teacher competence evaluation.")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--tag", type=str, default="pilot")
    ap.add_argument("--videos", type=int, default=2)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    out = run_competence(args.episodes, args.base_seed, args.tag,
                         n_videos=args.videos, device=args.device)
    print(json.dumps({k: out[k] for k in ["score_convention_validation", "aggregate"]}, indent=2))


if __name__ == "__main__":
    main()
