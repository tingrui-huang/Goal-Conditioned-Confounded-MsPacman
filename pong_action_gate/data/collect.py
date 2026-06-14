"""M4 — offline teacher dataset collection at the locked behaviour policy (T=2.0).

Pilot-first: collect 5 episodes, run full validation, and only then extend to 100.
NO class balancing, minority oversampling, goal sampling, critic training, or masking.

Behaviour policy: action ~ Categorical(logits / C.BEHAVIOR_TEMPERATURE).sample().
Recurrent state is reset (zeroed) at every episode start; frames never cross episodes.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .. import config as C
from ..objects import extract_pong_objects
from ..teacher.load_teacher import TeacherPolicy, load_teacher, make_env
from . import schema
from .schema import build_kstack, load_episode, save_episode

ART_ROOT = Path("artifacts/pong_action_gate/m4")
LEARNER_SIZE = 84


def _gray_learner(raw_rgb: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.resize(g, (LEARNER_SIZE, LEARNER_SIZE), interpolation=cv2.INTER_AREA)


def _raw_from_info(info: Dict[str, Any]) -> np.ndarray:
    arr = np.asarray(info["original_obs"])
    if arr.ndim == 4:
        arr = arr[0]
    return arr.astype(np.uint8)


def collect_episode(policy: TeacherPolicy, cfg: C.M1Config, seed: int, T: float,
                    episode_id: int) -> Dict[str, Any]:
    model = policy.model
    env = make_env(replace(cfg, seed=seed), num_envs=1)
    ale = env.env.ale
    obs, info = env.reset(seed=[seed])
    hx, cx = policy.initial_state(1)
    init_hx_norm, init_cx_norm = float(hx.norm()), float(cx.norm())

    cols = {k: [] for k in schema.ARRAY_FIELDS}
    agent_score = opp_score = 0
    t = 0
    while True:
        obs_slice = obs[:, C.TEACHER_OBS_SLICE, :, :]
        hx_in, cx_in = hx, cx
        with torch.no_grad():
            logits, _, (hx, cx) = model.predict_act_value(obs_slice, (hx_in, cx_in))
        scaled = logits / T
        probs_T = F.softmax(scaled, dim=1)[0]
        action = torch.distributions.Categorical(logits=scaled).sample()
        a = int(action.item())

        raw = _raw_from_info(info)                                   # aligned with current obs
        o = extract_pong_objects(ale.getRAM())                      # aligned with current obs
        teacher_u8 = ((obs[0] + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).numpy()

        cols["timestep"].append(t)
        cols["raw_rgb"].append(raw)
        cols["teacher_obs"].append(teacher_u8)
        cols["gray_learner"].append(_gray_learner(raw))
        cols["action"].append(a)
        cols["logits"].append(logits[0].numpy().astype(np.float32))
        cols["probs_T"].append(probs_T.numpy().astype(np.float32))
        cols["behavior_prob"].append(float(probs_T[a]))
        cols["score_diff"].append(agent_score - opp_score)          # obs-time score
        cols["ball_x"].append(np.float32(o.ball_x) if o.ball_present else np.float32(np.nan))
        cols["ball_y"].append(np.float32(o.ball_y) if o.ball_present else np.float32(np.nan))
        cols["player_y"].append(np.float32(o.player_y) if o.player_y is not None else np.float32(np.nan))
        cols["opp_y"].append(np.float32(o.opp_y) if o.opp_y is not None else np.float32(np.nan))
        cols["ball_present"].append(bool(o.ball_present))
        cols["player_valid"].append(o.player_y is not None)
        cols["opp_valid"].append(o.opp_y is not None)

        obs, rew, end, trunc, info = env.step(action)
        r = float(rew.item())
        if r > 0:
            agent_score += 1
        elif r < 0:
            opp_score += 1
        cols["reward"].append(np.float32(r))
        cols["terminated"].append(bool(end.item()))
        cols["truncated"].append(bool(trunc.item()))
        cols["is_scoring_event"].append(r != 0)
        cols["agent_score"].append(agent_score)
        cols["opp_score"].append(opp_score)
        t += 1
        if bool((end | trunc).item()) or t >= cfg.safety_step_cap:
            break
    env.close()

    ep = {
        "episode_id": np.int64(episode_id),
        "env_seed": np.int64(seed),
        "temperature": np.float32(T),
        "initial_hx_norm": np.float32(init_hx_norm),
        "initial_cx_norm": np.float32(init_cx_norm),
        "timestep": np.array(cols["timestep"], np.int32),
        "raw_rgb": np.array(cols["raw_rgb"], np.uint8),
        "teacher_obs": np.array(cols["teacher_obs"], np.uint8),
        "gray_learner": np.array(cols["gray_learner"], np.uint8),
        "action": np.array(cols["action"], np.int64),
        "logits": np.array(cols["logits"], np.float32),
        "probs_T": np.array(cols["probs_T"], np.float32),
        "behavior_prob": np.array(cols["behavior_prob"], np.float32),
        "reward": np.array(cols["reward"], np.float32),
        "terminated": np.array(cols["terminated"], bool),
        "truncated": np.array(cols["truncated"], bool),
        "score_diff": np.array(cols["score_diff"], np.int32),
        "agent_score": np.array(cols["agent_score"], np.int32),
        "opp_score": np.array(cols["opp_score"], np.int32),
        "is_scoring_event": np.array(cols["is_scoring_event"], bool),
        "ball_x": np.array(cols["ball_x"], np.float32),
        "ball_y": np.array(cols["ball_y"], np.float32),
        "player_y": np.array(cols["player_y"], np.float32),
        "opp_y": np.array(cols["opp_y"], np.float32),
        "ball_present": np.array(cols["ball_present"], bool),
        "player_valid": np.array(cols["player_valid"], bool),
        "opp_valid": np.array(cols["opp_valid"], bool),
    }
    return ep


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def validate_episode(path: Path) -> Dict[str, Any]:
    ep = load_episode(path)
    issues: List[str] = []
    T = len(ep["action"])

    # uniform lengths
    for k in schema.ARRAY_FIELDS:
        if len(ep[k]) != T:
            issues.append(f"length mismatch {k}: {len(ep[k])} != {T}")

    # action validity
    if not ((ep["action"] >= 0) & (ep["action"] < C.N_ACTIONS)).all():
        issues.append("invalid action id outside [0,N_ACTIONS)")

    # scoring-event detection
    if not np.array_equal(ep["is_scoring_event"], ep["reward"] != 0):
        issues.append("is_scoring_event != (reward!=0)")

    # exact score convention
    cum = float(ep["reward"].sum())
    final_diff = int(ep["agent_score"][-1] - ep["opp_score"][-1])
    if abs(cum - final_diff) > 1e-6:
        issues.append(f"cumulative reward {cum} != final score diff {final_diff}")
    rec_agent = int((ep["reward"] > 0).sum()); rec_opp = int((ep["reward"] < 0).sum())
    if rec_agent != int(ep["agent_score"][-1]) or rec_opp != int(ep["opp_score"][-1]):
        issues.append("reward-sign counts != stored agent/opp score")
    if bool(ep["terminated"][-1]) and max(int(ep["agent_score"][-1]), int(ep["opp_score"][-1])) != 21:
        issues.append("terminated episode does not end at 21")

    # behavior prob consistency + finiteness
    if not np.isfinite(ep["logits"]).all() or not np.isfinite(ep["probs_T"]).all():
        issues.append("non-finite logits/probs")
    if not np.allclose(ep["probs_T"].sum(1), 1.0, atol=1e-4):
        issues.append("probs_T rows do not sum to 1")
    bp = ep["probs_T"][np.arange(T), ep["action"]]
    if not np.allclose(bp, ep["behavior_prob"], atol=1e-5):
        issues.append("behavior_prob != probs_T[action]")

    # object validity flags consistent with nan coords
    if not np.array_equal(ep["ball_present"], np.isfinite(ep["ball_x"])):
        issues.append("ball_present inconsistent with ball_x nan")
    if not np.array_equal(ep["opp_valid"], np.isfinite(ep["opp_y"])):
        issues.append("opp_valid inconsistent with opp_y nan")

    # recurrent reset
    if float(ep["initial_hx_norm"]) != 0.0 or float(ep["initial_cx_norm"]) != 0.0:
        issues.append("recurrent state not zeroed at episode start")

    # raw <-> object alignment (sampled): ball pixel is bright
    rng = np.random.default_rng(0)
    present = np.where(ep["ball_present"])[0]
    checked = hit = 0
    for i in rng.choice(present, size=min(20, len(present)), replace=False) if len(present) else []:
        raw = ep["raw_rgb"][i]; bx = int(ep["ball_x"][i]); by = int(ep["ball_y"][i])
        if 0 <= by < 210 and 0 <= bx < 160:
            checked += 1
            if raw[max(0, by-4):by+5, max(0, bx-4):bx+5].max() > 200:
                hit += 1
    align_ok = (checked == 0) or (hit / checked >= 0.95)
    if not align_ok:
        issues.append(f"raw/object misalignment: {hit}/{checked} ball pixels bright")

    # k=4 stack: order + no cross-boundary (left-pad at start)
    s0 = build_kstack(ep["gray_learner"], 0, 4)
    if not all(np.array_equal(s0[j], ep["gray_learner"][0]) for j in range(4)):
        issues.append("k-stack at idx0 not left-padded with frame 0")
    if T > 5:
        s = build_kstack(ep["gray_learner"], 5, 4)
        if not all(np.array_equal(s[j], ep["gray_learner"][5 - 3 + j]) for j in range(4)):
            issues.append("k-stack order is not oldest->newest")

    # serialization round-trip (NaN-aware: absent-object coords are stored as NaN)
    def _arr_eq(a, b):
        if np.issubdtype(np.asarray(a).dtype, np.floating):
            return np.array_equal(a, b, equal_nan=True)
        return np.array_equal(a, b)
    ep2 = load_episode(path)
    if not all(_arr_eq(ep[k], ep2[k]) for k in schema.ARRAY_FIELDS):
        issues.append("serialization round-trip mismatch")

    return {"path": str(path), "transitions": T, "issues": issues, "ok": len(issues) == 0,
            "ball_align_checked": checked, "ball_align_hit": hit}


# --------------------------------------------------------------------------- #
# Collection + stats
# --------------------------------------------------------------------------- #
def collect_dataset(seeds: List[int], T: float, tag: str, device: str = "cpu") -> Dict[str, Any]:
    model, meta = load_teacher(device=device)
    policy = TeacherPolicy(model, device=device)
    cfg = replace(C.M1Config(), device=device)
    outdir = ART_ROOT / tag
    (outdir / "episodes").mkdir(parents=True, exist_ok=True)

    manifest_eps = []
    for i, s in enumerate(seeds):
        ep = collect_episode(policy, cfg, s, T, episode_id=i)
        path = outdir / "episodes" / f"ep_{i:04d}_seed{s}.npz"
        digest = save_episode(path, ep)
        manifest_eps.append({"episode_id": i, "seed": s, "path": path.name,
                             "transitions": int(len(ep["action"])),
                             "agent_score": int(ep["agent_score"][-1]),
                             "opp_score": int(ep["opp_score"][-1]),
                             "sha256": digest,
                             "bytes": int(path.stat().st_size)})
        print(f"[{tag}] ep {i+1}/{len(seeds)} seed={s} T={len(ep['action'])} "
              f"score={int(ep['agent_score'][-1])}-{int(ep['opp_score'][-1])} "
              f"{manifest_eps[-1]['bytes']//1024} KB")

    manifest = {"milestone": "M4", "tag": tag, "temperature": T, "seeds": seeds,
                "n_episodes": len(seeds), "episodes": manifest_eps,
                "teacher_meta": {k: meta[k] for k in ["ckpt_sha256", "teacher_source"]}}
    with open(outdir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def validate_and_stats(tag: str) -> Dict[str, Any]:
    outdir = ART_ROOT / tag
    paths = sorted((outdir / "episodes").glob("*.npz"))
    val = [validate_episode(p) for p in paths]
    all_ok = all(v["ok"] for v in val)

    # aggregate stats
    tot_T = sum(v["transitions"] for v in val)
    agent_pts = opp_pts = 0
    action_hist = np.zeros(C.N_ACTIONS, dtype=np.int64)
    bp_all = []
    sd_min, sd_max = 10**9, -10**9
    ball_present = ball_total = opp_ok = player_ok = 0
    boundary_violations = 0
    for p in paths:
        ep = load_episode(p)
        agent_pts += int((ep["reward"] > 0).sum())
        opp_pts += int((ep["reward"] < 0).sum())
        for a in ep["action"]:
            action_hist[a] += 1
        bp_all.append(ep["behavior_prob"])
        sd_min = min(sd_min, int(ep["score_diff"].min()))
        sd_max = max(sd_max, int(ep["score_diff"].max()))
        ball_present += int(ep["ball_present"].sum()); ball_total += len(ep["ball_present"])
        opp_ok += int(ep["opp_valid"].sum()); player_ok += int(ep["player_valid"].sum())
        # boundary check: a k-stack at idx 0..k-2 must not pull index < 0
        for idx in range(min(4, len(ep["gray_learner"]))):
            cols = [max(0, idx - 3 + j) for j in range(4)]
            if min(cols) < 0:
                boundary_violations += 1
    bp_all = np.concatenate(bp_all) if bp_all else np.array([])

    stats = {
        "episodes": len(paths),
        "total_transitions": tot_T,
        "agent_scoring_events": agent_pts,
        "opponent_scoring_events": opp_pts,
        "opponent_event_fraction": float(opp_pts / max(agent_pts + opp_pts, 1)),
        "score_diff_range": [sd_min, sd_max],
        "action_distribution": {C.ACTION_MEANINGS[i]: int(action_hist[i]) for i in range(C.N_ACTIONS)},
        "action_fraction": {C.ACTION_MEANINGS[i]: float(action_hist[i] / max(tot_T, 1)) for i in range(C.N_ACTIONS)},
        "behavior_prob_quantiles": {
            q: float(np.quantile(bp_all, v)) for q, v in
            {"p05": .05, "p25": .25, "p50": .5, "p75": .75, "p95": .95}.items()} if bp_all.size else {},
        "behavior_prob_mean": float(bp_all.mean()) if bp_all.size else None,
        "object_extraction": {
            "ball_present_rate": float(ball_present / max(ball_total, 1)),
            "ball_absent_rate": float(1 - ball_present / max(ball_total, 1)),
            "opp_valid_rate": float(opp_ok / max(ball_total, 1)),
            "player_valid_rate": float(player_ok / max(ball_total, 1)),
        },
        "episode_boundary_violations": boundary_violations,
        "validation": {"all_ok": all_ok,
                       "n_failed": sum(0 if v["ok"] else 1 for v in val),
                       "failures": [v for v in val if not v["ok"]]},
    }
    with open(outdir / "validation_stats.json", "w") as f:
        json.dump({"per_episode_validation": val, "stats": stats}, f, indent=2)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="M4 dataset collection at T=2.0.")
    ap.add_argument("--base-seed", type=int, default=6000)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--tag", type=str, default="pilot")
    ap.add_argument("--temp", type=float, default=C.BEHAVIOR_TEMPERATURE)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()
    seeds = [args.base_seed + i for i in range(args.episodes)]
    if not args.validate_only:
        collect_dataset(seeds, args.temp, args.tag, device=args.device)
    stats = validate_and_stats(args.tag)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
