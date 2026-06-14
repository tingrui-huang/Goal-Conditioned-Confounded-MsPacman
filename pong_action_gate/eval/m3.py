"""M3 — opponent dependence (A), conditional action support (B), outcome/goal support (C).

Diagnostics only. NO critic training, NO automatic exploration injection. Action
sampling for the behaviour we analyse is unchanged: `Categorical(logits).sample()`.

Object positions (ball, own/opponent paddle, scores) are read from the SAME ALE the
teacher steps (validated: 100% ball-pixel agreement, RAM scores == reward count).

Outputs land in artifacts/pong_action_gate/m3/<tag>/.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .. import config as C
from ..objects import extract_pong_objects, opp_paddle_box64, region_boxes64, PongObjects
from ..teacher.load_teacher import TeacherPolicy, load_teacher, make_env

ART_ROOT = Path("artifacts/pong_action_gate/m3")

# functional movement classes (Pong actions are behaviourally aliased)
#   STAY: NOOP, FIRE   |   UP: RIGHT, RIGHTFIRE   |   DOWN: LEFT, LEFTFIRE
MOVE_CLASS = {0: "STAY", 1: "STAY", 2: "UP", 4: "UP", 3: "DOWN", 5: "DOWN"}
MOVE_NAMES = ["STAY", "UP", "DOWN"]
SERVE_ACTIONS = {1, 4, 5}   # FIRE bit set


def _entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-(p * np.log2(p)).sum())


# --------------------------------------------------------------------------- #
# Recorder
# --------------------------------------------------------------------------- #
def record(policy: TeacherPolicy, cfg: C.M1Config, seeds: List[int],
           snap_every: int = 10, max_snaps: int = 1400) -> Tuple[List[Dict], List[Dict]]:
    """Run episodes; return (steps, snapshots).

    steps: light per-step records (for M3B/M3C).
    snapshots: a sampled subset with obs+hidden retained (for M3A ablation/saliency).
    """
    steps: List[Dict[str, Any]] = []
    snaps: List[Dict[str, Any]] = []
    for s in seeds:
        torch.manual_seed(s)
        env = make_env(replace(cfg, seed=s), num_envs=1)
        ale = env.env.ale
        obs, info = env.reset(seed=[s])
        hx, cx = policy.initial_state(1)
        prev = extract_pong_objects(ale.getRAM())
        agent_score = 0
        opp_score = 0
        t = 0
        while True:
            obs_slice = obs[:, C.TEACHER_OBS_SLICE, :, :]
            hx_in, cx_in = hx, cx                      # hidden state paired with this obs
            with torch.no_grad():
                logits, _, (hx, cx) = policy.model.predict_act_value(obs_slice, (hx_in, cx_in))
            probs = F.softmax(logits, dim=1)[0].numpy()
            action = torch.distributions.Categorical(logits=logits).sample()
            a = int(action.item())

            o = extract_pong_objects(ale.getRAM())     # RAM aligned with current obs
            ball_dx = (o.ball_x - prev.ball_x) if (o.ball_present and prev.ball_present) else None
            ball_dy = (o.ball_y - prev.ball_y) if (o.ball_present and prev.ball_present) else None
            rec = {
                "seed": s, "t": t, "action": a, "move_class": MOVE_CLASS[a],
                "logits": logits[0].tolist(),
                "ball_present": o.ball_present,
                "ball_x": o.ball_x, "ball_y": o.ball_y,
                "ball_dx": ball_dx, "ball_dy": ball_dy,
                "player_y": o.player_y, "opp_y": o.opp_y,
                "rel_ball_paddle": (o.ball_y - o.player_cy()) if (o.ball_present and o.player_y is not None) else None,
                "phase": "rally" if o.ball_present else "serve",
                "score_diff": agent_score - opp_score,
            }

            if (t % snap_every == 0) and (len(snaps) < max_snaps):
                snaps.append({
                    "obs_rgb": obs_slice.clone(),
                    "hx": hx_in.clone(), "cx": cx_in.clone(),
                    "objects": o, "logits": logits.clone(), "probs": probs,
                    "ball_dx": ball_dx, "phase": rec["phase"], "seed": s, "t": t,
                })

            obs, rew, end, trunc, info = env.step(action)
            r = float(rew.item())
            if r > 0:
                agent_score += 1
            elif r < 0:
                opp_score += 1
            rec["reward"] = r
            rec["is_scoring_event"] = int(r != 0)
            rec["agent_score"] = agent_score
            rec["opp_score"] = opp_score
            steps.append(rec)
            prev = o
            t += 1
            if bool((end | trunc).item()) or t >= cfg.safety_step_cap:
                break
        env.close()
    return steps, snaps


# --------------------------------------------------------------------------- #
# M3A — opponent dependence
# --------------------------------------------------------------------------- #
def _inpaint_opp(obs_rgb: torch.Tensor, obj: PongObjects) -> Optional[torch.Tensor]:
    box = opp_paddle_box64(obj, margin=1)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    out = obs_rgb.clone()
    # background fill: column just to the RIGHT of the paddle (playfield background)
    bx = min(x1 + 1, C.IMG_SIZE - 1)
    bg = out[:, :, y0:y1, bx:bx + 1].mean(dim=(2, 3), keepdim=True)
    out[:, :, y0:y1, x0:x1] = bg
    return out


def m3a_ablation(policy: TeacherPolicy, snaps: List[Dict]) -> Dict[str, Any]:
    model = policy.model
    rows = []
    for sn in snaps:
        if sn["objects"].opp_y is None:
            continue
        abl = _inpaint_opp(sn["obs_rgb"], sn["objects"])
        if abl is None:
            continue
        with torch.no_grad():
            l0, _, _ = model.predict_act_value(sn["obs_rgb"], (sn["hx"], sn["cx"]))
            l1, _, _ = model.predict_act_value(abl, (sn["hx"], sn["cx"]))
        p0 = F.softmax(l0, 1); p1 = F.softmax(l1, 1)
        kl = float((p0 * (p0.add(1e-9).log() - p1.add(1e-9).log())).sum())
        rows.append({
            "argmax_changed": int(l0.argmax().item() != l1.argmax().item()),
            "kl": kl,
            "mean_abs_dlogit": float((l0 - l1).abs().mean()),
            "ball_toward_agent": None if sn["ball_dx"] is None else int(sn["ball_dx"] > 0),
            "phase": sn["phase"],
        })
    if not rows:
        return {"n": 0}

    def agg(sub):
        if not sub:
            return None
        return {
            "n": len(sub),
            "argmax_disagree_rate": float(np.mean([r["argmax_changed"] for r in sub])),
            "mean_kl": float(np.mean([r["kl"] for r in sub])),
            "mean_abs_dlogit": float(np.mean([r["mean_abs_dlogit"] for r in sub])),
        }

    return {
        "n": len(rows),
        "overall": agg(rows),
        "by_ball_direction": {
            "toward_agent": agg([r for r in rows if r["ball_toward_agent"] == 1]),
            "toward_opponent": agg([r for r in rows if r["ball_toward_agent"] == 0]),
        },
        "by_phase": {
            "rally": agg([r for r in rows if r["phase"] == "rally"]),
            "serve": agg([r for r in rows if r["phase"] == "serve"]),
        },
    }


def _gauss_mask(cx: int, cy: int, r: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    m = np.zeros((C.IMG_SIZE, C.IMG_SIZE), dtype=np.float32)
    m[cy, cx] = 1.0
    m = gaussian_filter(m, sigma=r)
    return m / (m.max() + 1e-8)


def m3a_saliency(policy: TeacherPolicy, snaps: List[Dict], n_frames: int = 24,
                 d: int = 4, r: float = 3.0, save_dir: Optional[Path] = None,
                 n_save: int = 6) -> Dict[str, Any]:
    from scipy.ndimage import gaussian_filter
    model = policy.model
    idxs = np.linspace(0, len(snaps) - 1, min(n_frames, len(snaps))).astype(int)
    region_sal = {k: [] for k in ["ball", "own_paddle", "opp_paddle", "scoreboard", "background"]}
    to_render = []
    for rank, ix in enumerate(idxs):
        sn = snaps[int(ix)]
        obs = sn["obs_rgb"]
        blurred = torch.tensor(
            gaussian_filter(obs.numpy(), sigma=[0, 0, r, r]), dtype=obs.dtype)
        with torch.no_grad():
            L, _, _ = model.predict_act_value(obs, (sn["hx"], sn["cx"]))
        sal = np.zeros((C.IMG_SIZE, C.IMG_SIZE), dtype=np.float32)
        for i in range(0, C.IMG_SIZE, d):
            for j in range(0, C.IMG_SIZE, d):
                mask = torch.tensor(_gauss_mask(j, i, r))
                pert = obs * (1 - mask) + blurred * mask
                with torch.no_grad():
                    l, _, _ = model.predict_act_value(pert, (sn["hx"], sn["cx"]))
                sal[i:i + d, j:j + d] = 0.5 * float(((L - l) ** 2).sum())
        boxes = region_boxes64(sn["objects"])
        used = np.zeros((C.IMG_SIZE, C.IMG_SIZE), dtype=bool)
        for name in ["ball", "own_paddle", "opp_paddle", "scoreboard"]:
            b = boxes.get(name)
            if b is None:
                continue
            x0, y0, x1, y1 = b
            if x1 > x0 and y1 > y0:
                region_sal[name].append(float(sal[y0:y1, x0:x1].mean()))
                used[y0:y1, x0:x1] = True
        bg = sal[~used]
        if bg.size:
            region_sal["background"].append(float(bg.mean()))
        if save_dir is not None and rank < n_save:
            to_render.append((sn, sal.copy(), boxes))

    saved = []
    if save_dir is not None and to_render:
        saved = _render_saliency(to_render, save_dir)
    return {
        "n_frames": int(len(idxs)), "grid_d": d, "blur_r": r,
        "mean_saliency_by_region": {k: (float(np.mean(v)) if v else None) for k, v in region_sal.items()},
        "heatmaps": saved,
        "note": "Saliency = attended input regions, NOT a competence metric.",
    }


def _render_saliency(items, save_dir: Path) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    save_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    cols = ["ball", "own_paddle", "opp_paddle", "scoreboard"]
    colors = {"ball": "cyan", "own_paddle": "lime", "opp_paddle": "red", "scoreboard": "yellow"}
    for k, (sn, sal, boxes) in enumerate(items):
        rgb = ((sn["obs_rgb"][0].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.4))
        axes[0].imshow(rgb); axes[0].set_title("teacher input (64x64 RGB)"); axes[0].axis("off")
        axes[1].imshow(rgb)
        axes[1].imshow(sal, cmap="jet", alpha=0.55, interpolation="bilinear")
        for name in cols:
            b = boxes.get(name)
            if b:
                x0, y0, x1, y1 = b
                axes[1].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                            edgecolor=colors[name], lw=1.4))
        axes[1].set_title("actor-logit saliency"); axes[1].axis("off")
        p = save_dir / f"saliency_{k:02d}_seed{sn['seed']}_t{sn['t']}_{sn['phase']}.png"
        fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig)
        paths.append(str(p))
    return paths


# --------------------------------------------------------------------------- #
# M3B — conditional action support
# --------------------------------------------------------------------------- #
def _bin_stats(records: List[Dict]) -> Dict[str, Any]:
    n = len(records)
    a6 = np.zeros(C.N_ACTIONS)
    fc = {m: 0 for m in MOVE_NAMES}
    for r in records:
        a6[r["action"]] += 1
        fc[r["move_class"]] += 1
    p6 = a6 / max(n, 1)
    pf = np.array([fc[m] for m in MOVE_NAMES]) / max(n, 1)
    return {
        "count": n,
        "action6_dist": {C.ACTION_MEANINGS[i]: float(p6[i]) for i in range(C.N_ACTIONS)},
        "action6_entropy_bits": _entropy(p6),
        "supported_actions6": int((p6 >= 0.05).sum()),
        "dominant_action6_rate": float(p6.max()),
        "func_class_dist": {MOVE_NAMES[i]: float(pf[i]) for i in range(3)},
        "func_entropy_bits": _entropy(pf),
        "supported_func_classes": int((pf >= 0.05).sum()),
        "dominant_func_rate": float(pf.max()),
    }


def _binned(records: List[Dict], keyfn, names) -> Dict[str, Any]:
    groups: Dict[Any, List[Dict]] = {}
    for r in records:
        k = keyfn(r)
        if k is None:
            continue
        groups.setdefault(k, []).append(r)
    return {names.get(k, str(k)): _bin_stats(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def m3b_support(steps: List[Dict]) -> Dict[str, Any]:
    rally = [r for r in steps if r["phase"] == "rally"]
    serve = [r for r in steps if r["phase"] == "serve"]

    def qbin(v, edges):
        if v is None:
            return None
        for i, e in enumerate(edges):
            if v < e:
                return i
        return len(edges)

    bx_edges = [40, 75, 110]      # ball_x native
    by_edges = [70, 105, 140]     # ball_y native
    py_edges = [70, 105, 140]     # paddle_y native
    rel_edges = [-12, 12]         # ball above / aligned / below own paddle center

    out = {
        "global": _bin_stats(steps),
        "by_phase": {"rally": _bin_stats(rally) if rally else None,
                     "serve": _bin_stats(serve) if serve else None},
        "rally_by_ball_x": _binned(rally, lambda r: qbin(r["ball_x"], bx_edges), {0: "x<40", 1: "40-75", 2: "75-110", 3: "x>110"}),
        "rally_by_ball_y": _binned(rally, lambda r: qbin(r["ball_y"], by_edges), {0: "y<70", 1: "70-105", 2: "105-140", 3: "y>140"}),
        "rally_by_ball_dir_x": _binned(rally, lambda r: None if r["ball_dx"] is None else int(r["ball_dx"] > 0), {1: "toward_agent", 0: "toward_opp"}),
        "rally_by_ball_dir_y": _binned(rally, lambda r: None if r["ball_dy"] is None else int(r["ball_dy"] > 0), {1: "downward", 0: "upward"}),
        "rally_by_own_paddle_y": _binned(rally, lambda r: qbin(r["player_y"], py_edges), {0: "top", 1: "upper-mid", 2: "lower-mid", 3: "bottom"}),
        "rally_by_opp_paddle_y": _binned(rally, lambda r: qbin(r["opp_y"], py_edges), {0: "top", 1: "upper-mid", 2: "lower-mid", 3: "bottom"}),
        "rally_by_rel_ball_paddle": _binned(rally, lambda r: qbin(r["rel_ball_paddle"], rel_edges), {0: "ball_above", 1: "aligned", 2: "ball_below"}),
    }
    return out


# --------------------------------------------------------------------------- #
# M3C — outcome / goal support
# --------------------------------------------------------------------------- #
def m3c_outcome(steps: List[Dict], batch_size: int = 256) -> Dict[str, Any]:
    # group by episode
    by_ep: Dict[int, List[Dict]] = {}
    for r in steps:
        by_ep.setdefault(r["seed"], []).append(r)

    agent_pts = sum(1 for r in steps if r["reward"] > 0)
    opp_pts = sum(1 for r in steps if r["reward"] < 0)

    # next-scoring-outcome per transition + next_score_event goal value
    next_outcomes = []          # +1 agent / -1 opp
    d_next_vals = []            # achieved score diff right after next scoring event
    next_eq_curr_plus1 = []     # bool per transition
    score_diffs = [r["score_diff"] for r in steps]
    neg_diff_events = sum(1 for d in score_diffs if d < 0)

    neg_or_dec = 0
    for ep, rs in by_ep.items():
        # decreasing score-diff WITHIN an episode (do not cross the 21->0 reset)
        prev_d = None
        for r in rs:
            if prev_d is not None and r["score_diff"] < prev_d:
                neg_or_dec += 1
            prev_d = r["score_diff"]
        # next scoring event resolution
        ev_idx = [i for i, r in enumerate(rs) if r["is_scoring_event"]]
        for i, r in enumerate(rs):
            nxt = next((j for j in ev_idx if j >= i), None)
            if nxt is None:
                continue
            outcome = 1 if rs[nxt]["reward"] > 0 else -1
            next_outcomes.append(outcome)
            d_after = rs[nxt]["agent_score"] - rs[nxt]["opp_score"]   # diff AFTER the event
            d_next_vals.append(d_after)
            next_eq_curr_plus1.append(int(d_after == r["score_diff"] + 1))

    # conditional entropy of next scoring outcome (here unconditional; collapsed if all agent)
    no = np.array(next_outcomes)
    p_agent = float((no > 0).mean()) if len(no) else None
    outcome_entropy = _entropy(np.array([p_agent, 1 - p_agent])) if p_agent is not None else None

    # NCE duplicate goal rate under next_score_event: goal value = d_next
    vals, counts = np.unique(np.array(d_next_vals), return_counts=True) if d_next_vals else (np.array([]), np.array([]))
    pmf = counts / counts.sum() if counts.size else np.array([])
    dup_prob = float((pmf ** 2).sum()) if pmf.size else None   # P(two independent goals equal)

    return {
        "agent_points_total": agent_pts,
        "opponent_points_total": opp_pts,
        "episodes": len(by_ep),
        "score_diff_range": [int(min(score_diffs)), int(max(score_diffs))] if score_diffs else None,
        "negative_score_diff_events": neg_diff_events,
        "decreasing_score_diff_events": neg_or_dec,
        "fraction_decreasing": float(neg_or_dec / max(len(score_diffs), 1)),
        "next_scoring_outcome": {
            "p_agent": p_agent,
            "conditional_entropy_bits": outcome_entropy,
            "distinct_outcomes": int(len(set(next_outcomes))),
        },
        "next_score_event_eq_current_plus1_fraction": float(np.mean(next_eq_curr_plus1)) if next_eq_curr_plus1 else None,
        "next_event_goal_value_marginal": {int(v): int(c) for v, c in zip(vals.tolist(), counts.tolist())},
        "nce_duplicate_goal_probability": dup_prob,
        "nce_note": "P(a positive goal value equals an independently sampled negative goal value).",
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_m3(n_episodes: int, base_seed: int, tag: str, device: str = "cpu",
           sal_frames: int = 24) -> Dict[str, Any]:
    model, meta = load_teacher(device=device)
    policy = TeacherPolicy(model, device=device)
    cfg = replace(C.M1Config(), device=device)
    seeds = [base_seed + i for i in range(n_episodes)]

    outdir = ART_ROOT / tag
    outdir.mkdir(parents=True, exist_ok=True)

    steps, snaps = record(policy, cfg, seeds)
    a = m3a_ablation(policy, snaps)
    sal = m3a_saliency(policy, snaps, n_frames=sal_frames, save_dir=outdir / "saliency")
    b = m3b_support(steps)
    c = m3c_outcome(steps)

    # verdict
    collapsed = (
        c["opponent_points_total"] == 0
        and c["next_scoring_outcome"]["distinct_outcomes"] <= 1
        and (c["next_score_event_eq_current_plus1_fraction"] or 0) > 0.99
    )
    verdict = ("OUTCOME-COLLAPSED -- requires controlled diversity"
               if collapsed else
               "native teacher data has sufficient outcome support")

    out = {
        "milestone": "M3",
        "tag": tag,
        "base_seed": base_seed,
        "n_episodes": n_episodes,
        "n_steps_recorded": len(steps),
        "n_snapshots": len(snaps),
        "M3A_opponent_dependence": {"ablation": a, "saliency": sal},
        "M3B_conditional_action_support": b,
        "M3C_outcome_goal_support": c,
        "VERDICT": verdict,
        "proposal_if_collapsed": _diversity_proposal() if collapsed else None,
        "teacher_meta": {k: meta[k] for k in ["ckpt_sha256", "actor_linear_out", "teacher_source"]},
        "NOTE": "Diagnostics only. No exploration injected, no critic trained.",
    }
    with open(outdir / "m3_report.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


def _diversity_proposal() -> Dict[str, Any]:
    return {
        "summary": "Native data is outcome-collapsed; propose a small ordered pilot BEFORE any "
                   "large collection or critic training. Do NOT inject exploration automatically.",
        "ordered_options": [
            {"rank": 1, "name": "native Categorical(logits)",
             "desc": "baseline; already shown collapsed — included as control."},
            {"rank": 2, "name": "mild temperature scaling",
             "desc": "sample from softmax(logits / T) with small T>1 (e.g. 1.5, 2.0) to surface "
                     "occasional sub-optimal actions and let the opponent score, creating negative/"
                     "non-monotone score-diff outcomes — without destroying competence."},
            {"rank": 3, "name": "small epsilon mixture (last resort)",
             "desc": "with prob eps pick a uniform legal action; only if temperature is insufficient."},
        ],
        "reeval_per_setting": [
            "competence: win rate, final score diff, frac reaching +15 (M2 metrics)",
            "opponent dependence: re-run M3A ablation (argmax-disagree, KL) — must persist",
            "action support: re-run M3B conditional support + functional-class entropy",
            "outcome support: re-run M3C — require opponent_points>0, next-outcome entropy>0, "
            "non-trivial negative/decreasing score-diff events, lower NCE duplicate-goal probability",
        ],
        "guardrail": "Reject any setting that buys diversity by destroying competent goal-reaching "
                     "or erasing opponent->action dependence.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="M3 diagnostics (opponent dep / action support / outcome).")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--base-seed", type=int, default=2000)
    ap.add_argument("--tag", type=str, default="native")
    ap.add_argument("--sal-frames", type=int, default=24)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    out = run_m3(args.episodes, args.base_seed, args.tag, device=args.device, sal_frames=args.sal_frames)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
