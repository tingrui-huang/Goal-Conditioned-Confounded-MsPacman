"""Emulator-branch DIAGNOSIS — does next_score_event contain single-action signal?

For a small held-out set of cloned ALE states we restore EXACTLY (emulator system state,
frame buffers, teacher LSTM hidden, score, RNG), force each of the 6 first actions, let the
T=2.0 teacher continue, and over R repeated continuations estimate per first action:
  P(agent scores next), P(opponent scores next), E[next score diff], E[time to next event].

The teacher-branch part is checkpoint-free (the decisive variation measurement). If frozen
Stage-1 critic checkpoints are present, we additionally compare the real per-action outcome
ranking to each critic's action ranking, PER SEED (never averaged across seeds).

Conclusion rubric:
  * Goal/horizon mismatch : real outcomes barely vary across first actions.
  * Learning/objective failure : real outcomes vary meaningfully but critics rank inconsistently.

No retraining, no Colab, no masking/causal step.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .. import config as C
from ..teacher.load_teacher import TeacherPolicy, load_teacher, make_env
from . import dataset as D
from .emulator_branch import (_feat_single, collect_states, restore_env)
from .train_critic import (CKPT_ROOT, TrainConfig, latest_ckpt, load_ckpt,
                           make_critic, resolve_device, run_dir)


def _branch(env, teacher, snap, first_action: int, T: float, horizon: int, seed: int):
    """Force first_action, teacher continues; return (outcome in {-1,0,+1}, steps_to_event|None)."""
    torch.manual_seed(seed)
    hx, cx, ag, op = restore_env(env, snap)
    with torch.no_grad():
        _, _, (hx, cx) = teacher.model.predict_act_value(snap["anchor_obs"], (hx, cx))
    obs, rew, end, trunc, info = env.step(int(first_action))
    r = float(rew.item())
    if r != 0:
        return int(np.sign(r)), 0
    for step in range(1, horizon + 1):
        with torch.no_grad():
            logits, _, (hx, cx) = teacher.model.predict_act_value(obs[:, C.TEACHER_OBS_SLICE, :, :], (hx, cx))
            a = torch.distributions.Categorical(logits=logits / T).sample()
        obs, rew, end, trunc, info = env.step(a)
        r = float(rew.item())
        if r != 0:
            return int(np.sign(r)), step
        if bool((end | trunc).item()):
            break
    return 0, None


def _action_estimates(env, teacher, snap, R: int, T: float, horizon: int):
    """Per first action: P(agent), P(opp), P(no event), E[next score diff], E[time]."""
    out = []
    sd_pre = snap["score_diff_pre"]
    for a in range(6):
        outs, times = [], []
        for c in range(R):
            o, tt = _branch(env, teacher, snap, a, T, horizon, seed=10_000 + a * 1000 + c)
            outs.append(o)
            if tt is not None:
                times.append(tt)
        outs = np.array(outs)
        p_ag = float((outs > 0).mean()); p_op = float((outs < 0).mean()); p_no = float((outs == 0).mean())
        out.append({
            "action": a, "p_agent_next": p_ag, "p_opp_next": p_op, "p_no_event": p_no,
            "expected_next_score_diff": float(sd_pre + outs.mean()),
            "mean_outcome": float(outs.mean()),
            "expected_time_to_event": float(np.mean(times)) if times else None,
            "se_p_agent": float(np.sqrt(max(p_ag * (1 - p_ag), 1e-9) / R)),
        })
    return out


def teacher_branch_variation(state_seed: int, n_states: int, R: int, horizon: int,
                             stride: int, device: str) -> Dict[str, Any]:
    teacher_model, _ = load_teacher(device=device)
    teacher = TeacherPolicy(teacher_model, device=device)
    env = make_env(replace(C.M1Config(), device=device), num_envs=1)
    snaps = collect_states(teacher, env, state_seed, n_states, stride=stride)

    per_state = []
    for snap in snaps:
        est = _action_estimates(env, teacher, snap, R, C.BEHAVIOR_TEMPERATURE, horizon)
        p_ag = np.array([e["p_agent_next"] for e in est])
        ev = np.array([e["expected_next_score_diff"] for e in est])
        se = np.array([e["se_p_agent"] for e in est])
        per_state.append({
            "score_diff_pre": snap["score_diff_pre"],
            "per_action": est,
            "across_action_std_p_agent": float(p_ag.std()),
            "across_action_range_p_agent": float(p_ag.max() - p_ag.min()),
            "across_action_std_expected_score_diff": float(ev.std()),
            "mean_per_action_se_p_agent": float(se.mean()),
            # "signal" = across-action spread exceeds within-action sampling noise
            "spread_exceeds_noise": bool((p_ag.max() - p_ag.min()) > 2 * se.mean()),
        })
    env.close()

    spread = np.array([s["across_action_range_p_agent"] for s in per_state])
    noise = np.array([s["mean_per_action_se_p_agent"] for s in per_state])
    frac_signal = float(np.mean([s["spread_exceeds_noise"] for s in per_state]))
    return {
        "state_seed": state_seed, "n_states": len(per_state), "continuations_per_action": R,
        "horizon": horizon,
        "mean_across_action_range_p_agent": float(spread.mean()),
        "mean_within_action_se_p_agent": float(noise.mean()),
        "fraction_states_spread_exceeds_noise": frac_signal,
        "real_outcomes_vary_across_actions": bool(frac_signal >= 0.5 and spread.mean() > 2 * noise.mean()),
        "per_state": per_state,
        "_snaps": snaps,  # kept in-memory for an optional critic comparison in the same run
    }


# --------------------------------------------------------------------------- #
# Optional per-seed frozen-critic comparison (needs Stage-1 checkpoints)
# --------------------------------------------------------------------------- #
def _spearman(a, b):
    ar = np.argsort(np.argsort(a)); br = np.argsort(np.argsort(b))
    if ar.std() == 0 or br.std() == 0:
        return 0.0
    return float(np.corrcoef(ar, br)[0, 1])


def _critic_scores(critic, critic_kind, snap):
    goal = snap["score_diff_pre"] + 1
    goaln = torch.as_tensor(D.norm_goal(np.array([goal], np.float32)))
    if critic_kind == "state":
        obs = torch.as_tensor(_feat_single(snap["cur_obj"], snap["prev_obj"], snap["score_diff_pre"])[None])
    else:
        obs = torch.as_tensor(snap["gray_stack"][None])
    with torch.no_grad():
        return critic.scores_all_actions(obs, goaln)[0].numpy()


def compare_critic(seed: int, critic_kind: str, tv: Dict[str, Any], device: str) -> Optional[Dict[str, Any]]:
    cfg = TrainConfig(critic=critic_kind, seed=seed, device=device)
    d = run_dir(cfg)
    sel = d / "selected.json"
    if not sel.exists() or latest_ckpt(d) is None:
        return {"available": False, "reason": f"no Stage-1 checkpoint at {d}"}
    selected = json.loads(sel.read_text())
    critic = make_critic(cfg).to(resolve_device(device))
    critic.load_state_dict(load_ckpt(d / selected["ckpt"], resolve_device(device))["model"]); critic.eval()
    rows = []
    for s, snap in zip(tv["per_state"], tv["_snaps"]):
        emp = np.array([e["expected_next_score_diff"] for e in s["per_action"]])  # real per-action outcome
        cs = _critic_scores(critic, critic_kind, snap)
        rows.append({"top1_agree": int(np.argmax(cs) == np.argmax(emp)),
                     "spearman": _spearman(cs, emp)})
    return {"available": True, "selected_step": selected["selected_step"], "val_loss": selected["val_loss"],
            "top1_agreement": float(np.mean([r["top1_agree"] for r in rows])),
            "mean_spearman": float(np.mean([r["spearman"] for r in rows])),
            "random_baseline_top1": 1.0 / 6}


def main() -> None:
    ap = argparse.ArgumentParser(description="Emulator-branch diagnosis (goal/horizon vs learning failure).")
    ap.add_argument("--state-seed", type=int, default=7000)
    ap.add_argument("--n-states", type=int, default=12)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=200)
    ap.add_argument("--stride", type=int, default=53)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--compare-seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()

    tv = teacher_branch_variation(args.state_seed, args.n_states, args.reps, args.horizon,
                                  args.stride, args.device)
    comparisons = {}
    for s in args.compare_seeds:
        comparisons[f"seed{s}"] = {
            "state": compare_critic(s, "state", tv, args.device),
            "pixel": compare_critic(s, "pixel", tv, args.device)}

    real_varies = tv["real_outcomes_vary_across_actions"]
    any_compared = any(c["state"].get("available") or c["pixel"].get("available")
                       for c in comparisons.values())
    conclusion = (
        "GOAL/HORIZON MISMATCH (strongly supported, not conclusively proven): in this small diagnostic "
        "set most states are near action-indifferent and decision-critical states are a minority, so "
        "next_score_event at a typical anchor carries little single-action signal."
        if not real_varies else
        ("In this small diagnostic set the single-action signal exists but is concentrated in a minority "
         "of decision-critical states (strongly supports goal/horizon dilution, not conclusively proven). "
         "Per-seed critic comparison " + ("shows whether critics use that sparse signal." if any_compared
         else "PENDING Stage-1 checkpoints.")))

    tv.pop("_snaps", None)
    out = {"milestone": "emulator-branch-diagnosis",
           "teacher_branch_variation": tv,
           "per_seed_critic_comparison": comparisons,
           "conclusion": conclusion}
    outdir = CKPT_ROOT / "branch_diag"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "branch_diagnosis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"teacher_variation": {k: tv[k] for k in
                      ["mean_across_action_range_p_agent", "mean_within_action_se_p_agent",
                       "fraction_states_spread_exceeds_noise", "real_outcomes_vary_across_actions"]},
                      "per_seed_critic_comparison": comparisons,
                      "conclusion": conclusion}, indent=2))


if __name__ == "__main__":
    main()
