"""M3.5 — controlled behavior-diversity pilot (temperature scaling).

Goal: find the SMALLEST modification to the DIAMOND behavior policy that introduces
genuine score-outcome variation while preserving competence, opponent dependence, and
state-conditioned action support.

Behavior policy: action ~ Categorical(logits = original_logits / T).  No epsilon yet.
Checkpoint / preprocessing / LSTM handling / env / seeds are fixed across all T.

Pilot: 5 episodes per temperature, SAME five seeds for every setting. Not to be scaled
without approval. NO critic training, NO final dataset collection. Heatmaps are produced
only for the final candidate AFTER review (not here).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from .. import config as C
from ..objects import extract_pong_objects
from ..teacher.load_teacher import TeacherPolicy, load_teacher, make_env
from .m3 import (MOVE_CLASS, MOVE_NAMES, _entropy, _inpaint_opp, m3c_outcome, record)

ART_ROOT = Path("artifacts/pong_action_gate/m35")
TEMPS = [1.0, 1.25, 1.5, 2.0, 3.0]


# --------------------------------------------------------------------------- #
# Temperature rollout
# --------------------------------------------------------------------------- #
def record_temp(policy: TeacherPolicy, cfg: C.M1Config, seeds: List[int], T: float):
    """Roll out with Categorical(logits/T). Returns (steps, episodes).

    Each step records the exact behaviour probability of the sampled action under
    the temperature-scaled policy, plus per-state entropy and object state.
    """
    model = policy.model
    steps: List[Dict[str, Any]] = []
    episodes: List[Dict[str, Any]] = []
    for s in seeds:
        torch.manual_seed(s)
        env = make_env(replace(cfg, seed=s), num_envs=1)
        ale = env.env.ale
        obs, info = env.reset(seed=[s])
        hx, cx = policy.initial_state(1)
        agent_score = opp_score = 0
        first_plus15 = None
        ent_sum = 0.0
        t = 0
        while True:
            with torch.no_grad():
                logits, _, (hx, cx) = model.predict_act_value(
                    obs[:, C.TEACHER_OBS_SLICE, :, :], (hx, cx))
            scaled = logits / T
            pT = F.softmax(scaled, dim=1)[0]
            action = torch.distributions.Categorical(logits=scaled).sample()
            a = int(action.item())
            behavior_prob = float(pT[a])
            ent = _entropy(pT.numpy())
            ent_sum += ent

            o = extract_pong_objects(ale.getRAM())
            rec = {
                "seed": s, "t": t, "action": a, "move_class": MOVE_CLASS[a],
                "behavior_prob": behavior_prob, "state_entropy_bits": ent,
                "phase": "rally" if o.ball_present else "serve",
                "score_diff": agent_score - opp_score,
            }
            obs, rew, end, trunc, info = env.step(action)
            r = float(rew.item())
            if r > 0:
                agent_score += 1
            elif r < 0:
                opp_score += 1
            if first_plus15 is None and (agent_score - opp_score) >= C.GOAL_STAR:
                first_plus15 = t
            rec.update(reward=r, is_scoring_event=int(r != 0),
                       agent_score=agent_score, opp_score=opp_score)
            steps.append(rec)
            t += 1
            if bool((end | trunc).item()) or t >= cfg.safety_step_cap:
                break
        episodes.append({
            "seed": s, "episode_length": t,
            "final_score_diff": agent_score - opp_score,
            "agent_score": agent_score, "opp_score": opp_score,
            "win": int((agent_score - opp_score) > 0),
            "reached_plus15": int(first_plus15 is not None),
            "first_step_plus15": first_plus15,
            "mean_state_entropy_bits": ent_sum / max(t, 1),
        })
        env.close()
    return steps, episodes


# --------------------------------------------------------------------------- #
# Aggregations
# --------------------------------------------------------------------------- #
def competence(eps: List[Dict]) -> Dict[str, Any]:
    sd = np.array([e["final_score_diff"] for e in eps], float)
    return {
        "final_score_diff": {"mean": float(sd.mean()), "std": float(sd.std()),
                             "min": float(sd.min()), "max": float(sd.max()),
                             "values": [int(x) for x in sd]},
        "win_rate": float(np.mean([e["win"] for e in eps])),
        "fraction_reached_plus15": float(np.mean([e["reached_plus15"] for e in eps])),
        "episode_length": {"mean": float(np.mean([e["episode_length"] for e in eps])),
                           "min": int(min(e["episode_length"] for e in eps)),
                           "max": int(max(e["episode_length"] for e in eps))},
        "agent_points_total": int(sum(e["agent_score"] for e in eps)),
        "opponent_points_total": int(sum(e["opp_score"] for e in eps)),
    }


def behavior(steps: List[Dict], eps: List[Dict]) -> Dict[str, Any]:
    n = len(steps)
    a6 = np.zeros(C.N_ACTIONS)
    fc = {m: 0 for m in MOVE_NAMES}
    for r in steps:
        a6[r["action"]] += 1
        fc[r["move_class"]] += 1
    p6 = a6 / max(n, 1)
    pf = np.array([fc[m] for m in MOVE_NAMES]) / max(n, 1)
    return {
        "sampled_action_histogram": {C.ACTION_MEANINGS[i]: int(a6[i]) for i in range(C.N_ACTIONS)},
        "sampled_action_fraction": {C.ACTION_MEANINGS[i]: float(p6[i]) for i in range(C.N_ACTIONS)},
        "func_class_fraction": {MOVE_NAMES[i]: float(pf[i]) for i in range(3)},
        "supported_func_classes": int((pf >= 0.05).sum()),
        "mean_state_action_entropy_bits": float(np.mean([e["mean_state_entropy_bits"] for e in eps])),
        "mean_behavior_prob_of_sampled_action": float(np.mean([r["behavior_prob"] for r in steps])),
    }


def behavior_shift_on_matched(snaps: List[Dict], T: float) -> Dict[str, Any]:
    """KL(p_T || p_native) on the fixed matched native snapshot set."""
    kls = []
    ents = []
    for sn in snaps:
        l = sn["logits"]                      # (1,6) native logits
        p_nat = F.softmax(l, dim=1)
        p_T = F.softmax(l / T, dim=1)
        kl = float((p_T * (p_T.add(1e-9).log() - p_nat.add(1e-9).log())).sum())
        kls.append(kl)
        ents.append(_entropy(p_T[0].numpy()))
    return {"mean_kl_from_native": float(np.mean(kls)),
            "mean_state_entropy_bits": float(np.mean(ents))}


def opp_dependence_on_matched(policy: TeacherPolicy, snaps: List[Dict], T: float) -> Dict[str, Any]:
    """Opponent-removal ablation on the fixed matched snapshot set, under temperature T."""
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
        p0 = F.softmax(l0 / T, 1); p1 = F.softmax(l1 / T, 1)
        rows.append({
            "argmax_changed": int(l0.argmax().item() != l1.argmax().item()),  # T-invariant
            "kl": float((p0 * (p0.add(1e-9).log() - p1.add(1e-9).log())).sum()),
            "mean_abs_dlogit_scaled": float(((l0 - l1) / T).abs().mean()),
        })
    return {
        "n": len(rows),
        "argmax_disagree_rate": float(np.mean([r["argmax_changed"] for r in rows])) if rows else None,
        "mean_kl": float(np.mean([r["kl"] for r in rows])) if rows else None,
        "mean_abs_dlogit_scaled": float(np.mean([r["mean_abs_dlogit_scaled"] for r in rows])) if rows else None,
    }


# --------------------------------------------------------------------------- #
# Orchestration + selection
# --------------------------------------------------------------------------- #
def run_m35(seeds: List[int], temps: List[float], tag: str, device: str = "cpu") -> Dict[str, Any]:
    model, meta = load_teacher(device=device)
    policy = TeacherPolicy(model, device=device)
    cfg = replace(C.M1Config(), device=device)
    outdir = ART_ROOT / tag
    outdir.mkdir(parents=True, exist_ok=True)

    # fixed matched snapshot set from NATIVE (T=1.0) rollouts on the pilot seeds
    _, matched = record(policy, cfg, seeds)

    per_temp = {}
    for T in temps:
        steps, eps = record_temp(policy, cfg, seeds, T)
        comp = competence(eps)
        beh = behavior(steps, eps)
        out = m3c_outcome(steps)
        shift = behavior_shift_on_matched(matched, T)
        oppdep = opp_dependence_on_matched(policy, matched, T)
        per_temp[str(T)] = {
            "temperature": T,
            "competence": comp,
            "behavior": beh,
            "behavior_shift_vs_native_matched": shift,
            "opponent_dependence_matched": oppdep,
            "outcome_support": {
                "agent_points_total": out["agent_points_total"],
                "opponent_points_total": out["opponent_points_total"],
                "next_scoring_outcome": out["next_scoring_outcome"],
                "next_score_event_eq_current_plus1_fraction": out["next_score_event_eq_current_plus1_fraction"],
                "decreasing_score_diff_events": out["decreasing_score_diff_events"],
                "score_diff_range": out["score_diff_range"],
                "nce_duplicate_goal_probability": out["nce_duplicate_goal_probability"],
                "achieved_goal_marginal": out["next_event_goal_value_marginal"],
            },
        }
        # per-temp step log with behaviour probabilities (lightweight fields)
        with open(outdir / f"steps_T{T}.jsonl", "w") as f:
            for r in steps:
                f.write(json.dumps({k: r[k] for k in
                        ["seed", "t", "action", "behavior_prob", "reward", "score_diff"]}) + "\n")

    table = _tradeoff_table(per_temp)
    selection = _select_candidate(per_temp, temps)

    report = {
        "milestone": "M3.5",
        "tag": tag,
        "seeds": seeds,
        "temperatures": temps,
        "matched_snapshot_count": len(matched),
        "sampling": "Categorical(logits = original_logits / T).sample()",
        "tradeoff_table": table,
        "per_temperature": per_temp,
        "selection": selection,
        "teacher_meta": {k: meta[k] for k in ["ckpt_sha256", "actor_linear_out", "teacher_source"]},
        "NOTE": "Pilot only. No critic trained, no final dataset collected. Heatmaps deferred to "
                "the final candidate after human review.",
    }
    with open(outdir / "m35_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def _tradeoff_table(per_temp: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for k, v in per_temp.items():
        o = v["outcome_support"]
        rows.append({
            "T": v["temperature"],
            "win_rate": v["competence"]["win_rate"],
            "mean_final_score_diff": v["competence"]["final_score_diff"]["mean"],
            "frac_reach_+15": v["competence"]["fraction_reached_plus15"],
            "agent_pts": o["agent_points_total"],
            "opp_pts": o["opponent_points_total"],
            "next_outcome_entropy_bits": o["next_scoring_outcome"]["conditional_entropy_bits"],
            "distinct_outcomes": o["next_scoring_outcome"]["distinct_outcomes"],
            "frac_next_eq_cur+1": o["next_score_event_eq_current_plus1_fraction"],
            "decreasing_events": o["decreasing_score_diff_events"],
            "score_diff_range": o["score_diff_range"],
            "nce_dup_goal_prob": o["nce_duplicate_goal_probability"],
            "behavior_KL_from_native": v["behavior_shift_vs_native_matched"]["mean_kl_from_native"],
            "mean_state_entropy_bits": v["behavior"]["mean_state_action_entropy_bits"],
            "supported_func_classes": v["behavior"]["supported_func_classes"],
            "opp_ablation_argmax_disagree": v["opponent_dependence_matched"]["argmax_disagree_rate"],
            "opp_ablation_kl": v["opponent_dependence_matched"]["mean_kl"],
        })
    return rows


def _select_candidate(per_temp: Dict[str, Any], temps: List[float]) -> Dict[str, Any]:
    """Lowest T that produces BOTH next-scoring outcomes with nonzero outcome entropy.

    Competence / opponent-dependence sufficiency is presented as evidence for HUMAN
    judgement — no arbitrary numeric pass threshold is imposed here.
    """
    candidates = []
    for T in sorted(temps):
        v = per_temp[str(T)]
        o = v["outcome_support"]
        produces_both = o["agent_points_total"] > 0 and o["opponent_points_total"] > 0
        nonzero_entropy = (o["next_scoring_outcome"]["conditional_entropy_bits"] or 0) > 0
        if produces_both and nonzero_entropy:
            candidates.append(T)

    if not candidates:
        return {
            "candidate_T": None,
            "all_outcome_collapsed": True,
            "message": "All temperature variants remain outcome-collapsed (opponent never scores / "
                       "zero next-outcome entropy). STOP: propose a small epsilon-mixture pilot for "
                       "human approval — do NOT execute it.",
            "proposed_next_pilot": {
                "name": "epsilon mixture (NOT executed)",
                "spec": "with prob eps choose a uniform legal action, else native Categorical(logits); "
                        "sweep eps in {0.02, 0.05, 0.10}; re-evaluate competence, opponent dependence, "
                        "action support, outcome support exactly as in M3.5.",
                "guardrail": "reject any eps that destroys competence or erases opponent->action dependence.",
            },
        }
    lowest = min(candidates)
    return {
        "candidate_T": lowest,
        "rationale": "Lowest temperature that produces BOTH scoring outcomes with nonzero next-outcome "
                     "entropy. Competence and opponent-dependence evidence is in the trade-off table; "
                     "final sufficiency is for human review (no numeric threshold imposed).",
        "requires_human_review": True,
        "heatmaps": "Generate M3A-style saliency heatmaps for this candidate ONLY after approval.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="M3.5 temperature behavior-diversity pilot.")
    ap.add_argument("--base-seed", type=int, default=3000)
    ap.add_argument("--episodes", type=int, default=5, help="episodes (=seeds) per temperature")
    ap.add_argument("--temps", type=float, nargs="+", default=TEMPS)
    ap.add_argument("--tag", type=str, default="pilot")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    seeds = [args.base_seed + i for i in range(args.episodes)]
    report = run_m35(seeds, args.temps, args.tag, device=args.device)
    print(json.dumps({"tradeoff_table": report["tradeoff_table"],
                      "selection": report["selection"]}, indent=2))


if __name__ == "__main__":
    main()
