"""M3.6 — temperature confirmation: compare T=1.25 vs T=2.0 at higher episode count.

20 episodes per temperature on the SAME 20 seeds. Checkpoint / preprocessing / LSTM /
env / Categorical(logits / T).sample() all unchanged.

Crucially, opponent dependence of the *sampled policy* is measured with temperature-
SENSITIVE metrics (KL / JS / TV / coupled-uniform action disagreement / change in the
probability of the originally sampled action) on ONE fixed matched snapshot set — NOT
the temperature-invariant argmax disagreement used as a structural sanity check in M3.

No critic training, no final dataset collection. Stop after the confirmation report.
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
from ..teacher.load_teacher import TeacherPolicy, load_teacher, make_env  # noqa: F401
from .m3 import _inpaint_opp, m3c_outcome, record
from .m35 import behavior, competence, record_temp

ART_ROOT = Path("artifacts/pong_action_gate/m36")
COUPLING_SEED = 12345           # fixed reproducible coupling uniforms (shared across T)
MATCHED_SEEDS_N = 6             # native episodes used to build the fixed matched snapshot set


# --------------------------------------------------------------------------- #
# Divergences + coupled sampling
# --------------------------------------------------------------------------- #
def _kl_bits(p, q):
    p = np.clip(p, 1e-12, 1); q = np.clip(q, 1e-12, 1)
    return float((p * (np.log2(p) - np.log2(q))).sum())


def _js_bits(p, q):
    m = 0.5 * (p + q)
    return 0.5 * _kl_bits(p, m) + 0.5 * _kl_bits(q, m)


def _tv(p, q):
    return 0.5 * float(np.abs(p - q).sum())


def _inv_cdf(p, u):
    c = np.cumsum(p)
    return int(min(np.searchsorted(c, u, side="right"), len(p) - 1))


def sampled_opp_dependence(policy: TeacherPolicy, snaps: List[Dict], T: float) -> Dict[str, Any]:
    """Opponent-removal on a fixed matched set, measured on the temperature-T sampled policy.

    Uses fixed per-snapshot coupling uniforms (same across temperatures) so the
    sampled-action disagreement is reproducible and matched.
    """
    model = policy.model
    rng = np.random.default_rng(COUPLING_SEED)
    us = rng.random(len(snaps))

    kls, jss, tvs, coupled_dis, dprob_orig, argmax_dis = [], [], [], [], [], []
    for i, sn in enumerate(snaps):
        if sn["objects"].opp_y is None:
            continue
        abl = _inpaint_opp(sn["obs_rgb"], sn["objects"])
        if abl is None:
            continue
        with torch.no_grad():
            l0, _, _ = model.predict_act_value(sn["obs_rgb"], (sn["hx"], sn["cx"]))
            l1, _, _ = model.predict_act_value(abl, (sn["hx"], sn["cx"]))
        p0 = F.softmax(l0 / T, 1)[0].numpy()
        p1 = F.softmax(l1 / T, 1)[0].numpy()
        kls.append(_kl_bits(p0, p1))
        jss.append(_js_bits(p0, p1))
        tvs.append(_tv(p0, p1))
        u = us[i]
        a0 = _inv_cdf(p0, u)        # action sampled from original-obs policy (coupled u)
        a1 = _inv_cdf(p1, u)        # action sampled from opponent-removed policy (same u)
        coupled_dis.append(int(a0 != a1))
        dprob_orig.append(float(p0[a0] - p1[a0]))   # change in prob of the originally sampled action
        argmax_dis.append(int(np.argmax(p0) != np.argmax(p1)))   # T-invariant structural check

    n = len(kls)
    agg = lambda xs: float(np.mean(xs)) if xs else None
    return {
        "n": n,
        "mean_kl_bits": agg(kls),
        "mean_js_bits": agg(jss),
        "mean_tv": agg(tvs),
        "coupled_action_disagree_rate": agg(coupled_dis),
        "mean_change_in_prob_of_sampled_action": agg(dprob_orig),
        "argmax_disagree_rate_Tinvariant": agg(argmax_dis),
        "coupling": f"inverse-CDF with fixed uniforms (seed={COUPLING_SEED}), shared across T",
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _competence_extra(eps: List[Dict]) -> Dict[str, Any]:
    reached = [e for e in eps if e["reached_plus15"]]
    fs = [e["first_step_plus15"] for e in reached]
    return {
        "first_step_plus15": {
            "count_reached": len(reached),
            "mean": float(np.mean(fs)) if fs else None,
            "min": int(np.min(fs)) if fs else None,
            "max": int(np.max(fs)) if fs else None,
        },
        "final_score_diff_values": [e["final_score_diff"] for e in eps],
    }


def run_m36(seeds: List[int], temps: List[float], tag: str, device: str = "cpu") -> Dict[str, Any]:
    model, meta = load_teacher(device=device)
    policy = TeacherPolicy(model, device=device)
    cfg = replace(C.M1Config(), device=device)
    outdir = ART_ROOT / tag
    outdir.mkdir(parents=True, exist_ok=True)

    # one fixed matched snapshot set from native (T=1.0) rollouts on a fixed seed subset
    matched_seeds = seeds[:MATCHED_SEEDS_N]
    _, matched = record(policy, cfg, matched_seeds)

    per_temp = {}
    for T in temps:
        steps, eps = record_temp(policy, cfg, seeds, T)
        comp = competence(eps)
        comp.update(_competence_extra(eps))
        beh = behavior(steps, eps)
        out = m3c_outcome(steps)
        oppdep = sampled_opp_dependence(policy, matched, T)
        n_ep = len(eps)
        opp_pts = out["opponent_points_total"]
        agent_pts = out["agent_points_total"]
        per_temp[str(T)] = {
            "temperature": T,
            "n_episodes": n_ep,
            "competence": comp,
            "behavior": beh,
            "outcome_support": {
                "agent_points_total": agent_pts,
                "opponent_points_total": opp_pts,
                "opponent_point_fraction": float(opp_pts / max(agent_pts + opp_pts, 1)),
                "next_scoring_outcome": out["next_scoring_outcome"],
                "decreasing_score_diff_events": out["decreasing_score_diff_events"],
                "next_score_event_eq_current_plus1_fraction": out["next_score_event_eq_current_plus1_fraction"],
                "score_diff_range": out["score_diff_range"],
                "achieved_score_diff_marginal": out["next_event_goal_value_marginal"],
                "nce_duplicate_goal_probability": out["nce_duplicate_goal_probability"],
                "expected_opponent_events_per_100_episodes": float(opp_pts / max(n_ep, 1) * 100),
            },
            "sampled_policy_opponent_dependence": oppdep,
        }
        with open(outdir / f"episodes_T{T}.jsonl", "w") as f:
            for e in eps:
                f.write(json.dumps(e) + "\n")

    table = _table(per_temp)
    recommendation = _recommend(per_temp)
    report = {
        "milestone": "M3.6",
        "tag": tag,
        "seeds": seeds,
        "n_episodes_per_temp": len(seeds),
        "matched_snapshot_seeds": matched_seeds,
        "matched_snapshot_count": len(matched),
        "temperatures": temps,
        "tradeoff_table": table,
        "per_temperature": per_temp,
        "recommendation": recommendation,
        "teacher_meta": {k: meta[k] for k in ["ckpt_sha256", "actor_linear_out", "teacher_source"]},
        "NOTE": "Confirmation only. No critic trained, no final dataset collected.",
    }
    with open(outdir / "m36_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def _table(per_temp: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for v in per_temp.values():
        o = v["outcome_support"]; d = v["sampled_policy_opponent_dependence"]
        rows.append({
            "T": v["temperature"],
            "win_rate": v["competence"]["win_rate"],
            "mean_final_score_diff": round(v["competence"]["final_score_diff"]["mean"], 2),
            "frac_reach_+15": v["competence"]["fraction_reached_plus15"],
            "agent_pts": o["agent_points_total"],
            "opp_pts": o["opponent_points_total"],
            "opp_point_fraction": round(o["opponent_point_fraction"], 4),
            "next_outcome_entropy_bits": round(o["next_scoring_outcome"]["conditional_entropy_bits"], 4),
            "frac_next_eq_cur+1": round(o["next_score_event_eq_current_plus1_fraction"], 4),
            "decreasing_events": o["decreasing_score_diff_events"],
            "exp_opp_events_per_100ep": round(o["expected_opponent_events_per_100_episodes"], 1),
            "sampled_opp_KL_bits": round(d["mean_kl_bits"], 4),
            "sampled_opp_JS_bits": round(d["mean_js_bits"], 4),
            "sampled_opp_TV": round(d["mean_tv"], 4),
            "coupled_action_disagree": round(d["coupled_action_disagree_rate"], 4),
            "mean_dprob_sampled_action": round(d["mean_change_in_prob_of_sampled_action"], 4),
        })
    return rows


def _recommend(per_temp: Dict[str, Any]) -> Dict[str, Any]:
    a = per_temp.get("1.25"); b = per_temp.get("2.0")
    if a is None or b is None:
        return {"note": "expected T=1.25 and T=2.0"}
    oa = a["outcome_support"]
    reliable_125 = (oa["opponent_points_total"] > 0
                    and (oa["next_scoring_outcome"]["conditional_entropy_bits"] or 0) > 0
                    and a["competence"]["fraction_reached_plus15"] >= 0.9)
    stronger_dep_125 = (a["sampled_policy_opponent_dependence"]["coupled_action_disagree_rate"]
                        >= b["sampled_policy_opponent_dependence"]["coupled_action_disagree_rate"])
    return {
        "rule": "Recommend T=1.25 if it reliably produces nontrivial opponent scoring and nonzero "
                "outcome entropy while preserving near-complete +15 support AND stronger sampled-policy "
                "opponent dependence; recommend T=2.0 only if T=1.25 is too outcome-sparse.",
        "T1.25_reliable_outcome_variation": bool(reliable_125),
        "T1.25_stronger_sampled_opp_dependence_than_T2.0": bool(stronger_dep_125),
        "auto_suggestion": "T=1.25" if (reliable_125 and stronger_dep_125) else
                           ("T=2.0 (T=1.25 too sparse)" if not reliable_125 else
                            "T=1.25 vs T=2.0 — human call (see table)"),
        "requires_human_review": True,
        "note": "No arbitrary numeric pass threshold imposed; evidence presented for human decision.",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="M3.6 temperature confirmation (T=1.25 vs T=2.0).")
    ap.add_argument("--base-seed", type=int, default=4000)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--temps", type=float, nargs="+", default=[1.25, 2.0])
    ap.add_argument("--tag", type=str, default="confirm")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    seeds = [args.base_seed + i for i in range(args.episodes)]
    report = run_m36(seeds, args.temps, args.tag, device=args.device)
    print(json.dumps({"tradeoff_table": report["tradeoff_table"],
                      "recommendation": report["recommendation"]}, indent=2))


if __name__ == "__main__":
    main()
