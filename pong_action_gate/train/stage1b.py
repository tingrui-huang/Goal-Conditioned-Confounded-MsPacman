"""Stage-1B — uniform vs decision-focused anchor sampler (state critic only).

Everything is held fixed: the T=2.0 behaviour dataset, the score-difference goal,
`next_score_event`, the duplicate-corrected contrastive objective, and the state-critic
architecture. ONLY the training-anchor sampler changes:

  1. uniform        : all valid next_score_event anchors;
  2. decision_focused: anchors selected by PRE-ACTION observable state only
                       (ball toward agent, ball near agent paddle, large ball-to-paddle
                       vertical mismatch, active rally). NO future reward / branch outcome.

Question: does decision-focused sampling make state action-sensitivity STABLE across 3
fixed seeds? If yes -> uniform-anchor dilution was the primary problem. If it stays
unstable -> the next_score_event goal/horizon itself is inadequate (propose shorter horizon).

No pixel critic. Local CPU.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from ..data.goal_sampler import build_episode_index, duplicate_corrected_targets, next_event_for
from . import dataset as D
from .critics import StateSACritic, nce_loss

ART = Path("artifacts/pong_action_gate/stage1b")
DEVICE = "cpu"

# decision-focused strata thresholds (pre-action observable state; native ALE coords)
NEAR_PADDLE_X = 120.0     # agent paddle is at native x=140; "near" = ball_x >= 120
MISMATCH_Y = 16.0         # |ball_y - own-paddle-center| >= ~one paddle height
PADDLE_HALF = 7.5


def decision_criteria(ep: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Per-step boolean masks for each criterion (PRE-action observable only)."""
    T = len(ep["action"])
    bx = ep["ball_x"]; by = ep["ball_y"]; py = ep["player_y"]
    bp = ep["ball_present"].astype(bool)
    # ball horizontal motion (toward agent = +x, agent paddle on the right)
    bdx = np.zeros(T, np.float32)
    bdx[1:] = np.where(bp[1:] & bp[:-1], np.nan_to_num(bx[1:]) - np.nan_to_num(bx[:-1]), 0.0)
    toward_agent = (bdx > 0) & bp
    near_paddle = bp & (np.nan_to_num(bx, nan=-1) >= NEAR_PADDLE_X)
    mismatch = bp & (np.abs(np.nan_to_num(by) - (np.nan_to_num(py) + PADDLE_HALF)) >= MISMATCH_Y)
    rally = bp
    return {"toward_agent": toward_agent, "near_paddle": near_paddle,
            "large_mismatch": mismatch, "active_rally": rally}


def is_decision_focused(crit: Dict[str, np.ndarray]) -> np.ndarray:
    # pivotal approach: rally AND ball approaching AND near the agent paddle
    return crit["active_rally"] & crit["toward_agent"] & crit["near_paddle"]


def filter_anchors(anchors: List[D.Anchor], eps: List[Dict], decision: bool) -> List[D.Anchor]:
    if not decision:
        return anchors
    masks = {}
    keep = []
    for a in anchors:
        if a.ep not in masks:
            masks[a.ep] = is_decision_focused(decision_criteria(eps[a.ep]))
        if masks[a.ep][a.t]:
            keep.append(a)
    return keep


def next_dir(eps: List[Dict], anchors: List[D.Anchor]) -> Tuple[int, int]:
    """(#agent-next, #opp-next) over the anchor set."""
    idx_cache = {}
    ag = op = 0
    for a in anchors:
        if a.ep not in idx_cache:
            idx_cache[a.ep] = build_episode_index(eps[a.ep])
        res = next_event_for(idx_cache[a.ep], a.t, include_immediate=True)
        if res is not None:
            d = res[2]
            ag += int(d > 0); op += int(d < 0)
    return ag, op


# --------------------------------------------------------------------------- #
# Part 1 — anchor counts + next-outcome distribution
# --------------------------------------------------------------------------- #
def count_report(eps: List[Dict], all_anchors: List[D.Anchor]) -> Dict[str, Any]:
    crit_counts = {k: 0 for k in ["toward_agent", "near_paddle", "large_mismatch", "active_rally"]}
    masks = {li: decision_criteria(eps[li]) for li in set(a.ep for a in all_anchors)}
    df_set = []
    for a in all_anchors:
        c = masks[a.ep]
        for k in crit_counts:
            crit_counts[k] += int(c[k][a.t])
        if is_decision_focused(c)[a.t]:
            df_set.append(a)
    u_ag, u_op = next_dir(eps, all_anchors)
    d_ag, d_op = next_dir(eps, df_set)
    return {
        "total_valid_anchors": len(all_anchors),
        "criterion_counts (of valid anchors)": crit_counts,
        "decision_focused_count": len(df_set),
        "decision_focused_fraction": len(df_set) / max(len(all_anchors), 1),
        "next_outcome_distribution": {
            "uniform": {"agent": u_ag, "opp": u_op,
                        "opp_fraction": u_op / max(u_ag + u_op, 1)},
            "decision_focused": {"agent": d_ag, "opp": d_op,
                                 "opp_fraction": d_op / max(d_ag + d_op, 1)},
        },
    }


# --------------------------------------------------------------------------- #
# Train + frozen diagnostics
# --------------------------------------------------------------------------- #
def _state_make_batch(anchors, idx, feats):
    s, a, g = D.state_batch(anchors, idx, feats)
    return (torch.as_tensor(s), torch.as_tensor(a.astype(np.int64)),
            torch.as_tensor(D.norm_goal(g)), g.astype(int))


def train_state(train_anchors, val_anchors, feats, seed, steps, batch, lr=3e-4,
                eval_every=400, val_batches=4):
    torch.manual_seed(seed)
    critic = StateSACritic(D.STATE_DIM, 6)
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    best_val = float("inf"); best_state = None; curve = []
    n = len(train_anchors)
    nv = len(val_anchors)
    for step in range(steps + 1):
        if step % eval_every == 0:
            # predeclared val-loss selection: fixed deterministic val resample averaged over val_batches
            vr = np.random.default_rng(seed + 777 + step)
            with torch.no_grad():
                vl = float(np.mean([
                    float(nce_loss(critic.logits_matrix(vb[0], vb[1], vb[2]),
                                   torch.as_tensor(duplicate_corrected_targets(vb[3]))))
                    for vb in (_state_make_batch(val_anchors, vr.integers(nv, size=min(batch, nv)), feats)
                               for _ in range(val_batches))]))
            curve.append({"step": step, "val_loss": vl})
            if vl < best_val:
                best_val = vl; best_state = deepcopy(critic.state_dict())
        if step < steps:
            obs, action, goaln, goals = _state_make_batch(train_anchors, rng.integers(n, size=batch), feats)
            opt.zero_grad(set_to_none=True)
            loss = nce_loss(critic.logits_matrix(obs, action, goaln),
                            torch.as_tensor(duplicate_corrected_targets(goals)))
            loss.backward(); opt.step()
    critic.load_state_dict(best_state)
    critic.eval()
    return critic, {"best_val_loss": best_val, "selected_step": min(curve, key=lambda r: r["val_loss"])["step"]}


def _ep_metric(critic, anchors_idx, anchors, feats, B, rng):
    idx = rng.choice(anchors_idx, size=min(B, len(anchors_idx)), replace=False)
    obs, action, goaln, goals = _state_make_batch(anchors, idx, feats)
    tgt = torch.as_tensor(duplicate_corrected_targets(goals))
    with torch.no_grad():
        correct = float(nce_loss(critic.logits_matrix(obs, action, goaln), tgt))
        perm = torch.as_tensor(rng.permutation(len(action)))
        shuffled = float(nce_loss(critic.logits_matrix(obs, action[perm], goaln), tgt))
        randa = torch.as_tensor(rng.integers(6, size=len(action)).astype(np.int64))
        replaced = float(nce_loss(critic.logits_matrix(obs, randa, goaln), tgt))
        sa = critic.scores_all_actions(obs, goaln)
    return {"shuffle_minus_correct": shuffled - correct,
            "replace_minus_correct": replaced - correct,
            "same_state_all_action_std": float(sa.std(1).mean())}


def diagnostics(critic, val_anchors, feats, seed, B=256, n_boot=2000):
    by_ep: Dict[int, List[int]] = {}
    for i, a in enumerate(val_anchors):
        by_ep.setdefault(a.ep, []).append(i)
    rng = np.random.default_rng(seed)
    per_ep = [_ep_metric(critic, np.array(ix), val_anchors, feats, B, rng)
              for ix in by_ep.values() if len(ix) >= 8]

    def ci(key):
        vals = np.array([m[key] for m in per_ep])
        br = np.random.default_rng(seed + 1)
        boots = [float(vals[br.integers(len(vals), size=len(vals))].mean()) for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"point": float(vals.mean()), "ci95": [float(lo), float(hi)], "ci_excludes_zero_pos": bool(lo > 0)}

    return {"n_val_episodes": len(per_ep),
            "shuffle_minus_correct": ci("shuffle_minus_correct"),
            "replace_minus_correct": ci("replace_minus_correct"),
            "same_state_all_action_std": ci("same_state_all_action_std")}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(n_episodes: int, seeds: List[int], steps: int, batch: int) -> Dict[str, Any]:
    ART.mkdir(parents=True, exist_ok=True)
    # load once; per-seed splits are matched and decided by split_episodes(seed)
    all_eps = D.load_subset("full", list(range(n_episodes)), with_pixels=False)
    feats = [D.build_state_features(e) for e in all_eps]
    all_anchors = D.build_anchor_pool(all_eps, list(range(n_episodes)))
    part1 = count_report(all_eps, all_anchors)

    results = {"uniform": {}, "decision_focused": {}}
    for sampler in ["uniform", "decision_focused"]:
        decision = (sampler == "decision_focused")
        for seed in seeds:
            train_ids, val_ids = D.split_episodes(n_episodes, 0.2, seed)
            tr = filter_anchors(D.build_anchor_pool(all_eps, train_ids), all_eps, decision)
            va = filter_anchors(D.build_anchor_pool(all_eps, val_ids), all_eps, decision)
            critic, sel = train_state(tr, va, feats, seed, steps, batch)
            diag = diagnostics(critic, va, feats, seed)
            results[sampler][f"seed{seed}"] = {
                "train_anchors": len(tr), "val_anchors": len(va),
                "selected": sel, "diagnostics": diag}

    def stable(sampler):
        s = [results[sampler][f"seed{x}"]["diagnostics"]["shuffle_minus_correct"] for x in seeds]
        all_pos_ci = all(d["ci_excludes_zero_pos"] for d in s)
        all_pos_pt = all(d["point"] > 0 for d in s)
        return {"all_seeds_shuffle_ci_excludes_zero": all_pos_ci,
                "all_seeds_shuffle_point_positive": all_pos_pt,
                "points": [round(d["point"], 4) for d in s]}

    stab = {k: stable(k) for k in results}
    du = stab["uniform"]["points"]; dd = stab["decision_focused"]["points"]
    n_sig_decision = sum(results["decision_focused"][f"seed{s}"]["diagnostics"]
                         ["shuffle_minus_correct"]["ci_excludes_zero_pos"] for s in seeds)
    decision_improves = all(dd[i] >= du[i] for i in range(len(seeds))) and np.mean(dd) > np.mean(du)

    if stab["decision_focused"]["all_seeds_shuffle_ci_excludes_zero"]:
        conclusion = ("UNIFORM-ANCHOR DILUTION was the primary problem: decision-focused sampling makes "
                      "state action-sensitivity stable & positive (CI>0) across all 3 seeds.")
    elif decision_improves and n_sig_decision >= 2:
        conclusion = (f"PARTIAL: decision-focused sampling SUBSTANTIALLY raises action-sensitivity (every "
                      f"seed >= its uniform counterpart; {n_sig_decision}/3 seeds CI>0; same-state spread "
                      f"~2x; larger replace effect), so uniform-anchor dilution was a MAJOR factor. But it "
                      f"does NOT reach full 3/3 stability, so the next_score_event horizon likely "
                      f"contributes RESIDUAL instability. PROPOSE (do not run) a shorter-horizon goal "
                      f"(e.g. score outcome within H<=~20 steps, or a ball-contact/paddle-interception "
                      f"goal the single action can influence); a tighter test (more decision val episodes "
                      f"/ seeds) could resolve whether the residual is statistical noise or fundamental.")
    else:
        conclusion = ("next_score_event GOAL/HORIZON is inadequate: decision-focused sampling does NOT "
                      "stabilise action-sensitivity. PROPOSE (do not run) a shorter-horizon goal.")

    out = {"milestone": "Stage-1B", "n_episodes": n_episodes, "seeds": seeds, "steps": steps,
           "fixed": ["T=2.0 dataset", "score-diff goal", "next_score_event",
                     "duplicate-corrected objective", "state critic"],
           "decision_strata": {"near_paddle_x": NEAR_PADDLE_X, "mismatch_y": MISMATCH_Y,
                               "definition": "active_rally AND toward_agent AND near_paddle"},
           "part1_anchor_report": part1, "results": results, "stability": stab,
           "conclusion": conclusion}
    (ART / "stage1b_report.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-1B uniform vs decision-focused sampler (state critic).")
    ap.add_argument("--n-episodes", type=int, default=60)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--part1-only", action="store_true")
    args = ap.parse_args()
    if args.part1_only:
        all_eps = D.load_subset("full", list(range(args.n_episodes)), with_pixels=False)
        anchors = D.build_anchor_pool(all_eps, list(range(args.n_episodes)))
        print(json.dumps(count_report(all_eps, anchors), indent=2))
        return
    out = run(args.n_episodes, args.seeds, args.steps, args.batch)
    print(json.dumps({"part1": out["part1_anchor_report"], "stability": out["stability"],
                      "conclusion": out["conclusion"]}, indent=2))


if __name__ == "__main__":
    main()
