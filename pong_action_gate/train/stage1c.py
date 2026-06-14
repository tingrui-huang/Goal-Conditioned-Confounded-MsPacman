"""Stage-1C — mixed-anchor state critic: uniform vs 75/25 vs 50/50 (uniform/decision-focused).

Everything fixed except the training-anchor MIXTURE: dataset (T=2.0), score-diff goal,
next_score_event positive-goal semantics, duplicate-corrected objective, state-critic
architecture, step budget, val-loss checkpoint selection, episode splits.

Mixtures are EXACT and duplication-free: the focused fraction p is drawn from the focused
pool F; the remaining (1-p) is drawn from the NON-focused complement N = (all valid anchors
\\ F). So the effective focused fraction equals p exactly (the uniform baseline samples the
full pool, whose intrinsic focused fraction is ~7.8%). Every batch's composition is logged.

Because B×B goals come from the sampled anchors, changing the mixture also changes the batch
GOAL MARGINAL (focused anchors carry more opponent-direction events) — reported explicitly
per sampler so improvement is not attributed solely to state selection.

No goal redesign, no pixel critic, no Colab, no masking. Stop after Stage-1C.
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
from .stage1b import filter_anchors
from .stage1b_xeval import evaluate

ART = Path("artifacts/pong_action_gate/stage1c")
SAMPLERS = {"uniform": 0.0, "mix25": 0.25, "mix50": 0.50}   # focused fraction p


def split_focused(all_eps, ep_ids) -> Tuple[List[D.Anchor], List[D.Anchor], List[D.Anchor]]:
    alla = D.build_anchor_pool(all_eps, ep_ids)
    F = filter_anchors(alla, all_eps, decision=True)
    fset = set((a.ep, a.t) for a in F)
    N = [a for a in alla if (a.ep, a.t) not in fset]
    return alla, F, N


def anchor_directions(all_eps, anchors) -> np.ndarray:
    cache = {}
    dirs = np.zeros(len(anchors), np.int64)
    for i, a in enumerate(anchors):
        if a.ep not in cache:
            cache[a.ep] = build_episode_index(all_eps[a.ep])
        res = next_event_for(cache[a.ep], a.t, include_immediate=True)
        dirs[i] = res[2] if res is not None else 0
    return dirs


def goal_dir_marginal(anchors, dirs) -> Dict[str, Any]:
    goals = np.array([a.goal for a in anchors])
    vals, counts = np.unique(goals, return_counts=True)
    n = len(anchors)
    return {"n": n, "opp_event_fraction": float((dirs < 0).mean()),
            "agent_event_fraction": float((dirs > 0).mean()),
            "goal_value_mean": float(goals.mean()),
            "goal_value_marginal": {int(v): int(c) for v, c in zip(vals, counts)}}


def mixture_marginal(F, N, dirsF, dirsN, p) -> Dict[str, Any]:
    """Induced batch marginal = (1-p) over N + p over F (reported, not sampled)."""
    if p == 0.0:    # uniform baseline = full pool
        A = F + N; dA = np.concatenate([dirsF, dirsN])
        m = goal_dir_marginal(A, dA)
        m["composition"] = "full uniform pool"
        return m
    mF = goal_dir_marginal(F, dirsF); mN = goal_dir_marginal(N, dirsN)
    return {"composition": f"{int((1-p)*100)}% complement-N + {int(p*100)}% focused-F",
            "opp_event_fraction": float((1 - p) * mN["opp_event_fraction"] + p * mF["opp_event_fraction"]),
            "agent_event_fraction": float((1 - p) * mN["agent_event_fraction"] + p * mF["agent_event_fraction"]),
            "goal_value_mean": float((1 - p) * mN["goal_value_mean"] + p * mF["goal_value_mean"])}


def _batch(anchors, idx, feats):
    s, a, g = D.state_batch(anchors, idx, feats)
    return (torch.as_tensor(s), torch.as_tensor(a.astype(np.int64)),
            torch.as_tensor(D.norm_goal(g)), g.astype(int))


def mixed_idx(F, N, p, B, rng) -> Tuple[List[D.Anchor], int, int]:
    n_f = int(round(B * p)); n_u = B - n_f
    fa = [F[i] for i in rng.integers(len(F), size=n_f)] if n_f and F else []
    na = [N[i] for i in rng.integers(len(N), size=n_u)]
    return fa + na, n_f, n_u


def train_mixed(F_tr, N_tr, all_tr, F_va, N_va, p, feats, seed, steps, batch, lr=3e-4, eval_every=400):
    torch.manual_seed(seed)
    critic = StateSACritic(D.STATE_DIM, 6)
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    best_val = float("inf"); best_state = None; curve = []
    realized = []
    for step in range(steps + 1):
        if step % eval_every == 0:
            vr = np.random.default_rng(seed + 777 + step)
            with torch.no_grad():
                vls = []
                for _ in range(4):
                    if p == 0.0:
                        anc, _, _ = mixed_idx([], F_va + N_va, 0.0, batch, vr)   # full val pool
                    else:
                        anc, _, _ = mixed_idx(F_va, N_va, p, batch, vr)
                    ob, ac, gn, gi = _batch(anc, np.arange(len(anc)), feats)
                    vls.append(float(nce_loss(critic.logits_matrix(ob, ac, gn),
                                              torch.as_tensor(duplicate_corrected_targets(gi)))))
                vl = float(np.mean(vls))
            curve.append({"step": step, "val_loss": vl})
            if vl < best_val:
                best_val = vl; best_state = deepcopy(critic.state_dict())
        if step < steps:
            if p == 0.0:
                anc, nf, nu = mixed_idx([], all_tr, 0.0, batch, rng)
            else:
                anc, nf, nu = mixed_idx(F_tr, N_tr, p, batch, rng)
            if step < 3:
                realized.append({"step": step, "n_focused": nf, "n_complement": nu, "B": batch})
            ob, ac, gn, gi = _batch(anc, np.arange(len(anc)), feats)
            opt.zero_grad(set_to_none=True)
            loss = nce_loss(critic.logits_matrix(ob, ac, gn), torch.as_tensor(duplicate_corrected_targets(gi)))
            loss.backward(); opt.step()
    critic.load_state_dict(best_state); critic.eval()
    return critic, {"best_val_loss": best_val,
                    "selected_step": min(curve, key=lambda r: r["val_loss"])["step"],
                    "batch_composition_first_steps": realized,
                    "configured_focused_fraction": p}


def run(n_episodes: int, seeds: List[int], steps: int, batch: int) -> Dict[str, Any]:
    ART.mkdir(parents=True, exist_ok=True)
    all_eps = D.load_subset("full", list(range(n_episodes)), with_pixels=False)
    feats = [D.build_state_features(e) for e in all_eps]

    # induced goal/direction marginal per sampler (over the full dataset pools)
    allA, F_all, N_all = split_focused(all_eps, list(range(n_episodes)))
    dF = anchor_directions(all_eps, F_all); dN = anchor_directions(all_eps, N_all)
    goal_marginals = {name: mixture_marginal(F_all, N_all, dF, dN, p) for name, p in SAMPLERS.items()}

    results = {name: {} for name in SAMPLERS}
    for name, p in SAMPLERS.items():
        for seed in seeds:
            tr_ids, va_ids = D.split_episodes(n_episodes, 0.2, seed)
            _, F_tr, N_tr = split_focused(all_eps, tr_ids)
            all_tr = D.build_anchor_pool(all_eps, tr_ids)
            _, F_va, N_va = split_focused(all_eps, va_ids)
            critic, sel = train_mixed(F_tr, N_tr, all_tr, F_va, N_va, p, feats, seed, steps, batch)
            # evaluate on BOTH held-out subsets (same val episodes)
            uni_eval = D.build_anchor_pool(all_eps, va_ids)
            dec_eval = filter_anchors(uni_eval, all_eps, decision=True)
            results[name][f"seed{seed}"] = {
                "selected": sel,
                "eval_uniform": evaluate(critic, uni_eval, feats, seed),
                "eval_decision": evaluate(critic, dec_eval, feats, seed)}

    # stability + target assessment
    def shuffle_pts(name, ev):
        return [results[name][f"seed{s}"][ev]["shuffle_minus_correct"] for s in seeds]
    assess = {}
    for name in SAMPLERS:
        dec = shuffle_pts(name, "eval_decision")
        uni = shuffle_pts(name, "eval_uniform")
        assess[name] = {
            "decision_eval_shuffle_points": [round(d["point"], 4) for d in dec],
            "decision_eval_all_seeds_ci_pos": all(d["ci_excludes_zero_pos"] for d in dec),
            "decision_eval_n_seeds_ci_pos": sum(d["ci_excludes_zero_pos"] for d in dec),
            "uniform_eval_correct_loss": [round(results[name][f"seed{s}"]["eval_uniform"]["correct_loss_mean"], 2) for s in seeds],
            "uniform_eval_shuffle_points": [round(d["point"], 4) for d in uni]}

    out = {"milestone": "Stage-1C",
           "samplers": {k: f"focused_fraction={v}" for k, v in SAMPLERS.items()},
           "mixing": "exact, duplication-free: (1-p) from complement N, p from focused F (uniform=full pool)",
           "induced_goal_direction_marginal_per_sampler": goal_marginals,
           "results": results, "assessment": assess,
           "target": "preserve uniform-eval performance while producing STABLE action sensitivity on "
                     "decision-focused evaluation"}
    (ART / "stage1c_report.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-1C mixed-anchor state critic (uniform/75-25/50-50).")
    ap.add_argument("--n-episodes", type=int, default=80)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    out = run(args.n_episodes, args.seeds, args.steps, args.batch)
    print(json.dumps({"induced_goal_direction_marginal_per_sampler": out["induced_goal_direction_marginal_per_sampler"],
                      "assessment": out["assessment"]}, indent=2))


if __name__ == "__main__":
    main()
