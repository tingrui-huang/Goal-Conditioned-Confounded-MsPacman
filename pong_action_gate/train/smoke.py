"""M6 smoke: matched state + pixel contrastive-critic checks (NO full training run).

Order: (1) episode-level split; (2) deterministic matched anchors/goals shared by both
critics; (3) tiny-batch overfit + gradient tests; (4) state-feature critic smoke FIRST;
(5) pixel critic smoke ONLY if the state critic shows action sensitivity.

Primary task: next_score_event with the NATURAL uniform anchor distribution. The
duplicate-corrected B×B target (target[i,j]=1 iff goal[j]==goal[i]) is the objective;
diagonal-only is reported separately as a diagnostic baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import torch

from ..data.goal_sampler import diagonal_targets, duplicate_corrected_targets
from . import dataset as D
from .critics import PixelSACritic, StateSACritic, nce_loss

ART = Path("artifacts/pong_action_gate/m6")
DEVICE = "cpu"


def _t(x):
    return torch.as_tensor(x, device=DEVICE)


def _targets(goals: np.ndarray, kind: str = "corrected") -> torch.Tensor:
    m = duplicate_corrected_targets(goals) if kind == "corrected" else diagonal_targets(len(goals))
    return _t(m)


def param_count(m) -> int:
    return int(sum(p.numel() for p in m.parameters()))


# --------------------------------------------------------------------------- #
# Tests + diagnostics (critic-agnostic via make_batch)
# --------------------------------------------------------------------------- #
def gradient_tests(critic, obs, action, goaln, obs_grad_param) -> Dict[str, Any]:
    critic.zero_grad(set_to_none=True)
    goals_int = np.round(goaln.numpy() * 23 - 2).astype(int)
    logits = critic.logits_matrix(obs, action, goaln)
    loss = nce_loss(logits, _targets(goals_int))
    loss.backward()
    g_action = critic.action_embed.weight.grad
    g_obs = obs_grad_param.grad
    g_psi = critic.psi.net[0].weight.grad
    # zero-action changes outputs?
    with torch.no_grad():
        d0 = torch.diagonal(critic.logits_matrix(obs, action, goaln))
        dz = torch.diagonal(critic.logits_matrix(obs, action, goaln, zero_action=True))
    return {
        "action_pathway_grad_norm": float(g_action.norm()),
        "obs_pathway_grad_norm": float(g_obs.norm()) if g_obs is not None else None,
        "goal_pathway_grad_norm": float(g_psi.norm()) if g_psi is not None else None,
        "action_pathway_grad_nonzero": bool(g_action.norm() > 0),
        "zero_action_changes_output": float((d0 - dz).abs().mean()) > 1e-6,
        "mean_abs_diag_change_zero_action": float((d0 - dz).abs().mean()),
    }


def tiny_overfit(critic, obs, action, goaln, steps: int = 400, lr: float = 1e-3) -> Dict[str, Any]:
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    goals_int = np.round(goaln.numpy() * 23 - 2).astype(int)
    tgt = _targets(goals_int)
    init = None
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = critic.logits_matrix(obs, action, goaln)
        loss = nce_loss(logits, tgt)
        if s == 0:
            init = loss.item()
        loss.backward(); opt.step()
    with torch.no_grad():
        logits = critic.logits_matrix(obs, action, goaln)
        pos = logits[tgt > 0].mean(); neg = logits[tgt == 0].mean()
    return {"init_loss": init, "final_loss": loss.item(),
            "pos_logit_mean": float(pos), "neg_logit_mean": float(neg),
            "pos_minus_neg": float(pos - neg)}


def action_diagnostics(critic, obs, action, goaln, rng: np.random.Generator) -> Dict[str, Any]:
    goals_int = np.round(goaln.numpy() * 23 - 2).astype(int)
    tgt = _targets(goals_int)
    with torch.no_grad():
        correct = float(nce_loss(critic.logits_matrix(obs, action, goaln), tgt))
        perm = _t(rng.permutation(len(action)))
        shuffled = float(nce_loss(critic.logits_matrix(obs, action[perm], goaln), tgt))
        rand_a = _t(rng.integers(critic.n_actions, size=len(action)).astype(np.int64))
        replaced = float(nce_loss(critic.logits_matrix(obs, rand_a, goaln), tgt))
        zeroed = float(nce_loss(critic.logits_matrix(obs, action, goaln, zero_action=True), tgt))
        sa = critic.scores_all_actions(obs, goaln)             # (B, n_actions)
        per_row_std = sa.std(dim=1)
    return {
        "loss_correct": correct, "loss_action_shuffled": shuffled,
        "loss_action_replaced": replaced, "loss_action_zeroed": zeroed,
        "shuffle_minus_correct": shuffled - correct,
        "zeroed_minus_correct": zeroed - correct,
        "same_state_all_action_score_std_mean": float(per_row_std.mean()),
        "fraction_near_flat_rows(<1e-3)": float((per_row_std < 1e-3).float().mean()),
    }


def smoke_train(critic, make_batch: Callable, train_anchors, val_anchors,
                steps: int, B: int, seed: int, lr: float = 3e-4):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    n = len(train_anchors)
    init_loss = None
    for s in range(steps):
        idx = rng.integers(n, size=B)
        obs, action, goaln, goals_int = make_batch(train_anchors, idx)
        opt.zero_grad(set_to_none=True)
        loss = nce_loss(critic.logits_matrix(obs, action, goaln), _t(duplicate_corrected_targets(goals_int)))
        if s == 0:
            init_loss = loss.item()
        loss.backward(); opt.step()
    # validation on held-out episodes
    nv = len(val_anchors)
    vidx = rng.integers(nv, size=min(B, nv))
    vobs, vaction, vgoaln, vgoals = make_batch(val_anchors, vidx)
    with torch.no_grad():
        val_loss = float(nce_loss(critic.logits_matrix(vobs, vaction, vgoaln),
                                  _t(duplicate_corrected_targets(vgoals))))
    return {"init_train_loss": init_loss, "final_train_loss": loss.item(),
            "val_loss": val_loss}, (vobs, vaction, vgoaln)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_critic(name, critic, obs_grad_param, make_batch, train_anchors, val_anchors,
               steps, B, seed) -> Dict[str, Any]:
    rng = np.random.default_rng(seed + 1)
    # tiny overfit + gradient tests on a small fixed batch
    fixed_idx = rng.integers(len(train_anchors), size=64)
    fobs, faction, fgoaln, _ = make_batch(train_anchors, fixed_idx)
    grad = gradient_tests(critic_fresh := critic.__class__(*critic._init_args), fobs, faction, fgoaln,
                          obs_grad_param(critic_fresh))
    overfit = tiny_overfit(critic.__class__(*critic._init_args), fobs, faction, fgoaln)
    # smoke training
    train_metrics, valbatch = smoke_train(critic, make_batch, train_anchors, val_anchors, steps, B, seed)
    diag = action_diagnostics(critic, *valbatch, np.random.default_rng(seed + 7))
    sensitive = (diag["shuffle_minus_correct"] > 0 and diag["zeroed_minus_correct"] > 0
                 and diag["same_state_all_action_score_std_mean"] > 1e-3)
    return {
        "param_count": param_count(critic),
        "gradient_tests": grad,
        "tiny_overfit": overfit,
        "smoke_train": train_metrics,
        "action_diagnostics_val": diag,
        "action_sensitive": bool(sensitive),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="M6 smoke: matched state + pixel critic checks.")
    ap.add_argument("--tag", type=str, default="full")
    ap.add_argument("--n-episodes", type=int, default=30)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    ART.mkdir(parents=True, exist_ok=True)

    # (1) episode-level split
    train_ids, val_ids = D.split_episodes(args.n_episodes, args.val_frac, args.seed)
    all_ids = sorted(train_ids + val_ids)

    # state critic needs no pixels; load small arrays for the subset
    eps_s = D.load_subset(args.tag, all_ids, with_pixels=False)
    id_to_local = {g: i for i, g in enumerate(all_ids)}
    train_local = [id_to_local[g] for g in train_ids]
    val_local = [id_to_local[g] for g in val_ids]
    state_feats = [D.build_state_features(ep) for ep in eps_s]

    # (2) deterministic matched anchors/goals (shared)
    train_anchors = D.build_anchor_pool(eps_s, train_local)
    val_anchors = D.build_anchor_pool(eps_s, val_local)

    def state_make_batch(anchors, idx):
        s, a, g = D.state_batch(anchors, idx, state_feats)
        return _t(s), _t(a.astype(np.int64)), _t(D.norm_goal(g)), g.astype(int)

    report: Dict[str, Any] = {
        "milestone": "M6-smoke", "tag": args.tag,
        "split": {"n_episodes": args.n_episodes, "train_ids": train_ids, "val_ids": val_ids,
                  "train_anchors": len(train_anchors), "val_anchors": len(val_anchors)},
        "task": "next_score_event (uniform, natural distribution); duplicate-corrected B×B target",
        "state_dim": D.STATE_DIM, "state_feature_names": D.STATE_FEATURE_NAMES,
        "batch": args.batch, "steps": args.steps,
    }

    # (4) state critic FIRST
    sc = StateSACritic(D.STATE_DIM, 6)
    sc._init_args = (D.STATE_DIM, 6)
    report["state_critic"] = run_critic(
        "state", sc, lambda c: c.trunk[0].weight, state_make_batch,
        train_anchors, val_anchors, args.steps, args.batch, args.seed)

    # (5) pixel critic ONLY if state critic is action sensitive
    if report["state_critic"]["action_sensitive"]:
        eps_p = D.load_subset(args.tag, all_ids, with_pixels=True)

        def pixel_make_batch(anchors, idx):
            f, a, g = D.pixel_batch(anchors, idx, eps_p)
            return _t(f), _t(a.astype(np.int64)), _t(D.norm_goal(g)), g.astype(int)

        pc = PixelSACritic(4, 6)
        pc._init_args = (4, 6)
        report["pixel_critic"] = run_critic(
            "pixel", pc, lambda c: c.conv[0].weight, pixel_make_batch,
            train_anchors, val_anchors, args.steps, args.batch, args.seed)
    else:
        report["pixel_critic"] = {"skipped": True,
                                  "reason": "state critic did not show action sensitivity"}

    with open(ART / "m6_smoke_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
