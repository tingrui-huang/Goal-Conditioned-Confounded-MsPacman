"""Stage-1D — rich short-horizon future-STATE goal (diagnostic only).

Hypothesis: under the ORIGINAL uniform anchor distribution, replacing the coarse, distant
scalar score goal (next_score_event) with a rich fixed-horizon (H=8) future object-state goal
makes the state critic reliably use the current action.

Fixed: T=2.0 dataset, existing episodes/alignment, STATE critic only, UNIFORM anchors,
existing state encoder + action pathway, contrastive sigmoid-BCE objective + Adam settings,
episode-level split, val-loss checkpoint selection, action diagnostics only after selection.
Only the GOAL changes: the scalar goal encoder is replaced by an MLP over the rich goal
vector g(t+8). NOT the final +15 goal; no pixel critic; no decision-focused/mixed sampling.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from . import dataset as D
from .critics import StateSACritic, nce_loss
from .stage1b import decision_criteria, is_decision_focused

ART = Path("artifacts/pong_action_gate/stage1d")
H = 8                      # fixed horizon (teacher transitions)
GOAL_DIM = 10             # 7 continuous + 3 masks
CONT = 7                  # ball_x,ball_y,ball_vx,ball_vy,agent_y,opp_y,score_diff


# --------------------------------------------------------------------------- #
# Rich goal critic = existing state encoder + action pathway, vector goal MLP
# --------------------------------------------------------------------------- #
class RichGoalEncoder(nn.Module):
    def __init__(self, in_dim: int, repr_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, repr_dim))

    def forward(self, g):                          # g: (B, in_dim)
        return self.net(g)


class RichStateCritic(StateSACritic):
    def __init__(self, state_dim, n_actions, goal_dim=GOAL_DIM, repr_dim=128, action_dim=32):
        super().__init__(state_dim, n_actions, repr_dim, action_dim)
        self.psi = RichGoalEncoder(goal_dim, repr_dim)   # replace ONLY the scalar goal encoder

    def g_repr(self, g):                            # g: (B, goal_dim) already normalized
        return self.psi(g)


# --------------------------------------------------------------------------- #
# Goal construction (pre-action aligned object states at t+8)
# --------------------------------------------------------------------------- #
def _raw_goals(ep: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Per future-index tf, the raw continuous components + per-component validity."""
    T = len(ep["action"])
    bx = ep["ball_x"]; by = ep["ball_y"]; py = ep["player_y"]; oy = ep["opp_y"]
    bp = ep["ball_present"].astype(bool); ov = ep["opp_valid"].astype(bool)
    sd = ep["score_diff"].astype(np.float32)
    vx = np.zeros(T, np.float32); vy = np.zeros(T, np.float32)
    velok = np.zeros(T, bool)
    velok[1:] = bp[1:] & bp[:-1]
    vx[1:] = np.where(velok[1:], np.nan_to_num(bx[1:]) - np.nan_to_num(bx[:-1]), 0.0)
    vy[1:] = np.where(velok[1:], np.nan_to_num(by[1:]) - np.nan_to_num(by[:-1]), 0.0)
    cont = np.stack([np.nan_to_num(bx), np.nan_to_num(by), vx, vy,
                     np.nan_to_num(py), np.nan_to_num(oy), sd], axis=1)   # (T,7)
    valid = np.stack([bp, bp, velok, velok, ep["player_valid"].astype(bool), ov, np.ones(T, bool)], axis=1)
    masks = np.stack([bp, ov, velok], axis=1).astype(np.float32)         # (T,3)
    return {"cont": cont.astype(np.float32), "valid": valid, "masks": masks}


def build_arrays(eps, ep_ids, stats=None) -> Dict[str, Any]:
    """Flat per-anchor arrays for uniform valid anchors (t+8 inside same episode)."""
    feats_per_ep = {li: D.build_state_features(eps[li]) for li in ep_ids}
    raw_per_ep = {li: _raw_goals(eps[li]) for li in ep_ids}
    A_state, A_action, A_cont, A_valid, A_masks = [], [], [], [], []
    A_ep, A_t, A_event, A_decision = [], [], [], []
    for li in ep_ids:
        ep = eps[li]; T = len(ep["action"])
        rg = raw_per_ep[li]; feat = feats_per_ep[li]
        dec = is_decision_focused(decision_criteria(ep))
        ise = ep["is_scoring_event"].astype(bool)
        for t in range(0, T - H):                  # t+H valid, same episode
            tf = t + H
            A_state.append(feat[t]); A_action.append(int(ep["action"][t]))
            A_cont.append(rg["cont"][tf]); A_valid.append(rg["valid"][tf]); A_masks.append(rg["masks"][tf])
            A_ep.append(li); A_t.append(t)
            A_event.append(bool(ise[t:tf].any()))
            A_decision.append(bool(dec[t]))
    cont = np.array(A_cont, np.float32); valid = np.array(A_valid, bool); masks = np.array(A_masks, np.float32)

    if stats is None:                              # train: normalization from these anchors only
        mean = np.zeros(CONT, np.float64); std = np.ones(CONT, np.float64)
        for c in range(CONT):
            v = cont[valid[:, c], c]
            if len(v):
                mean[c] = v.mean(); std[c] = max(v.std(), 1e-6)
        stats = {"mean": mean.tolist(), "std": std.tolist()}
    mean = np.array(stats["mean"]); std = np.array(stats["std"])
    norm = (cont - mean) / std
    norm = np.where(valid, norm, 0.0)              # neutral 0 for missing components
    goal = np.concatenate([norm.astype(np.float32), masks], axis=1)   # (Na, 10)

    # exact-duplicate key from raw integer goal (missing->0) + masks
    keyint = np.concatenate([np.round(np.where(valid, cont, 0.0)).astype(np.int64), masks.astype(np.int64)], axis=1)
    keys = np.array([hash(tuple(row)) for row in keyint])

    return {"state": np.array(A_state, np.float32), "action": np.array(A_action, np.int64),
            "goal": goal, "keys": keys, "ep": np.array(A_ep), "t": np.array(A_t),
            "event": np.array(A_event), "decision": np.array(A_decision),
            "stats": stats, "n_anchors": len(A_action),
            "missing": {"ball_absent_frac": float(1 - masks[:, 0].mean()),
                        "opp_absent_frac": float(1 - masks[:, 1].mean()),
                        "vel_invalid_frac": float(1 - masks[:, 2].mean())}}


def duplicate_rate(keys: np.ndarray) -> Dict[str, float]:
    vals, counts = np.unique(keys, return_counts=True)
    pmf = counts / counts.sum()
    frac_with_dup = float(np.mean(counts[np.searchsorted(vals, keys)] > 1))
    return {"pairwise_collision_prob": float((pmf ** 2).sum()),
            "fraction_anchors_with_exact_duplicate": frac_with_dup}


def _targets(keys_batch: np.ndarray) -> torch.Tensor:
    k = keys_batch.reshape(-1)
    return torch.as_tensor((k[:, None] == k[None, :]).astype(np.float32))


# --------------------------------------------------------------------------- #
# Train (uniform anchors) + val-loss checkpoint selection
# --------------------------------------------------------------------------- #
def _critic_logits(critic, A, idx):
    s = torch.as_tensor(A["state"][idx]); a = torch.as_tensor(A["action"][idx]); g = torch.as_tensor(A["goal"][idx])
    return critic.logits_matrix(s, a, g), A["keys"][idx]


def train(A_tr, A_va, seed, steps, batch, lr=3e-4, eval_every=400, goal_dim=GOAL_DIM):
    torch.manual_seed(seed)
    critic = RichStateCritic(D.STATE_DIM, 6, goal_dim=goal_dim)
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    best = float("inf"); best_state = None; curve = []
    n = A_tr["n_anchors"]; nv = A_va["n_anchors"]
    for step in range(steps + 1):
        if step % eval_every == 0:
            vr = np.random.default_rng(seed + 777 + step)
            with torch.no_grad():
                vls = []
                for _ in range(4):
                    vidx = vr.integers(nv, size=min(batch, nv))
                    logits, keys = _critic_logits(critic, A_va, vidx)
                    vls.append(float(nce_loss(logits, _targets(keys))))
                vl = float(np.mean(vls))
            curve.append({"step": step, "val_loss": vl})
            if vl < best:
                best = vl; best_state = deepcopy(critic.state_dict())
        if step < steps:
            idx = rng.integers(n, size=batch)
            logits, keys = _critic_logits(critic, A_tr, idx)
            opt.zero_grad(set_to_none=True)
            loss = nce_loss(logits, _targets(keys))
            loss.backward(); opt.step()
    critic.load_state_dict(best_state); critic.eval()
    return critic, {"best_val_loss": best, "selected_step": min(curve, key=lambda r: r["val_loss"])["step"],
                    "val_curve": curve}


# --------------------------------------------------------------------------- #
# Frozen diagnostics (episode-level bootstrap CI)
# --------------------------------------------------------------------------- #
def _ep_metric(critic, A, ep_idx, B, rng):
    idx = rng.choice(ep_idx, size=min(B, len(ep_idx)), replace=False)
    s = torch.as_tensor(A["state"][idx]); a = torch.as_tensor(A["action"][idx]); g = torch.as_tensor(A["goal"][idx])
    tgt = _targets(A["keys"][idx])
    with torch.no_grad():
        correct = float(nce_loss(critic.logits_matrix(s, a, g), tgt))
        perm = torch.as_tensor(rng.permutation(len(a)))
        shuffled = float(nce_loss(critic.logits_matrix(s, a[perm], g), tgt))
        randa = torch.as_tensor(rng.integers(6, size=len(a)).astype(np.int64))
        replaced = float(nce_loss(critic.logits_matrix(s, randa, g), tgt))
        sa = critic.scores_all_actions(s, g)
    return {"correct": correct, "shuffle_delta": shuffled - correct,
            "replace_delta": replaced - correct, "same_state_std": float(sa.std(1).mean())}


def diagnose(critic, A, seed, B=256, n_boot=2000, label="", mask=None):
    sel = np.arange(A["n_anchors"]) if mask is None else np.where(mask)[0]
    by_ep: Dict[int, List[int]] = {}
    for i in sel:
        by_ep.setdefault(int(A["ep"][i]), []).append(int(i))
    rng = np.random.default_rng(seed)
    per_ep = [_ep_metric(critic, A, np.array(ix), B, rng) for ix in by_ep.values() if len(ix) >= 8]
    if not per_ep:
        return {"label": label, "n_episodes": 0, "n_anchors": int(len(sel))}

    def ci(key):
        vals = np.array([m[key] for m in per_ep]); br = np.random.default_rng(seed + 1)
        bs = [float(vals[br.integers(len(vals), size=len(vals))].mean()) for _ in range(n_boot)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return {"point": float(vals.mean()), "ci95": [float(lo), float(hi)], "ci_excludes_zero_pos": bool(lo > 0)}

    return {"label": label, "n_episodes": len(per_ep), "n_anchors": int(len(sel)),
            "correct_loss_mean": float(np.mean([m["correct"] for m in per_ep])),
            "shuffle_delta": ci("shuffle_delta"), "replace_delta": ci("replace_delta"),
            "same_state_all_action_std": ci("same_state_std")}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_seed(all_eps, n_episodes, seed, steps, batch) -> Dict[str, Any]:
    tr_ids, va_ids = D.split_episodes(n_episodes, 0.2, seed)
    A_tr = build_arrays(all_eps, tr_ids)
    A_va = build_arrays(all_eps, va_ids, stats=A_tr["stats"])   # reuse train normalization
    critic, sel = train(A_tr, A_va, seed, steps, batch)

    diags = {
        "uniform_all": diagnose(critic, A_va, seed, label="uniform_all"),
        "score_event_in_window": diagnose(critic, A_va, seed, label="score_event_in_window", mask=A_va["event"]),
        "no_score_event_in_window": diagnose(critic, A_va, seed, label="no_event", mask=~A_va["event"]),
        "ordinary_anchors": diagnose(critic, A_va, seed, label="ordinary", mask=~A_va["decision"]),
        "decision_focused": diagnose(critic, A_va, seed, label="decision_focused", mask=A_va["decision"]),
    }
    out = {
        "seed": seed, "H": H, "goal_dim": GOAL_DIM,
        "config": {"steps": steps, "batch": batch, "lr": 3e-4, "uniform_anchors": True,
                   "state_critic": True, "rich_goal": True},
        "episode_split": {"train": tr_ids, "val": va_ids},
        "normalization_stats": A_tr["stats"],
        "goal_schema": ["ball_x", "ball_y", "ball_vx", "ball_vy", "agent_paddle_y(player,right,x=140)",
                        "opponent_paddle_y(enemy,left,x=16)", "score_diff_pre",
                        "mask_ball", "mask_opp", "mask_vel"],
        "duplicate_rate_train": duplicate_rate(A_tr["keys"]),
        "object_missing_train": A_tr["missing"], "object_missing_val": A_va["missing"],
        "n_train_anchors": A_tr["n_anchors"], "n_val_anchors": A_va["n_anchors"],
        "selected": {k: sel[k] for k in ["best_val_loss", "selected_step"]},
        "val_curve": sel["val_curve"],
        "diagnostics": diags,
    }
    d = ART / f"h{H}" / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "report.json").write_text(json.dumps(out, indent=2))
    torch.save(critic.state_dict(), d / "critic.pt")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-1D rich short-horizon future-state goal.")
    ap.add_argument("--n-episodes", type=int, default=80)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    all_eps = D.load_subset("full", list(range(args.n_episodes)), with_pixels=False)
    for seed in args.seeds:
        out = run_seed(all_eps, args.n_episodes, seed, args.steps, args.batch)
        u = out["diagnostics"]["uniform_all"]
        print(f"seed{seed}: uniform shuffle {u['shuffle_delta']['point']:+.4f} CI{u['shuffle_delta']['ci95']} "
              f"replace {u['replace_delta']['point']:+.4f} sameStd {u['same_state_all_action_std']['point']:.4f} "
              f"correctL {u['correct_loss_mean']:.2f} dupRate {out['duplicate_rate_train']['fraction_anchors_with_exact_duplicate']:.4f}")


if __name__ == "__main__":
    main()
