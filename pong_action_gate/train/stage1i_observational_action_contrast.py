"""Stage-1I — observational same-state action contrast (can it recover the interventional effect?).

Stage-1H showed the critic CAN learn the action-conditioned future under explicit interventional
(cloned-branch) supervision. Stage-1I asks the harder, deployable question: trained on OBSERVATIONAL
trajectories only, with heuristic same-state alternative-action negatives, does the learned action
preference align with the REAL interventional action effect (emulator branches), or does it merely
imitate the behavior policy?

Construction (observational sample (s, a_obs, g_obs)):
    positive:     (s, a_obs, g_obs)
    alternatives: (s, a_alt, g_obs) for the other two semantic groups   [HEURISTIC negatives]
A != a_obs does NOT prove A could not also produce g_obs -> alternatives are heuristic negatives
(false negatives possible). Training accuracy is therefore NOT evidence of causal alignment; the
emulator-branch test is.

Hard rules: TRAIN on observational data only; cloned branches are EVAL-ONLY (never touch train/val
loss, checkpoint selection, normalization, or any tuning). goal_dim==9 no-self; H never an input;
H=12 and H=16 trained separately; reuse RichStateCritic unchanged; episode-level split. No causal
identification is claimed. No pixel/confounder/causal/Seaquest/chunk/policy/+15 work. Nothing committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import dataset as D
from .critics import nce_loss
from .stage1b import decision_criteria, is_decision_focused
from .stage1d_rich_goal import RichStateCritic, _raw_goals
from .stage1h_interventional_branch_critic import (CANON_IDS, _norm9, build_branches as h_build_branches,
                                                   canon_logits, diversity_per_base, evaluate as h_evaluate,
                                                   generate_or_load_branches)

RESULTS = Path("pong_action_gate/results/stage1i_observational_action_contrast")
HORIZONS = [12, 16]
GROUP_NAMES = ["STAY", "UP", "DOWN"]               # STAY == NOOP group
GIDX = {0: 0, 1: 0, 2: 1, 4: 1, 3: 2, 5: 2}        # exact ALE id -> semantic group index
NS_CONT = [0, 1, 2, 3, 5, 6]                        # no-self cont cols of _raw_goals (-> raw6)
TAU = 1.0
LAMBDA = 1.0
SEM_BASE = 1.0 / 3
# Stage-1H branch cache identity (EVAL ONLY) — must match the generated config.
BRANCH_STATE_SEED, BRANCH_N_DEC, BRANCH_N_ORD = 8000, 700, 150


# --------------------------------------------------------------------------- #
# Step 0 — audit
# --------------------------------------------------------------------------- #
def audit(eps, ep_ids) -> Dict[str, Any]:
    allc = np.zeros(3, int); decc = np.zeros(3, int); ordc = np.zeros(3, int); exact = np.zeros(6, int)
    nanch = ndec = 0
    for li in ep_ids:
        e = eps[li]; T = len(e["action"]); dec = is_decision_focused(decision_criteria(e))
        for t in range(T - max(HORIZONS)):
            a = int(e["action"][t]); g = GIDX[a]; exact[a] += 1; allc[g] += 1; nanch += 1
            (decc if dec[t] else ordc)[g] += 1; ndec += int(dec[t])
    prop = lambda c: (c / max(c.sum(), 1)).round(4).tolist()
    return {
        "ale_action_meanings": ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"],
        "semantic_groups_ids": {"STAY": [0, 1], "UP": [2, 4], "DOWN": [3, 5]},
        "canonical_action_per_group": {"STAY": 0, "UP": 2, "DOWN": 3},
        "alias_note": "STAY=NOOP/FIRE, UP=RIGHT/RIGHTFIRE, DOWN=LEFT/LEFTFIRE; Pong has NO UPFIRE/DOWNFIRE.",
        "exact_action_counts": exact.tolist(),
        "aliases_present": {"FIRE": int(exact[1]), "RIGHTFIRE": int(exact[4]), "LEFTFIRE": int(exact[5])},
        "semantic_counts": {"all": allc.tolist(), "decision": decc.tolist(), "ordinary": ordc.tolist()},
        "semantic_proportions": {"all": prop(allc), "decision": prop(decc), "ordinary": prop(ordc)},
        "decision_fraction": float(ndec / max(nanch, 1)),
        "policy_support_note": ("T=2.0 behavior policy gives non-degenerate support to all 3 groups in "
                                "decision states (min group prop reported above) -> heuristic same-state "
                                "negatives are available; majority-group prop sets the imitation floor."),
        "decision_selector": "stage1b: active_rally & toward_agent(ball dx>0) & near_paddle(ball_x>=120)",
        "goal9": "stage1f/1h no-self [ball_x,ball_y,ball_vx,ball_vy,opp_y,score_diff,mask_ball,mask_opp,mask_vel]",
        "normalization": "6-dim mean/std on observational TRAIN episodes only; reused for val+emulator",
        "episode_split": "3-way episode-level (disjoint); cloned eval states from independent rollout seed 8000",
        "emulator_eval": "Stage-1H branch cache (clone/restore, canonical 0/2/3); col k -> group k (aligned)",
    }


# --------------------------------------------------------------------------- #
# Step 1 — observational dataset (raw, then normalized per horizon)
# --------------------------------------------------------------------------- #
def build_obs_raw(eps, ep_ids, H, decision_only) -> Dict[str, Any]:
    st, ax, gp, ep_a, dec_a, wt = [], [], [], [], [], []
    raw6, valid6, masks3 = [], [], []
    for li in ep_ids:
        e = eps[li]; T = len(e["action"]); rg = _raw_goals(e); f = D.build_state_features(e)
        dec = is_decision_focused(decision_criteria(e))
        bx = np.nan_to_num(e["ball_x"]); bp = e["ball_present"].astype(bool)
        bdx = np.zeros(T); bdx[1:] = np.where(bp[1:] & bp[:-1], bx[1:] - bx[:-1], 0.0)
        for t in range(0, T - H):
            if decision_only and not dec[t]:
                continue
            if (not decision_only) and dec[t]:
                continue
            tf = t + H
            a = int(e["action"][t])
            st.append(f[t]); ax.append(a); gp.append(GIDX[a]); ep_a.append(li); dec_a.append(bool(dec[t]))
            raw6.append(rg["cont"][tf][NS_CONT]); valid6.append(rg["valid"][tf][NS_CONT]); masks3.append(rg["masks"][tf])
            # decision-strength weight from OBSERVABLE state only (no counterfactual signal):
            # ball approaching (dx>0) AND near the agent paddle (x in [120,160]).
            strength = float(np.clip((bx[t] - 120.0) / 40.0, 0, 1) * np.clip(bdx[t] / 4.0, 0, 1))
            wt.append(0.1 + 0.9 * strength)
    return {"state": np.array(st, np.float32), "a_exact": np.array(ax, np.int64),
            "group": np.array(gp, np.int64), "ep": np.array(ep_a), "decision": np.array(dec_a),
            "weight": np.array(wt, np.float32), "raw6": np.array(raw6, np.float32),
            "valid6": np.array(valid6, np.float32), "masks3": np.array(masks3, np.float32), "n": len(st)}


def fit_stats(B, idx) -> Dict[str, list]:
    raw = B["raw6"][idx]; val = B["valid6"][idx]
    mean = np.zeros(6); std = np.ones(6)
    for c in range(6):
        v = raw[val[:, c] > 0, c]
        if len(v):
            mean[c] = v.mean(); std[c] = max(v.std(), 1e-6)
    return {"mean": mean.tolist(), "std": std.tolist()}


def finalize(B, stats) -> Dict[str, Any]:
    goal = np.stack([_norm9(B["raw6"][i], B["valid6"][i], B["masks3"][i], stats) for i in range(B["n"])]).astype(np.float32)
    keyint = np.concatenate([np.round(np.where(B["valid6"] > 0, B["raw6"], 0.0)).astype(np.int64),
                             B["masks3"].astype(np.int64)], axis=1)
    key = np.array([hash(tuple(r)) for r in keyint])
    out = dict(B); out["goal"] = goal; out["key"] = key
    return out


def dup_and_support(B) -> Dict[str, Any]:
    vals, counts = np.unique(B["key"], return_counts=True)
    frac_dup = float(np.mean(counts[np.searchsorted(vals, B["key"])] > 1))
    sc = np.bincount(B["group"], minlength=3)
    sdiff = B["raw6"][:, 5]  # score_diff comp
    return {"n_anchors": int(B["n"]), "n_episodes": int(len(set(B["ep"].tolist()))),
            "semantic_counts": sc.tolist(), "semantic_props": (sc / max(sc.sum(), 1)).round(4).tolist(),
            "frac_visible_ball": float(B["masks3"][:, 0].mean()), "frac_valid_velocity": float(B["masks3"][:, 2].mean()),
            "score_diff_mean_std": [float(sdiff.mean()), float(sdiff.std())],
            "n_exact_duplicate_goals": int((counts > 1).sum()), "n_unique_goals": int(len(vals)),
            "frac_anchors_with_dup": frac_dup,
            "decision_strength_weight_mean": float(B["weight"].mean())}


# --------------------------------------------------------------------------- #
# Step 2 — losses + training (observational only; checkpoint = observational val)
# --------------------------------------------------------------------------- #
def _targets(keys) -> torch.Tensor:
    k = np.asarray(keys).reshape(-1)
    return torch.as_tensor((k[:, None] == k[None, :]).astype(np.float32))


def _loss(critic, B, idx, kind):
    s = torch.as_tensor(B["state"][idx]); g = torch.as_tensor(B["goal"][idx])
    y = torch.as_tensor(B["group"][idx]); ae = torch.as_tensor(B["a_exact"][idx])
    nce = nce_loss(critic.logits_matrix(s, ae, g), _targets(B["key"][idx])) if kind in ("nce", "combined", "weighted") else 0.0
    if kind == "nce":
        return nce
    ce_none = F.cross_entropy(canon_logits(critic, s, g) / TAU, y, reduction="none")
    if kind == "action":
        return ce_none.mean()
    if kind == "combined":
        return nce + LAMBDA * ce_none.mean()
    if kind == "weighted":
        w = torch.as_tensor(B["weight"][idx])
        return nce + LAMBDA * (w * ce_none).sum() / (w.sum() + 1e-8)
    raise ValueError(kind)


def train_critic(Btr, Bva, seed, kind, steps, batch, lr=3e-4, eval_every=200):
    torch.manual_seed(seed)
    critic = RichStateCritic(D.STATE_DIM, 6, goal_dim=9)
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    best = [float("inf")]; best_state = [None]; curve = []
    for step in range(steps + 1):
        if step % eval_every == 0:
            vr = np.random.default_rng(seed + 777 + step)
            with torch.no_grad():
                vl = float(np.mean([float(_loss(critic, Bva, vr.integers(Bva["n"], size=min(batch, Bva["n"])), kind))
                                    for _ in range(4)]))
            curve.append({"step": step, "val_loss": vl})
            if vl < best[0]:
                best[0] = vl; best_state[0] = deepcopy(critic.state_dict())
        if step < steps:
            idx = rng.integers(Btr["n"], size=min(batch, Btr["n"]))
            opt.zero_grad(set_to_none=True); _loss(critic, Btr, idx, kind).backward(); opt.step()
    critic.load_state_dict(best_state[0]); critic.eval()
    return critic, {"best_val_loss": best[0], "selected_step": min(curve, key=lambda r: r["val_loss"])["step"],
                    "val_curve": curve}


class MLPClf(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.net(x)


def train_clf(Btr, Bva, seed, use_goal, steps=2000, batch=256, lr=3e-4):
    torch.manual_seed(seed + 13)
    in_dim = D.STATE_DIM + (9 if use_goal else 0)
    clf = MLPClf(in_dim); opt = torch.optim.Adam(clf.parameters(), lr=lr); rng = np.random.default_rng(seed + 5)
    feat = lambda B, idx: torch.as_tensor(np.concatenate([B["state"][idx], B["goal"][idx]], 1) if use_goal else B["state"][idx])
    best = [float("inf")]; best_state = [None]
    for step in range(steps + 1):
        if step % 200 == 0:
            with torch.no_grad():
                vi = np.random.default_rng(seed + step).integers(Bva["n"], size=min(batch, Bva["n"]))
                vl = float(F.cross_entropy(clf(feat(Bva, vi)), torch.as_tensor(Bva["group"][vi])))
            if vl < best[0]:
                best[0] = vl; best_state[0] = deepcopy(clf.state_dict())
        if step < steps:
            idx = rng.integers(Btr["n"], size=batch)
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(clf(feat(Btr, idx)), torch.as_tensor(Btr["group"][idx])).backward(); opt.step()
    clf.load_state_dict(best_state[0]); clf.eval()

    def acc(B):
        with torch.no_grad():
            pred = clf(feat(B, np.arange(B["n"]))).argmax(1).numpy()
        return float((pred == B["group"]).mean())
    return clf, acc


# --------------------------------------------------------------------------- #
# Step 3 — observational held-out diagnostics
# --------------------------------------------------------------------------- #
def obs_action_metrics(critic, B) -> Dict[str, Any]:
    with torch.no_grad():
        sc = canon_logits(critic, torch.as_tensor(B["state"]), torch.as_tensor(B["goal"])).numpy()
    pred = sc.argmax(1); y = B["group"]
    top1 = float((pred == y).mean())
    recalls = [float((pred[y == k] == k).mean()) if (y == k).any() else float("nan") for k in range(3)]
    bal = float(np.nanmean(recalls))
    maj = int(np.bincount(y, minlength=3).argmax()); maj_acc = float((y == maj).mean())
    return {"obs_action_top1": top1, "balanced_acc": bal, "per_group_recall": recalls,
            "majority_acc": maj_acc, "ce_loss": float(F.cross_entropy(torch.as_tensor(sc) / TAU, torch.as_tensor(y)))}


def shuffle_cross_deltas(critic, B, seed) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(B["n"], size=min(2048, B["n"]))
    s = torch.as_tensor(B["state"][idx]); g = torch.as_tensor(B["goal"][idx])
    ae = B["a_exact"][idx]; tgt = _targets(B["key"][idx])
    perm = rng.permutation(len(idx))
    cross = np.array([[a for a in range(6) if GIDX[a] != GIDX[int(x)]][rng.integers(4)] for x in ae], np.int64)
    with torch.no_grad():
        base = float(nce_loss(critic.logits_matrix(s, torch.as_tensor(ae), g), tgt))
        shuf = float(nce_loss(critic.logits_matrix(s, torch.as_tensor(ae[perm]), g), tgt))
        crs = float(nce_loss(critic.logits_matrix(s, torch.as_tensor(cross), g), tgt))
    return {"nce_correct": base, "action_shuffle_delta": shuf - base, "cross_group_replace_delta": crs - base}


# --------------------------------------------------------------------------- #
# Step 4 — emulator-branch evaluation (EVAL ONLY; reuse Stage-1H cache)
# --------------------------------------------------------------------------- #
def emul_eval(critic, cache, base_ids, hi, stats, seed) -> Dict[str, Any]:
    B = h_build_branches(cache, base_ids, hi, stats)
    ev = h_evaluate(critic, B, cache, base_ids, hi, stats, seed)             # top1/pairwise/margin/diversity/corr (base bootstrap)
    with torch.no_grad():
        sc = canon_logits(critic, torch.as_tensor(B["state"]), torch.as_tensor(B["goal"])).numpy()
        sc6 = critic.scores_all_actions(torch.as_tensor(B["state"]), torch.as_tensor(B["goal"])).numpy()
    pred = sc.argmax(1); y = B["group"]
    conf = np.zeros((3, 3), int)
    for t, p in zip(y, pred):
        conf[t, p] += 1
    recall = [float(conf[k, k] / conf[k].sum()) if conf[k].sum() else float("nan") for k in range(3)]
    # alias consistency: within-group |alias score diff| vs canonical between-group spread
    within = np.mean([np.abs(sc6[:, 0] - sc6[:, 1]), np.abs(sc6[:, 2] - sc6[:, 4]), np.abs(sc6[:, 3] - sc6[:, 5])])
    between = float(sc[:, [0, 1, 2]].std(1).mean())
    # action-shuffle (random group label) margin delta
    rng = np.random.default_rng(seed + 9); yperm = rng.permutation(y)
    m_true = np.mean([sc[i, y[i]] - np.mean([sc[i, j] for j in range(3) if j != y[i]]) for i in range(len(y))])
    m_shuf = np.mean([sc[i, yperm[i]] - np.mean([sc[i, j] for j in range(3) if j != yperm[i]]) for i in range(len(y))])
    ev["confusion_true_pred"] = conf.tolist(); ev["per_group_recall"] = recall
    ev["alias_within_vs_between"] = {"within_group_alias_meandiff": float(within), "between_group_spread": between}
    ev["action_shuffle_margin_delta"] = float(m_true - m_shuf)
    ev["n_base_states_eval"] = int(len(base_ids))
    return ev


def gate_1i(dec_ev, ord_ev) -> Dict[str, Any]:
    t = dec_ev["semantic_top1"]; p = dec_ev["pairwise"]; m = dec_ev["margin"]
    om = ord_ev["margin"]; ot = ord_ev["semantic_top1"]
    ord_margin_includes0 = bool(om["ci95"][0] <= 0 <= om["ci95"][1])
    ord_safe = bool(ot["point"] <= 0.40 and ord_margin_includes0)
    strict = bool(t["point"] > 0.50 and t.get("ci_excludes_baseline") and p["point"] > 0.60
                  and m["median"] > 0 and m.get("ci_excludes_baseline")
                  and dec_ev["stronger_in_high_diversity"] and ord_safe)
    weak = bool(t["point"] > 0.40 and p["point"] > 0.55 and m["point"] > 0 and ord_safe)
    return {"strict_pass": strict, "weakly_positive": bool(weak and not strict),
            "ordinary_safe": ord_safe, "label": "PASS" if strict else ("WEAK" if weak else "FAIL")}


def behavior_vs_branch(obs_top1, maj_acc, dec_top1, dec_top1_ci_excl) -> Dict[str, Any]:
    high_obs = obs_top1 > maj_acc + 0.05
    high_branch = bool(dec_top1_ci_excl and dec_top1 > 0.40)
    if high_obs and high_branch:
        q = "high_obs+high_branch (predicts behavior AND aligns with real action effect)"
    elif high_obs and not high_branch:
        q = "high_obs+chance_branch => BEHAVIOR-POLICY SHORTCUT (imitation, not action-effect)"
    elif (not high_obs) and high_branch:
        q = "low_obs+high_branch (captures dynamics without modeling teacher choice)"
    else:
        q = "low_obs+low_branch (method failed)"
    return {"obs_action_top1": obs_top1, "majority_acc": maj_acc, "branch_dec_top1": dec_top1,
            "branch_above_baseline": high_branch, "quadrant": q}


# --------------------------------------------------------------------------- #
# Step 5 — toy tests (validate: low training loss != causal alignment)
# --------------------------------------------------------------------------- #
def _toyB(state, a_exact, goal):
    grp = np.array([GIDX[int(a)] for a in a_exact], np.int64)
    keyint = np.round(goal[:, :6] * 10).astype(np.int64)
    key = np.array([hash(tuple(r)) for r in keyint])
    return {"state": state.astype(np.float32), "a_exact": a_exact.astype(np.int64), "group": grp,
            "goal": goal.astype(np.float32), "key": key,
            "weight": np.ones(len(grp), np.float32), "ep": np.arange(len(grp)) // 3, "n": len(grp)}


def _branch_align(critic, states, goals_by_group):
    """states:(K,11); goals_by_group: list of 3 arrays (K,9). Returns semantic top-1 over K*3."""
    correct = 0; tot = 0
    for k in range(states.shape[0]):
        s = torch.as_tensor(np.repeat(states[k][None], 3, 0))
        g = torch.as_tensor(np.stack([goals_by_group[j][k] for j in range(3)]))
        with torch.no_grad():
            sc = canon_logits(critic, s, g).numpy()        # row j = goal from group j
        correct += int(sc[0].argmax() == 0) + int(sc[1].argmax() == 1) + int(sc[2].argmax() == 2); tot += 3
    return correct / tot


def run_toy_tests(seed=0, K=120, steps=500) -> Dict[str, Any]:
    rng = np.random.default_rng(seed); res = {}
    sc = np.array([[2.0, 0.0, 0.0]]); res["margin_sign_positive"] = bool(sc[0, 0] - sc[0, 1:].mean() > 0)
    gmap = np.eye(3, 9).astype(np.float32) * 4.0
    canon = np.array(CANON_IDS)

    # Toy 1: action-conditioned future, random state -> obs high, branch high.
    base = rng.standard_normal((K, D.STATE_DIM)).astype(np.float32)
    grp = np.tile([0, 1, 2], K); states = np.repeat(base, 3, 0)
    g1 = gmap[grp] + rng.standard_normal((3 * K, 9)).astype(np.float32) * 0.05
    B1 = _toyB(states, canon[grp], g1)
    c1, _ = train_critic(B1, B1, seed, "action", steps, 96, eval_every=steps)
    gbg = [gmap[[j] * K] + rng.standard_normal((K, 9)).astype(np.float32) * 0.05 for j in range(3)]
    res["toy1_action_conditioned"] = {"obs_top1": round(obs_action_metrics(c1, B1)["obs_action_top1"], 3),
                                       "branch_align": round(_branch_align(c1, base, gbg), 3)}

    # Toy 2/3: behavior shortcut — STATE predicts action, future action-INDEPENDENT.
    # obs action accuracy can be high (state routes action) yet branch alignment is chance.
    grp_by_state = rng.integers(3, size=K)                        # each base state has a fixed "policy" action
    states3 = np.repeat(base, 3, 0); a3 = canon[np.repeat(grp_by_state, 3)]
    gconst = gmap[[0] * (3 * K)] + rng.standard_normal((3 * K, 9)).astype(np.float32) * 0.05
    B3 = _toyB(states3, a3, gconst)
    c3, _ = train_critic(B3, B3, seed, "action", steps, 96, eval_every=steps)
    gbg_const = [gmap[[0] * K] + rng.standard_normal((K, 9)).astype(np.float32) * 0.05 for _ in range(3)]
    res["toy3_behavior_shortcut"] = {"obs_top1": round(obs_action_metrics(c3, B3)["obs_action_top1"], 3),
                                     "majority_acc": round(float(np.bincount(B3["group"], minlength=3).max() / B3["n"]), 3),
                                     "branch_align": round(_branch_align(c3, base, gbg_const), 3)}

    # Toy 4: state FIXED, action determines future -> must use goal -> branch high.
    fixed = np.repeat(rng.standard_normal((1, D.STATE_DIM)).astype(np.float32), 3 * K, 0)
    B4 = _toyB(fixed, canon[grp], g1)
    c4, _ = train_critic(B4, B4, seed, "action", steps, 96, eval_every=steps)
    res["toy4_state_fixed_action_future"] = {"obs_top1": round(obs_action_metrics(c4, B4)["obs_action_top1"], 3),
                                             "branch_align": round(_branch_align(c4, fixed[:K], gbg), 3)}

    # Toy 5: permuted observed labels -> branch alignment destroyed.
    permg = rng.permutation(grp)
    B5 = _toyB(states, canon[permg], g1)
    c5, _ = train_critic(B5, B5, seed, "action", steps, 96, eval_every=steps)
    res["toy5_permuted_labels"] = {"obs_top1": round(obs_action_metrics(c5, B5)["obs_action_top1"], 3),
                                   "branch_align": round(_branch_align(c5, base, gbg), 3)}

    res["passed"] = bool(res["margin_sign_positive"]
                         and res["toy1_action_conditioned"]["branch_align"] > 0.9
                         and res["toy3_behavior_shortcut"]["obs_top1"] > 0.8
                         and res["toy3_behavior_shortcut"]["branch_align"] < 0.45
                         and res["toy4_state_fixed_action_future"]["branch_align"] > 0.9
                         and res["toy5_permuted_labels"]["branch_align"] < 0.45)
    return res


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def split_episodes3(n, seed, fracs=(0.7, 0.15, 0.15)):
    rng = np.random.default_rng(seed); perm = rng.permutation(n)
    a = int(round(n * fracs[0])); b = a + int(round(n * fracs[1]))
    return sorted(perm[:a].tolist()), sorted(perm[a:b].tolist()), sorted(perm[b:].tolist())


def _sha(critic) -> str:
    return hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in critic.state_dict().values())).hexdigest()[:16]


MODELS = ["nce", "action", "combined", "weighted"]


def run_seed(eps, n_episodes, seed, steps, batch, cache, dec_clone_ids, ord_clone_ids) -> Dict[str, Any]:
    tr_ep, va_ep, te_ep = split_episodes3(n_episodes, seed)
    assert not (set(tr_ep) & set(va_ep)) and not (set(tr_ep) & set(te_ep)) and not (set(va_ep) & set(te_ep))
    sdir = RESULTS / f"seed{seed}"; sdir.mkdir(parents=True, exist_ok=True)
    out = {"seed": seed, "episode_split": {"train": tr_ep, "val": va_ep, "test": te_ep}, "by_horizon": {}}
    for hi, H in enumerate(HORIZONS):
        # observational datasets (decision primary; ordinary control)
        tr = build_obs_raw(eps, tr_ep, H, decision_only=True)
        stats = fit_stats(tr, np.arange(tr["n"]))                       # TRAIN ONLY
        tr = finalize(tr, stats)
        va = finalize(build_obs_raw(eps, va_ep, H, True), stats)
        te = finalize(build_obs_raw(eps, te_ep, H, True), stats)
        tr_o = finalize(build_obs_raw(eps, tr_ep, H, decision_only=False), stats)
        va_o = finalize(build_obs_raw(eps, va_ep, H, False), stats)
        manifest = {"decision": {k: dup_and_support(b) for k, b in [("train", tr), ("val", va), ("test", te)]},
                    "ordinary": {"train": dup_and_support(tr_o), "val": dup_and_support(va_o)},
                    "near_zero_var_dims": [i for i, s in enumerate(stats["std"]) if s < 1e-3]}
        # diagnostics: state-only and state+goal action classifiers (no leakage: clf sees state[+goal], not action)
        _, acc_so = train_clf(tr, va, seed, use_goal=False)
        _, acc_sg = train_clf(tr, va, seed, use_goal=True)
        diag = {"state_only_action_acc_test": acc_so(te), "state_goal_action_acc_test": acc_sg(te),
                "goal_added_improvement": acc_sg(te) - acc_so(te),
                "majority_acc_test": float(np.bincount(te["group"], minlength=3).max() / te["n"])}
        hres = {"normalization": stats, "dataset_manifest": manifest, "diagnostics": diag, "models": {}}
        for kind in MODELS:
            critic, sel = train_critic(tr, va, seed, kind, steps, batch)
            obsm = obs_action_metrics(critic, te)
            deltas = shuffle_cross_deltas(critic, va, seed)
            dec_ev = emul_eval(critic, cache, dec_clone_ids, hi, stats, seed)
            ord_ev = emul_eval(critic, cache, ord_clone_ids, hi, stats, seed)
            g = gate_1i(dec_ev, ord_ev)
            bvb = behavior_vs_branch(obsm["obs_action_top1"], obsm["majority_acc"],
                                     dec_ev["semantic_top1"]["point"], dec_ev["semantic_top1"].get("ci_excludes_baseline"))
            torch.save(critic.state_dict(), sdir / f"{kind}_H{H}.pt")
            hres["models"][kind] = {"selected_step": sel["selected_step"], "sha16": _sha(critic),
                                    "obs_action": obsm, "obs_nce_deltas": deltas,
                                    "emulator_decision": dec_ev, "emulator_ordinary": ord_ev,
                                    "gate": g, "behavior_vs_branch": bvb}
        out["by_horizon"][str(H)] = hres
    (sdir / "report.json").write_text(json.dumps(out, indent=2))
    return out


def run(seeds, n_episodes, steps, batch) -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    eps = D.load_subset("full", list(range(n_episodes)), with_pixels=False)
    aud = audit(eps, list(range(n_episodes)))
    toy = run_toy_tests()
    cache = generate_or_load_branches(BRANCH_STATE_SEED, BRANCH_N_DEC, BRANCH_N_ORD)   # EVAL-ONLY (cache hit)
    assert "raw6" in cache, "branch cache missing"
    dec_clone_ids = np.where(cache["decision"])[0]; ord_clone_ids = np.where(~cache["decision"])[0]
    (RESULTS / "audit_report.json").write_text(json.dumps(aud, indent=2))
    (RESULTS / "toy_test_report.json").write_text(json.dumps(toy, indent=2))

    per_seed = {}
    s0 = run_seed(eps, n_episodes, seeds[0], steps, batch, cache, dec_clone_ids, ord_clone_ids)
    per_seed[str(seeds[0])] = s0

    def non_baseline_promising(sd):
        for H in HORIZONS:
            for k in ("action", "combined", "weighted"):
                g = sd["by_horizon"][str(H)]["models"][k]["gate"]
                if g["label"] in ("PASS", "WEAK") and g["ordinary_safe"]:
                    return True
        return False

    ran = [seeds[0]]
    if non_baseline_promising(s0):
        for seed in seeds[1:]:
            per_seed[str(seed)] = run_seed(eps, n_episodes, seed, steps, batch, cache, dec_clone_ids, ord_clone_ids)
            ran.append(seed)

    out = {"milestone": "Stage-1I-observational-action-contrast",
           "goal_dim": 9, "future_self_paddle_in_goal": False, "horizon_input_to_model": False,
           "horizons_separate": HORIZONS, "canonical_actions": {GROUP_NAMES[i]: CANON_IDS[i] for i in range(3)},
           "lambda": LAMBDA, "temperature": TAU, "training_data": "observational only",
           "emulator_branches": "evaluation only (Stage-1H cache)", "branch_cache": cache["cache_path"],
           "config_hash": cache["config_hash"], "audit": aud, "toy_tests": toy,
           "random_baselines": {"semantic": SEM_BASE, "pairwise": 0.5}, "seeds_run": ran, "per_seed": per_seed,
           "interpretation_note": "Heuristic same-state negatives; NO causal identification claimed. Emulator "
                                  "alignment (not training accuracy) is the evidence. Outcomes A/B/C/D/E."}
    (RESULTS / "config.json").write_text(json.dumps({"seeds": seeds, "n_episodes": n_episodes, "steps": steps,
                                                     "batch": batch, "horizons": HORIZONS, "lambda": LAMBDA,
                                                     "tau": TAU, "branch_state_seed": BRANCH_STATE_SEED}, indent=2))
    (RESULTS / "stage1i_report.json").write_text(json.dumps(out, indent=2))
    _summary(out)
    return out


def _summary(out) -> None:
    L = ["Stage-1I observational same-state action contrast. sem base 0.333, pairwise 0.5. TRAIN=observational; "
         "emulator branches=EVAL ONLY.",
         f"canonical {out['canonical_actions']}; H={out['horizons_separate']} separate; goal_dim=9 no-self; H not input.\n",
         f"toy passed: {out['toy_tests']['passed']} | shortcut toy: obs={out['toy_tests']['toy3_behavior_shortcut']['obs_top1']} "
         f"branch={out['toy_tests']['toy3_behavior_shortcut']['branch_align']} (high obs + chance branch = shortcut detector works)\n"]
    for s, d in out["per_seed"].items():
        L.append(f"== seed {s} ==")
        for H in out["horizons_separate"]:
            h = d["by_horizon"][str(H)]; dg = h["diagnostics"]
            L.append(f" H{H}: state_only_act_acc={dg['state_only_action_acc_test']:.3f} "
                     f"state+goal={dg['state_goal_action_acc_test']:.3f} (goal_add={dg['goal_added_improvement']:+.3f}) "
                     f"majority={dg['majority_acc_test']:.3f}")
            for k in MODELS:
                m = h["models"][k]; de = m["emulator_decision"]; oe = m["emulator_ordinary"]
                t = de["semantic_top1"]; st = "*" if t.get("ci_excludes_baseline") else ""
                L.append(f"   {k:>9}: obsAcc={m['obs_action']['obs_action_top1']:.3f} | "
                         f"DEC top1={t['point']:.3f}{st} pair={de['pairwise']['point']:.3f} "
                         f"marg={de['margin']['point']:+.3f}[{de['margin']['ci95'][0]:+.2f},{de['margin']['ci95'][1]:+.2f}] "
                         f"hi-lo={(de['by_diversity']['high_div_margin'] or 0)-(de['by_diversity']['low_div_margin'] or 0):+.2f} | "
                         f"ORD top1={oe['semantic_top1']['point']:.3f} marg={oe['margin']['point']:+.3f} -> "
                         f"{m['gate']['label']} | {m['behavior_vs_branch']['quadrant'].split('(')[0].strip()}")
        L.append("")
    L.append(f"seeds run: {out['seeds_run']}")
    (RESULTS / "summary.txt").write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-1I observational same-state action contrast.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-episodes", type=int, default=80)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--toy-only", action="store_true")
    ap.add_argument("--audit-only", action="store_true")
    args = ap.parse_args()
    if args.toy_only:
        print(json.dumps(run_toy_tests(), indent=2)); return
    if args.audit_only:
        eps = D.load_subset("full", list(range(args.n_episodes)), with_pixels=False)
        print(json.dumps(audit(eps, list(range(args.n_episodes))), indent=2)); return
    out = run(args.seeds, args.n_episodes, args.steps, args.batch)
    print(open(RESULTS / "summary.txt").read())


if __name__ == "__main__":
    main()
