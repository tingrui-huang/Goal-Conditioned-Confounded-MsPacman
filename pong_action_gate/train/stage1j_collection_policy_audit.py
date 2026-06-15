"""Stage-1J — collection-policy audit: does a small GLOBAL epsilon-mixture improve local
action overlap without breaking the confounded-demonstrator design?

Stage-1I found that the T=2.0 collector's teacher action is ~0.83 predictable from the
learner-visible state -> poor LOCAL conditional action overlap (few same-state alternative
actions), which is why observational action contrast collapses to behavior imitation.

This stage is a COLLECTION-POLICY PILOT + AUDIT only (no critic / no learner training). The only
allowed change is a single global, state-independent epsilon-mixture:
    pi_b^eps(a|s,u) = (1-eps) pi_teacher,T=2(a|s,u) + eps Uniform(A_exact)
evaluated at eps in {0.0, 0.1, 0.2}. The teacher (checkpoint/obs/recurrence/T/preprocessing) is
unchanged; the exploration coin is global, per-decision, uniform over the 6 EXACT ALE actions, and
never exposed to the learner. We audit four properties per epsilon:
  A. local conditional action overlap (propensity, kNN, cross-action NN, effective support)
  B. teacher competence (episode-level, bootstrap CIs)
  C. hidden-information dependence S->A vs (S,U)->A  (U = opponent paddle, the masking target)
  D. external action-effect preservation (Stage-1F clone/restore CRN branch diversity + decoder)

S (learner-visible) EXCLUDES the opponent paddle; U (hidden) IS the opponent paddle. Improving
overlap does NOT identify P(G|S,do(A)); this stage only checks alternative-action support.
No critic-based selection. Nothing committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.distributions.categorical import Categorical

from .. import config as C
from ..objects import extract_pong_objects
from ..teacher.load_teacher import TeacherPolicy, load_teacher, make_env
from .emulator_branch import _feat_single, restore_env, snapshot_env
from .stage1e_branch_alignment import _gray_stub
from .stage1f_horizon_scan import diversity_ratio_state, goal_at, nearest_centroid_state, normalize, run_to_hmax
from .stage1h_interventional_branch_critic import run_to_H

RESULTS = Path("pong_action_gate/results/stage1j_collection_policy_audit")
CACHE = Path("artifacts/pong_action_gate/stage1j")          # gitignored raw trajectories + branch caches

EPSILONS = [0.0, 0.1, 0.2]
GIDX = {0: 0, 1: 0, 2: 1, 4: 1, 3: 2, 5: 2}                  # exact ALE id -> semantic group
GROUP_NAMES = ["STAY", "UP", "DOWN"]
CANON_IDS = [0, 2, 3]
# 11-dim _feat_single layout: 0 ball_x,1 ball_y,2 ball_present,3 ball_dx,4 ball_dy,5 player_y,
#                             6 player_dy,7 opp_y,8 opp_present,9 opp_dy,10 score_diff
S_COLS = [0, 1, 2, 3, 4, 5, 6, 10]                          # learner-visible (NO opponent)
U_COLS = [7, 8, 9]                                          # hidden opponent paddle (masking target)
BR_H = [12, 16]
BR_R = 4                                                     # CRN reps per (clone, canonical action)
SEM_BASE = 1.0 / 3
GEN = "v1"


# --------------------------------------------------------------------------- #
# Step 0 — audit
# --------------------------------------------------------------------------- #
def audit_report(meta, action_meanings) -> Dict[str, Any]:
    return {
        "teacher_action_path": "logits,_,(hx,cx)=model.predict_act_value(obs[:,4:7],(hx,cx)); "
                               "a=Categorical(logits=logits / BEHAVIOR_TEMPERATURE).sample()",
        "temperature_T": C.BEHAVIOR_TEMPERATURE, "temperature_location": "logits / T before Categorical",
        "sampling": "categorical (NOT argmax)",
        "recurrent_advance": "predict_act_value called every step on obs[:,4:7]; hidden advances from "
                             "observation only (independent of which action is executed)",
        "teacher_obs_slice": [C.TEACHER_OBS_SLICE.start, C.TEACHER_OBS_SLICE.stop],
        "ale_action_meanings": list(action_meanings), "n_actions": C.N_ACTIONS,
        "semantic_groups": {"STAY": [0, 1], "UP": [2, 4], "DOWN": [3, 5]},
        "canonical_per_group": dict(zip(GROUP_NAMES, CANON_IDS)),
        "alias_note": "STAY=NOOP/FIRE, UP=RIGHT/RIGHTFIRE, DOWN=LEFT/LEFTFIRE; no UPFIRE/DOWNFIRE. "
                      "Uniform over 6 EXACT actions => equal mass on the 3 groups (2 aliases each).",
        "reset_life": {"done_on_life_loss": C.M1Config().done_on_life_loss,
                       "note": "Pong has no ALE lives; episodes end on game over / truncation"},
        "frame_skip": C.FRAME_SKIP, "num_stack": C.NUM_STACK, "img_size": C.IMG_SIZE,
        "decision_state_selector": "ball_present & ball moving right (dx>0) & ball_x>=120 "
                                   "(Stage-1F/1H clone selector; ball-only, computable from S)",
        "learner_visible_S": {"cols": S_COLS, "names": ["ball_x", "ball_y", "ball_present", "ball_dx",
                              "ball_dy", "player_y", "player_dy", "score_diff"]},
        "hidden_U": {"cols": U_COLS, "names": ["opp_y", "opp_present", "opp_dy"],
                     "role": "confounder/masking target (full-obs teacher sees it; learner will not)"},
        "existing_exploration_in_action_path": False,
        "teacher_meta_audit_only": {"ckpt_sha256": meta["ckpt_sha256"], "n_params": meta["n_actor_critic_params"],
                                    "arch": meta["arch"]},
        "note": "epsilon is the ONLY changed mechanism; teacher config hash identical across epsilon.",
    }


def _teacher_cfg_hash(meta) -> str:
    blob = json.dumps({"sha": meta["ckpt_sha256"], "arch": meta["arch"], "T": C.BEHAVIOR_TEMPERATURE,
                       "slice": [C.TEACHER_OBS_SLICE.start, C.TEACHER_OBS_SLICE.stop],
                       "frame_skip": C.FRAME_SKIP, "num_stack": C.NUM_STACK}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Step 1 — collection with global epsilon-mixture
# --------------------------------------------------------------------------- #
def collect_epsilon(teacher, env, epsilon, seeds, snap_stride, max_snaps, max_steps=4000) -> Dict[str, Any]:
    ale = env.env.ale
    rows = {k: [] for k in ["ep", "t", "a_exact", "group", "src", "probs", "S", "U",
                            "reward", "agent", "opp", "term", "trunc", "life", "decision", "ball_vis", "vel"]}
    ep_metrics = []
    snaps = []
    dec_count = 0
    for ei, seed in enumerate(seeds):
        torch.manual_seed(seed)
        coin = np.random.default_rng(seed + 999983)            # coin+random-action stream (same across eps)
        obs, info = env.reset(seed=[seed]); hx, cx = teacher.initial_state(1)
        prev = extract_pong_objects(ale.getRAM()); ag = op = 0; t = 0
        ret = 0.0; max_diff = 0; min_diff = 0; ag_pts = op_pts = 0; lifes = 0
        last_score_t = 0; rallies = []; capped = False
        ep_start_idx = len(rows["ep"])
        while True:
            o = extract_pong_objects(ale.getRAM())
            full = _feat_single(o, prev, int(ag - op))
            bdx = (o.ball_x - prev.ball_x) if (o.ball_present and prev.ball_present) else None
            is_dec = bool(o.ball_present and bdx is not None and bdx > 0 and o.ball_x is not None and o.ball_x >= 120)
            with torch.no_grad():
                logits, _, (hx2, cx2) = teacher.model.predict_act_value(obs[:, C.TEACHER_OBS_SLICE], (hx, cx))
                probs = torch.softmax(logits / C.BEHAVIOR_TEMPERATURE, dim=-1)[0].numpy()
                u = coin.random()
                if u < epsilon:
                    a = int(coin.integers(C.N_ACTIONS)); src = 1            # uniform over 6 EXACT actions
                else:
                    a = int(Categorical(logits=logits / C.BEHAVIOR_TEMPERATURE).sample().item()); src = 0
            if is_dec and (dec_count % snap_stride == 0) and len(snaps) < max_snaps:
                sn = snapshot_env(env, hx, cx, ag, op, obs[:, C.TEACHER_OBS_SLICE], _gray_stub(), o, prev)
                sn["decision"] = True; sn["anchor_state"] = full
                snaps.append(sn)
            if is_dec:
                dec_count += 1
            hx, cx = hx2, cx2
            obs2, rew, end, trunc, info = env.step(a)
            r = float(rew.item()); ag += int(r > 0); op += int(r < 0); ret += r
            if r != 0:
                rallies.append(t - last_score_t); last_score_t = t
                ag_pts += int(r > 0); op_pts += int(r < 0)
            max_diff = max(max_diff, ag - op); min_diff = min(min_diff, ag - op)
            life = bool(info.get("life_loss", False)); lifes += int(life)
            for k, v in [("ep", ei), ("t", t), ("a_exact", a), ("group", GIDX[a]), ("src", src),
                         ("probs", probs), ("S", full[S_COLS]), ("U", full[U_COLS]), ("reward", r),
                         ("agent", ag), ("opp", op), ("term", bool(end.item())), ("trunc", bool(trunc.item())),
                         ("life", life), ("decision", is_dec), ("ball_vis", bool(o.ball_present)),
                         ("vel", bool(bdx is not None))]:
                rows[k].append(v)
            prev = o; obs = obs2; t += 1
            if bool((end | trunc).item()) or t >= max_steps:
                capped = bool(t >= max_steps and not bool((end | trunc).item()))
                break
        ep_metrics.append({"ep": ei, "seed": seed, "len": t, "return": ret, "win": int(ret > 0),
                           "capped": int(capped),
                           "loss": int(ret < 0), "max_score_diff": int(max_diff), "min_score_diff": int(min_diff),
                           "reach15": int(max_diff >= 15), "agent_points": ag_pts, "opp_points": op_pts,
                           "life_loss": lifes, "truncated": int(bool(trunc.item())), "terminated": int(bool(end.item())),
                           "mean_rally": float(np.mean(rallies)) if rallies else float(t),
                           "median_rally": float(np.median(rallies)) if rallies else float(t),
                           "n_rows": len(rows["ep"]) - ep_start_idx})
    out = {k: np.array(v) for k, v in rows.items()}
    out["episode_metrics"] = ep_metrics
    return out, snaps


# --------------------------------------------------------------------------- #
# B — competence (episode-level bootstrap)
# --------------------------------------------------------------------------- #
def _boot_ci(vals, seed=0, n_boot=2000):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return None
    rng = np.random.default_rng(seed)
    bs = [vals[rng.integers(len(vals), size=len(vals))].mean() for _ in range(n_boot)]
    return {"mean": float(vals.mean()), "median": float(np.median(vals)), "std": float(vals.std()),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def competence(data) -> Dict[str, Any]:
    em = data["episode_metrics"]; g = lambda k: np.array([m[k] for m in em], float)
    src = data["src"]; ex = np.bincount(data["a_exact"], minlength=6); se = np.bincount(data["group"], minlength=3)
    return {"n_episodes": len(em),
            "return": _boot_ci(g("return")), "win_rate": _boot_ci(g("win")), "loss_rate": _boot_ci(g("loss")),
            "reach15_rate": _boot_ci(g("reach15")), "final_score_diff": _boot_ci(g("max_score_diff")),
            "mean_rally": _boot_ci(g("mean_rally")), "median_rally_per_ep": _boot_ci(g("median_rally")),
            "agent_points": _boot_ci(g("agent_points")), "opp_points": _boot_ci(g("opp_points")),
            "episode_len": _boot_ci(g("len")), "life_loss_total": int(g("life_loss").sum()),
            "truncation_rate": float(g("truncated").mean()), "termination_rate": float(g("terminated").mean()),
            "episode_cap_rate": float(g("capped").mean()) if em and "capped" in em[0] else 0.0,
            "action_source": {"teacher": float((src == 0).mean()), "epsilon_random": float((src == 1).mean())},
            "exact_action_freq": (ex / ex.sum()).round(4).tolist(),
            "semantic_action_freq": (se / se.sum()).round(4).tolist()}


# --------------------------------------------------------------------------- #
# A/C — propensity, overlap, hidden-information (sklearn, episode-disjoint cross-fit)
# --------------------------------------------------------------------------- #
def _crossfit_oof(X, y, groups, seed=0, n_splits=3):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    oof = np.full((len(y), 3), 1 / 3)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        clf = HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.1,
                                             l2_regularization=1.0, random_state=seed)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])
        tmp = np.full((len(te), 3), 1e-6)
        for j, cls in enumerate(clf.classes_):
            tmp[:, cls] = p[:, j]
        oof[te] = tmp / tmp.sum(1, keepdims=True)
    return oof


def _clf_metrics(oof, y) -> Dict[str, Any]:
    pred = oof.argmax(1); n = len(y)
    recalls = [float((pred[y == k] == k).mean()) if (y == k).any() else None for k in range(3)]
    onehot = np.eye(3)[y]
    nll = float(-np.mean(np.log(np.clip(oof[np.arange(n), y], 1e-9, 1))))
    brier = float(np.mean(np.sum((oof - onehot) ** 2, 1)))
    conf = np.clip(oof.max(1), 0, 1); correct = (pred == y).astype(float)
    ece = 0.0
    for lo in np.linspace(0, 1, 11)[:-1]:
        m = (conf >= lo) & (conf < lo + 0.1)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    maj = int(np.bincount(y, minlength=3).argmax())
    return {"top1": float((pred == y).mean()), "balanced_acc": float(np.nanmean([r for r in recalls if r is not None])),
            "nll": nll, "brier": brier, "ece": float(ece), "majority_acc": float((y == maj).mean()),
            "per_action_recall": recalls}


def action_support(data, seed=0) -> Dict[str, Any]:
    S = data["S"]; y = data["group"]; ep = data["ep"]; dec = data["decision"]
    oof = _crossfit_oof(S, y, ep, seed=seed)
    out = {"state_only_propensity": {"all": _clf_metrics(oof, y),
                                     "decision": _clf_metrics(oof[dec], y[dec]),
                                     "ordinary": _clf_metrics(oof[~dec], y[~dec])}}
    # A2 propensity distribution by subset
    def prop_dist(mask):
        p = oof[mask]; yy = y[mask]
        if len(yy) == 0:
            return None
        pobs = p[np.arange(len(yy)), yy]; pmin = p.min(1); pmax = p.max(1)
        ent = -np.sum(p * np.log(np.clip(p, 1e-9, 1)), 1)
        return {"n": int(len(yy)), "p_obs_mean": float(pobs.mean()), "p_obs_median": float(np.median(pobs)),
                "min_prop_mean": float(pmin.mean()), "max_prop_mean": float(pmax.mean()),
                "action_entropy_mean": float(ent.mean()), "norm_entropy_mean": float(ent.mean() / np.log(3)),
                "frac_min_lt_0.01": float((pmin < 0.01).mean()), "frac_min_lt_0.05": float((pmin < 0.05).mean()),
                "frac_min_lt_0.10": float((pmin < 0.10).mean()),
                "frac_max_gt_0.90": float((pmax > 0.90).mean()), "frac_max_gt_0.80": float((pmax > 0.80).mean())}
    out["propensity_distribution"] = {"all": prop_dist(np.ones(len(y), bool)),
                                      "decision": prop_dist(dec), "ordinary": prop_dist(~dec)}
    # A5 effective sample support (per action, inverse-propensity, clipped)
    ess = {}
    for k in range(3):
        m = y == k
        if m.sum() == 0:
            continue
        w = 1.0 / np.clip(oof[m, k], 0.02, 1.0)
        ess[GROUP_NAMES[k]] = {"n": int(m.sum()), "ess": float(w.sum() ** 2 / (w ** 2).sum()),
                               "ess_frac": float((w.sum() ** 2 / (w ** 2).sum()) / m.sum()), "clip_min": 0.02}
    out["effective_sample_support"] = ess
    out["oof_propensity"] = oof                                  # returned for caching predictions
    return out


def knn_overlap(data, seed=0, ks=(25, 50), n_query=800, window=8) -> Dict[str, Any]:
    from sklearn.neighbors import NearestNeighbors
    S = data["S"]; y = data["group"]; ep = data["ep"]; t = data["t"]; dec = data["decision"]
    tr_ep = np.array(sorted(set(ep.tolist()))[: max(1, int(0.7 * len(set(ep.tolist()))))])
    trm = np.isin(ep, tr_ep)
    mu = S[trm].mean(0); sd = S[trm].std(0) + 1e-6
    Z = (S - mu) / sd
    dec_idx = np.where(dec)[0]
    rng = np.random.default_rng(seed)
    q = dec_idx if len(dec_idx) <= n_query else rng.choice(dec_idx, n_query, replace=False)
    K = max(ks) + 3 * window + 5
    nn = NearestNeighbors(n_neighbors=min(K, len(Z))).fit(Z)
    dist, nbr = nn.kneighbors(Z[q])
    res = {}
    for k in ks:
        groups_present, minority, all3, atleast2, ent = [], [], [], [], []
        cross_ratio = []
        for qi, (ds, ns) in enumerate(zip(dist, nbr)):
            keep = [j for j in ns if not (ep[j] == ep[q[qi]] and abs(int(t[j]) - int(t[q[qi]])) <= window)][:k]
            if len(keep) < k:
                continue
            gl = y[keep]; cnt = np.bincount(gl, minlength=3)
            groups_present.append(int((cnt > 0).sum())); minority.append(int(cnt.min()))
            all3.append(int((cnt > 0).sum() == 3)); atleast2.append(int((cnt > 0).sum() >= 2))
            p = cnt / cnt.sum(); ent.append(float(-np.sum(p[p > 0] * np.log(p[p > 0])) / np.log(3)))
        res[f"k={k}"] = {"n_query": len(groups_present),
                         "mean_groups_present": float(np.mean(groups_present)),
                         "mean_minority_count": float(np.mean(minority)),
                         "frac_all3_groups": float(np.mean(all3)), "frac_ge2_groups": float(np.mean(atleast2)),
                         "mean_local_norm_entropy": float(np.mean(ent))}
    # A4 cross-action nearest-neighbor distance (normalized Z), aggregate by observed action
    per_group_nn = {}
    idx_by_g = {k: np.where(y == k)[0] for k in range(3)}
    nn_by_g = {k: NearestNeighbors(n_neighbors=2).fit(Z[idx_by_g[k]]) for k in range(3) if len(idx_by_g[k]) > 2}
    for k in range(3):
        if k not in nn_by_g:
            continue
        qg = q[y[q] == k]
        if len(qg) == 0:
            continue
        same = nn_by_g[k].kneighbors(Z[qg])[0][:, 1]            # skip self
        cross = []
        for kk in range(3):
            if kk != k and kk in nn_by_g:
                cross.append(nn_by_g[kk].kneighbors(Z[qg])[0][:, 0])
        cross = np.min(cross, 0) if cross else np.full(len(qg), np.nan)
        per_group_nn[GROUP_NAMES[k]] = {"n": int(len(qg)), "same_action_dist": float(np.mean(same)),
                                        "cross_action_dist": float(np.nanmean(cross)),
                                        "cross_over_same_ratio": float(np.nanmean(cross) / (np.mean(same) + 1e-9))}
    res["cross_action_nn"] = per_group_nn
    res["overall_cross_over_same_ratio"] = float(np.nanmean([v["cross_over_same_ratio"] for v in per_group_nn.values()])) if per_group_nn else None
    return res


def hidden_information(data, seed=0) -> Dict[str, Any]:
    S = data["S"]; U = data["U"]; y = data["group"]; ep = data["ep"]; dec = data["decision"]
    SU = np.concatenate([S, U], 1)
    oof_s = _crossfit_oof(S, y, ep, seed=seed)
    oof_su = _crossfit_oof(SU, y, ep, seed=seed)
    # permutation importance of U: shuffle U rows, recompute via fresh crossfit
    rngp = np.random.default_rng(seed + 7)
    SUp = np.concatenate([S, U[rngp.permutation(len(U))]], 1)
    oof_sup = _crossfit_oof(SUp, y, ep, seed=seed)

    def block(mask):
        ms = _clf_metrics(oof_s[mask], y[mask]); msu = _clf_metrics(oof_su[mask], y[mask])
        msup = _clf_metrics(oof_sup[mask], y[mask])
        return {"S_only": ms, "S_plus_U": msu, "S_plus_U_permuted": msup,
                "U_nll_reduction": ms["nll"] - msu["nll"], "U_acc_improvement": msu["top1"] - ms["top1"],
                "U_permutation_importance_nll": msup["nll"] - msu["nll"]}
    return {"all": block(np.ones(len(y), bool)), "decision": block(dec), "ordinary": block(~dec),
            "per_action_U_recall_gain": [
                float((_clf_metrics(oof_su, y)["per_action_recall"][k] or 0) - (_clf_metrics(oof_s, y)["per_action_recall"][k] or 0))
                for k in range(3)]}


# --------------------------------------------------------------------------- #
# D — external action-effect preservation (Stage-1F clone/restore CRN branches)
# --------------------------------------------------------------------------- #
def _nc_decode(byA, n_train) -> float:
    """Replicate-split nearest-centroid semantic decoder for one clone state (BR_R-compatible)."""
    trX, trY, teX, teY = [], [], [], []
    for gi, a in enumerate(CANON_IDS):
        for ri, v in enumerate(byA[a]):
            (trX if ri < n_train else teX).append(v); (trY if ri < n_train else teY).append(gi)
    if not teX or len(set(trY)) < 2:
        return None
    cents = {gi: np.mean([trX[i] for i in range(len(trX)) if trY[i] == gi], 0)
             for gi in set(trY)}
    gl = list(cents); correct = 0
    for x, yy in zip(teX, teY):
        pred = gl[int(np.argmin([np.sum((x - cents[gi]) ** 2) for gi in gl]))]
        correct += int(pred == yy)
    return correct / len(teX)


def branch_effect(teacher, env, snaps, seed=0) -> Dict[str, Any]:
    Hmax = max(BR_H)
    per_h = {str(H): {"ratios": [], "decoder": [], "cross_dist": [], "nonzero": [], "n": 0} for H in BR_H}
    cens = 0; total = 0
    for sn in snaps:
        gba = {a: {str(H): [] for H in BR_H} for a in CANON_IDS}
        for rep in range(BR_R):
            rep_seed = 90000 + rep
            for a in CANON_IDS:
                steps = run_to_H(env, teacher, sn, a, rep_seed, Hmax)
                for H in BR_H:
                    total += 1; g = goal_at(steps, H)
                    if g is None:
                        cens += 1; continue
                    gba[a][str(H)].append(np.concatenate([g["raw6"], g["masks3"]]))
        for H in BR_H:
            byA = {a: gba[a][str(H)] for a in CANON_IDS}
            if sum(len(v) for v in byA.values()) < 6 or any(len(v) == 0 for v in byA.values()):
                continue
            ratio = diversity_ratio_state(byA)
            nc = _nc_decode(byA, n_train=BR_R // 2)
            means = [np.mean(byA[a], 0) for a in CANON_IDS]
            dists = [np.linalg.norm(means[i] - means[j]) for i in range(3) for j in range(i + 1, 3)]
            per_h[str(H)]["ratios"].append(ratio); per_h[str(H)]["cross_dist"].append(float(np.median(dists)))
            per_h[str(H)]["nonzero"].append(float(np.median(dists) > 0))
            if nc is not None:
                per_h[str(H)]["decoder"].append(nc)
            per_h[str(H)]["n"] += 1
    out = {"n_clone_states": len(snaps), "censor_rate": float(cens / max(total, 1)), "per_horizon": {}}
    for H in BR_H:
        d = per_h[str(H)]; r = np.array([x for x in d["ratios"] if x is not None])
        out["per_horizon"][str(H)] = {
            "n_usable": d["n"], "diversity_ratio_median": float(np.median(r)) if len(r) else None,
            "frac_diversity_gt1": float(np.mean(r > 1)) if len(r) else None,
            "median_cross_action_distance": float(np.median(d["cross_dist"])) if d["cross_dist"] else None,
            "frac_nonzero_distance": float(np.mean(d["nonzero"])) if d["nonzero"] else None,
            "decoder_accuracy": _boot_ci(d["decoder"]) if d["decoder"] else None,
            "decoder_baseline": SEM_BASE}
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _cfg_hash(d) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def _reload_eps(eps, chash):
    """Resume: reload a fully-completed epsilon block from disk (no recollection)."""
    edir = RESULTS / f"epsilon_{eps}"; req = ["episode_metrics", "action_support_metrics", "propensity_metrics",
                                             "knn_overlap_metrics", "hidden_information_metrics", "branch_effect_metrics"]
    cpath = CACHE / f"traj_eps{eps}_{chash}.npz"
    if not (cpath.exists() and all((edir / f"{r}.json").exists() for r in req)):
        return None
    j = lambda r: json.loads((edir / f"{r}.json").read_text())
    z = np.load(cpath, allow_pickle=True)
    brn = j("branch_effect_metrics")
    return {"competence": j("episode_metrics"), "support": j("action_support_metrics"),
            "knn": j("knn_overlap_metrics"), "hidden": j("hidden_information_metrics"), "branch": brn,
            "n_snaps": brn["n_clone_states"], "n_rows": int(z["S"].shape[0]), "n_decision": int(z["decision"].sum())}


def run(n_episodes, base_seed, snap_stride, max_snaps, max_steps=4000, device="cpu") -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    teacher_model, meta = load_teacher(device=device)
    teacher = TeacherPolicy(teacher_model, device=device)
    env = make_env(replace(C.M1Config(), device=device), num_envs=1)
    aud = audit_report(meta, env.action_meanings)
    thash = _teacher_cfg_hash(meta)
    seeds = [base_seed + i for i in range(n_episodes)]
    cfg = {"n_episodes": n_episodes, "base_seed": base_seed, "seeds": seeds, "epsilons": EPSILONS,
           "snap_stride": snap_stride, "max_snaps": max_snaps, "teacher_hash": thash, "BR_H": BR_H,
           "BR_R": BR_R, "S_COLS": S_COLS, "U_COLS": U_COLS, "gen": GEN,
           "preprocessing": {"frame_skip": C.FRAME_SKIP, "num_stack": C.NUM_STACK, "img_size": C.IMG_SIZE,
                             "T": C.BEHAVIOR_TEMPERATURE}}
    chash = _cfg_hash(cfg)

    per_eps = {}
    for eps in EPSILONS:
        cached = _reload_eps(str(eps), chash)            # resume completed blocks (e.g. eps=0.0/0.1)
        if cached is not None:
            per_eps[str(eps)] = cached
            continue
        # Collect fresh (live ALE snapshots are needed in-process for branch eval and are not cached).
        # Trajectory arrays are saved under the gitignored cache for provenance/reproducibility.
        data, snaps = collect_epsilon(teacher, env, eps, seeds, snap_stride, max_snaps, max_steps=max_steps)
        cpath = CACHE / f"traj_eps{eps}_{chash}.npz"
        np.savez(cpath, **{k: v for k, v in data.items() if k != "episode_metrics"},
                 episode_metrics=np.array(data["episode_metrics"], dtype=object))
        comp = competence(data)
        supp = action_support(data)
        knn = knn_overlap(data)
        hid = hidden_information(data)
        brn = branch_effect(teacher, env, snaps)
        edir = RESULTS / f"epsilon_{eps}"; edir.mkdir(parents=True, exist_ok=True)
        (edir / "episode_metrics.json").write_text(json.dumps(comp, indent=2))
        (edir / "action_support_metrics.json").write_text(json.dumps({k: v for k, v in supp.items() if k != "oof_propensity"}, indent=2))
        (edir / "propensity_metrics.json").write_text(json.dumps(supp["propensity_distribution"], indent=2))
        (edir / "knn_overlap_metrics.json").write_text(json.dumps(knn, indent=2))
        (edir / "hidden_information_metrics.json").write_text(json.dumps(hid, indent=2))
        (edir / "branch_effect_metrics.json").write_text(json.dumps(brn, indent=2))
        np.savez(edir / "predictions.npz", oof_propensity=supp["oof_propensity"],
                 group=data["group"], decision=data["decision"], ep=data["ep"])
        per_eps[str(eps)] = {"competence": comp, "support": {k: v for k, v in supp.items() if k != "oof_propensity"},
                             "knn": knn, "hidden": hid, "branch": brn, "n_snaps": len(snaps),
                             "n_rows": int(data["S"].shape[0]), "n_decision": int(data["decision"].sum())}
    env.close()

    table = _comparison_table(per_eps)
    outcome = _select_outcome(per_eps)
    out = {"milestone": "Stage-1J-collection-policy-audit", "audit": aud, "config": cfg, "config_hash": chash,
           "teacher_hash_identical_across_epsilon": True, "only_changed_mechanism": "global epsilon-mixture",
           "epsilons": EPSILONS, "per_epsilon": per_eps, "comparison_table": table,
           "outcome": outcome,
           "interpretation_note": "Overlap diagnostics only. Improved overlap does NOT identify P(G|S,do(A)); "
                                  "hidden U may still make P(G|S,A)!=P(G|S,do(A)). No causal claim; no critic selection."}
    (RESULTS / "audit_report.json").write_text(json.dumps(aud, indent=2))
    (RESULTS / "config.json").write_text(json.dumps(cfg, indent=2))
    (RESULTS / "collection_manifest.json").write_text(json.dumps(
        {"config_hash": chash, "teacher_hash": thash, "seeds": seeds, "epsilons": EPSILONS,
         "n_rows_per_eps": {e: per_eps[e]["n_rows"] for e in per_eps},
         "n_decision_per_eps": {e: per_eps[e]["n_decision"] for e in per_eps},
         "cache_dir": str(CACHE), "teacher_ckpt_sha256": meta["ckpt_sha256"]}, indent=2))
    (RESULTS / "comparison_table.json").write_text(json.dumps(table, indent=2))
    (RESULTS / "stage1j_report.json").write_text(json.dumps(out, indent=2))
    _summary(out)
    return out


def _comparison_table(per_eps) -> Dict[str, Any]:
    def row(fn):
        return {e: fn(per_eps[e]) for e in per_eps}
    g = lambda d, *ks: _dig(d, ks)
    return {
        "state_only_action_acc_decision": row(lambda d: d["support"]["state_only_propensity"]["decision"]["top1"]),
        "norm_local_entropy_k50": row(lambda d: d["knn"]["k=50"]["mean_local_norm_entropy"]),
        "frac_neighborhoods_all3_groups_k50": row(lambda d: d["knn"]["k=50"]["frac_all3_groups"]),
        "frac_min_prop_lt0.05_decision": row(lambda d: d["support"]["propensity_distribution"]["decision"]["frac_min_lt_0.05"]),
        "cross_action_nn_ratio": row(lambda d: d["knn"].get("overall_cross_over_same_ratio")),
        "win_rate": row(lambda d: d["competence"]["win_rate"]["mean"]),
        "reach15_rate": row(lambda d: d["competence"]["reach15_rate"]["mean"]),
        "avg_return": row(lambda d: d["competence"]["return"]["mean"]),
        "mean_rally": row(lambda d: d["competence"]["mean_rally"]["mean"]),
        "U_added_nll_improvement_decision": row(lambda d: d["hidden"]["decision"]["U_nll_reduction"]),
        "U_added_acc_improvement_decision": row(lambda d: d["hidden"]["decision"]["U_acc_improvement"]),
        "H12_branch_diversity_gt1": row(lambda d: d["branch"]["per_horizon"]["12"]["frac_diversity_gt1"]),
        "H16_branch_diversity_gt1": row(lambda d: d["branch"]["per_horizon"]["16"]["frac_diversity_gt1"]),
        "H12_branch_decoder_acc": row(lambda d: _dig(d, ("branch", "per_horizon", "12", "decoder_accuracy", "mean"))),
        "H16_branch_decoder_acc": row(lambda d: _dig(d, ("branch", "per_horizon", "16", "decoder_accuracy", "mean"))),
    }


def _dig(d, ks):
    for k in ks:
        if d is None:
            return None
        d = d.get(k) if isinstance(d, dict) else None
    return d


def _select_outcome(per_eps) -> Dict[str, Any]:
    base = per_eps["0.0"]
    base_acc = base["support"]["state_only_propensity"]["decision"]["top1"]
    base_win = base["competence"]["win_rate"]["mean"]
    base_ret = base["competence"]["return"]["mean"]

    def material_overlap(e):
        d = per_eps[e]
        acc_drop = base_acc - d["support"]["state_only_propensity"]["decision"]["top1"]
        ent_up = d["knn"]["k=50"]["mean_local_norm_entropy"] - base["knn"]["k=50"]["mean_local_norm_entropy"]
        lowp_drop = (base["support"]["propensity_distribution"]["decision"]["frac_min_lt_0.05"]
                     - d["support"]["propensity_distribution"]["decision"]["frac_min_lt_0.05"])
        all3_up = d["knn"]["k=50"]["frac_all3_groups"] - base["knn"]["k=50"]["frac_all3_groups"]
        return {"acc_drop": acc_drop, "entropy_up": ent_up, "lowprop_drop": lowp_drop, "all3_up": all3_up,
                "material": bool(acc_drop > 0.05 and ent_up > 0.02 and all3_up > 0.02)}

    def competent(e):
        d = per_eps[e]["competence"]
        return {"win_rate": d["win_rate"]["mean"], "reach15": d["reach15_rate"]["mean"],
                "return": d["return"]["mean"], "mean_rally": d["mean_rally"]["mean"],
                # competence "acceptable" = not a collapse vs eps=0
                "acceptable": bool(d["return"]["mean"] > base_ret - 8 and d["mean_rally"]["mean"] > 3)}

    def hidden_ok(e):
        h = per_eps[e]["hidden"]["decision"]
        return {"U_nll_reduction": h["U_nll_reduction"], "U_acc_improvement": h["U_acc_improvement"],
                "nontrivial": bool(h["U_nll_reduction"] > 0.01 or h["U_acc_improvement"] > 0.01)}

    def effect_ok(e):
        b = per_eps[e]["branch"]["per_horizon"]
        d12 = b["12"]["frac_diversity_gt1"] or 0; d16 = b["16"]["frac_diversity_gt1"] or 0
        return {"H12_div_gt1": d12, "H16_div_gt1": d16, "present": bool(max(d12, d16) > 0.4)}

    checks = {e: {"overlap": material_overlap(e), "competence": competent(e), "hidden": hidden_ok(e),
                  "effect": effect_ok(e)} for e in per_eps}

    def viable(e):
        c = checks[e]
        return c["overlap"]["material"] and c["competence"]["acceptable"] and c["hidden"]["nontrivial"] and c["effect"]["present"]

    if viable("0.1"):
        label = "B"; sel = "0.1"
    elif viable("0.2"):
        label = "C"; sel = "0.2"
    elif not checks["0.0"]["overlap"]["material"] and not viable("0.1") and not viable("0.2") and base_acc < 0.55:
        label = "A"; sel = "0.0"
    elif not viable("0.1") and not viable("0.2"):
        label = "D"; sel = None
    else:
        label = "D"; sel = None
    # A specifically: eps=0 already adequate (low state-only decision accuracy)
    if base_acc < 0.45:
        label = "A"; sel = "0.0"
    return {"label": label, "selected_epsilon": sel, "checks": checks,
            "baseline_state_only_decision_acc": base_acc,
            "note": {"A": "eps=0 already adequate overlap", "B": "eps=0.1 minimal viable",
                     "C": "eps=0.2 minimal viable", "D": "no tested epsilon acceptable",
                     "E": "implementation/audit failure"}[label]}


def _summary(out) -> None:
    t = out["comparison_table"]; eps = [str(e) for e in out["epsilons"]]
    L = ["Stage-1J collection-policy audit (global epsilon-mixture; T=2 teacher unchanged). EVAL-ONLY branches.",
         f"S=learner-visible(no opponent), U=opponent paddle. decision selector = ball approaching & near paddle.\n",
         f"{'metric':<40}" + "".join(f"{e:>12}" for e in eps)]
    for k, row in t.items():
        L.append(f"{k:<40}" + "".join(f"{(row[e] if row[e] is not None else float('nan')):>12.4f}"
                                       if isinstance(row.get(e), (int, float)) else f"{str(row.get(e)):>12}" for e in eps))
    o = out["outcome"]
    L.append(f"\nOUTCOME {o['label']}: {o['note']}  (selected epsilon = {o['selected_epsilon']})")
    L.append(f"baseline state-only DECISION action acc (eps=0): {o['baseline_state_only_decision_acc']:.3f}")
    (RESULTS / "summary.txt").write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-1J collection-policy epsilon audit.")
    ap.add_argument("--n-episodes", type=int, default=60)
    ap.add_argument("--base-seed", type=int, default=20000)
    ap.add_argument("--snap-stride", type=int, default=10)
    ap.add_argument("--max-snaps", type=int, default=80)
    ap.add_argument("--max-ep-steps", type=int, default=4000,
                    help="per-episode agent-step cap (bounds runaway high-epsilon rallies)")
    args = ap.parse_args()
    out = run(args.n_episodes, args.base_seed, args.snap_stride, args.max_snaps, max_steps=args.max_ep_steps)
    print(open(RESULTS / "summary.txt").read())


if __name__ == "__main__":
    main()
