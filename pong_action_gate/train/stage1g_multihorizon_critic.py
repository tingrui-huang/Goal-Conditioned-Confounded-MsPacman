"""Stage-1G — unified multi-horizon external future-state critic for Pong.

Hypothesis: ONE shared contrastive critic f(s_t, a_t, g_f), trained on UNIFORMLY sampled
states and a BALANCED MIXTURE of rich no-self future-state goals over horizons
H in [4,8,12,16,24,32] (horizon-marginalized, H NOT given as input), can keep global state
coverage while learning to use the action specifically in the state/horizon regions where the
environment actually contains an external action effect (decision-focused states, H~12-16) and
stay action-agnostic where different actions produce indistinguishable external futures.

Locks (unchanged vs prior stages): T=2.0 DIAMOND dataset, episode-level split, existing state
encoder (StateSACritic trunk) + action pathway + contrastive sigmoid-BCE, Adam, val-loss
checkpoint selection, exact-duplicate multi-positive targets. NO decision-focused/oversampled
anchors, NO future agent_paddle_y in the goal, NO horizon ID, NO scalar score goal, NO
score-event sampling, NO auxiliary/horizon-classification losses, NO pixel critic, NO masking,
NO Seaquest. The ONLY change vs Stage-1D no-self is the balanced multi-horizon future mixture.

Emulator-branch alignment reuses the Stage-1F environment branches (clone/restore + T=2.0
teacher continuation, CRN), generated ONCE to H_max and cached, then scored with the frozen
multi-horizon critic. Goal layout is identical to Stage-1E/1F no-self (9-dim).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .. import config as C
from ..teacher.load_teacher import TeacherPolicy, load_teacher, make_env
from . import dataset as D
from .critics import nce_loss
from .stage1b import decision_criteria, is_decision_focused
from .stage1d_ablation import VARIANTS
from .stage1d_rich_goal import (RichStateCritic, _critic_logits, _raw_goals,
                                _targets, duplicate_rate)
from .stage1e_branch_alignment import (GROUP_NAMES, GROUPS, _branch_metrics,
                                       _cluster_ci, collect_states, summarize)
from .stage1f_horizon_scan import (diversity_ratio_state, goal_at, normalize,
                                   run_to_hmax)

ART = Path("artifacts/pong_action_gate/stage1g/multihorizon_no_self")
BRANCH_CACHE = ART / "stage1f_branches_cache.npz"

HORIZONS = [4, 8, 12, 16, 24, 32]
GOAL_DIM = 9
NS_CONT = VARIANTS["no_self"]["cont"]          # full-7 cols used by no-self: [0,1,2,3,5,6]
R = 16                                          # CRN replicates per (state, forced action)
HMAX = 32                                       # max training/eval horizon

GROUP_MEMBERS = {"STAY": [0, 1], "UP": [2, 4], "DOWN": [3, 5]}
ALIAS = {0: 1, 1: 0, 2: 4, 4: 2, 3: 5, 5: 3}    # same-movement-group FIRE alias swap
SEM_BASE = 1.0 / 3
EXACT_BASE = 1.0 / 6


# --------------------------------------------------------------------------- #
# 1. Balanced multi-horizon training arrays (no-self goal, no horizon input)
# --------------------------------------------------------------------------- #
def build_mh(eps, ep_ids, stats=None) -> Dict[str, Any]:
    """All (anchor t, horizon H) records with t+H inside the same episode; no-self 9-dim goal.

    Normalization (continuous components) is pooled over ALL six horizons, train episodes only.
    The 7-col stats array is kept (NS_CONT indexes it) so it is byte-compatible with the
    Stage-1F / Stage-1E `normalize` helper used for emulator-branch scoring.
    """
    raw = {li: _raw_goals(eps[li]) for li in ep_ids}
    feats = {li: D.build_state_features(eps[li]) for li in ep_ids}
    dec = {li: is_decision_focused(decision_criteria(eps[li])) for li in ep_ids}

    state, action, cont7, valid7, masks = [], [], [], [], []
    ep_a, t_a, H_a, event, decision = [], [], [], [], []
    for H in HORIZONS:
        for li in ep_ids:
            ep = eps[li]; T = len(ep["action"]); rg = raw[li]; f = feats[li]
            ise = ep["is_scoring_event"].astype(bool)
            for t in range(0, T - H):
                tf = t + H
                state.append(f[t]); action.append(int(ep["action"][t]))
                cont7.append(rg["cont"][tf]); valid7.append(rg["valid"][tf]); masks.append(rg["masks"][tf])
                ep_a.append(li); t_a.append(t); H_a.append(H)
                event.append(bool(ise[t:tf].any())); decision.append(bool(dec[li][t]))

    cont7 = np.array(cont7, np.float32); valid7 = np.array(valid7, bool)
    masks = np.array(masks, np.float32); H_a = np.array(H_a, np.int64)

    if stats is None:                                  # pooled over all horizons, train-only
        mean = np.zeros(cont7.shape[1]); std = np.ones(cont7.shape[1])
        for c in range(cont7.shape[1]):
            v = cont7[valid7[:, c], c]
            if len(v):
                mean[c] = v.mean(); std[c] = max(v.std(), 1e-6)
        stats = {"mean": mean.tolist(), "std": std.tolist()}

    cont_ns = cont7[:, NS_CONT]; valid_ns = valid7[:, NS_CONT]
    mean_ns = np.array(stats["mean"])[NS_CONT]; std_ns = np.array(stats["std"])[NS_CONT]
    norm = np.where(valid_ns, (cont_ns - mean_ns) / std_ns, 0.0).astype(np.float32)
    goal = np.concatenate([norm, masks], axis=1)       # (N, 9)

    keyint = np.concatenate([np.round(np.where(valid_ns, cont_ns, 0.0)).astype(np.int64),
                             masks.astype(np.int64)], axis=1)
    keys = np.array([hash(tuple(row)) for row in keyint])

    idx_by_H = {H: np.where(H_a == H)[0] for H in HORIZONS}
    return {"state": np.array(state, np.float32), "action": np.array(action, np.int64),
            "goal": goal, "keys": keys, "ep": np.array(ep_a), "t": np.array(t_a),
            "H": H_a, "event": np.array(event), "decision": np.array(decision),
            "idx_by_H": idx_by_H, "stats": stats, "n_anchors": len(action),
            "horizon_counts": {int(H): int(len(idx_by_H[H])) for H in HORIZONS},
            "dup_by_horizon": {int(H): duplicate_rate(keys[idx_by_H[H]]) for H in HORIZONS},
            "dup_overall": duplicate_rate(keys)}


def balanced_idx(idx_by_H, batch, rng) -> np.ndarray:
    """Sample H uniformly per slot, then an anchor uniformly within that horizon's pool."""
    Hs = rng.integers(len(HORIZONS), size=batch)
    out = np.empty(batch, np.int64)
    for i, hi in enumerate(Hs):
        pool = idx_by_H[HORIZONS[hi]]
        out[i] = pool[rng.integers(len(pool))]
    return out


# --------------------------------------------------------------------------- #
# 2. Train (balanced sampler) + val-loss checkpoint selection
# --------------------------------------------------------------------------- #
def train_mh(A_tr, A_va, seed, steps, batch, lr=3e-4, eval_every=400):
    torch.manual_seed(seed)
    critic = RichStateCritic(D.STATE_DIM, 6, goal_dim=GOAL_DIM)
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    best = float("inf"); best_state = None; curve = []
    sampled_H = np.zeros(len(HORIZONS), np.int64)
    for step in range(steps + 1):
        if step % eval_every == 0:
            vr = np.random.default_rng(seed + 777 + step)
            with torch.no_grad():
                vls = []
                for _ in range(4):
                    vidx = balanced_idx(A_va["idx_by_H"], min(batch, A_va["n_anchors"]), vr)
                    logits, keys = _critic_logits(critic, A_va, vidx)
                    vls.append(float(nce_loss(logits, _targets(keys))))
                vl = float(np.mean(vls))
            curve.append({"step": step, "val_loss": vl})
            if vl < best:
                best = vl; best_state = deepcopy(critic.state_dict())
        if step < steps:
            idx = balanced_idx(A_tr["idx_by_H"], batch, rng)
            for hi in A_tr["H"][idx]:
                sampled_H[HORIZONS.index(int(hi))] += 1
            logits, keys = _critic_logits(critic, A_tr, idx)
            opt.zero_grad(set_to_none=True)
            loss = nce_loss(logits, _targets(keys))
            loss.backward(); opt.step()
    critic.load_state_dict(best_state); critic.eval()
    freq = {int(HORIZONS[i]): int(sampled_H[i]) for i in range(len(HORIZONS))}
    sel = {"best_val_loss": best, "selected_step": min(curve, key=lambda r: r["val_loss"])["step"],
           "val_curve": curve, "sampled_horizon_freq": freq}
    return critic, sel


def overfit_check(A_tr, seed=0, n=128, steps=300, lr=1e-3) -> Dict[str, float]:
    """Tiny fixed-batch overfit: a working critic should drive BCE far below its init."""
    torch.manual_seed(seed)
    critic = RichStateCritic(D.STATE_DIM, 6, goal_dim=GOAL_DIM)
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    idx = balanced_idx(A_tr["idx_by_H"], min(n, A_tr["n_anchors"]), rng)
    tgt = _targets(A_tr["keys"][idx])
    first = last = None
    for st in range(steps):
        logits, _ = _critic_logits(critic, A_tr, idx)
        loss = nce_loss(logits, tgt)
        if st == 0:
            first = float(loss)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        last = float(loss)
    return {"init_loss": first, "final_loss": last, "dropped": first - last}


# --------------------------------------------------------------------------- #
# 3. Offline held-out diagnostics by horizon and state subset
# --------------------------------------------------------------------------- #
def _bucket_ep_metric(critic, A, ix, B, rng):
    idx = rng.choice(ix, size=min(B, len(ix)), replace=False)
    s = torch.as_tensor(A["state"][idx]); a = torch.as_tensor(A["action"][idx]); g = torch.as_tensor(A["goal"][idx])
    tgt = _targets(A["keys"][idx])
    an = A["action"][idx]
    perm = torch.as_tensor(rng.permutation(len(idx)))
    alias = torch.as_tensor(np.array([ALIAS[int(x)] for x in an], np.int64))
    cross = np.empty(len(idx), np.int64)
    for i, x in enumerate(an):
        other = [b for b in range(6) if GROUPS[b] != GROUPS[int(x)]]
        cross[i] = other[rng.integers(len(other))]
    cross = torch.as_tensor(cross)
    with torch.no_grad():
        correct = float(nce_loss(critic.logits_matrix(s, a, g), tgt))
        shuffled = float(nce_loss(critic.logits_matrix(s, a[perm], g), tgt))
        same = float(nce_loss(critic.logits_matrix(s, alias, g), tgt))
        crossl = float(nce_loss(critic.logits_matrix(s, cross, g), tgt))
        sa = critic.scores_all_actions(s, g).numpy()        # (B,6)
    grp = np.stack([sa[:, GROUP_MEMBERS[gn]].mean(1) for gn in GROUP_NAMES], axis=1)  # (B,3)
    return {"correct": correct, "shuffle_delta": shuffled - correct,
            "cross_group_delta": crossl - correct, "same_group_alias_delta": same - correct,
            "sem_score_spread": float(grp.std(1).mean())}


def diag_bucket(critic, A, seed, H, subset, B=256, n_boot=2000) -> Dict[str, Any]:
    mask = (A["H"] == H)
    if subset == "ordinary":
        mask &= ~A["decision"]
    elif subset == "decision":
        mask &= A["decision"]
    sel = np.where(mask)[0]
    by_ep: Dict[int, List[int]] = {}
    for i in sel:
        by_ep.setdefault(int(A["ep"][i]), []).append(int(i))
    rng = np.random.default_rng(seed + H)
    per_ep = [_bucket_ep_metric(critic, A, np.array(ix), B, rng) for ix in by_ep.values() if len(ix) >= 8]
    if not per_ep:
        return {"H": H, "subset": subset, "n_episodes": 0, "n_anchors": int(len(sel))}

    def ci(key, pos=True):
        vals = np.array([m[key] for m in per_ep]); br = np.random.default_rng(seed + H + 1)
        bs = [float(vals[br.integers(len(vals), size=len(vals))].mean()) for _ in range(n_boot)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return {"point": float(vals.mean()), "ci95": [float(lo), float(hi)],
                "ci_excludes_zero_pos": bool(lo > 0)}

    return {"H": H, "subset": subset, "n_episodes": len(per_ep), "n_anchors": int(len(sel)),
            "correct_loss": float(np.mean([m["correct"] for m in per_ep])),
            "shuffle_delta": ci("shuffle_delta"), "cross_group_delta": ci("cross_group_delta"),
            "same_group_alias_delta": ci("same_group_alias_delta"),
            "sem_score_spread": ci("sem_score_spread")}


def offline_diagnostics(critic, A_va, seed) -> Dict[str, Any]:
    out = {}
    for H in HORIZONS:
        out[str(H)] = {sub: diag_bucket(critic, A_va, seed, H, sub)
                       for sub in ("all", "ordinary", "decision")}
    return out


# --------------------------------------------------------------------------- #
# 4. Stage-1F emulator branches: generate ONCE to H_max, cache, reuse per seed
# --------------------------------------------------------------------------- #
def generate_or_load_branches(state_seed, n_each, device="cpu") -> Dict[str, Any]:
    if BRANCH_CACHE.exists():
        z = np.load(BRANCH_CACHE, allow_pickle=False)
        if int(z["state_seed"]) == state_seed and int(z["n_each"]) == n_each:
            return {k: z[k] for k in z.files}
    ART.mkdir(parents=True, exist_ok=True)
    teacher_model, _ = load_teacher(device=device)
    teacher = TeacherPolicy(teacher_model, device=device)
    env = make_env(replace(C.M1Config(), device=device), num_envs=1)
    snaps = collect_states(teacher, env, state_seed, n_each=n_each)
    S = len(snaps); nH = len(HORIZONS)
    anchor_state = np.stack([sn["anchor_state"] for sn in snaps]).astype(np.float32)
    decision = np.array([bool(sn["decision"]) for sn in snaps])
    raw6 = np.zeros((S, 6, R, nH, 6), np.float32)
    masks3 = np.zeros((S, 6, R, nH, 3), np.float32)
    valid6 = np.zeros((S, 6, R, nH, 6), np.float32)
    censored = np.ones((S, 6, R, nH), bool)
    for si, sn in enumerate(snaps):
        for rep in range(R):
            rep_seed = 90000 + rep                       # CRN, shared across forced actions
            for a in range(6):
                steps = run_to_hmax(env, teacher, sn, a, rep_seed)
                for hi, H in enumerate(HORIZONS):
                    g = goal_at(steps, H)
                    if g is None:
                        continue
                    censored[si, a, rep, hi] = False
                    raw6[si, a, rep, hi] = g["raw6"]; masks3[si, a, rep, hi] = g["masks3"]
                    valid6[si, a, rep, hi] = g["valid6"]
    env.close()
    cache = {"state_seed": np.array(state_seed), "n_each": np.array(n_each),
             "horizons": np.array(HORIZONS), "anchor_state": anchor_state, "decision": decision,
             "raw6": raw6, "masks3": masks3, "valid6": valid6, "censored": censored}
    np.savez(BRANCH_CACHE, **cache)
    return cache


def score_branches(critic, stats, cache) -> Dict[str, Any]:
    """For every realized no-self branch goal, score all 6 actions; aggregate per state/horizon."""
    anchor = cache["anchor_state"]; decision = cache["decision"]
    raw6 = cache["raw6"]; masks3 = cache["masks3"]; valid6 = cache["valid6"]; censored = cache["censored"]
    S = anchor.shape[0]
    per_h = {}
    for hi, H in enumerate(HORIZONS):
        rows = []
        for si in range(S):
            ms = []
            gba = {a: [] for a in range(6)}
            for a in range(6):
                for rep in range(R):
                    if censored[si, a, rep, hi]:
                        continue
                    g9 = normalize(raw6[si, a, rep, hi], valid6[si, a, rep, hi], masks3[si, a, rep, hi], stats)
                    gba[a].append(g9)
                    ms.append(_branch_metrics(critic, anchor[si], g9, a))
            if len(ms) < 6:
                continue
            agg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
            agg["decision"] = bool(decision[si]); agg["n_branches"] = len(ms)
            agg["diversity"] = diversity_ratio_state(gba) or 0.0
            agg["state_idx"] = si
            rows.append(agg)
        per_h[str(H)] = _summarize_horizon(rows)
    return per_h


def _summarize_horizon(rows) -> Dict[str, Any]:
    dec = [r for r in rows if r["decision"]]; ordi = [r for r in rows if not r["decision"]]
    div = np.array([r["diversity"] for r in rows]); marg = np.array([r["group_margin"] for r in rows])
    corr = float(np.corrcoef(div, marg)[0, 1]) if len(rows) >= 3 and div.std() > 0 and marg.std() > 0 else None
    hi_div = [r for r in rows if r["diversity"] > np.median(div)] if len(rows) >= 4 else []
    lo_div = [r for r in rows if r["diversity"] <= np.median(div)] if len(rows) >= 4 else []
    return {
        "n_states": len(rows),
        "overall": summarize(rows) if len(rows) >= 4 else {"n": len(rows)},
        "decision": summarize(dec) if len(dec) >= 4 else {"n": len(dec)},
        "ordinary": summarize(ordi) if len(ordi) >= 4 else {"n": len(ordi)},
        "env_diversity_vs_group_margin_corr": corr,
        "group_margin_high_div": float(np.mean([r["group_margin"] for r in hi_div])) if hi_div else None,
        "group_margin_low_div": float(np.mean([r["group_margin"] for r in lo_div])) if lo_div else None,
        "diversity_median": float(np.median(div)) if len(rows) else None,
    }


# --------------------------------------------------------------------------- #
# 5. Decision gate (seed 0) and three-seed aggregation
# --------------------------------------------------------------------------- #
def _peak_ok(emb, off) -> Dict[str, Any]:
    """Seed-0 gate evidence on decision-focused H=12/H=16 branches + offline alias/cross check."""
    checks = {}
    peak_pass = []
    for H in (12, 16):
        d = emb[str(H)]["decision"]
        if "sem_top1" not in d:
            checks[H] = {"available": False}; continue
        sem_pt = d["sem_top1"]["point"]; pair_pt = d["pairwise"]["point"]; gm = d["group_margin"]["point"]
        ci_excl = bool(d["sem_top1"].get("above_baseline_ci") or d["pairwise"].get("above_half"))
        ok = bool(sem_pt > SEM_BASE and pair_pt > 0.5 and gm > 0)
        checks[H] = {"available": True, "sem_top1": sem_pt, "pairwise": pair_pt, "group_margin": gm,
                     "directional_pass": ok, "ci_excludes_baseline": ci_excl}
        peak_pass.append((ok, ci_excl, H))
    # stronger alignment in higher-diversity states (at the peak horizons)
    div_mono = []
    for H in (12, 16):
        s = emb[str(H)]; hi = s.get("group_margin_high_div"); lo = s.get("group_margin_low_div")
        if hi is not None and lo is not None:
            div_mono.append(hi >= lo)
    # same-group alias effect < cross-group effect (decision subset, peak horizons)
    alias_lt_cross = []
    for H in (12, 16):
        dd = off.get(str(H), {}).get("decision", {})
        if "same_group_alias_delta" in dd:
            alias_lt_cross.append(dd["same_group_alias_delta"]["point"] < dd["cross_group_delta"]["point"])
    cond1 = any(checks[H].get("directional_pass") for H in (12, 16) if checks[H].get("available"))
    cond2 = any(ci for (_, ci, _) in peak_pass)
    cond3 = bool(div_mono) and all(div_mono)
    cond4 = bool(alias_lt_cross) and all(alias_lt_cross)
    promising = bool(cond1 and cond2 and cond3 and cond4)
    return {"peak_checks": checks, "cond1_directional_peak": cond1, "cond2_ci_excludes_base": cond2,
            "cond3_stronger_with_diversity": cond3, "cond4_alias_lt_cross": cond4,
            "cond5_no_self_paddle_in_goal": True, "promising": promising}


def three_seed_label(per_seed) -> str:
    seeds = list(per_seed)
    dir_ok = {s: [] for s in seeds}; marg_pos = {s: [] for s in seeds}; ci_peak = {s: 0 for s in seeds}
    for s in seeds:
        emb = per_seed[s]["emulator"]
        for H in (12, 16):
            d = emb[str(H)]["decision"]
            if "sem_top1" not in d:
                continue
            dir_ok[s].append(d["sem_top1"]["point"] > SEM_BASE or d["pairwise"]["point"] > 0.5)
            marg_pos[s].append(d["group_margin"]["point"] > 0)
            if d["sem_top1"].get("above_baseline_ci") or d["pairwise"].get("above_half"):
                ci_peak[s] += 1
    all_dir = all(any(dir_ok[s]) for s in seeds)
    all_marg = all(all(marg_pos[s]) and marg_pos[s] for s in seeds)
    n_ci = sum(ci_peak[s] > 0 for s in seeds)
    diversity_tracks = all(
        (per_seed[s]["emulator"][str(H)].get("group_margin_high_div") or 0) >=
        (per_seed[s]["emulator"][str(H)].get("group_margin_low_div") or 0)
        for s in seeds for H in (12, 16)
        if per_seed[s]["emulator"][str(H)].get("group_margin_high_div") is not None)
    clearly_exceed = sum(any(per_seed[s]["emulator"][str(H)]["decision"].get("sem_top1", {}).get("point", 0) > SEM_BASE
                             for H in (12, 16)) for s in seeds)
    if all_dir and all_marg and n_ci >= 2 and diversity_tracks:
        return "STRONG PASS"
    if all_dir and clearly_exceed >= 2 and diversity_tracks:
        return "CANDIDATE PASS"
    return "FAIL"


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _sha_state(critic) -> str:
    buf = b"".join(v.cpu().numpy().tobytes() for v in critic.state_dict().values())
    return hashlib.sha256(buf).hexdigest()[:16]


def run_seed(all_eps, n_episodes, seed, steps, batch, cache, do_overfit=False) -> Dict[str, Any]:
    tr_ids, va_ids = D.split_episodes(n_episodes, 0.2, seed)
    A_tr = build_mh(all_eps, tr_ids)
    A_va = build_mh(all_eps, va_ids, stats=A_tr["stats"])     # reuse train normalization
    of = overfit_check(A_tr, seed) if do_overfit else None
    critic, sel = train_mh(A_tr, A_va, seed, steps, batch)
    off = offline_diagnostics(critic, A_va, seed)
    emb = score_branches(critic, A_tr["stats"], cache)

    d = ART / f"seed{seed}"; d.mkdir(parents=True, exist_ok=True)
    torch.save(critic.state_dict(), d / "critic.pt")
    res = {
        "seed": seed, "checkpoint_sha16": _sha_state(critic),
        "selected_step": sel["selected_step"], "best_val_loss": sel["best_val_loss"],
        "sampled_horizon_freq": sel["sampled_horizon_freq"],
        "anchor_horizon_counts": A_tr["horizon_counts"],
        "dup_overall": A_tr["dup_overall"], "dup_by_horizon": A_tr["dup_by_horizon"],
        "n_train_anchors": A_tr["n_anchors"], "n_val_anchors": A_va["n_anchors"],
        "normalization_stats": A_tr["stats"], "episode_split": {"train": tr_ids, "val": va_ids},
        "val_curve": sel["val_curve"], "overfit_check": of,
        "offline": off, "emulator": emb,
    }
    (d / "report.json").write_text(json.dumps(res, indent=2))
    return res


def run(seeds, n_episodes, steps, batch, state_seed, n_each, device="cpu") -> Dict[str, Any]:
    ART.mkdir(parents=True, exist_ok=True)
    all_eps = D.load_subset("full", list(range(n_episodes)), with_pixels=False)
    cache = generate_or_load_branches(state_seed, n_each, device=device)

    per_seed = {}
    s0 = run_seed(all_eps, n_episodes, seeds[0], steps, batch, cache, do_overfit=True)
    per_seed[str(seeds[0])] = s0
    gate = _peak_ok(s0["emulator"], s0["offline"])

    ran = [seeds[0]]
    if gate["promising"]:
        for seed in seeds[1:]:
            per_seed[str(seed)] = run_seed(all_eps, n_episodes, seed, steps, batch, cache)
            ran.append(seed)
        label = three_seed_label(per_seed)
    else:
        label = "FAIL"

    out = {
        "milestone": "Stage-1G-multihorizon-no-self-critic",
        "horizons": HORIZONS, "goal_dim": GOAL_DIM, "horizon_input_to_model": False,
        "future_self_paddle_in_goal": False,
        "goal_schema_no_self": ["ball_x", "ball_y", "ball_vx", "ball_vy", "opponent_paddle_y",
                                "score_diff_pre", "mask_ball", "mask_opp", "mask_vel"],
        "config": {"n_episodes": n_episodes, "steps": steps, "batch": batch, "lr": 3e-4,
                   "state_seed": state_seed, "n_each": n_each, "R": R,
                   "uniform_anchors": True, "decision_focused_sampling": False,
                   "balanced_horizon_sampler": True, "val_loss_checkpoint_selection": True},
        "branch_cache": str(BRANCH_CACHE),
        "random_baselines": {"semantic": SEM_BASE, "exact": EXACT_BASE, "pairwise": 0.5},
        "seed0_gate": gate, "seeds_run": ran, "decision": label,
        "per_seed": per_seed,
        "interpretation_note": "Controlled diagnostic. A pass supports a unified multi-horizon no-self "
                               "critic conditioning action use on the state/timescale where the ENVIRONMENT "
                               "has a real action effect; it does NOT prove +15 reaching, next-score "
                               "prediction, pixels, confounding robustness, causal ID, policy improvement, "
                               "or optimality of this horizon mixture.",
    }
    (ART / "stage1g_report.json").write_text(json.dumps(out, indent=2))
    _write_summary(out)
    return out


def _write_summary(out) -> None:
    L = ["Stage-1G unified multi-horizon no-self critic (H in [4,8,12,16,24,32], H not an input).",
         "Semantic baseline 0.333, pairwise 0.5, exact 0.167. Emulator = Stage-1F branches.\n"]
    for s, d in out["per_seed"].items():
        L.append(f"== seed {s} (sel step {d['selected_step']}, val {d['best_val_loss']:.3f}, "
                 f"dup {d['dup_overall']['fraction_anchors_with_exact_duplicate']:.3f}) ==")
        L.append(f"{'H':>4} {'dec_sem':>8} {'dec_pair':>9} {'dec_gmarg':>10} {'ord_sem':>8} "
                 f"{'div~marg':>9} {'hi-lo_marg':>10}")
        for H in out["horizons"]:
            e = d["emulator"][str(H)]; dec = e["decision"]; ordi = e["ordinary"]
            ds = dec.get("sem_top1", {}).get("point", float("nan"))
            dp = dec.get("pairwise", {}).get("point", float("nan"))
            dg = dec.get("group_margin", {}).get("point", float("nan"))
            os_ = ordi.get("sem_top1", {}).get("point", float("nan"))
            cr = e.get("env_diversity_vs_group_margin_corr")
            hi = e.get("group_margin_high_div"); lo = e.get("group_margin_low_div")
            hl = (hi - lo) if (hi is not None and lo is not None) else float("nan")
            star = "*" if dec.get("sem_top1", {}).get("above_baseline_ci") else ""
            L.append(f"{H:>4} {ds:>7.3f}{star:<1} {dp:>9.3f} {dg:>+10.4f} {os_:>8.3f} "
                     f"{(cr if cr is not None else float('nan')):>9.3f} {hl:>+10.4f}")
        L.append("")
    g = out["seed0_gate"]
    L.append(f"seed0 gate: cond1_dir={g['cond1_directional_peak']} cond2_ci={g['cond2_ci_excludes_base']} "
             f"cond3_div={g['cond3_stronger_with_diversity']} cond4_alias<cross={g['cond4_alias_lt_cross']} "
             f"-> promising={g['promising']}")
    L.append(f"seeds run: {out['seeds_run']}    DECISION: {out['decision']}")
    (ART / "summary.txt").write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-1G unified multi-horizon no-self critic.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-episodes", type=int, default=80)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--state-seed", type=int, default=8000)
    ap.add_argument("--n-each", type=int, default=12)
    ap.add_argument("--dry-run", action="store_true", help="schema + sampler + tiny overfit only (no teacher)")
    args = ap.parse_args()
    if args.dry_run:
        _dry_run(args.n_episodes)
        return
    out = run(args.seeds, args.n_episodes, args.steps, args.batch, args.state_seed, args.n_each)
    print(open(ART / "summary.txt").read())


def _dry_run(n_episodes) -> None:
    all_eps = D.load_subset("full", list(range(n_episodes)), with_pixels=False)
    tr_ids, va_ids = D.split_episodes(n_episodes, 0.2, 0)
    A_tr = build_mh(all_eps, tr_ids)
    A_va = build_mh(all_eps, va_ids, stats=A_tr["stats"])
    print("train anchors:", A_tr["n_anchors"], "| horizon counts:", A_tr["horizon_counts"])
    print("val anchors:", A_va["n_anchors"], "| goal shape:", A_tr["goal"].shape,
          "| state shape:", A_tr["state"].shape)
    rng = np.random.default_rng(0)
    idx = balanced_idx(A_tr["idx_by_H"], 256, rng)
    sampled = {int(H): int((A_tr["H"][idx] == H).sum()) for H in HORIZONS}
    print("one balanced batch horizon freq:", sampled)
    print("dup overall:", A_tr["dup_overall"])
    print("dup by horizon:", {H: round(A_tr["dup_by_horizon"][H]["fraction_anchors_with_exact_duplicate"], 4)
                              for H in [int(h) for h in HORIZONS]})
    of = overfit_check(A_tr, 0)
    print("overfit check:", {k: round(v, 4) for k, v in of.items()})
    assert A_tr["goal"].shape[1] == GOAL_DIM, "goal must be 9-dim no-self"
    print("OK: goal_dim == 9 (no future self-paddle, no horizon channel)")


if __name__ == "__main__":
    main()
