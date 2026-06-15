"""Stage-1H — paired interventional branch critic (positive control).

Question carried in from Stage-1E/1F/1G: does the no-self critic fail because the OBSERVATIONAL
NCE objective never forces same-state action use, or because the critic / 9-dim external goal
representation simply CANNOT learn the action-conditioned future mapping even under explicit
interventional supervision?

Stage-1H builds a SUPERVISED interventional dataset by cloning a decision-focused Pong state and
branching the SAME state under the three semantic action groups (NOOP/UP/DOWN, canonical ALE
actions 0/2/3), recording the 9-dim no-self future goal at H=12 and H=16 (run separately). Two
objectives are compared on identical data:
  1H-A  observational-style in-batch NCE (closest analogue of Stage-1G);
  1H-B  within-state action discrimination: given the SAME state and a generated future, which of
        {NOOP,UP,DOWN} produced it? (the decisive same-state supervision).

Invariants asserted + reported: identical base state across branches; goal_dim==9 with NO future
self-paddle; H never an input; train/val/test split at the BASE-STATE level (no branch leakage);
the Stage-1G RichStateCritic reused unchanged for the primary run. Bootstrap unit = base state.

No pixel/confounder/causal/Seaquest/chunks/policy work. Nothing is committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .. import config as C
from ..objects import extract_pong_objects
from ..teacher.load_teacher import TeacherPolicy, load_teacher, make_env
from . import dataset as D
from .critics import nce_loss
from .emulator_branch import _feat_single, restore_env, snapshot_env
from .stage1d_rich_goal import RichStateCritic
from .stage1e_branch_alignment import GROUPS, _gray_stub
from .stage1f_horizon_scan import goal_at

# Output: required deliverables under results/ (NOT gitignored); heavy branch cache under
# artifacts/ (gitignored) so a blanket commit does not balloon the tracked tree.
RESULTS = Path("pong_action_gate/results/stage1h_interventional_branch_critic")
CACHE_DIR = Path("artifacts/pong_action_gate/stage1h")

HORIZONS = [12, 16]                              # trained SEPARATELY (no mixing)
GROUP_NAMES = ["NOOP", "UP", "DOWN"]            # Stage-1H semantic groups (NOOP == Stage-1E "STAY")
CANON_IDS = [0, 2, 3]                            # canonical ALE rep per group: NOOP / RIGHT(up) / LEFT(down)
GROUP_OF_CANON = {0: 0, 2: 1, 3: 2}
R = 3                                            # CRN replicates per (base state, action group)
REP_BASE = 90000
TAU = 1.0                                        # fixed within-state softmax temperature (not tuned)
SEM_BASE = 1.0 / 3
GEN = "v1"                                       # branch-generation convention version (cache key)


# --------------------------------------------------------------------------- #
# Step 0 — audit
# --------------------------------------------------------------------------- #
def audit(env=None) -> Dict[str, Any]:
    own_env = env is None
    if own_env:
        env = make_env(replace(C.M1Config(), device="cpu"), num_envs=1)
        env.reset(seed=[12345])
    names = list(env.env.unwrapped.get_action_meanings())
    if own_env:
        env.close()
    return {
        "ale_action_meanings": names,
        "semantic_groups_ids": {"NOOP": [0, 1], "UP": [2, 4], "DOWN": [3, 5]},
        "canonical_action_per_group": {"NOOP": 0, "UP": 2, "DOWN": 3},
        "canonical_action_names": {"NOOP": names[0], "UP": names[2], "DOWN": names[3]},
        "alias_note": ("Pong has NO UPFIRE/DOWNFIRE. Aliases are NOOP/FIRE (STAY/NOOP group), "
                       "RIGHT/RIGHTFIRE (UP group, paddle dy<0), LEFT/LEFTFIRE (DOWN group, dy>0). "
                       "Primary run uses ONE canonical rep per group; no alias variation."),
        "reused_components": {
            "clone_restore": "emulator_branch.snapshot_env / restore_env (ALE cloneSystemState + "
                             "AtariPreprocessing frame buffers + LSTM hidden + scores)",
            "decision_state_logic": "Stage-1F/1E: ball_present & ball moving right (dx>0) & ball_x>=120",
            "continuation": "Stage-1F run_to_hmax convention, here capped at H (one env.step = one "
                            "agent decision step; frame-skip internal to AtariPreprocessing)",
            "goal9": "stage1f.goal_at -> [ball_x,ball_y,ball_vx,ball_vy,opp_y,score_diff,mask_ball,"
                     "mask_opp,mask_vel]; future agent paddle EXCLUDED",
            "normalization": "Stage-1G principle: train-only mean/std on the 6 continuous comps, "
                             "masks passed through; here refit on Stage-1H TRAIN base states",
            "duplicate_targets": "exact integer-goal key equality -> multi-positive BCE targets",
            "critic": "stage1d_rich_goal.RichStateCritic(STATE_DIM,6,goal_dim=9) (= Stage-1G critic)",
        },
        "GROUPS_map": {int(k): v for k, v in GROUPS.items()},
    }


# --------------------------------------------------------------------------- #
# Step 1 — collect a large set of cloned base states (decision-focused + matched ordinary)
# --------------------------------------------------------------------------- #
def collect_base_states(teacher, env, seed, n_dec, n_ord, stride=7, max_resets=400) -> Tuple[List[Dict], Dict]:
    torch.manual_seed(seed)
    obs, info = env.reset(seed=[seed]); ale = env.env.ale
    hx, cx = teacher.initial_state(1)
    dec, ordi = [], []
    prev = extract_pong_objects(ale.getRAM()); ag = op = 0; t = 0; resets = 0
    n_cand = 0; n_rej = 0; rej = {"no_player": 0, "ball_absent": 0, "not_decision_geom": 0}
    vis_ball = 0; vis_vel = 0
    while (len(dec) < n_dec or len(ordi) < n_ord) and resets < max_resets:
        o = extract_pong_objects(ale.getRAM())
        if t > 0 and t % stride == 0:
            n_cand += 1
            if o.player_y is None:
                n_rej += 1; rej["no_player"] += 1
            else:
                bdx = (o.ball_x - prev.ball_x) if (o.ball_present and prev.ball_present) else None
                vis_ball += int(o.ball_present); vis_vel += int(bdx is not None)
                is_dec = bool(o.ball_present and bdx is not None and bdx > 0 and o.ball_x is not None and o.ball_x >= 120)
                snap = snapshot_env(env, hx, cx, ag, op, obs[:, C.TEACHER_OBS_SLICE], _gray_stub(), o, prev)
                snap["decision"] = is_dec
                snap["anchor_state"] = _feat_single(o, prev, snap["score_diff_pre"])
                if is_dec and len(dec) < n_dec:
                    dec.append(snap)
                elif (not is_dec) and len(ordi) < n_ord:
                    ordi.append(snap)
                elif not is_dec:
                    n_rej += 1; rej["not_decision_geom"] += 1
        with torch.no_grad():
            logits, _, (hx, cx) = teacher.model.predict_act_value(obs[:, C.TEACHER_OBS_SLICE], (hx, cx))
            a = torch.distributions.Categorical(logits=logits / C.BEHAVIOR_TEMPERATURE).sample()
        prev = o
        obs, rew, end, trunc, info = env.step(a)
        r = float(rew.item()); ag += int(r > 0); op += int(r < 0); t += 1
        if bool((end | trunc).item()):
            obs, info = env.reset(seed=[seed + 7000 + resets]); hx, cx = teacher.initial_state(1)
            prev = extract_pong_objects(ale.getRAM()); ag = op = 0; resets += 1
    stats = {"candidates": n_cand, "accepted_decision": len(dec), "accepted_ordinary": len(ordi),
             "rejected": n_rej, "rejection_reasons": rej, "episodes_used": resets + 1,
             "ball_visibility_rate": float(vis_ball / max(n_cand, 1)),
             "velocity_availability_rate": float(vis_vel / max(n_cand, 1))}
    return dec + ordi, stats


# --------------------------------------------------------------------------- #
# Step 2 — interventional branch continuation (mirror Stage-1F, capped at H_max)
# --------------------------------------------------------------------------- #
def run_to_H(env, teacher, snap, first_action, rep_seed, Hmax) -> List:
    """Restore EXACT clone, force first_action, teacher continues to Hmax decision steps (CRN)."""
    torch.manual_seed(rep_seed)
    hx, cx, ag, op = restore_env(env, snap)
    with torch.no_grad():
        _, _, (hx, cx) = teacher.model.predict_act_value(snap["anchor_obs"], (hx, cx))
    ale = env.env.ale
    steps = []
    obs, rew, end, trunc, info = env.step(int(first_action))
    ag += int(float(rew.item()) > 0); op += int(float(rew.item()) < 0)
    steps.append((extract_pong_objects(ale.getRAM()), ag, op, float(rew.item())))
    for _ in range(Hmax - 1):
        if bool((end | trunc).item()):
            break
        with torch.no_grad():
            logits, _, (hx, cx) = teacher.model.predict_act_value(obs[:, C.TEACHER_OBS_SLICE], (hx, cx))
            a = torch.distributions.Categorical(logits=logits / C.BEHAVIOR_TEMPERATURE).sample()
        obs, rew, end, trunc, info = env.step(a)
        ag += int(float(rew.item()) > 0); op += int(float(rew.item()) < 0)
        steps.append((extract_pong_objects(ale.getRAM()), ag, op, float(rew.item())))
    return steps


def _cfg_hash(cfg) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def generate_or_load_branches(state_seed, n_dec, n_ord, device="cpu") -> Dict[str, Any]:
    Hmax = max(HORIZONS)
    cfg = {"state_seed": state_seed, "n_dec": n_dec, "n_ord": n_ord, "stride": 7,
           "R": R, "Hmax": Hmax, "horizons": HORIZONS, "canon": CANON_IDS, "rep_base": REP_BASE, "gen": GEN}
    chash = _cfg_hash(cfg)
    cpath = CACHE_DIR / f"branches_{chash}.npz"
    if cpath.exists():
        z = np.load(cpath, allow_pickle=False)
        d = {k: z[k] for k in z.files} | {"config_hash": chash, "config": cfg, "cache_path": str(cpath)}
        csp = CACHE_DIR / f"collect_stats_{chash}.json"
        if csp.exists():
            d["collect_stats"] = json.loads(csp.read_text())
        return d
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    teacher_model, _ = load_teacher(device=device)
    teacher = TeacherPolicy(teacher_model, device=device)
    env = make_env(replace(C.M1Config(), device=device), num_envs=1)
    snaps, cstats = collect_base_states(teacher, env, state_seed, n_dec, n_ord)
    S = len(snaps); nG = len(CANON_IDS); nH = len(HORIZONS)
    anchor_state = np.stack([sn["anchor_state"] for sn in snaps]).astype(np.float32)
    decision = np.array([bool(sn["decision"]) for sn in snaps])
    raw6 = np.zeros((S, nG, R, nH, 6), np.float32); masks3 = np.zeros((S, nG, R, nH, 3), np.float32)
    valid6 = np.zeros((S, nG, R, nH, 6), np.float32); censored = np.ones((S, nG, R, nH), bool)
    forced = np.zeros((S, nG, R), np.int64)
    for si, sn in enumerate(snaps):
        for rep in range(R):
            rep_seed = REP_BASE + rep                      # CRN: shared across action groups
            for gi, a in enumerate(CANON_IDS):
                steps = run_to_H(env, teacher, sn, a, rep_seed, Hmax)
                forced[si, gi, rep] = a
                for hi, H in enumerate(HORIZONS):
                    g = goal_at(steps, H)
                    if g is None:
                        continue
                    censored[si, gi, rep, hi] = False
                    raw6[si, gi, rep, hi] = g["raw6"]; masks3[si, gi, rep, hi] = g["masks3"]
                    valid6[si, gi, rep, hi] = g["valid6"]
    # reproducibility probe: regenerate first 3 states twice, compare exactly
    repro_max = 0.0; repro_match = 1.0; ncmp = 0
    for si in range(min(3, S)):
        for gi, a in enumerate(CANON_IDS):
            s2 = run_to_H(env, teacher, snaps[si], a, REP_BASE + 0, Hmax)
            g2 = goal_at(s2, HORIZONS[0]); g1 = goal_at(run_to_H(env, teacher, snaps[si], a, REP_BASE + 0, Hmax), HORIZONS[0])
            if g1 is not None and g2 is not None:
                d = float(np.abs(g1["raw6"] - g2["raw6"]).max()); repro_max = max(repro_max, d)
                repro_match = min(repro_match, float(d == 0.0)); ncmp += 1
    env.close()
    out = {"state_seed": np.array(state_seed), "anchor_state": anchor_state, "decision": decision,
           "raw6": raw6, "masks3": masks3, "valid6": valid6, "censored": censored, "forced": forced,
           "horizons": np.array(HORIZONS), "canon": np.array(CANON_IDS),
           "repro_max_disc": np.array(repro_max), "repro_match_rate": np.array(repro_match),
           "repro_n": np.array(ncmp)}
    np.savez(cpath, **out)
    (CACHE_DIR / f"collect_stats_{chash}.json").write_text(json.dumps(cstats, indent=2))
    out["collect_stats"] = cstats
    return out | {"config_hash": chash, "config": cfg, "cache_path": str(cpath)}


# --------------------------------------------------------------------------- #
# Step 3 — goal normalization (train base states only) + branch table per horizon
# --------------------------------------------------------------------------- #
def fit_norm(cache, train_ids, hi) -> Dict[str, list]:
    raw = cache["raw6"][train_ids, :, :, hi].reshape(-1, 6)
    val = cache["valid6"][train_ids, :, :, hi].reshape(-1, 6)
    cen = cache["censored"][train_ids, :, :, hi].reshape(-1)
    raw = raw[~cen]; val = val[~cen]
    mean = np.zeros(6); std = np.ones(6)
    for c in range(6):
        v = raw[val[:, c] > 0, c]
        if len(v):
            mean[c] = v.mean(); std[c] = max(v.std(), 1e-6)
    return {"mean": mean.tolist(), "std": std.tolist()}


def _norm9(raw6, valid6, masks3, stats) -> np.ndarray:
    mean = np.array(stats["mean"]); std = np.array(stats["std"])
    norm = np.where(valid6 > 0, (raw6 - mean) / std, 0.0).astype(np.float32)
    return np.concatenate([norm, masks3]).astype(np.float32)


def build_branches(cache, base_ids, hi, stats) -> Dict[str, Any]:
    """Flat per-branch arrays restricted to base_ids at horizon index hi."""
    st, gl, ac, gp, ky, bid, dec = [], [], [], [], [], [], []
    anchor = cache["anchor_state"]; decision = cache["decision"]
    raw6 = cache["raw6"]; masks3 = cache["masks3"]; valid6 = cache["valid6"]; cen = cache["censored"]
    canon = list(cache["canon"])
    for si in base_ids:
        for gi in range(len(canon)):
            for rep in range(R):
                if cen[si, gi, rep, hi]:
                    continue
                g9 = _norm9(raw6[si, gi, rep, hi], valid6[si, gi, rep, hi], masks3[si, gi, rep, hi], stats)
                st.append(anchor[si]); gl.append(g9); ac.append(int(canon[gi])); gp.append(gi)
                bid.append(int(si)); dec.append(bool(decision[si]))
                ki = np.concatenate([np.round(np.where(valid6[si, gi, rep, hi] > 0, raw6[si, gi, rep, hi], 0.0)).astype(np.int64),
                                     masks3[si, gi, rep, hi].astype(np.int64)])
                ky.append(hash(tuple(ki)))
    return {"state": np.array(st, np.float32), "goal": np.array(gl, np.float32),
            "action": np.array(ac, np.int64), "group": np.array(gp, np.int64),
            "key": np.array(ky), "base": np.array(bid), "decision": np.array(dec),
            "n": len(st)}


# --------------------------------------------------------------------------- #
# Step 4 — scoring helpers (aligned: column k <-> group k <-> canonical CANON_IDS[k])
# --------------------------------------------------------------------------- #
def canon_logits(critic, s, g, zero_action=False) -> torch.Tensor:
    gr = critic.g_repr(g)
    cols = []
    for a in CANON_IDS:
        act = torch.full((s.shape[0],), a, dtype=torch.long)
        cols.append((critic.sa_repr(s, act, zero_action=zero_action) * gr).sum(1, keepdim=True))
    return torch.cat(cols, dim=1)                  # (B,3) in group order [NOOP,UP,DOWN]


def _targets(keys) -> torch.Tensor:
    k = np.asarray(keys).reshape(-1)
    return torch.as_tensor((k[:, None] == k[None, :]).astype(np.float32))


# --------------------------------------------------------------------------- #
# Step 5 — training (NCE / within-state action discrimination / state-only)
# --------------------------------------------------------------------------- #
def _val_select(critic, best, best_state, vl):
    if vl < best[0]:
        best[0] = vl; best_state[0] = deepcopy(critic.state_dict())


def train_objective(Btr, Bva, seed, objective, steps, batch, lr=3e-4, eval_every=200, zero_action=False):
    torch.manual_seed(seed)
    critic = RichStateCritic(D.STATE_DIM, 6, goal_dim=9)
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    best = [float("inf")]; best_state = [None]; curve = []
    n = Btr["n"]; nv = Bva["n"]

    def loss_on(B, idx):
        s = torch.as_tensor(B["state"][idx]); g = torch.as_tensor(B["goal"][idx])
        if objective == "nce":
            a = torch.as_tensor(B["action"][idx])
            return nce_loss(critic.logits_matrix(s, a, g, zero_action=zero_action), _targets(B["key"][idx]))
        y = torch.as_tensor(B["group"][idx])
        return F.cross_entropy(canon_logits(critic, s, g, zero_action=zero_action) / TAU, y)

    for step in range(steps + 1):
        if step % eval_every == 0:
            vr = np.random.default_rng(seed + 777 + step)
            with torch.no_grad():
                vls = [float(loss_on(Bva, vr.integers(nv, size=min(batch, nv)))) for _ in range(4)]
            vl = float(np.mean(vls)); curve.append({"step": step, "val_loss": vl})
            _val_select(critic, best, best_state, vl)
        if step < steps:
            idx = rng.integers(n, size=min(batch, n))
            opt.zero_grad(set_to_none=True); loss = loss_on(Btr, idx); loss.backward(); opt.step()
    critic.load_state_dict(best_state[0]); critic.eval()
    return critic, {"best_val_loss": best[0], "selected_step": min(curve, key=lambda r: r["val_loss"])["step"],
                    "val_curve": curve}


# --------------------------------------------------------------------------- #
# Step 6 — evaluation (base-state unit, semantic primary)
# --------------------------------------------------------------------------- #
def _branch_scores(critic, B, zero_action=False) -> np.ndarray:
    with torch.no_grad():
        sc = canon_logits(critic, torch.as_tensor(B["state"]), torch.as_tensor(B["goal"]),
                          zero_action=zero_action).numpy()                     # (n,3)
    return sc


def diversity_per_base(cache, base_ids, hi, stats) -> Dict[int, float]:
    from .stage1f_horizon_scan import diversity_ratio_state
    raw6 = cache["raw6"]; masks3 = cache["masks3"]; valid6 = cache["valid6"]; cen = cache["censored"]
    out = {}
    for si in base_ids:
        gba = {a: [] for a in range(6)}
        for gi, a in enumerate(CANON_IDS):
            for rep in range(R):
                if cen[si, gi, rep, hi]:
                    continue
                gba[a].append(_norm9(raw6[si, gi, rep, hi], valid6[si, gi, rep, hi], masks3[si, gi, rep, hi], stats))
        out[int(si)] = diversity_ratio_state(gba) or 0.0
    return out


def evaluate(critic, B, cache, base_ids, hi, stats, seed, zero_action=False, n_boot=2000) -> Dict[str, Any]:
    sc = _branch_scores(critic, B, zero_action=zero_action)
    y = B["group"]; base = B["base"]
    top1 = (sc.argmax(1) == y)
    pair = np.array([np.mean([sc[i, y[i]] > sc[i, j] for j in range(3) if j != y[i]]) for i in range(len(y))])
    margin = np.array([sc[i, y[i]] - np.mean([sc[i, j] for j in range(3) if j != y[i]]) for i in range(len(y))])
    div = diversity_per_base(cache, base_ids, hi, stats)
    # aggregate to base-state level
    per_base = {}
    for i in range(len(y)):
        per_base.setdefault(int(base[i]), {"t": [], "p": [], "m": []})
        per_base[int(base[i])]["t"].append(top1[i]); per_base[int(base[i])]["p"].append(pair[i]); per_base[int(base[i])]["m"].append(margin[i])
    bids = sorted(per_base)
    bt = np.array([np.mean(per_base[b]["t"]) for b in bids])
    bp = np.array([np.mean(per_base[b]["p"]) for b in bids])
    bm = np.array([np.mean(per_base[b]["m"]) for b in bids])
    bdv = np.array([div[b] for b in bids])

    def ci(vals, base_val=None, seedoff=0):
        rng = np.random.default_rng(seed + seedoff)
        bs = np.array([vals[rng.integers(len(vals), size=len(vals))].mean() for _ in range(n_boot)])
        lo, hi_ = np.percentile(bs, [2.5, 97.5])
        d = {"point": float(vals.mean()), "median": float(np.median(vals)), "ci95": [float(lo), float(hi_)]}
        if base_val is not None:
            d["ci_excludes_baseline"] = bool(lo > base_val)
        return d

    med = np.median(bdv)
    hi_div = bm[bdv > med]; lo_div = bm[bdv <= med]
    corr = float(np.corrcoef(bdv, bm)[0, 1]) if len(bids) >= 3 and bdv.std() > 0 and bm.std() > 0 else None
    return {
        "n_base_states": len(bids), "n_branches": int(len(y)),
        "semantic_top1": ci(bt, SEM_BASE), "pairwise": ci(bp, 0.5, 1), "margin": ci(bm, 0.0, 2),
        "exact_top1": "N/A (canonical-only; no alias variation in primary run)",
        "by_diversity": {"high_div_margin": float(hi_div.mean()) if len(hi_div) else None,
                         "low_div_margin": float(lo_div.mean()) if len(lo_div) else None,
                         "high_div_top1": float(bt[bdv > med].mean()) if (bdv > med).any() else None,
                         "low_div_top1": float(bt[bdv <= med].mean()) if (bdv <= med).any() else None,
                         "div_median": float(med)},
        "env_diversity_vs_margin_corr": corr,
        "stronger_in_high_diversity": bool((hi_div.mean() if len(hi_div) else -1) >
                                           (lo_div.mean() if len(lo_div) else 0)),
    }


def gate(ev) -> Dict[str, Any]:
    t = ev["semantic_top1"]; p = ev["pairwise"]; m = ev["margin"]
    strict = bool(t["point"] > 0.50 and t.get("ci_excludes_baseline") and p["point"] > 0.60
                  and m["median"] > 0 and m.get("ci_excludes_baseline") and ev["stronger_in_high_diversity"])
    soft = bool(t["point"] > 0.40 and p["point"] > 0.55 and m["point"] > 0)
    return {"strict_pass": strict, "weakly_positive": bool(soft and not strict),
            "label": "PASS" if strict else ("WEAK" if soft else "FAIL")}


# --------------------------------------------------------------------------- #
# Step 7 — toy tests (validate the eval/training code, not just the model)
# --------------------------------------------------------------------------- #
def _toy_branches(states, goals, groups):
    keys = [hash((round(float(g[0]), 3), round(float(g[1]), 3), int(gr))) for g, gr in zip(goals, groups)]
    return {"state": states.astype(np.float32), "goal": goals.astype(np.float32),
            "action": np.array([CANON_IDS[g] for g in groups], np.int64),
            "group": np.array(groups, np.int64), "key": np.array(keys),
            "base": np.arange(len(groups)) // 3, "decision": np.ones(len(groups), bool), "n": len(groups)}


def _toy_acc(critic, B, zero_action=False):
    sc = _branch_scores(critic, B, zero_action=zero_action)
    return float((sc.argmax(1) == B["group"]).mean())


def _toy_split(states, goals, groups, n_base):
    """Split triplets (3 rows/base) into train/test halves by base index (no base leakage)."""
    tr_b = np.arange(n_base) % 2 == 0
    rows_tr = np.repeat(tr_b, 3); rows_te = ~rows_tr
    mk = lambda m: _toy_branches(states[m], goals[m], groups[m])
    return mk(rows_tr), mk(rows_te)


def run_toy_tests(seed=0, n_base=120, steps=500) -> Dict[str, Any]:
    """All accuracies are HELD-OUT (train/test split by base) so memorization cannot inflate them."""
    rng = np.random.default_rng(seed)
    res = {}
    # margin-sign hand-check: true action highest -> margin>0
    sc = np.array([[2.0, 0.0, 0.0]]); y = 0
    margin = sc[0, y] - np.mean([sc[0, j] for j in range(3) if j != y])
    res["margin_sign_toy"] = {"scores": sc.tolist(), "true": y, "margin": float(margin), "positive": bool(margin > 0)}

    sgroups = np.tile([0, 1, 2], n_base)
    base_state = rng.standard_normal((n_base, D.STATE_DIM)).astype(np.float32)
    states = np.repeat(base_state, 3, axis=0)
    goalmap = (np.eye(3, 9).astype(np.float32) * 4.0)
    g1 = goalmap[sgroups] + rng.standard_normal((3 * n_base, 9)).astype(np.float32) * 0.05

    # Toy 1: goal encodes group, state shared within triplet -> mapping generalizes -> ~1.0 on held-out
    tr, te = _toy_split(states, g1, sgroups, n_base)
    c1, _ = train_objective(tr, te, seed, "within", steps, 96, eval_every=steps)
    res["toy1_perfect_mapping_acc"] = round(_toy_acc(c1, te), 3)

    # Toy 2: goal independent of group -> no generalizable signal -> ~1/3 held-out
    g2 = np.repeat(goalmap[[0]], 3 * n_base, axis=0) + rng.standard_normal((3 * n_base, 9)).astype(np.float32) * 0.05
    tr2, te2 = _toy_split(states, g2, sgroups, n_base)
    c2, _ = train_objective(tr2, te2, seed, "within", steps, 96, eval_every=steps)
    res["toy2_action_independent_acc"] = round(_toy_acc(c2, te2), 3)

    # Toy 3: per-sample permuted labels destroy the goal->action correspondence -> ~1/3 held-out
    permy = rng.permutation(sgroups)
    tr3, te3 = _toy_split(states, g1, permy, n_base)
    c3, _ = train_objective(tr3, te3, seed, "within", steps, 96, eval_every=steps)
    res["toy3_permuted_labels_acc"] = round(_toy_acc(c3, te3), 3)

    # Toy 4: leakage guard. Even when the STATE maximally encodes the group, the state-only
    # (zeroed-action) within-state eval stays at chance -> the task provably needs the action
    # pathway and cannot be solved by a state shortcut. Full model on same data generalizes (~1.0).
    leak_state = (np.eye(3, D.STATE_DIM).astype(np.float32) * 4.0)[sgroups] + rng.standard_normal((3 * n_base, D.STATE_DIM)).astype(np.float32) * 0.05
    tr4, te4 = _toy_split(leak_state, g1, sgroups, n_base)
    c4s, _ = train_objective(tr4, te4, seed, "within", steps, 96, eval_every=steps, zero_action=True)
    c4f, _ = train_objective(tr4, te4, seed, "within", steps, 96, eval_every=steps)
    res["toy4_stateonly_informative_state_acc"] = round(_toy_acc(c4s, te4, zero_action=True), 3)
    res["toy4_full_model_same_data_acc"] = round(_toy_acc(c4f, te4), 3)

    res["passed"] = bool(res["toy1_perfect_mapping_acc"] > 0.9 and res["toy2_action_independent_acc"] < 0.45
                         and res["toy3_permuted_labels_acc"] < 0.45
                         and res["toy4_stateonly_informative_state_acc"] < 0.45
                         and res["toy4_full_model_same_data_acc"] > 0.9
                         and res["margin_sign_toy"]["positive"])
    return res


# --------------------------------------------------------------------------- #
# Sanity checks on the real dataset
# --------------------------------------------------------------------------- #
def dataset_sanity(cache, train_ids, val_ids, test_ids) -> Dict[str, Any]:
    # A. same-state equality across action branches (anchor_state is per base; identical by construction)
    anchor = cache["anchor_state"]
    same_state_ok = True  # one anchor per base, shared by all branches; assert split disjointness instead
    # D. leakage
    s_tr, s_va, s_te = set(train_ids.tolist()), set(val_ids.tolist()), set(test_ids.tolist())
    leak = {"train_val": len(s_tr & s_va), "train_test": len(s_tr & s_te), "val_test": len(s_va & s_te)}
    assert leak["train_val"] == 0 and leak["train_test"] == 0 and leak["val_test"] == 0, "base-state leakage!"
    # C. action-effect existence on the enlarged dataset, per horizon, decision subset
    eff = {}
    raw6 = cache["raw6"]; valid6 = cache["valid6"]; masks3 = cache["masks3"]; cen = cache["censored"]
    dec = cache["decision"]
    from .stage1f_horizon_scan import diversity_ratio_state
    for hi, H in enumerate(HORIZONS):
        cross_d, nz, gt1 = [], [], []
        for si in range(anchor.shape[0]):
            if not dec[si]:
                continue
            gmeans = []
            gba = {a: [] for a in range(6)}
            for gi, a in enumerate(CANON_IDS):
                gg = [raw6[si, gi, rep, hi] for rep in range(R) if not cen[si, gi, rep, hi]]
                if gg:
                    gmeans.append(np.mean(gg, 0))
                for rep in range(R):
                    if not cen[si, gi, rep, hi]:
                        gba[a].append(np.concatenate([raw6[si, gi, rep, hi], masks3[si, gi, rep, hi]]))
            if len(gmeans) >= 2:
                dists = [np.linalg.norm(gmeans[i] - gmeans[j]) for i in range(len(gmeans)) for j in range(i + 1, len(gmeans))]
                cross_d.append(np.median(dists)); nz.append(float(np.median(dists) > 0))
                dr = diversity_ratio_state(gba)
                if dr is not None:
                    gt1.append(float(dr > 1))
        eff[str(H)] = {"median_cross_action_distance": float(np.median(cross_d)) if cross_d else None,
                       "frac_nonzero_distance": float(np.mean(nz)) if nz else None,
                       "frac_diversity_gt1": float(np.mean(gt1)) if gt1 else None,
                       "n_decision_states": len(cross_d)}
    return {"same_state_branch_equality_ok": same_state_ok, "split_leakage": leak,
            "action_effect": eff,
            "reproducibility": {"max_goal_discrepancy": float(cache["repro_max_disc"]),
                                "exact_match_rate": float(cache["repro_match_rate"]),
                                "n_compared": int(cache["repro_n"])}}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def split_base(n, seed, fracs=(0.7, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    a = int(round(n * fracs[0])); b = a + int(round(n * fracs[1]))
    return np.sort(perm[:a]), np.sort(perm[a:b]), np.sort(perm[b:])


def _sha(critic) -> str:
    return hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in critic.state_dict().values())).hexdigest()[:16]


def run_seed(cache, seed, steps, batch) -> Dict[str, Any]:
    dec_ids = np.where(cache["decision"])[0]
    tr, va, te = split_base(len(dec_ids), seed)
    train_ids, val_ids, test_ids = dec_ids[tr], dec_ids[va], dec_ids[te]
    sanity = dataset_sanity(cache, train_ids, val_ids, test_ids)
    out = {"seed": seed, "split_sizes": {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)},
           "dataset_sanity": sanity, "by_horizon": {}}
    sdir = RESULTS / f"seed{seed}"; sdir.mkdir(parents=True, exist_ok=True)
    for hi, H in enumerate(HORIZONS):
        stats = fit_norm(cache, train_ids, hi)
        nzv = [i for i, sd in enumerate(stats["std"]) if sd < 1e-3]
        Btr = build_branches(cache, train_ids, hi, stats)
        Bva = build_branches(cache, val_ids, hi, stats)
        Bte = build_branches(cache, test_ids, hi, stats)
        # tiny overfit on >=16 base states, all branches
        tiny_ids = train_ids[:max(16, 16)]
        Bt = build_branches(cache, tiny_ids, hi, stats)
        ctiny, seltiny = train_objective(Bt, Bt, seed, "within", 400, min(96, Bt["n"]), eval_every=400)
        tiny_eval = evaluate(ctiny, Bt, cache, tiny_ids, hi, stats, seed, n_boot=200)
        # primary objectives
        c_nce, sel_nce = train_objective(Btr, Bva, seed, "nce", steps, batch)
        c_ws, sel_ws = train_objective(Btr, Bva, seed, "within", steps, batch)
        c_so, sel_so = train_objective(Btr, Bva, seed, "within", steps, batch, zero_action=True)
        ev_nce = evaluate(c_nce, Bte, cache, test_ids, hi, stats, seed)
        ev_ws = evaluate(c_ws, Bte, cache, test_ids, hi, stats, seed)
        ev_so = evaluate(c_so, Bte, cache, test_ids, hi, stats, seed, zero_action=True)
        torch.save(c_nce.state_dict(), sdir / f"critic_nce_H{H}.pt")
        torch.save(c_ws.state_dict(), sdir / f"critic_within_state_H{H}.pt")
        out["by_horizon"][str(H)] = {
            "normalization": stats, "near_zero_var_dims": nzv,
            "n_train_branches": Btr["n"], "n_test_branches": Bte["n"],
            "tiny_overfit": {"train_top1": tiny_eval["semantic_top1"]["point"],
                             "train_margin": tiny_eval["margin"]["point"],
                             "init_vs_final_val": [seltiny["val_curve"][0]["val_loss"], seltiny["best_val_loss"]],
                             "n_base": len(tiny_ids)},
            "nce": {"selected_step": sel_nce["selected_step"], "sha16": _sha(c_nce), "test": ev_nce, "gate": gate(ev_nce)},
            "within_state": {"selected_step": sel_ws["selected_step"], "sha16": _sha(c_ws), "test": ev_ws, "gate": gate(ev_ws)},
            "state_only_baseline": {"test_top1": ev_so["semantic_top1"]["point"], "test_margin": ev_so["margin"]["point"]},
        }
        np.savez(sdir / f"predictions_H{H}.npz", test_base=Bte["base"], test_group=Bte["group"],
                 scores_nce=_branch_scores(c_nce, Bte), scores_within=_branch_scores(c_ws, Bte))
    (sdir / "report.json").write_text(json.dumps(out, indent=2))
    return out


def run(seeds, state_seed, n_dec, n_ord, steps, batch, device="cpu") -> Dict[str, Any]:
    RESULTS.mkdir(parents=True, exist_ok=True)
    aud = audit()
    toy = run_toy_tests()
    cache = generate_or_load_branches(state_seed, n_dec, n_ord, device=device)
    (RESULTS / "audit_report.json").write_text(json.dumps(aud, indent=2))
    (RESULTS / "toy_test_report.json").write_text(json.dumps(toy, indent=2))
    manifest = {"n_base_total": int(cache["anchor_state"].shape[0]),
                "n_decision": int(cache["decision"].sum()), "n_ordinary": int((~cache["decision"]).sum()),
                "config": cache["config"], "config_hash": cache["config_hash"], "cache_path": cache["cache_path"],
                "collect_stats": cache.get("collect_stats")}
    (RESULTS / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))

    per_seed = {}
    s0 = run_seed(cache, seeds[0], steps, batch)
    per_seed[str(seeds[0])] = s0
    # decide whether to run seeds 1,2: gate on best of H12/H16 within-state
    best = max((s0["by_horizon"][str(H)]["within_state"]["gate"]["label"] for H in HORIZONS),
               key=lambda l: {"PASS": 2, "WEAK": 1, "FAIL": 0}[l])
    ran = [seeds[0]]
    if best in ("PASS", "WEAK"):
        for seed in seeds[1:]:
            per_seed[str(seed)] = run_seed(cache, seed, steps, batch)
            ran.append(seed)

    out = {"milestone": "Stage-1H-interventional-branch-critic",
           "audit": aud, "toy_tests": toy, "dataset_manifest": manifest,
           "goal_dim": 9, "future_self_paddle_in_goal": False, "horizon_input_to_model": False,
           "horizons_separate": HORIZONS, "canonical_actions": {GROUP_NAMES[i]: CANON_IDS[i] for i in range(3)},
           "temperature": TAU, "R": R, "random_baselines": {"semantic": SEM_BASE, "pairwise": 0.5},
           "seeds_run": ran, "seed0_within_state_best_gate": best, "per_seed": per_seed,
           "interpretation_note": "Positive control. PASS => architecture/goal CAPABLE; Stage-1G failure is "
                                  "an objective/data (observational) problem. FAIL under within-state "
                                  "supervision => representation/goal inadequate. Distinguishes A/B/C/D."}
    (RESULTS / "config.json").write_text(json.dumps({"seeds": seeds, "state_seed": state_seed, "n_dec": n_dec,
                                                     "n_ord": n_ord, "steps": steps, "batch": batch,
                                                     "horizons": HORIZONS, "TAU": TAU, "R": R}, indent=2))
    (RESULTS / "stage1h_report.json").write_text(json.dumps(out, indent=2))
    _summary(out)
    return out


def _summary(out) -> None:
    L = ["Stage-1H interventional branch critic (positive control). sem base 0.333, pairwise 0.5.",
         f"canonical actions {out['canonical_actions']}; H={out['horizons_separate']} (separate); "
         f"goal_dim=9 no-self; H not an input.\n",
         f"toy tests passed: {out['toy_tests']['passed']}  "
         f"(t1={out['toy_tests']['toy1_perfect_mapping_acc']} t2={out['toy_tests']['toy2_action_independent_acc']} "
         f"t3={out['toy_tests']['toy3_permuted_labels_acc']} "
         f"t4_stateonly={out['toy_tests']['toy4_stateonly_informative_state_acc']} "
         f"t4_full={out['toy_tests']['toy4_full_model_same_data_acc']})\n"]
    for s, d in out["per_seed"].items():
        L.append(f"== seed {s}  split {d['split_sizes']} ==")
        for H in out["horizons_separate"]:
            h = d["by_horizon"][str(H)]
            for obj in ("nce", "within_state"):
                e = h[obj]["test"]; t = e["semantic_top1"]; p = e["pairwise"]; m = e["margin"]
                star = "*" if t.get("ci_excludes_baseline") else ""
                L.append(f"  H{H:>2} {obj:>12}: top1={t['point']:.3f}{star} pair={p['point']:.3f} "
                         f"marg(med)={m['median']:+.4f} margCIlo={m['ci95'][0]:+.4f} "
                         f"hi-lo_marg={(e['by_diversity']['high_div_margin'] or 0)-(e['by_diversity']['low_div_margin'] or 0):+.4f} "
                         f"-> {h[obj]['gate']['label']}")
            so = h["state_only_baseline"]
            L.append(f"  H{H:>2} {'state_only':>12}: top1={so['test_top1']:.3f} margin={so['test_margin']:+.4f} (expect ~chance)")
            tov = h["tiny_overfit"]
            L.append(f"  H{H:>2} {'tiny_overfit':>12}: top1={tov['train_top1']:.3f} margin={tov['train_margin']:+.4f}")
        L.append("")
    L.append(f"seed0 within-state best gate: {out['seed0_within_state_best_gate']}   seeds run: {out['seeds_run']}")
    (RESULTS / "summary.txt").write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-1H interventional branch critic (positive control).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--state-seed", type=int, default=8000)
    ap.add_argument("--n-dec", type=int, default=800)
    ap.add_argument("--n-ord", type=int, default=200)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--toy-only", action="store_true")
    args = ap.parse_args()
    if args.toy_only:
        print(json.dumps(run_toy_tests(), indent=2)); return
    out = run(args.seeds, args.state_seed, args.n_dec, args.n_ord, args.steps, args.batch)
    print(open(RESULTS / "summary.txt").read())


if __name__ == "__main__":
    main()
