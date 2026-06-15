"""Stage-1K — expert-policy validation & perturbation-saliency audit for the DIAMOND teacher.

Purpose: directly inspect WHAT the current recurrent DIAMOND actor uses to choose actions, to
separate four things the prior stages conflated — (1) competence, (2) integration validity,
(3) general policy plausibility (ball/own-paddle), (4) whether the action actually depends on the
opponent/scoreboard information hidden from the learner. NO critic / learner / causal training.

Method: perturbation saliency, faithfully adapted from the upstream teacher repo
(`saliency/saliency.py`, Greydanus-2017): occlude(I,m)=I*(1-m)+gaussian_filter(I,σ=3)*m with a soft
circular mask get_mask(center,size,r), density-d grid on the 64×64 RGB teacher input, pre-forward
recurrent state taken from the cached (obs, hx_cx) and never mutated. The ONLY deviation from
upstream is the SCORE: upstream uses 0.5·||L−l||² on raw logits; we make the PRIMARY score the
Jensen–Shannon divergence of the SEMANTIC action distribution (STAY/UP/DOWN) and keep logit-L2 +
a full metric suite as secondaries. Parameters reuse upstream: radius=4, density=4, σ=3.

Parts: 0 provenance audit · replay validation (stop if cached logits not reproduced) · A grid
heatmaps · B object-region perturbations (+area-matched bg control) · C temporal/recurrent opponent
replay (K=1,4,8,16) · D recurrent-state ablation. T=2 (behavior) primary, T=1 (raw) secondary.
Improved/observed saliency does NOT identify P(G|S,do(A)); no causal claim. Nothing committed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from .. import config as C
from ..objects import extract_pong_objects
from ..teacher.external_teacher import load_external_teacher
from ..teacher.load_teacher import TeacherPolicy, load_teacher, make_env

RESULTS = Path("pong_action_gate/results/stage1k_teacher_policy_saliency")
CACHE = Path("artifacts/pong_action_gate/stage1k")
GROUPS = {0: 0, 1: 0, 2: 1, 4: 1, 3: 2, 5: 2}           # exact ALE -> semantic group
GROUP_MEMBERS = {0: [0, 1], 1: [2, 4], 2: [3, 5]}        # STAY / UP / DOWN
GROUP_NAMES = ["STAY", "UP", "DOWN"]
RADIUS, DENSITY, BLUR_SIGMA = 4, 4, 3                    # upstream Pong saliency params
SX, SY = 64 / 160.0, 64 / 210.0                         # native(160w,210h) -> 64x64 (no crop)
PADDLE_AGENT_X, PADDLE_OPP_X = 140.0, 16.0              # native paddle x


# --------------------------------------------------------------------------- #
# Semantic distributions + saliency metrics
# --------------------------------------------------------------------------- #
def _softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def semantic_probs(logits, T):
    p = _softmax(np.asarray(logits, float) / T)          # (...,6)
    out = np.stack([p[..., GROUP_MEMBERS[g]].sum(-1) for g in range(3)], -1)
    return out, p


def _js(p, q):
    p = np.clip(p, 1e-12, 1); q = np.clip(q, 1e-12, 1)
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * (np.log2(a) - np.log2(b)), -1)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)               # base-2 JS in [0,1]


def metrics(L, l, T):
    """Saliency metrics between baseline logits L and perturbed logits l (both 1-D, 6 actions)."""
    Ps, Pe = semantic_probs(L, T); Qs, Qe = semantic_probs(l, T)
    pref = int(np.argmax(Ps))
    return {"js_sem": float(_js(Qs, Ps)), "js_exact": float(_js(Qe, Pe)),
            "l1_sem": float(np.abs(Qs - Ps).sum()),
            "logit_l2": float(np.sqrt(((np.asarray(l, float) - L) ** 2).sum())),
            "top1_flip": float(int(np.argmax(Qs) != pref)),
            "pref_drop": float(Ps[pref] - Qs[pref])}


# --------------------------------------------------------------------------- #
# Perturbations (upstream-faithful)
# --------------------------------------------------------------------------- #
def get_mask(center, size=(64, 64), r=RADIUS):
    y, x = np.ogrid[-center[0]:size[0] - center[0], -center[1]:size[1] - center[1]]
    m = np.zeros(size); m[x * x + y * y <= 1] = 1
    m = gaussian_filter(m, sigma=r)
    return (m / m.max()).astype(np.float32)              # (64,64) in [0,1]


def _blur(o):
    return gaussian_filter(o, sigma=(0, BLUR_SIGMA, BLUR_SIGMA)).astype(np.float32)   # spatial only


def occlude(o, m, o_blur):
    return (o * (1 - m) + o_blur * m).astype(np.float32)


def bgfill(o, m, bg):
    return (o * (1 - m) + bg[:, None, None] * m).astype(np.float32)


def box_mask(cx, cy, w, h, soft=1.0):
    m = np.zeros((64, 64), np.float32)
    x0, x1 = int(max(0, cx - w / 2)), int(min(64, cx + w / 2 + 1))
    y0, y1 = int(max(0, cy - h / 2)), int(min(64, cy + h / 2 + 1))
    m[y0:y1, x0:x1] = 1.0
    if soft:
        m = gaussian_filter(m, sigma=soft)
        m = m / (m.max() + 1e-8)
    return m.astype(np.float32)


# --------------------------------------------------------------------------- #
# Batched teacher forward (broadcast pre-forward hidden; never mutates inputs)
# --------------------------------------------------------------------------- #
def batch_logits(model, obs_arr, h, bs=64):
    hx, cx = h
    out = []
    with torch.no_grad():
        for s in range(0, len(obs_arr), bs):
            ob = torch.as_tensor(obs_arr[s:s + bs])
            B = ob.shape[0]
            lg, _, _ = model.predict_act_value(ob, (hx.repeat(B, 1), cx.repeat(B, 1)))
            out.append(lg.numpy())
    return np.concatenate(out, 0)


def one_logits(model, obs, h):
    with torch.no_grad():
        lg, _, _ = model.predict_act_value(torch.as_tensor(obs[None]), (h[0], h[1]))
    return lg.numpy()[0]


# --------------------------------------------------------------------------- #
# Object boxes (native RAM -> 64x64 teacher-input coords)
# --------------------------------------------------------------------------- #
def object_boxes(o):
    boxes = {}
    if o.ball_present and o.ball_x is not None:
        boxes["ball"] = (o.ball_x * SX, o.ball_y * SY, max(2, 3 * SX) + 2, max(2, 5 * SY) + 2)
    if o.player_y is not None:
        boxes["agent_paddle"] = (PADDLE_AGENT_X * SX, o.player_y * SY, 3, 16 * SY + 2)
    if o.opp_y is not None:
        boxes["opp_paddle"] = (PADDLE_OPP_X * SX, o.opp_y * SY, 3, 16 * SY + 2)
    boxes["scoreboard"] = (32.0, 6.0, 60.0, 12.0)        # Pong score band at top
    # area-matched empty background control: mid-field, away from ball/paddles
    boxes["background"] = (32.0, 50.0, 3.0, 16 * SY + 2)
    return boxes


# --------------------------------------------------------------------------- #
# Step 0 — provenance audit
# --------------------------------------------------------------------------- #
def provenance_audit(meta) -> Dict[str, Any]:
    ext = load_external_teacher()
    root = Path(ext.root)
    sal = root / "saliency" / "saliency.py"
    return {
        "checkpoint": {"path": meta["ckpt_path"], "sha256": meta["ckpt_sha256"],
                       "n_params": meta["n_actor_critic_params"], "source": "HuggingFace eloialonso/diamond Pong.pt"},
        "teacher_provenance": ext.provenance, "teacher_root": ext.root, "stubbed": ext.stubbed_modules,
        "input_tensor": {"full_obs_shape": [1, 7, 64, 64], "teacher_input": "obs[:,4:7] (latest RGB frame)",
                         "shape": [1, 3, 64, 64], "dtype": "float32", "range": "~[-1,1] (x/255*2-1)",
                         "channel_order": "RGB", "frame_stack": "NONE for teacher (single frame); ch0-3 are "
                         "grayscale stack used only by student/sebulba", "resize": "cv2.resize 210x160->64x64 "
                         "INTER_AREA, NO crop -> x64=x_nat*0.4, y64=y_nat*0.305"},
        "recurrent_state": {"hx_cx_shape": [1, 512], "lstm_dim": 512, "reset": "zeros at episode start (None->zeros)",
                            "semantics": "predict_act_value(obs, (hx,cx)) takes PRE-forward hidden, returns "
                            "POST-forward hidden; saliency varies obs with pre-forward hidden fixed"},
        "logits": {"shape": [1, 6], "ale_mapping": C.ACTION_MEANINGS, "semantic_groups": GROUP_MEMBERS},
        "sampling": "Categorical(logits / BEHAVIOR_TEMPERATURE).sample()  (T applied to logits before sampling)",
        "temperature": {"T_behavior": C.BEHAVIOR_TEMPERATURE, "raw_policy_is_T1": True,
                        "T_location": "logits/T before softmax/sample; forward pass identical for T1/T2"},
        "epsilon_in_teacher": False,
        "upstream_saliency": {"exists": sal.exists(), "file": str(sal),
                              "method": "Greydanus-2017 occlusion: I*(1-m)+gaussian_filter(I,sigma=3)*m, "
                              "get_mask circular, score_frame density-grid, score=0.5*||L-l||^2 on raw logits",
                              "params_used_upstream_pong": {"radius": 4, "density": 4, "mode": "actor/critic"},
                              "reuse": "perturbation/mask/density/sigma reused EXACTLY; PRIMARY score changed to "
                              "semantic-JS (+ full metric suite); upstream logit-L2 kept as secondary"},
        "deviation_from_upstream": "score metric only (semantic JS primary vs upstream logit-L2); "
                                   "perturbation, mask, grid, recurrent handling identical",
    }


# --------------------------------------------------------------------------- #
# Collection of a dedicated saliency dataset (eps=0, unchanged teacher path)
# --------------------------------------------------------------------------- #
def _category_flags(o, prev, ag, op, max_diff, just_scored, just_served):
    dx = (o.ball_x - prev.ball_x) if (o.ball_present and prev.ball_present and prev.ball_x is not None) else None
    decision = bool(o.ball_present and dx is not None and dx > 0 and o.ball_x is not None and o.ball_x >= 120)
    defensive = bool(o.ball_present and dx is not None and dx > 0 and o.ball_x is not None and o.ball_x >= 80)
    opp_side = bool(o.ball_present and ((dx is not None and dx < 0) or (o.ball_x is not None and o.ball_x < 80)))
    score_restart = bool(just_scored or just_served or abs(ag - op) >= 10 or max_diff >= 13)
    return {"decision": decision, "defensive": defensive, "opp_side": opp_side,
            "score_restart": score_restart, "ordinary": not decision, "ball_dx": dx}


def collect_states(model, teacher, env, seed, quotas, window=16) -> Tuple[List[Dict], Dict]:
    ale = env.env.ale
    want = dict(quotas); picked = []
    by_cat = {k: 0 for k in quotas}
    dec_by_grp = {0: 0, 1: 0, 2: 0}
    rng = np.random.default_rng(seed)
    ep = 0; stats = {"episodes": 0, "candidates": 0}
    while sum(by_cat.values()) < sum(want.values()) * 2 and ep < 40:
        torch.manual_seed(seed + ep)
        obs, info = env.reset(seed=[seed + ep]); hx, cx = teacher.initial_state(1)
        prev = extract_pong_objects(ale.getRAM()); ag = op = 0; t = 0; max_diff = 0
        hist_obs, hist_hx, hist_cx = [], [], []
        served_at = 0
        while True:
            o = extract_pong_objects(ale.getRAM())
            ti = obs[:, C.TEACHER_OBS_SLICE].numpy()[0].copy()       # (3,64,64) pre-forward input
            with torch.no_grad():
                lg, _, (hx2, cx2) = model.predict_act_value(obs[:, C.TEACHER_OBS_SLICE], (hx, cx))
            logits = lg.numpy()[0]
            hist_obs.append(ti); hist_hx.append(hx.numpy()[0].copy()); hist_cx.append(cx.numpy()[0].copy())
            a = int(torch.distributions.Categorical(logits=lg / C.BEHAVIOR_TEMPERATURE).sample().item())
            just_served = bool((not prev.ball_present) and o.ball_present)
            if just_served:
                served_at = t
            obs2, rew, end, trunc, info = env.step(a)
            r = float(rew.item()); ag += int(r > 0); op += int(r < 0); max_diff = max(max_diff, ag - op)
            flags = _category_flags(o, prev, ag, op, max_diff, r != 0, just_served)
            stats["candidates"] += 1
            # decide whether to keep this state (fill the neediest matching category)
            cats = [c for c in ["decision", "defensive", "opp_side", "score_restart", "ordinary"]
                    if flags[c] and by_cat[c] < want[c] * 2]
            keep = o.player_y is not None and t >= window and (t - served_at) >= 0 and len(cats) > 0 and rng.random() < 0.5
            if keep and flags["decision"]:
                g = GROUPS[a]
                if dec_by_grp[g] >= max(want["decision"], 1):   # balance executed semantic action
                    keep = bool(rng.random() < 0.2)
            if keep:
                cat = cats[0]
                win_o = np.stack(hist_obs[-window - 1:]) if len(hist_obs) > window else None
                rec = {"ep": ep, "t": t, "obs": ti, "hx": hx.numpy()[0].copy(), "cx": cx.numpy()[0].copy(),
                       "logits": logits, "action": a, "sem": GROUPS[a],
                       "ball_x": o.ball_x, "ball_y": o.ball_y, "ball_present": bool(o.ball_present),
                       "player_y": o.player_y, "opp_y": o.opp_y, "opp_present": o.opp_y is not None,
                       "ball_dx": flags["ball_dx"], "agent": ag, "opp": op, "category": cat,
                       "flags": {k: bool(flags[k]) for k in ["decision", "defensive", "opp_side", "score_restart", "ordinary"]},
                       "render": np.asarray(env.env.render()).astype(np.uint8),
                       "win_obs": win_o,
                       "win_hx": np.stack(hist_hx[-window - 1:]) if win_o is not None else None,
                       "win_cx": np.stack(hist_cx[-window - 1:]) if win_o is not None else None}
                picked.append(rec); by_cat[cat] += 1
                if flags["decision"]:
                    dec_by_grp[GROUPS[a]] += 1
            prev = o; obs = obs2; hx, cx = hx2, cx2; t += 1
            if bool((end | trunc).item()) or t >= 4000:
                break
        ep += 1; stats["episodes"] = ep
    stats["by_category_collected"] = by_cat; stats["decision_by_group"] = dec_by_grp
    return picked, stats


def select_manifest(picked, quotas) -> List[Dict]:
    """Trim to ~sum(quotas) states honoring per-category minimums and decision STAY/UP/DOWN balance."""
    chosen = []; used = set()
    # decision balanced first
    dec = [p for p in picked if p["flags"]["decision"]]
    for g in range(3):
        gg = [p for p in dec if p["sem"] == g and (p["ep"], p["t"]) not in used]
        for p in gg[:max(1, quotas["decision"] // 3 + 1)]:
            chosen.append(p); used.add((p["ep"], p["t"]))
    for cat in ["defensive", "opp_side", "score_restart", "ordinary"]:
        cc = [p for p in picked if p["flags"][cat] and (p["ep"], p["t"]) not in used]
        for p in cc[:quotas[cat]]:
            chosen.append(p); used.add((p["ep"], p["t"]))
    return chosen


# --------------------------------------------------------------------------- #
# Replay validation
# --------------------------------------------------------------------------- #
def replay_validation(model, states, n=20) -> Dict[str, Any]:
    rows = []
    for st in states[:n]:
        h = (torch.as_tensor(st["hx"][None]), torch.as_tensor(st["cx"][None]))
        l = one_logits(model, st["obs"], h)
        Ls, _ = semantic_probs(st["logits"], C.BEHAVIOR_TEMPERATURE); ls, _ = semantic_probs(l, C.BEHAVIOR_TEMPERATURE)
        rows.append({"max_abs_logit_diff": float(np.abs(l - st["logits"]).max()),
                     "sem_prob_l1": float(np.abs(ls - Ls).sum()),
                     "top1_agree": int(np.argmax(ls) == np.argmax(Ls))})
    mx = max(r["max_abs_logit_diff"] for r in rows)
    valid = bool(mx < 1e-4 and all(r["top1_agree"] for r in rows))
    return {"n_states": len(rows), "max_abs_logit_diff": mx,
            "mean_sem_prob_l1": float(np.mean([r["sem_prob_l1"] for r in rows])),
            "top1_agreement_rate": float(np.mean([r["top1_agree"] for r in rows])),
            "valid": valid, "tolerance": 1e-4}


# --------------------------------------------------------------------------- #
# Part A — instantaneous grid saliency
# --------------------------------------------------------------------------- #
def grid_saliency(model, st):
    o = st["obs"]; o_blur = _blur(o); h = (torch.as_tensor(st["hx"][None]), torch.as_tensor(st["cx"][None]))
    L = st["logits"]
    centers = [(i, j) for i in range(0, 64, DENSITY) for j in range(0, 64, DENSITY)]
    perturbed = np.empty((len(centers), 3, 64, 64), np.float32)
    for k, (i, j) in enumerate(centers):
        perturbed[k] = occlude(o, get_mask((i, j))[None], o_blur)
    lg = batch_logits(model, perturbed, h)               # (N,6)
    gw = 64 // DENSITY + (1 if 64 % DENSITY else 0)
    heat = {T: np.zeros((gw, gw), np.float32) for T in ("T2", "T1")}
    for k, (i, j) in enumerate(centers):
        m2 = metrics(L, lg[k], C.BEHAVIOR_TEMPERATURE); m1 = metrics(L, lg[k], 1.0)
        heat["T2"][i // DENSITY, j // DENSITY] = m2["js_sem"]; heat["T1"][i // DENSITY, j // DENSITY] = m1["js_sem"]
    return heat, float(heat["T2"].max()), float(heat["T1"].max())


# --------------------------------------------------------------------------- #
# Part B — object-region perturbations
# --------------------------------------------------------------------------- #
def _bg_color(o):
    return np.median(o.reshape(3, -1), axis=1).astype(np.float32)


def region_perturbations(model, st):
    o = st["obs"]; o_blur = _blur(o); bg = _bg_color(o)
    h = (torch.as_tensor(st["hx"][None]), torch.as_tensor(st["cx"][None])); L = st["logits"]
    cur = extract_pong_objects_from_state(st)
    boxes = object_boxes(cur)
    bg_area = boxes["background"][2] * boxes["background"][3]
    res = {}
    for name, (cx, cy, w, hh) in boxes.items():
        m = box_mask(cx, cy, w, hh)
        area = float(w * hh)
        rr = {"area": area}
        for pert, fn in [("blur", lambda: occlude(o, m[None], o_blur)), ("bgfill", lambda: bgfill(o, m[None], bg))]:
            l = one_logits(model, fn(), h)
            mm = metrics(L, l, C.BEHAVIOR_TEMPERATURE)
            rr[pert] = {"js_sem": float(mm["js_sem"]), "pref_drop": float(mm["pref_drop"]),
                        "top1_flip": float(mm["top1_flip"]), "logit_l2": float(mm["logit_l2"])}
        rr["js_sem_mean"] = 0.5 * (rr["blur"]["js_sem"] + rr["bgfill"]["js_sem"])
        rr["js_per_area"] = rr["js_sem_mean"] / (area + 1e-6)
        res[name] = rr
    bgjs = res["background"]["js_sem_mean"]
    for name in res:
        res[name]["js_rel_background"] = res[name]["js_sem_mean"] - bgjs
    return res


class _Obj:
    pass


def extract_pong_objects_from_state(st):
    o = _Obj()
    o.ball_x = st["ball_x"]; o.ball_y = st["ball_y"]; o.ball_present = st["ball_present"]
    o.player_y = st["player_y"]; o.opp_y = st["opp_y"]
    return o


# --------------------------------------------------------------------------- #
# Part C — temporal / recurrent opponent audit
# --------------------------------------------------------------------------- #
def temporal_opponent(model, st, Ks=(1, 4, 8, 16)):
    if st["win_obs"] is None:
        return None
    win = st["win_obs"]; whx = st["win_hx"]; wcx = st["win_cx"]   # (W+1, ...) ending at t
    W = win.shape[0] - 1
    cur = extract_pong_objects_from_state(st); boxes = object_boxes(cur)
    om = box_mask(*boxes["opp_paddle"])[None] if "opp_paddle" in boxes else None
    sm = box_mask(*boxes["scoreboard"])[None]
    out = {}
    for K in Ks:
        if K > W:
            continue
        start = (W + 1) - 1 - K                                   # index of t-K in window
        h0 = (torch.as_tensor(whx[start][None]), torch.as_tensor(wcx[start][None]))

        def replay(mask):
            h = h0; o_blur_cache = {}
            for s in range(start, W + 1):
                ob = win[s].copy()
                if mask is not None:
                    if s not in o_blur_cache:
                        o_blur_cache[s] = _blur(win[s])
                    ob = occlude(win[s], mask, o_blur_cache[s])
                with torch.no_grad():
                    lg, _, h = model.predict_act_value(torch.as_tensor(ob[None]), h)
            return lg.numpy()[0], (h[0].numpy()[0], h[1].numpy()[0])

        L, hL = replay(None)
        rec = {}
        for tag, mask in [("opponent", om), ("scoreboard", sm)]:
            if mask is None:
                continue
            l, hl = replay(mask)
            mm = metrics(L, l, C.BEHAVIOR_TEMPERATURE)
            hcat_o = np.concatenate([hL[0], hL[1]]); hcat_p = np.concatenate([hl[0], hl[1]])
            cos = float(np.dot(hcat_o, hcat_p) / (np.linalg.norm(hcat_o) * np.linalg.norm(hcat_p) + 1e-9))
            rec[tag] = {"js_sem": float(mm["js_sem"]), "top1_flip": float(mm["top1_flip"]),
                        "pref_drop": float(mm["pref_drop"]),
                        "recur_l2": float(np.linalg.norm(hcat_o - hcat_p)), "recur_cos_dist": 1 - cos}
        out[f"K={K}"] = rec
    return out


# --------------------------------------------------------------------------- #
# Part D — recurrent-state ablation
# --------------------------------------------------------------------------- #
def recurrent_ablation(model, st, mismatch_hx=None):
    o = st["obs"]; L = st["logits"]
    conds = {"saved": (torch.as_tensor(st["hx"][None]), torch.as_tensor(st["cx"][None])),
             "zero": (torch.zeros(1, 512), torch.zeros(1, 512))}
    if mismatch_hx is not None:
        conds["mismatch"] = (torch.as_tensor(mismatch_hx[0][None]), torch.as_tensor(mismatch_hx[1][None]))
    res = {}
    for name, h in conds.items():
        l = one_logits(model, o, h)
        mm = metrics(L, l, C.BEHAVIOR_TEMPERATURE)
        Ps, _ = semantic_probs(l, C.BEHAVIOR_TEMPERATURE)
        res[name] = {"logit_l2_vs_saved": float(mm["logit_l2"]), "js_sem_vs_saved": float(mm["js_sem"]),
                     "top1_sem": GROUP_NAMES[int(np.argmax(Ps))], "top1_flip_vs_saved": float(mm["top1_flip"])}
    return res


# --------------------------------------------------------------------------- #
# Aggregation + bootstrap
# --------------------------------------------------------------------------- #
def _boot(vals, seed=0, n=2000):
    vals = np.asarray([v for v in vals if v is not None], float)
    if len(vals) == 0:
        return None
    rng = np.random.default_rng(seed)
    bs = [vals[rng.integers(len(vals), size=len(vals))].mean() for _ in range(n)]
    return {"mean": float(vals.mean()), "median": float(np.median(vals)),
            "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "n": int(len(vals))}


def region_summary(states, regions_per_state):
    def collect(region, cat=None, key="js_sem_mean"):
        vals = []
        for st, rg in zip(states, regions_per_state):
            if rg is None or region not in rg:
                continue
            if cat is not None and not st["flags"].get(cat, False):
                continue
            vals.append(rg[region][key])
        return vals
    def collect_blur(region, cat, key):
        return [rg[region]["blur"][key] for st, rg in zip(states, regions_per_state)
                if rg and region in rg and (cat is None or st["flags"].get(cat, False))]
    rows = {}
    for label, region, cat in [("ball_all", "ball", None), ("own_paddle_all", "agent_paddle", None),
                               ("opponent_all", "opp_paddle", None), ("opponent_opp_side", "opp_paddle", "opp_side"),
                               ("opponent_decision", "opp_paddle", "decision"), ("scoreboard_all", "scoreboard", None),
                               ("scoreboard_restart", "scoreboard", "score_restart"), ("background", "background", None)]:
        rows[label] = {"js_sem": _boot(collect(region, cat, "js_sem_mean")),
                       "js_rel_background": _boot(collect(region, cat, "js_rel_background")),
                       "top1_flip": _boot(collect_blur(region, cat, "top1_flip")),
                       "pref_drop": _boot(collect_blur(region, cat, "pref_drop"))}
    return rows


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def _save_panel(st, heat, path, boxes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = st["render"]
    fig, ax = plt.subplots(1, 3, figsize=(11, 4))
    ax[0].imshow(frame); ax[0].set_title(f"{st['category']} ep{st['ep']} t{st['t']}\nexec={C.ACTION_MEANINGS[st['action']]}"); ax[0].axis("off")
    disp = ((st["obs"].transpose(1, 2, 0) + 1) / 2).clip(0, 1)
    for T, col in [("T2", 1), ("T1", 2)]:
        ax[col].imshow(disp)
        up = np.kron(heat[T], np.ones((DENSITY, DENSITY)))[:64, :64]
        ax[col].imshow(up, cmap="hot", alpha=0.5, extent=(0, 64, 64, 0))
        ax[col].set_title(f"semantic JS saliency {T}"); ax[col].axis("off")
    Ps, _ = semantic_probs(st["logits"], C.BEHAVIOR_TEMPERATURE)
    ax[0].set_xlabel("")
    fig.suptitle(f"P_sem(T2)=STAY {Ps[0]:.2f} UP {Ps[1]:.2f} DOWN {Ps[2]:.2f}  ball_dx={st['ball_dx']}", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=70); plt.close(fig)


def _save_box_check(st, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    disp = ((st["obs"].transpose(1, 2, 0) + 1) / 2).clip(0, 1)
    boxes = object_boxes(extract_pong_objects_from_state(st))
    fig, ax = plt.subplots(figsize=(4, 4)); ax.imshow(disp)
    cols = {"ball": "yellow", "agent_paddle": "lime", "opp_paddle": "red", "scoreboard": "cyan", "background": "white"}
    for n, (cx, cy, w, h) in boxes.items():
        ax.add_patch(mpatches.Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, edgecolor=cols[n], lw=1.2, label=n))
    ax.legend(fontsize=6, loc="upper right"); ax.set_title(f"object boxes ep{st['ep']} t{st['t']}"); ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def _save_montage(states, heats, path, title, maxn=9):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    states, heats = states[:maxn], heats[:maxn]
    n = len(states)
    if n == 0:
        return
    c = min(3, n); r = (n + c - 1) // c
    fig, ax = plt.subplots(r, c, figsize=(3 * c, 3 * r), squeeze=False)
    for k in range(r * c):
        a = ax[k // c][k % c]; a.axis("off")
        if k >= n:
            continue
        st = states[k]; disp = ((st["obs"].transpose(1, 2, 0) + 1) / 2).clip(0, 1)
        a.imshow(disp)
        up = np.kron(heats[k]["T2"], np.ones((DENSITY, DENSITY)))[:64, :64]
        a.imshow(up, cmap="hot", alpha=0.5, extent=(0, 64, 64, 0))
        a.set_title(f"{st['category']} {C.ACTION_MEANINGS[st['action']]}", fontsize=7)
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=70); plt.close(fig)


def temporal_summary(states, temporal_per_state):
    out = {}
    for K in ("K=1", "K=4", "K=8", "K=16"):
        for tag in ("opponent", "scoreboard"):
            js = [tp[K][tag]["js_sem"] for tp in temporal_per_state if tp and K in tp and tag in tp[K]]
            flip = [tp[K][tag]["top1_flip"] for tp in temporal_per_state if tp and K in tp and tag in tp[K]]
            rl2 = [tp[K][tag]["recur_l2"] for tp in temporal_per_state if tp and K in tp and tag in tp[K]]
            out[f"{tag}_{K}"] = {"js_sem": _boot(js), "top1_flip": _boot(flip), "recur_l2": _boot(rl2)}
    return out


def _ci_pos(b):
    return bool(b and b["ci95"][0] > 0)


def decide_outcome(replay, region, temporal) -> Dict[str, Any]:
    if not replay["valid"]:
        return {"label": "D", "name": "TEACHER_OR_INTEGRATION_INVALID",
                "reason": f"replay does not reproduce cached logits (max diff {replay['max_abs_logit_diff']:.2e})"}
    ball = region["ball_all"]["js_sem"]; own = region["own_paddle_all"]["js_sem"]; bg = region["background"]["js_sem"]
    sensible = bool(ball and bg and ball["mean"] > 2 * max(bg["mean"], 1e-6))
    if not sensible:
        return {"label": "D", "name": "TEACHER_OR_INTEGRATION_INVALID",
                "reason": "ball saliency not clearly above background -> attends to nothing sensible"}
    opp_all = region["opponent_all"]["js_rel_background"]
    opp_opp = region["opponent_opp_side"]["js_rel_background"]
    opp_dec = region["opponent_decision"]["js_rel_background"]
    inst_opp = _ci_pos(opp_all) or _ci_pos(opp_opp) or _ci_pos(opp_dec)
    t16 = temporal.get("opponent_K=16", {}).get("js_sem"); t1 = temporal.get("opponent_K=1", {}).get("js_sem")
    temporal_opp = bool(t16 and t1 and t16["mean"] > max(2 * t1["mean"], bg["mean"]) and t16["mean"] > 0.01)
    opp_small = bool(region["opponent_all"]["js_sem"]["mean"] < 0.2 * ball["mean"])
    if inst_opp and not opp_small:
        label, name, reason = "A", "VALID_OPPONENT_SENSITIVE_TEACHER", \
            "opponent perturbation has a reproducible action-distribution effect above background"
    elif (inst_opp and opp_small) or temporal_opp:
        label, name, reason = "B", "TEMPORAL_OR_LOCAL_OPPONENT_DEPENDENCE", \
            "global instantaneous opponent effect weak/small; opponent matters locally or via recurrent history"
    else:
        label, name, reason = "C", "COMPETENT_BUT_OPPONENT_INSENSITIVE", \
            "opponent (and scoreboard) perturbations do not materially change actions, instantaneous or temporal"
    return {"label": label, "name": name, "reason": reason,
            "evidence": {"ball_js": ball["mean"], "own_paddle_js": own["mean"], "background_js": bg["mean"],
                         "opp_rel_bg_all_ci": opp_all["ci95"] if opp_all else None,
                         "opp_rel_bg_oppside_ci": opp_opp["ci95"] if opp_opp else None,
                         "opp_rel_bg_decision_ci": opp_dec["ci95"] if opp_dec else None,
                         "opp_temporal_K16_js": t16["mean"] if t16 else None,
                         "opp_temporal_K1_js": t1["mean"] if t1 else None}}


def downstream_answers(replay, rsum, tsum, rabl, outcome):
    opp_inst = (_ci_pos(rsum["opponent_all"]["js_rel_background"]) or _ci_pos(rsum["opponent_opp_side"]["js_rel_background"])
                or _ci_pos(rsum["opponent_decision"]["js_rel_background"]))
    t16 = tsum.get("opponent_K=16", {}).get("js_sem"); t1 = tsum.get("opponent_K=1", {}).get("js_sem")
    opp_temporal = bool(t16 and t1 and t16["mean"] > 2 * t1["mean"] and t16["mean"] > 0.01)
    score_inst = _ci_pos(rsum["scoreboard_all"]["js_rel_background"])
    recurrent_dep = bool(rabl["zero_hidden_js_sem"] and rabl["zero_hidden_js_sem"]["mean"] > 0.05)
    return {
        "1_integration_valid": bool(replay["valid"]),
        "2_plausible_pong_expert": bool(rsum["ball_all"]["js_sem"]["mean"] > 2 * rsum["background"]["js_sem"]["mean"]),
        "3_uses_opponent_for_actions": bool(opp_inst or opp_temporal),
        "4_uses_scoreboard": bool(score_inst),
        "5_dependence_type": ("instantaneous" if opp_inst else ("recurrent/temporal" if opp_temporal else "none-detected")),
        "6_opponent_mask_defensible_confounder": bool(opp_inst or opp_temporal),
        "7_recommendation": ("keep teacher; redefine U (opponent history/velocity/region pixels/recurrent ctx)"
                             if outcome["label"] == "B" else
                             "keep teacher & current opponent confounder is plausible" if outcome["label"] == "A" else
                             "keep teacher but choose a DIFFERENT hidden variable (opponent not an action confounder)"
                             if outcome["label"] == "C" else "repair integration pipeline / re-verify checkpoint"),
        "8_agrees_with_stage1j_zero_U": bool(not opp_inst),
        "9_justified_to_resume_critic_training": False,
        "9_reason": "Stage-1K is diagnostic; resuming critic training requires a defensible confounder (see outcome).",
        "recurrent_dependence_strong": recurrent_dep,
    }


def _summary(out, table_rows):
    L = ["Stage-1K teacher policy saliency (perturbation; upstream Greydanus occlusion, semantic-JS primary).",
         f"checkpoint sha {out['provenance']['checkpoint']['sha256'][:16]} | replay valid: {out['replay_validation']['valid']} "
         f"(max logit diff {out['replay_validation']['max_abs_logit_diff']:.2e})",
         f"states={out['config']['n_states']} radius={RADIUS} density={DENSITY} T_behavior={C.BEHAVIOR_TEMPERATURE}\n",
         "REGION SUMMARY (T2 semantic JS, base-2; bootstrap over states):",
         f"{'region':<26}{'T2_JS':>26}{'rel_background':>26}{'top1_flip':>22}"]
    for r in table_rows:
        L.append(f"{r['region']:<26}{r['T2_JS']:>26}{r['rel_background']:>26}{r['top1_flip']:>22}")
    L.append("\nTEMPORAL OPPONENT (semantic JS by K):")
    for K in ("K=1", "K=4", "K=8", "K=16"):
        b = out["temporal_opponent"].get(f"opponent_{K}", {}).get("js_sem")
        L.append(f"  opponent {K}: {('%.4f' % b['mean']) if b else 'NA'} CI{[round(x, 4) for x in b['ci95']] if b else None}")
    L.append(f"\nrecurrent ablation: zero-hidden JS {out['recurrent_ablation']['zero_hidden_js_sem']['mean']:.4f} "
             f"(recurrent dependence {'STRONG' if out['downstream_answers']['recurrent_dependence_strong'] else 'weak'})")
    o = out["outcome"]; L.append(f"\nOUTCOME {o['label']} — {o['name']}\n  {o['reason']}")
    a = out["downstream_answers"]
    L.append(f"\nDownstream: integration_valid={a['1_integration_valid']} plausible_expert={a['2_plausible_pong_expert']} "
             f"uses_opponent={a['3_uses_opponent_for_actions']} uses_scoreboard={a['4_uses_scoreboard']} "
             f"dependence={a['5_dependence_type']}")
    L.append(f"  recommendation: {a['7_recommendation']}")
    L.append(f"  agrees with Stage-1J ~0 U gain: {a['8_agrees_with_stage1j_zero_U']} | resume critic now: {a['9_justified_to_resume_critic_training']}")
    (RESULTS / "summary.txt").write_text("\n".join(L))


def run(seed, quotas, smoke_only=False) -> Dict[str, Any]:
    import csv
    RESULTS.mkdir(parents=True, exist_ok=True); CACHE.mkdir(parents=True, exist_ok=True)
    for sub in ["quantitative", "heatmaps/instantaneous", "heatmaps/latest_frame_only", "heatmaps/temporal",
                "heatmaps/montages", "object_checks/bounding_box_examples"]:
        (RESULTS / sub).mkdir(parents=True, exist_ok=True)
    model, meta = load_teacher(device="cpu")
    teacher = TeacherPolicy(model, device="cpu")
    env = make_env(replace(C.M1Config(), device="cpu"), num_envs=1)
    prov = provenance_audit(meta)
    (RESULTS / "provenance_audit.json").write_text(json.dumps(prov, indent=2))
    (RESULTS / "config.json").write_text(json.dumps(
        {"seed": seed, "quotas": quotas, "radius": RADIUS, "density": DENSITY, "blur_sigma": BLUR_SIGMA,
         "T_behavior": C.BEHAVIOR_TEMPERATURE, "checkpoint_sha256": meta["ckpt_sha256"]}, indent=2))

    picked, cstats = collect_states(model, teacher, env, seed, quotas)
    states = select_manifest(picked, quotas)
    manifest = [{"idx": i, "ep": s["ep"], "t": s["t"], "category": s["category"], "flags": s["flags"],
                 "exec_action": C.ACTION_MEANINGS[s["action"]], "exec_semantic": GROUP_NAMES[s["sem"]],
                 "ball": [s["ball_x"], s["ball_y"]], "ball_dx": s["ball_dx"], "opp_y": s["opp_y"],
                 "score": [s["agent"], s["opp"]], "has_window": s["win_obs"] is not None} for i, s in enumerate(states)]
    (RESULTS / "state_manifest.json").write_text(json.dumps(
        {"n_states": len(states), "collect_stats": cstats, "quotas": quotas, "states": manifest}, indent=2))

    replay = replay_validation(model, states)
    (RESULTS / "replay_validation.json").write_text(json.dumps(replay, indent=2))
    if not replay["valid"]:
        env.close()
        out = {"milestone": "Stage-1K", "provenance": prov, "replay_validation": replay,
               "outcome": {"label": "D", "name": "TEACHER_OR_INTEGRATION_INVALID", "reason": "replay validation failed"}}
        (RESULTS / "stage1k_report.json").write_text(json.dumps(out, indent=2))
        (RESULTS / "summary.txt").write_text(f"Stage-1K INVALID: replay max logit diff {replay['max_abs_logit_diff']:.2e}\n")
        return out

    for s in states[:4]:
        _save_box_check(s, RESULTS / "object_checks/bounding_box_examples" / f"box_ep{s['ep']}_t{s['t']}.png")

    n = len(states); patches = (64 // DENSITY) ** 2
    est_fwd = n * patches + n * 10 + n * (1 + 4 + 8 + 16) * 2 + n * 3
    print(f"[smoke] states={n} grid_patches/state={patches} est_forward_passes~{est_fwd} "
          f"(grid {n*patches}, temporal ~{n*58}, region {n*10}, ablation {n*3})")
    for s in states[:3]:
        h, _, _ = grid_saliency(model, s)
        _save_panel(s, h, RESULTS / "heatmaps/instantaneous" / f"SMOKE_ep{s['ep']}_t{s['t']}.png",
                    object_boxes(extract_pong_objects_from_state(s)))
    smoke_rep = {"n_states": n, "patches_per_state": patches, "est_forward_passes": int(est_fwd),
                 "smoke_panels": 3, "replay_valid": replay["valid"]}
    (RESULTS / "smoke_test_report.json").write_text(json.dumps(smoke_rep, indent=2))
    print("[smoke] 3 heatmaps saved")
    if smoke_only:
        env.close()
        return {"smoke": smoke_rep}

    heats, regions, temporals, ablations, per_state_rows = [], [], [], [], []
    mismatch = (states[n // 2]["hx"], states[n // 2]["cx"])
    for i, s in enumerate(states):
        h, t2max, t1max = grid_saliency(model, s)
        rg = region_perturbations(model, s)
        tp = temporal_opponent(model, s)
        ab = recurrent_ablation(model, s, mismatch_hx=mismatch if i != n // 2 else None)
        heats.append(h); regions.append(rg); temporals.append(tp); ablations.append(ab)
        np.savez(CACHE / f"heat_{i}.npz", T2=h["T2"], T1=h["T1"])
        _save_panel(s, h, RESULTS / "heatmaps/instantaneous" / f"state{i:02d}_{s['category']}.png",
                    object_boxes(extract_pong_objects_from_state(s)))
        per_state_rows.append({"idx": i, "ep": s["ep"], "t": s["t"], "category": s["category"],
                               "exec_sem": GROUP_NAMES[s["sem"]], "grid_t2_max_js": round(t2max, 4),
                               "grid_t1_max_js": round(t1max, 4),
                               "ball_js": round(rg["ball"]["js_sem_mean"], 4) if "ball" in rg else None,
                               "opp_js": round(rg["opp_paddle"]["js_sem_mean"], 4) if "opp_paddle" in rg else None,
                               "opp_rel_bg": round(rg["opp_paddle"]["js_rel_background"], 4) if "opp_paddle" in rg else None,
                               "score_js": round(rg["scoreboard"]["js_sem_mean"], 4),
                               "ablation_zero_js": round(ab["zero"]["js_sem_vs_saved"], 4)})
        if (i + 1) % 5 == 0 or i == n - 1:
            print(f"[full] processed {i+1}/{n} states")
    env.close()

    rsum = region_summary(states, regions)
    tsum = temporal_summary(states, temporals)
    tcomp = {"grid_max_js_T2": _boot([r["grid_t2_max_js"] for r in per_state_rows]),
             "grid_max_js_T1": _boot([r["grid_t1_max_js"] for r in per_state_rows]),
             "note": "same raw logits; T applied only at softmax. T2 is the behavior policy used for collection."}
    rabl = {"zero_hidden_js_sem": _boot([a["zero"]["js_sem_vs_saved"] for a in ablations]),
            "zero_hidden_logit_l2": _boot([a["zero"]["logit_l2_vs_saved"] for a in ablations]),
            "zero_hidden_flip_rate": _boot([a["zero"]["top1_flip_vs_saved"] for a in ablations]),
            "mismatch_hidden_js_sem": _boot([a["mismatch"]["js_sem_vs_saved"] for a in ablations if "mismatch" in a])}
    outcome = decide_outcome(replay, rsum, tsum)

    cats = {"decision": [], "defensive": [], "opp_side": [], "score_restart": []}
    grp = {0: [], 1: [], 2: []}
    for s, h in zip(states, heats):
        for cc in cats:
            if s["flags"].get(cc):
                cats[cc].append((s, h))
        grp[s["sem"]].append((s, h))
    for cc, lst in cats.items():
        if lst:
            _save_montage([x[0] for x in lst], [x[1] for x in lst],
                          RESULTS / "heatmaps/montages" / f"montage_{cc}.png", f"{cc} states (T2 semantic JS)")
    for g, lst in grp.items():
        if lst:
            _save_montage([x[0] for x in lst], [x[1] for x in lst],
                          RESULTS / "heatmaps/montages" / f"montage_pred_{GROUP_NAMES[g]}.png",
                          f"executed {GROUP_NAMES[g]} (T2 semantic JS)")

    with open(RESULTS / "quantitative/per_state_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_state_rows[0].keys())); w.writeheader(); w.writerows(per_state_rows)
    (RESULTS / "quantitative/region_summary.json").write_text(json.dumps(rsum, indent=2))
    (RESULTS / "quantitative/temperature_comparison.json").write_text(json.dumps(tcomp, indent=2))
    (RESULTS / "quantitative/recurrent_ablation.json").write_text(json.dumps(rabl, indent=2))
    (RESULTS / "quantitative/temporal_opponent_audit.json").write_text(json.dumps(tsum, indent=2))
    (RESULTS / "quantitative/bootstrap_intervals.json").write_text(json.dumps(
        {"region_summary": rsum, "temporal": tsum, "temperature": tcomp, "recurrent_ablation": rabl}, indent=2))

    def cell(b):
        return f"{b['mean']:.4f} [{b['ci95'][0]:.4f},{b['ci95'][1]:.4f}]" if b else "NA"
    table_rows = []
    for label, key in [("Ball — all", "ball_all"), ("Own paddle — all", "own_paddle_all"),
                       ("Opponent — all", "opponent_all"), ("Opponent — opp-side", "opponent_opp_side"),
                       ("Opponent — decision", "opponent_decision"), ("Scoreboard — all", "scoreboard_all"),
                       ("Scoreboard — restart", "scoreboard_restart"), ("Empty background", "background")]:
        r = rsum[key]
        table_rows.append({"region": label, "T2_JS": cell(r["js_sem"]), "rel_background": cell(r["js_rel_background"]),
                           "top1_flip": cell(r["top1_flip"]), "pref_drop": cell(r["pref_drop"])})
    with open(RESULTS / "summary_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table_rows[0].keys())); w.writeheader(); w.writerows(table_rows)

    answers = downstream_answers(replay, rsum, tsum, rabl, outcome)
    out = {"milestone": "Stage-1K-teacher-policy-saliency", "provenance": prov, "replay_validation": replay,
           "config": {"seed": seed, "quotas": quotas, "radius": RADIUS, "density": DENSITY, "blur_sigma": BLUR_SIGMA,
                      "T_behavior": C.BEHAVIOR_TEMPERATURE, "n_states": n},
           "region_summary": rsum, "temporal_opponent": tsum, "temperature_comparison": tcomp,
           "recurrent_ablation": rabl, "outcome": outcome, "downstream_answers": answers,
           "interpretation_note": "Perturbation saliency on the behavior policy. Action-distribution sensitivity "
                                  "to a region is NOT causal confounding of the FUTURE; no P(G|S,do(A)) claim."}
    (RESULTS / "stage1k_report.json").write_text(json.dumps(out, indent=2))
    _summary(out, table_rows)
    return out


def main():
    ap = argparse.ArgumentParser(description="Stage-1K teacher policy perturbation-saliency audit.")
    ap.add_argument("--seed", type=int, default=30000)
    ap.add_argument("--dec", type=int, default=21); ap.add_argument("--ord", type=int, default=15)
    ap.add_argument("--defn", type=int, default=10); ap.add_argument("--opp", type=int, default=12)
    ap.add_argument("--score", type=int, default=6); ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()
    quotas = {"decision": args.dec, "ordinary": args.ord, "defensive": args.defn,
              "opp_side": args.opp, "score_restart": args.score}
    out = run(args.seed, quotas, smoke_only=args.smoke_only)
    if not args.smoke_only and "outcome" in out:
        print(open(RESULTS / "summary.txt").read())


if __name__ == "__main__":
    main()
