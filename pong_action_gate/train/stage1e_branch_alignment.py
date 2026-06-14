"""Stage-1E — emulator-branch alignment for the H=8 NO-SELF-PADDLE critic.

Question: starting from the SAME cloned emulator state, does the frozen no-self-paddle critic
assign higher compatibility to the action that actually generated a particular H=8 future
ball/opponent/score goal? Checkpoint-only validation (no retraining unless the deterministic
no-self checkpoints are absent — they are, so they are reproduced ONCE from the locked config).

Branches are environment-only (T=2.0 teacher + ALE clone/restore), generated once; all three
no-self critic seeds are scored on the SAME branches. No self-paddle in the goal, no H change,
no pixel/masking/Seaquest/+15.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .. import config as C
from ..objects import extract_pong_objects
from ..teacher.load_teacher import TeacherPolicy, load_teacher, make_env
from . import dataset as D
from .emulator_branch import _feat_single, restore_env, snapshot_env
from .stage1b import decision_criteria, is_decision_focused
from .stage1d_ablation import VARIANTS, build_full, make_variant_A, norm_stats
from .stage1d_rich_goal import RichStateCritic, train

ART = Path("artifacts/pong_action_gate/stage1e/h8_no_self")
CKPT = ART / "ckpts"
H = 8
R = 16
# no-self continuous = full-7 columns [0,1,2,3,5,6]; masks = validity cols [ball=0, opp=5, vel=2]
NS_CONT = VARIANTS["no_self"]["cont"]          # [0,1,2,3,5,6]
NS_MASK = VARIANTS["no_self"]["masks"]         # [0,5,2]
GROUPS = {0: "STAY", 1: "STAY", 2: "UP", 4: "UP", 3: "DOWN", 5: "DOWN"}   # verified in step 2
GROUP_NAMES = ["STAY", "UP", "DOWN"]


# --------------------------------------------------------------------------- #
# Step 1 — reproduce/load the three frozen no-self critics + per-seed stats
# --------------------------------------------------------------------------- #
def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def reproduce_or_load(seeds, n_episodes=80, steps=4000, batch=256):
    CKPT.mkdir(parents=True, exist_ok=True)
    all_eps = D.load_subset("full", list(range(n_episodes)), with_pixels=False)
    out = {}
    repro = []
    for seed in seeds:
        cpath = CKPT / f"no_self_seed{seed}.pt"
        spath = CKPT / f"no_self_seed{seed}_meta.json"
        tr_ids, va_ids = D.split_episodes(n_episodes, 0.2, seed)
        full_tr = build_full(all_eps, tr_ids)
        stats = norm_stats(full_tr)                       # train-only normalization (per seed)
        critic = RichStateCritic(D.STATE_DIM, 6, goal_dim=9)
        if cpath.exists():
            critic.load_state_dict(torch.load(cpath, map_location="cpu", weights_only=True))
            meta = json.loads(spath.read_text()); how = "loaded"
        else:
            A_tr, gdim = make_variant_A(full_tr, "no_self", stats)
            full_va = build_full(all_eps, va_ids); A_va, _ = make_variant_A(full_va, "no_self", stats)
            critic, sel = train(A_tr, A_va, seed, steps, batch, goal_dim=9)
            torch.save(critic.state_dict(), cpath)
            meta = {"selected_step": sel["selected_step"], "stats": stats,
                    "split": {"train": tr_ids, "val": va_ids}, "goal_dim": 9}
            spath.write_text(json.dumps(meta)); how = "reproduced (deterministic, locked config)"
            repro.append(seed)
        critic.eval()
        out[seed] = {"critic": critic, "stats": stats, "meta": meta, "ckpt": str(cpath), "sha": _sha(cpath)}
    return out, repro


# --------------------------------------------------------------------------- #
# Step 2 — action semantics (measure one-step paddle displacement)
# --------------------------------------------------------------------------- #
def action_semantics(teacher, env, n_probe=40, seed=8100) -> Dict[str, Any]:
    meanings = env.env.unwrapped.get_action_meanings()
    torch.manual_seed(seed)
    obs, info = env.reset(seed=[seed]); ale = env.env.ale
    hx, cx = teacher.initial_state(1)
    disp = {a: [] for a in range(6)}
    # gather some mid-rally snapshots, then probe each action one step from each
    snaps = []
    t = 0
    while len(snaps) < n_probe:
        o = extract_pong_objects(ale.getRAM())
        if o.player_y is not None and o.ball_present and t % 7 == 0:
            snaps.append(snapshot_env(env, hx, cx, 0, 0, obs[:, C.TEACHER_OBS_SLICE], _gray_stub(), o, o))
        with torch.no_grad():
            logits, _, (hx, cx) = teacher.model.predict_act_value(obs[:, C.TEACHER_OBS_SLICE], (hx, cx))
            a = torch.distributions.Categorical(logits=logits / C.BEHAVIOR_TEMPERATURE).sample()
        obs, *_ = env.step(a); t += 1
    for sn in snaps:
        y0 = sn["cur_obj"].player_y
        for a in range(6):
            restore_env(env, sn)
            env.step(int(a))
            y1 = extract_pong_objects(ale.getRAM()).player_y
            if y0 is not None and y1 is not None:
                disp[a].append(y1 - y0)
    mean_disp = {meanings[a]: float(np.mean(disp[a])) if disp[a] else None for a in range(6)}
    return {"legal_action_ids": list(range(6)), "ale_action_names": list(meanings),
            "mean_one_step_paddle_dy": mean_disp,
            "semantic_groups": {GROUP_NAMES[0]: [0, 1], GROUP_NAMES[1]: [2, 4], GROUP_NAMES[2]: [3, 5]},
            "exact_random_baseline": 1.0 / 6, "semantic_random_baseline": 1.0 / 3,
            "note": "UP=RIGHT/RIGHTFIRE (paddle moves up, dy<0), DOWN=LEFT/LEFTFIRE (dy>0), STAY=NOOP/FIRE; "
                    "FIRE-aliases share movement -> 6 IDs collapse to 3 movement groups."}


def _gray_stub():
    from collections import deque
    d = deque(maxlen=4)
    for _ in range(4):
        d.append(np.zeros((84, 84), np.uint8))
    return d


# --------------------------------------------------------------------------- #
# Step 3 — collect balanced cloned states (pre-action observable only)
# --------------------------------------------------------------------------- #
def collect_states(teacher, env, seed, n_each=12, stride=11) -> List[Dict]:
    torch.manual_seed(seed)
    obs, info = env.reset(seed=[seed]); ale = env.env.ale
    hx, cx = teacher.initial_state(1)
    dec_states, ord_states = [], []
    prev = extract_pong_objects(ale.getRAM())
    ag = op = 0
    t = 0
    while len(dec_states) < n_each or len(ord_states) < n_each:
        o = extract_pong_objects(ale.getRAM())
        if t > 0 and t % stride == 0 and o.player_y is not None:
            bdx = (o.ball_x - prev.ball_x) if (o.ball_present and prev.ball_present) else None
            is_dec = bool(o.ball_present and bdx is not None and bdx > 0 and o.ball_x is not None and o.ball_x >= 120)
            snap = snapshot_env(env, hx, cx, ag, op, obs[:, C.TEACHER_OBS_SLICE], _gray_stub(), o, prev)
            snap["decision"] = is_dec
            snap["anchor_state"] = _feat_single(o, prev, snap["score_diff_pre"])
            if is_dec and len(dec_states) < n_each:
                dec_states.append(snap)
            elif (not is_dec) and len(ord_states) < n_each:
                ord_states.append(snap)
        with torch.no_grad():
            logits, _, (hx, cx) = teacher.model.predict_act_value(obs[:, C.TEACHER_OBS_SLICE], (hx, cx))
            a = torch.distributions.Categorical(logits=logits / C.BEHAVIOR_TEMPERATURE).sample()
        prev = o
        obs, rew, end, trunc, info = env.step(a)
        r = float(rew.item()); ag += int(r > 0); op += int(r < 0)
        t += 1
        if bool((end | trunc).item()):
            obs, info = env.reset(seed=[seed + 7000 + t]); hx, cx = teacher.initial_state(1)
            prev = extract_pong_objects(ale.getRAM()); ag = op = 0
    return dec_states + ord_states


# --------------------------------------------------------------------------- #
# Step 4 — generate matched H=8 branches (raw no-self goal), CRN per replicate
# --------------------------------------------------------------------------- #
def _branch_raw_goal(env, teacher, snap, first_action, rep_seed):
    """Force first_action, teacher controls 7 more steps; return raw no-self goal + censored."""
    torch.manual_seed(rep_seed)                         # CRN: same stream across forced actions
    hx, cx, ag, op = restore_env(env, snap)
    with torch.no_grad():
        _, _, (hx, cx) = teacher.model.predict_act_value(snap["anchor_obs"], (hx, cx))   # advance hidden
    ale = env.env.ale
    obs, rew, end, trunc, info = env.step(int(first_action))
    ag += int(float(rew.item()) > 0); op += int(float(rew.item()) < 0)
    ball_hist = [extract_pong_objects(ale.getRAM())]
    censored = False
    for _ in range(H - 1):                              # 7 teacher steps -> reach t+8
        if bool((end | trunc).item()):
            censored = True; break
        with torch.no_grad():
            logits, _, (hx, cx) = teacher.model.predict_act_value(obs[:, C.TEACHER_OBS_SLICE], (hx, cx))
            a = torch.distributions.Categorical(logits=logits / C.BEHAVIOR_TEMPERATURE).sample()
        obs, rew, end, trunc, info = env.step(a)
        ag += int(float(rew.item()) > 0); op += int(float(rew.item()) < 0)
        ball_hist.append(extract_pong_objects(ale.getRAM()))
    if censored or len(ball_hist) < 2:
        return None
    o8 = ball_hist[-1]; o7 = ball_hist[-2]
    mb = float(o8.ball_present); mopp = float(o8.opp_y is not None)
    mvel = float(o8.ball_present and o7.ball_present)
    bx = o8.ball_x if o8.ball_present else 0.0; by = o8.ball_y if o8.ball_present else 0.0
    vx = (o8.ball_x - o7.ball_x) if mvel else 0.0; vy = (o8.ball_y - o7.ball_y) if mvel else 0.0
    oy = o8.opp_y if o8.opp_y is not None else 0.0
    raw6 = np.array([bx, by, vx, vy, oy, ag - op], np.float32)         # ball_x,y,vx,vy,opp_y,score
    valid6 = np.array([mb, mb, mvel, mvel, mopp, 1.0], np.float32)
    masks3 = np.array([mb, mopp, mvel], np.float32)
    return {"raw6": raw6, "valid6": valid6, "masks3": masks3}


def generate_branches(teacher, env, snaps) -> List[Dict]:
    branches = []
    for si, sn in enumerate(snaps):
        cell = {"state_idx": si, "decision": sn["decision"], "anchor_state": sn["anchor_state"],
                "per_action": {a: [] for a in range(6)}, "censored": {a: 0 for a in range(6)}}
        for rep in range(R):
            rep_seed = 90000 + rep                       # CRN seed, shared across actions
            for a in range(6):
                g = _branch_raw_goal(env, teacher, sn, a, rep_seed)
                if g is None:
                    cell["censored"][a] += 1
                else:
                    cell["per_action"][a].append(g)
        branches.append(cell)
    return branches


# --------------------------------------------------------------------------- #
# Step 5/6/7 — normalize per seed, score, metrics, diversity, bootstrap
# --------------------------------------------------------------------------- #
def _norm_goal(raw6, valid6, masks3, stats):
    mean = np.array(stats["mean"])[NS_CONT]; std = np.array(stats["std"])[NS_CONT]
    norm = np.where(valid6 > 0, (raw6 - mean) / std, 0.0).astype(np.float32)
    return np.concatenate([norm, masks3]).astype(np.float32)


def diversity_ratio(cell) -> float:
    """Environment-only between/within branch dispersion on globally-standardized raw goals."""
    allg = [g["raw6"] for a in range(6) for g in cell["per_action"][a]]
    if len(allg) < 6:
        return 0.0
    allg = np.array(allg); gstd = allg.std(0) + 1e-6
    per_action_mean = []; within = []
    for a in range(6):
        ga = np.array([g["raw6"] for g in cell["per_action"][a]])
        if len(ga) < 2:
            continue
        gz = ga / gstd
        per_action_mean.append(gz.mean(0)); within.append(gz.var(0).mean())
    if len(per_action_mean) < 2:
        return 0.0
    between = np.array(per_action_mean).var(0).mean()
    return float(between / (np.mean(within) + 1e-9))


def _branch_metrics(critic, anchor_state, goal9, a_gen):
    s = torch.as_tensor(anchor_state[None]); g = torch.as_tensor(goal9[None])
    with torch.no_grad():
        sc = critic.scores_all_actions(s, g)[0].numpy()   # (6,)
    wrong = [a for a in range(6) if a != a_gen]
    grp = GROUPS[a_gen]; same = [a for a in range(6) if GROUPS[a] == grp]; other = [a for a in range(6) if GROUPS[a] != grp]
    return {
        "exact_top1": int(np.argmax(sc) == a_gen),
        "rank": int(1 + (sc > sc[a_gen]).sum()),
        "pairwise": float(np.mean([sc[a_gen] > sc[w] for w in wrong])),
        "margin": float(sc[a_gen] - np.mean([sc[w] for w in wrong])),
        "hardest_margin": float(sc[a_gen] - max(sc[w] for w in wrong)),
        "sem_top1": int(GROUPS[int(np.argmax(sc))] == grp),
        "sem_pairwise": float(np.mean([sc[a_gen] > sc[w] for w in other])) if other else 1.0,
        "group_margin": float(max(sc[a] for a in same) - max(sc[a] for a in other)) if other else 0.0,
    }


def per_state_metrics(critic, stats, branches):
    rows = []
    for cell in branches:
        ms = []
        for a in range(6):
            for g in cell["per_action"][a]:
                goal9 = _norm_goal(g["raw6"], g["valid6"], g["masks3"], stats)
                ms.append(_branch_metrics(critic, cell["anchor_state"], goal9, a))
        if not ms:
            continue
        agg = {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}
        agg["decision"] = cell["decision"]; agg["diversity"] = diversity_ratio(cell); agg["n_branches"] = len(ms)
        rows.append(agg)
    return rows


def _cluster_ci(rows, key, n_boot=2000, seed=0):
    vals = np.array([r[key] for r in rows]); rng = np.random.default_rng(seed)
    bs = [float(vals[rng.integers(len(vals), size=len(vals))].mean()) for _ in range(n_boot)]
    return {"point": float(vals.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def summarize(rows, exact_base=1/6, sem_base=1/3):
    keys = ["exact_top1", "sem_top1", "pairwise", "sem_pairwise", "margin", "group_margin", "rank", "hardest_margin"]
    ci = {k: _cluster_ci(rows, k) for k in keys}
    ci["exact_top1"]["above_baseline_ci"] = bool(ci["exact_top1"]["ci95"][0] > exact_base)
    ci["sem_top1"]["above_baseline_ci"] = bool(ci["sem_top1"]["ci95"][0] > sem_base)
    ci["pairwise"]["above_half"] = bool(ci["pairwise"]["ci95"][0] > 0.5)
    return ci


def split_reports(rows, seed):
    dec = [r for r in rows if r["decision"]]; ordi = [r for r in rows if not r["decision"]]
    div = np.array([r["diversity"] for r in rows]); q1, q2 = np.percentile(div, [33.3, 66.6])
    ter = {"low": [r for r in rows if r["diversity"] <= q1],
           "med": [r for r in rows if q1 < r["diversity"] <= q2],
           "high": [r for r in rows if r["diversity"] > q2]}
    return {
        "decision_focused": summarize(dec) if len(dec) >= 4 else {"n": len(dec)},
        "ordinary": summarize(ordi) if len(ordi) >= 4 else {"n": len(ordi)},
        "diversity_tertiles": {k: ({"n": len(v), "sem_top1": _cluster_ci(v, "sem_top1", seed=seed)["point"],
                                    "pairwise": _cluster_ci(v, "pairwise", seed=seed)["point"]} if len(v) >= 3
                                   else {"n": len(v)}) for k, v in ter.items()},
        "diversity_tertile_edges": [float(q1), float(q2)]}


def decision_label(per_seed):
    sems = [per_seed[s]["overall"]["sem_top1"] for s in per_seed]
    pair = [per_seed[s]["overall"]["pairwise"] for s in per_seed]
    grpm = [per_seed[s]["overall"]["group_margin"] for s in per_seed]
    all_sem_ci = all(d["above_baseline_ci"] for d in sems)
    n_sem_ci = sum(d["above_baseline_ci"] for d in sems)
    all_pair = all(d["point"] > 0.5 for d in pair)
    all_pos_margin = all(d["point"] > 0 for d in grpm)
    all_dir = all(d["point"] > 1/3 for d in sems)
    if all_sem_ci and all_pair and all_pos_margin:
        return "STRONG PASS"
    if all_dir and n_sem_ci >= 2 and all_pos_margin:
        return "CANDIDATE PASS"
    return "FAIL"


def run(seeds, state_seed, n_each, device="cpu") -> Dict[str, Any]:
    ART.mkdir(parents=True, exist_ok=True)
    critics, repro = reproduce_or_load(seeds)
    teacher_model, _ = load_teacher(device=device)
    teacher = TeacherPolicy(teacher_model, device=device)
    env = make_env(replace(C.M1Config(), device=device), num_envs=1)

    sem = action_semantics(teacher, env)
    snaps = collect_states(teacher, env, state_seed, n_each=n_each)
    branches = generate_branches(teacher, env, snaps)
    env.close()

    cens = {a: sum(c["censored"][a] for c in branches) for a in range(6)}
    n_states = len(branches)
    per_seed = {}
    for s in seeds:
        rows = per_state_metrics(critics[s]["critic"], critics[s]["stats"], branches)
        per_seed[str(s)] = {"n_states_used": len(rows), "overall": summarize(rows), "splits": split_reports(rows, s)}

    out = {
        "milestone": "Stage-1E-branch-alignment", "H": H, "R": R, "n_cloned_states": n_states,
        "reproduced_checkpoints": repro,
        "checkpoints": {str(s): {"path": critics[s]["ckpt"], "sha256_16": critics[s]["sha"],
                                  "selected_step": critics[s]["meta"]["selected_step"]} for s in seeds},
        "goal_schema_no_self": ["ball_x", "ball_y", "ball_vx", "ball_vy", "opponent_paddle_y",
                                "score_diff_pre", "mask_ball", "mask_opp", "mask_vel"],
        "action_semantics": sem,
        "censoring_per_action": cens,
        "censoring_rate": float(sum(cens.values()) / (n_states * 6 * R)),
        "per_seed": per_seed,
        "random_baselines": {"exact": 1/6, "semantic": 1/3, "pairwise": 0.5},
        "decision": decision_label({s: per_seed[str(s)] for s in seeds}),
        "interpretation_note": "Controlled diagnostic on a small cloned-state set; does NOT prove +15, "
                               "pixels, confounding, causal ID, or H-optimality.",
    }
    (ART / "stage1e_report.json").write_text(json.dumps(out, indent=2))
    _write_table(out)
    return out


def _write_table(out):
    lines = ["Stage-1E branch alignment (H=8, no-self goal) — per seed (sem baseline 0.333, exact 0.167)\n"]
    for s, d in out["per_seed"].items():
        o = d["overall"]
        lines.append(f"seed {s}: states={d['n_states_used']}  "
                     f"sem_top1={o['sem_top1']['point']:.3f} CI{[round(x,3) for x in o['sem_top1']['ci95']]}"
                     f"{'*' if o['sem_top1'].get('above_baseline_ci') else ''}  "
                     f"exact_top1={o['exact_top1']['point']:.3f}  pairwise={o['pairwise']['point']:.3f}  "
                     f"sem_pairwise={o['sem_pairwise']['point']:.3f}  group_margin={o['group_margin']['point']:+.4f}")
    lines.append(f"\ncensoring rate: {out['censoring_rate']:.4f}   DECISION: {out['decision']}")
    (ART / "summary.txt").write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-1E emulator-branch alignment (no-self critic).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--state-seed", type=int, default=8000)
    ap.add_argument("--n-each", type=int, default=12)
    args = ap.parse_args()
    out = run(args.seeds, args.state_seed, args.n_each)
    print(open(ART / "summary.txt").read())
    print("\nreproduced:", out["reproduced_checkpoints"], "| censoring:", out["censoring_rate"])


if __name__ == "__main__":
    main()
