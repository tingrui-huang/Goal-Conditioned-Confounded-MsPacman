"""Stage-1F — environment-only horizon feasibility scan for Pong (NO critic).

Question: is there ANY horizon at which one forced first action leaves an externally
distinguishable footprint in the no-self future goal under the stochastic T=2.0 teacher?

Environment-only: clone/restore + teacher continuation. No critic trained or evaluated.
Each continuation is run ONCE to H_max=48 (recording the no-self state at every step), so the
goal at every horizon is read off the same matched, common-random-number branch. The 24 cloned
states are the exact Stage-1E set (deterministic collect_states(seed=8000)).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from .. import config as C
from ..objects import extract_pong_objects
from ..teacher.load_teacher import TeacherPolicy, load_teacher, make_env
from .emulator_branch import restore_env
from .stage1d_ablation import VARIANTS
from .stage1e_branch_alignment import GROUPS, GROUP_NAMES, collect_states

ART = Path("artifacts/pong_action_gate/stage1f/environment_horizon_scan")
HORIZONS = [4, 8, 12, 16, 24, 32, 48]
HMAX = 48
R = 16
NS_CONT = VARIANTS["no_self"]["cont"]          # full-7 cols used by no-self: [0,1,2,3,5,6]
PADDLE_X = 140.0                                # agent paddle native x (contact detection)


def _stats_seed0():
    p = Path("artifacts/pong_action_gate/stage1e/h8_no_self/ckpts/no_self_seed0_meta.json")
    if p.exists():
        return json.loads(p.read_text())["stats"]
    return None


def run_to_hmax(env, teacher, snap, first_action, rep_seed):
    """Force first action; teacher controls the rest to H_max. Record per-step external state."""
    torch.manual_seed(rep_seed)
    hx, cx, ag, op = restore_env(env, snap)
    with torch.no_grad():
        _, _, (hx, cx) = teacher.model.predict_act_value(snap["anchor_obs"], (hx, cx))
    ale = env.env.ale
    steps = []
    obs, rew, end, trunc, info = env.step(int(first_action))
    ag += int(float(rew.item()) > 0); op += int(float(rew.item()) < 0)
    o = extract_pong_objects(ale.getRAM())
    steps.append((o, ag, op, float(rew.item())))
    for _ in range(HMAX - 1):
        if bool((end | trunc).item()):
            break
        with torch.no_grad():
            logits, _, (hx, cx) = teacher.model.predict_act_value(obs[:, C.TEACHER_OBS_SLICE], (hx, cx))
            a = torch.distributions.Categorical(logits=logits / C.BEHAVIOR_TEMPERATURE).sample()
        obs, rew, end, trunc, info = env.step(a)
        ag += int(float(rew.item()) > 0); op += int(float(rew.item()) < 0)
        steps.append((extract_pong_objects(ale.getRAM()), ag, op, float(rew.item())))
    return steps     # len <= HMAX; steps[i] is the state at t+(i+1)


def goal_at(steps, H):
    """Raw no-self goal at horizon H (None if branch censored before H)."""
    if len(steps) < H:
        return None
    o8, ag, op, _ = steps[H - 1]
    o7 = steps[H - 2][0] if H >= 2 else o8
    mb = float(o8.ball_present); mopp = float(o8.opp_y is not None)
    mvel = float(o8.ball_present and o7.ball_present)
    bx = o8.ball_x if o8.ball_present else 0.0; by = o8.ball_y if o8.ball_present else 0.0
    vx = (o8.ball_x - o7.ball_x) if mvel else 0.0; vy = (o8.ball_y - o7.ball_y) if mvel else 0.0
    oy = o8.opp_y if o8.opp_y is not None else 0.0
    raw6 = np.array([bx, by, vx, vy, oy, ag - op], np.float32)
    masks3 = np.array([mb, mopp, mvel], np.float32)
    return {"raw6": raw6, "masks3": masks3, "valid6": np.array([mb, mb, mvel, mvel, mopp, 1.0], np.float32)}


def normalize(raw6, valid6, masks3, stats):
    if stats is None:
        return np.concatenate([raw6, masks3]).astype(np.float32)
    mean = np.array(stats["mean"])[NS_CONT]; std = np.array(stats["std"])[NS_CONT]
    norm = np.where(valid6 > 0, (raw6 - mean) / std, 0.0).astype(np.float32)
    return np.concatenate([norm, masks3]).astype(np.float32)


# --------------------------------------------------------------------------- #
# Per-horizon environment statistics (semantic-group based)
# --------------------------------------------------------------------------- #
def diversity_ratio_state(goals_by_action):
    """semantic between-group / within-group dispersion on normalized goals for one state."""
    grp_goals = {g: [] for g in GROUP_NAMES}
    for a, gs in goals_by_action.items():
        for v in gs:
            grp_goals[GROUPS[a]].append(v)
    mus = {}; allv = []
    for g, vs in grp_goals.items():
        if vs:
            mus[g] = np.mean(vs, 0); allv += vs
    if len(mus) < 2 or len(allv) < 6:
        return None
    grand = np.mean(allv, 0)
    between = np.mean([np.sum((mus[g] - grand) ** 2) for g in mus])
    within = np.mean([np.sum((v - mus[GROUPS[a]]) ** 2) for a, gs in goals_by_action.items() for v in gs
                      if GROUPS[a] in mus])
    return float(between / (within + 1e-9))


def nearest_centroid_state(goals_by_action):
    """Replicate-split nearest-centroid semantic decoder for one state. Returns (acc, balanced_acc)."""
    train_X, train_y, test_X, test_y = [], [], [], []
    for a, gs in goals_by_action.items():
        for ri, v in enumerate(gs):
            (train_X if ri < R // 2 else test_X).append(v)
            (train_y if ri < R // 2 else test_y).append(GROUPS[a])
    if not test_X:
        return None
    cents = {}
    for g in GROUP_NAMES:
        xs = [x for x, y in zip(train_X, train_y) if y == g]
        if xs:
            cents[g] = np.mean(xs, 0)
    if len(cents) < 2:
        return None
    glist = list(cents)
    correct = 0; per_g = {g: [0, 0] for g in GROUP_NAMES}
    for x, y in zip(test_X, test_y):
        pred = glist[int(np.argmin([np.sum((x - cents[g]) ** 2) for g in glist]))]
        per_g[y][1] += 1; per_g[y][0] += int(pred == y); correct += int(pred == y)
    acc = correct / len(test_X)
    recalls = [per_g[g][0] / per_g[g][1] for g in GROUP_NAMES if per_g[g][1] > 0]
    return acc, float(np.mean(recalls))


def _ci(vals, seed=0):
    vals = np.array([v for v in vals if v is not None])
    if len(vals) == 0:
        return None
    rng = np.random.default_rng(seed)
    bs = [float(vals[rng.integers(len(vals), size=len(vals))].mean()) for _ in range(2000)]
    return {"point": float(vals.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}


def run(state_seed=8000, n_each=12, device="cpu") -> Dict[str, Any]:
    ART.mkdir(parents=True, exist_ok=True)
    stats = _stats_seed0()
    teacher_model, _ = load_teacher(device=device)
    teacher = TeacherPolicy(teacher_model, device=device)
    env = make_env(replace(C.M1Config(), device=device), num_envs=1)
    snaps = collect_states(teacher, env, state_seed, n_each=n_each)

    # run all continuations to H_max once
    raw = []   # per state: {decision, per_action_rep: {a: [steps,...]}}
    for sn in snaps:
        cell = {"decision": sn["decision"], "steps": {a: [] for a in range(6)},
                "contact": {a: [] for a in range(6)}, "score_evt": {a: [] for a in range(6)}}
        for rep in range(R):
            rep_seed = 90000 + rep
            for a in range(6):
                st = run_to_hmax(env, teacher, sn, a, rep_seed)
                cell["steps"][a].append(st)
        raw.append(cell)
    env.close()

    horizon_results = {}
    for H in HORIZONS:
        # build goals per state/action (normalized + raw), track censoring + components
        per_state_ratio, per_state_dec = [], []
        per_state_acc, per_state_bacc = [], []
        ratios_ord, ratios_dec, acc_ord, acc_dec = [], [], [], []
        cens = 0; total = 0
        comp_between = []   # per-state component between-group var (normalized)
        frac_contact = []; frac_score = []; frac_missing = []
        for cell in raw:
            gba_norm = {a: [] for a in range(6)}; gba_raw = {a: [] for a in range(6)}
            cset = 0; sset = 0; mset = 0; nb = 0
            for a in range(6):
                for st in cell["steps"][a]:
                    total += 1
                    g = goal_at(st, H)
                    if g is None:
                        cens += 1; continue
                    nb += 1
                    gba_norm[a].append(normalize(g["raw6"], g["valid6"], g["masks3"], stats))
                    gba_raw[a].append(np.concatenate([g["raw6"], g["masks3"]]))
                    # env footprint up to H
                    maxbx = max((s[0].ball_x or 0) for s in st[:H])
                    cset += int(maxbx >= PADDLE_X - 5)
                    sset += int(any(s[3] != 0 for s in st[:H]))
                    mset += int(g["masks3"][0] == 0)
            if nb < 6:
                continue
            ratio = diversity_ratio_state(gba_norm)
            nc = nearest_centroid_state(gba_norm)
            per_state_ratio.append(ratio); per_state_dec.append(cell["decision"])
            if nc:
                per_state_acc.append(nc[0]); per_state_bacc.append(nc[1])
            (ratios_dec if cell["decision"] else ratios_ord).append(ratio)
            if nc:
                (acc_dec if cell["decision"] else acc_ord).append(nc[0])
            # componentwise between-group var (normalized 9-dim)
            grp = {g: [] for g in GROUP_NAMES}
            for a in range(6):
                for v in gba_norm[a]:
                    grp[GROUPS[a]].append(v)
            mus = {g: np.mean(vs, 0) for g, vs in grp.items() if vs}
            if len(mus) >= 2:
                grand = np.mean([v for vs in grp.values() for v in vs], 0)
                diffs = np.array([(mus[g] - grand) ** 2 for g in mus])   # (n_groups, 9)
                comp_between.append(diffs.mean(0))                        # per-component between-group var
            frac_contact.append(cset / nb); frac_score.append(sset / nb); frac_missing.append(mset / nb)

        ratios = [r for r in per_state_ratio if r is not None]
        comp = np.mean(comp_between, 0) if comp_between else None
        comp_names = ["ball_x", "ball_y", "ball_vx", "ball_vy", "opp_y", "score", "mask_ball", "mask_opp", "mask_vel"]
        horizon_results[str(H)] = {
            "censoring_rate": float(cens / max(total, 1)),
            "branch_diversity_ratio": {
                "mean": float(np.mean(ratios)), "median": float(np.median(ratios)),
                "p25": float(np.percentile(ratios, 25)), "p75": float(np.percentile(ratios, 75)),
                "frac_states_gt1": float(np.mean([r > 1 for r in ratios])),
                "n_states_gt1": int(np.sum([r > 1 for r in ratios])),
                "ci_mean": _ci(ratios),
                "ordinary_median": float(np.median([r for r in ratios_ord if r is not None])) if ratios_ord else None,
                "decision_median": float(np.median([r for r in ratios_dec if r is not None])) if ratios_dec else None},
            "nearest_centroid_decoder": {
                "accuracy": _ci(per_state_acc), "balanced_accuracy": _ci(per_state_bacc),
                "baseline": 1 / 3,
                "ordinary_accuracy": _ci(acc_ord), "decision_accuracy": _ci(acc_dec)},
            "componentwise_between_group_var": dict(zip(comp_names, [round(float(x), 4) for x in comp])) if comp is not None else None,
            "env_footprint": {"frac_branches_paddle_contact": float(np.mean(frac_contact)),
                              "frac_branches_score_event": float(np.mean(frac_score)),
                              "frac_branches_missing_ball_at_H": float(np.mean(frac_missing))},
        }

    # decision rule per horizon
    viable = {}
    for H in HORIZONS:
        r = horizon_results[str(H)]
        dec = r["nearest_centroid_decoder"]["accuracy"]
        decode_above = bool(dec and dec["ci95"][0] > 1 / 3)
        div_ok = r["branch_diversity_ratio"]["median"] > 1 or r["branch_diversity_ratio"]["frac_states_gt1"] >= 0.5
        ord_acc = r["nearest_centroid_decoder"]["ordinary_accuracy"]
        not_only_decision = bool(ord_acc and ord_acc["point"] > 1 / 3)
        comp = r["componentwise_between_group_var"]
        ball_effect = bool(comp and (comp["ball_x"] + comp["ball_y"] + comp["ball_vx"] + comp["ball_vy"]) >
                           comp["score"] + 1e-6) if comp else False
        viable[str(H)] = {"decode_ci_above_base": decode_above, "diversity_ok": div_ok,
                          "not_only_decision": not_only_decision, "ball_components_effect": ball_effect,
                          "VIABLE": bool(decode_above and div_ok and not_only_decision and ball_effect)}

    any_viable = [H for H in HORIZONS if viable[str(H)]["VIABLE"]]
    decision_only = [H for H in HORIZONS if (not viable[str(H)]["VIABLE"]) and viable[str(H)]["diversity_ok"] is False
                     and (horizon_results[str(H)]["nearest_centroid_decoder"]["decision_accuracy"] or {}).get("ci95", [0])[0] > 1/3]
    if any_viable:
        label = "VIABLE HORIZON FOUND"
    elif decision_only:
        label = "LIMITED TO DECISION STATES"
    else:
        label = "NO VIABLE HORIZON"

    out = {"milestone": "Stage-1F-horizon-scan", "horizons": HORIZONS, "R": R,
           "n_cloned_states": len(snaps), "state_seed": state_seed,
           "normalization": "fixed Stage-1D no-self seed-0 training stats (single scheme, all horizons)",
           "semantic_groups": {GROUP_NAMES[0]: [0, 1], GROUP_NAMES[1]: [2, 4], GROUP_NAMES[2]: [3, 5]},
           "random_baseline_semantic": 1 / 3,
           "per_horizon": horizon_results, "viability_per_horizon": viable,
           "viable_horizons": any_viable, "decision_only_horizons": decision_only,
           "recommendation": label,
           "note": "24 cloned states; controlled diagnostic, NOT a population estimate. A positive result "
                   "shows only that a first-action footprint is externally distinguishable at some timescale."}
    (ART / "stage1f_report.json").write_text(json.dumps(out, indent=2))
    _table(out)
    return out


def _table(out):
    L = ["Stage-1F environment-only horizon scan (no critic). diversity ratio>1 = action effect exceeds noise.",
         "decode = nearest-centroid semantic decoder (baseline 0.333).\n",
         f"{'H':>4} {'div_med':>8} {'frac>1':>7} {'decode_acc':>11} {'decode_CI':>16} {'ord_acc':>8} {'dec_acc':>8} {'contact':>8} {'VIABLE':>7}"]
    for H in out["horizons"]:
        r = out["per_horizon"][str(H)]; v = out["viability_per_horizon"][str(H)]
        d = r["nearest_centroid_decoder"]; acc = d["accuracy"]; oa = d["ordinary_accuracy"]; da = d["decision_accuracy"]
        L.append(f"{H:>4} {r['branch_diversity_ratio']['median']:>8.3f} {r['branch_diversity_ratio']['frac_states_gt1']:>7.2f} "
                 f"{acc['point']:>11.3f} {str([round(x,3) for x in acc['ci95']]):>16} "
                 f"{(oa['point'] if oa else float('nan')):>8.3f} {(da['point'] if da else float('nan')):>8.3f} "
                 f"{r['env_footprint']['frac_branches_paddle_contact']:>8.2f} {str(v['VIABLE']):>7}")
    L.append(f"\nRECOMMENDATION: {out['recommendation']}")
    (ART / "summary.txt").write_text("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-1F environment-only horizon scan (no critic).")
    ap.add_argument("--state-seed", type=int, default=8000)
    ap.add_argument("--n-each", type=int, default=12)
    args = ap.parse_args()
    out = run(args.state_seed, args.n_each)
    print(open(ART / "summary.txt").read())


if __name__ == "__main__":
    main()
