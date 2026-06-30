"""Re-evaluate the stored oxygen-aware pilot under the INTENDED rule: the FIRST life loss is terminal.
Reuses artifacts/seaquest/pilot_oxyaware/data/*.npz (truncates each at first life loss); only the
matched-branch and the pure-teacher baseline are re-run fresh (stored data can't provide them).

First-life-loss index T1 = first step with death_respawn != 0 (death animation onset). Valid single-
life trajectory = [0, T1). All post-respawn frames and respawn-trigger surfacings are dropped.
"""
import glob
import json
import os
import sys
from unittest.mock import MagicMock
for _m in ("envpool", "gym"):
    sys.modules.setdefault(_m, MagicMock())
import numpy as np
import ocatari.ram.seaquest as _sq
_orig = _sq._detect_objects_ram
_sq._detect_objects_ram = lambda o, r, h: _orig(o, np.asarray(r, np.int64), h)
from seaquest_ccrl.scripts.g0_closed_loop_eval import SeaquestPort, TEACHER_CKPT, TEACHER_SRC
from seaquest_ccrl.scripts.g0_diag_oxygen_trigger_sweep import prime_len
from seaquest_ccrl.scripts.g0_pilot_oxyaware import read_state
from seaquest_stage_s0.teacher_adapter import CleanRLSeaquestTeacher

DATA = "artifacts/seaquest/pilot_oxyaware/data"
OUT = "artifacts/seaquest/pilot_oxyaware"
TRIGGER, REFILLED, SURFACE_ACTION = 20, 58, 10
UPFAM = {2, 6, 7, 10, 14, 15}
HORIZONS = (16, 32, 64)
AT_SURF_C = 56.0          # center-y at/above this == at the surface (top~46 -> center~51.5)
MAX_STEPS = 2000


def first_life_loss(dr):
    nz = np.where(dr != 0)[0]
    return int(nz[0]) if len(nz) else len(dr)


def classify_terminal(surf, cy, divers, oxy):
    if oxy is not None and oxy <= 2:
        return "oxygen_drown"
    if not surf:
        return "normal_teacher"
    if cy is not None and cy <= AT_SURF_C and divers == 0:
        return "nodiver_surface"
    if cy is not None and cy > AT_SURF_C:
        return "enemy_ascent"
    return "ascent_other"


def _cohen(a, b):
    if a.size < 2 or b.size < 2:
        return np.nan
    sp = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def _boot(a, b, n=2000):
    if a.size < 2 or b.size < 2:
        return [np.nan, np.nan]
    rng = np.random.default_rng(0); d = [rng.choice(a, a.size).mean() - rng.choice(b, b.size).mean() for _ in range(n)]
    return [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)]


def analyze_stored():
    eps = []
    for f in sorted(glob.glob(f"{DATA}/*.npz")):
        z = np.load(f)
        dr = z["death_respawn"]; T1 = first_life_loss(dr)
        sl = slice(0, T1)
        ep = {"T1": T1, "act": z["actions"][sl], "oxy": z["oxygen"][sl].astype(float),
              "surf": z["surfacing"][sl], "py": z["player_pos"][sl, 1].astype(float),
              "divers": z["divers"][sl], "terminated_in_ep": bool(T1 < len(dr))}
        # terminal death state = last valid step before the death (T1-1)
        if ep["terminated_in_ep"] and T1 >= 1:
            j = T1 - 1
            ep["terminal"] = classify_terminal(bool(z["surfacing"][j]), float(z["player_pos"][j, 1]),
                                                int(z["divers"][j]), float(z["oxygen"][j]))
        else:
            ep["terminal"] = "censored_no_death"      # first life survived the 2000-step cap
        eps.append(ep)
    return eps


def metrics_from_eps(eps, label):
    up_lo = up_hi = nlo = nhi = 0; trig = 0; anchors = 0; clean = 0; surf_steps = 0
    lengths = []; tcause = {}
    dyl = {H: [] for H in HORIZONS}; dyh = {H: [] for H in HORIZONS}
    for ep in eps:
        a, oxy, surf, py, div, T1 = ep["act"], ep["oxy"], ep["surf"], ep["py"], ep["divers"], ep["T1"]
        lengths.append(T1); anchors += T1
        tcause[ep["terminal"]] = tcause.get(ep["terminal"], 0) + 1
        surf_steps += int(surf.sum())
        for i in range(len(a)):
            if not (0 <= oxy[i] <= 63):
                continue
            isup = int(a[i]) in UPFAM
            if oxy[i] < TRIGGER:
                nlo += 1; up_lo += isup
                if i > 0 and oxy[i - 1] >= TRIGGER:
                    trig += 1
            else:
                nhi += 1; up_hi += isup
        # clean refills within the single life: surfacing block that reaches surface & refills to >=REFILLED
        t = 0
        while t < T1:
            if not surf[t]:
                t += 1; continue
            s0 = t
            while t < T1 and surf[t]:
                t += 1
            blk_py = py[s0:t]
            reached = bool(np.isfinite(blk_py).any() and np.nanmin(blk_py) <= AT_SURF_C)
            exit_oxy = oxy[min(t, T1 - 1)]
            if reached and exit_oxy >= REFILLED:
                clean += 1
        # oxygen->future, censoring anchors whose horizon crosses T1
        for H in HORIZONS:
            for tt in range(T1 - H):
                if not np.isfinite(py[tt]) or not np.isfinite(py[tt + H]):
                    continue
                dy = py[tt + H] - py[tt]
                (dyl if oxy[tt] < TRIGGER else dyh)[H].append(dy)
    plo = up_lo / nlo if nlo else float("nan"); phi = up_hi / nhi if nhi else float("nan")
    fut = {}
    for H in HORIZONS:
        lo, hi = np.array(dyl[H]), np.array(dyh[H])
        fut[H] = {"n_low": int(lo.size), "n_high": int(hi.size),
                  "mean_dy_low": round(float(lo.mean()), 2) if lo.size else None,
                  "mean_dy_high": round(float(hi.mean()), 2) if hi.size else None,
                  "diff": round(float(lo.mean() - hi.mean()), 2) if lo.size and hi.size else None,
                  "cohen_d": round(float(_cohen(lo, hi)), 3), "ci95_diff": _boot(lo, hi)}
    override_term = sum(tcause.get(k, 0) for k in ("nodiver_surface", "enemy_ascent", "ascent_other"))
    return {"label": label, "n_ep": len(eps), "median_life_len": int(np.median(lengths)),
            "mean_life_len": round(float(np.mean(lengths)), 1), "total_anchors": anchors,
            "P_up_low": round(plo, 4), "P_up_high": round(phi, 4), "gradient": round(plo - phi, 4),
            "low_oxygen_events": trig, "clean_refills": clean, "override_fraction": round(surf_steps / anchors, 4),
            "terminal_cause": tcause, "frac_terminated_by_override": round(override_term / len(eps), 3),
            "future_y": fut}


# ---- fresh runs (terminal rule) ----
def run_terminal(teacher, seed, policy):
    """policy='P1' (wrapper) or 'teacher'. Stop at FIRST life loss. Returns per-step arrays to T1."""
    pr = prime_len(teacher, seed)
    p = SeaquestPort(sticky=0.0, full_action_space=True, seed=seed); p.reset(seed=seed, noop_max=0)
    rng = np.random.default_rng(seed)
    for _ in range(pr):
        p.agent_step(int(teacher.sample_action(p.teacher_obs(), teacher.gumbel_from_uniform(rng.uniform(size=18)))[0]))
    surfacing = False; prev_lives = None
    act, oxyA, surfA, pyA, divA = [], [], [], [], []
    term_state = None
    for s in range(MAX_STEPS):
        P, oxy, oxbar, divers, lives, _ = read_state(p)
        if prev_lives is not None and lives is not None and lives < prev_lives:   # FIRST life loss -> terminal
            cy = pyA[-1] if pyA else None
            term_state = classify_terminal(surfA[-1] if surfA else False, cy, divA[-1] if divA else 0,
                                           oxyA[-1] if oxyA else None)
            break
        if P is None:                       # death animation has begun -> terminal at last valid frame
            term_state = classify_terminal(surfA[-1] if surfA else False, pyA[-1] if pyA else None,
                                           divA[-1] if divA else 0, oxyA[-1] if oxyA else None)
            break
        prev_lives = lives if lives is not None else prev_lives
        obs = p.teacher_obs()
        teacher_a = int(teacher.sample_action(obs, teacher.gumbel_from_uniform(rng.uniform(size=18)))[0])
        if policy == "P1":
            wo = oxy if oxbar else -1
            if wo >= 0:
                if surfacing and wo >= REFILLED:
                    surfacing = False
                elif not surfacing and wo < TRIGGER:
                    surfacing = True
            a = SURFACE_ACTION if surfacing else teacher_a
        else:
            a = teacher_a; surfacing = False
        act.append(a); oxyA.append(float(oxy if oxbar else last(oxyA)))
        surfA.append(surfacing); pyA.append(P[1] + P[3] / 2.0); divA.append(divers)
        p.agent_step(a)
    return {"T1": len(act), "act": np.array(act), "oxy": np.array(oxyA, float), "surf": np.array(surfA, bool),
            "py": np.array(pyA, float), "divers": np.array(divA), "terminal": term_state or "censored_no_death",
            "terminated_in_ep": term_state is not None}


def last(a):
    return a[-1] if a else 0.0


def matched_terminal(teacher, seeds, H=32, every=70):
    surv_diffs = []; lo_deaths = hi_deaths = npairs = 0
    for seed in seeds:
        pr = prime_len(teacher, seed)
        p = SeaquestPort(sticky=0.0, full_action_space=True, seed=seed); p.reset(seed=seed, noop_max=0)
        rng = np.random.default_rng(seed)
        for _ in range(pr):
            p.agent_step(int(teacher.sample_action(p.teacher_obs(), teacher.gumbel_from_uniform(rng.uniform(size=18)))[0]))
        for s in range(1200):
            P, oxy, oxbar, _, lives, _ = read_state(p)
            if P is not None and oxbar and oxy >= TRIGGER and s % every == 0:
                snap = p.snapshot()
                ylo, dlo = _branch_term(p, snap, "low", H, lives)
                yhi, dhi = _branch_term(p, snap, "high", H, lives)
                p.restore(snap); npairs += 1; lo_deaths += dlo; hi_deaths += dhi
                if not dlo and not dhi:
                    surv_diffs.append(ylo - yhi)
            a = int(teacher.sample_action(p.teacher_obs(), teacher.gumbel_from_uniform(rng.uniform(size=18)))[0])
            P2, _, _, _, lv2, _ = read_state(p)
            if p.agent_step(a)["terminated"]:
                break
        # stop scanning this seed at its own first life loss to stay pre-terminal
    d = np.array(surv_diffs)
    return {"n_anchors": npairs, "n_survived_both": int(d.size),
            "mean_future_y_low_minus_high_survivors": round(float(d.mean()), 2) if d.size else None,
            "low_branch_death_rate": round(lo_deaths / npairs, 3) if npairs else None,
            "high_branch_death_rate": round(hi_deaths / npairs, 3) if npairs else None}


_T = None
def _branch_term(p, snap, kind, H, start_lives):
    p.restore(snap)
    for _ in range(H):
        P, _, _, _, lives, _ = read_state(p)
        if P is None or (start_lives is not None and lives is not None and lives < start_lives):
            return None, 1                                  # died within H
        a = SURFACE_ACTION if kind == "low" else int(_T.greedy_action(p.teacher_obs())[0])
        p.agent_step(a)
    P, *_ = read_state(p)
    return ((P[1] + P[3] / 2.0) if P is not None else None), 0


def main():
    global _T
    teacher = CleanRLSeaquestTeacher(TEACHER_CKPT, TEACHER_SRC, mod_name="cleanrl_src_A"); _T = teacher
    seeds = list(range(7000, 7014))

    p1 = metrics_from_eps(analyze_stored(), "P1 (stored, truncated@first-life-loss)")
    # fresh pure-teacher baseline under the same terminal rule
    t_eps = [_to_ep(run_terminal(teacher, sd, "teacher")) for sd in seeds]
    tb = metrics_from_eps(t_eps, "HF teacher (terminal)")
    mb = matched_terminal(teacher, seeds[:6])

    summary = {"rule": "first life loss is terminal; no post-respawn frames; goals/horizons cannot cross it",
               "P1": p1, "teacher_baseline": tb, "matched_branch_terminal": mb}
    json.dump(summary, open(f"{OUT}/pilot_terminal_recompute.json", "w"), indent=2)

    print("\n=== PILOT under FIRST-LIFE-LOSS terminal rule ===")
    for m in (p1, tb):
        print(f"\n[{m['label']}]")
        print(f"  life length: median={m['median_life_len']} mean={m['mean_life_len']}  total_anchors={m['total_anchors']}")
        print(f"  oxygen->action: P(UP|lo)={m['P_up_low']} P(UP|hi)={m['P_up_high']} grad={m['gradient']:+}  low_oxy_events={m['low_oxygen_events']}")
        print(f"  clean_refills={m['clean_refills']}  override_frac={m['override_fraction']}")
        print(f"  terminal causes={m['terminal_cause']}  frac_terminated_by_override={m['frac_terminated_by_override']}")
        print(f"  future_y: " + " ".join(f"H{H} d={m['future_y'][H]['diff']}(CI{m['future_y'][H]['ci95_diff']})" for H in HORIZONS))
    print(f"\n[matched branch terminal] {mb}")
    print(f"\nWROTE {OUT}/pilot_terminal_recompute.json")


def _to_ep(r):
    return {"T1": r["T1"], "act": r["act"], "oxy": r["oxy"], "surf": r["surf"], "py": r["py"],
            "divers": r["divers"], "terminal": r["terminal"], "terminated_in_ep": r["terminated_in_ep"]}


if __name__ == "__main__":
    main()
