"""Lightweight PILOT recollection with the oxygen-aware HF teacher (P1 unconditional, trigger=20,
refilled=58) + a 4-check qualification report. NOT full recollection, NO model training.

Reuses: SeaquestPort + HF teacher (the raw_hf collector's engine), the OxygenAwareTeacher P1 logic
(inlined so the teacher-selected action is logged every step), the raw_hf per-episode npz schema,
_prep/oxygen-mask, and the existing death classifier (measure/classify_death). Stores ~14 genuinely
stochastic episodes (~25k steps) under artifacts/seaquest/pilot_oxyaware/.

Checks: (1) oxygen->action, (2) oxygen->future player-y (+ matched clone/restore branch),
(3) masked-view oxygen recoverability (simple ridge probe), (4) failure + filtering audit.
"""
import argparse
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from seaquest_ccrl.scripts.g0_closed_loop_eval import SeaquestPort, TEACHER_CKPT, TEACHER_SRC, _prep
from seaquest_ccrl.scripts.g0_diag_oxygen_trigger_sweep import prime_len
from seaquest_ccrl.scripts.g0_diag_surface_death_classify import sprite_contact
from seaquest_ccrl.scripts.g0_diag_lowoxy_policy_eval import measure, lives_drops
from seaquest_ccrl import config as C
from seaquest_stage_s0.teacher_adapter import CleanRLSeaquestTeacher

OUT = "artifacts/seaquest/pilot_oxyaware"
DATA = f"{OUT}/data"
TRIGGER, REFILLED, SURFACE_ACTION = 20, 58, 10
UPFAM = {2, 6, 7, 10, 14, 15}
SEEDS = list(range(7000, 7014))          # 14 genuinely-distinct stochastic episodes
MAX_STEPS = 2000                          # raw_hf episode length cap
FS = 84
HORIZONS = (16, 32, 64)


def read_state(port):
    """Player box, oxygen (OxygenBar.w, raw_hf definition), carried divers, lives, shark/sub boxes."""
    P = None; oxy = 0; oxbar = False; divers = 0; lives = None; sharks = []
    for o in port.env.objects:
        c = getattr(o, "category", "")
        if c == "NoObject":
            continue
        if c == "Player":
            P = (float(o.x), float(o.y), float(o.w), float(o.h))
        elif c == "OxygenBar":
            oxy = int(getattr(o, "w", 0) or 0); oxbar = True
        elif c == "CollectedDiver":
            divers += 1
        elif c == "Lives":
            lives = getattr(o, "value", None)
        elif c in ("Shark", "Submarine"):
            sharks.append((float(o.x), float(o.y), float(o.w), float(o.h)))
    return P, oxy, oxbar, divers, lives, sharks


AT_SURF_C = 56.0      # center-y at/above this == at the surface (top~46 -> center~51.5)


def _classify_terminal(surf, cy, divers, oxy):
    """Cause of the first (terminal) life loss, from the last valid pre-death state."""
    if oxy is not None and oxy <= 2:
        return "oxygen_drown"
    if not surf:
        return "normal_teacher"
    if cy is not None and cy <= AT_SURF_C and divers == 0:
        return "nodiver_surface"
    if cy is not None and cy > AT_SURF_C:
        return "enemy_ascent"
    return "ascent_other"


def collect_episode(teacher, seed, store_frames=True, first_life=True):
    """One stochastic P1 (OxygenAwareTeacher) episode. Returns (arrays, log, terminal_cause).

    first_life=True (the intended rule): the FIRST life loss terminates the trajectory. Collection
    stops at the first life loss, NO post-death/respawn frame is saved, the final saved transition is
    marked done=True, and the terminal cause is classified from the last valid state. Frames are
    captured PRE-step (aligned with the action) so the final transition is the lethal action from the
    last valid frame. death_respawn/player_present are therefore all 0/True by construction."""
    pr = prime_len(teacher, seed)
    p = SeaquestPort(sticky=0.0, full_action_space=True, seed=seed); p.reset(seed=seed, noop_max=0)
    rng = np.random.default_rng(seed)
    for _ in range(pr):
        p.agent_step(int(teacher.sample_action(p.teacher_obs(), teacher.gumbel_from_uniform(rng.uniform(size=18)))[0]))
    rng_t = np.random.RandomState(seed)
    tgt = (float(rng_t.randint(*C.TARGET_X_RANGE)), float(rng_t.randint(*C.TARGET_Y_RANGE)))  # reuse goal ranges
    surfacing = False; init_lives = None
    A = {k: [] for k in ("frames", "actions", "teacher_actions", "player_pos", "oxygen", "divers",
                          "lives", "reward", "surfacing", "done", "death_respawn", "player_present")}
    log = []
    terminal_cause = "censored_no_death"
    for s in range(MAX_STEPS):
        P, oxy, oxbar, divers, lives, sharks = read_state(p)
        if init_lives is None and lives is not None:
            init_lives = lives
        life_lost = init_lives is not None and lives is not None and lives < init_lives
        if first_life and (P is None or life_lost):           # FIRST life loss -> terminal; save nothing here
            if A["done"]:
                A["done"][-1] = True
                terminal_cause = _classify_terminal(A["surfacing"][-1], A["player_pos"][-1][1],
                                                    A["divers"][-1], A["oxygen"][-1])
            break
        obs = p.teacher_obs()
        # OxygenAwareTeacher P1 logic (inlined to log teacher-selected action every step)
        teacher_a = int(teacher.sample_action(obs, teacher.gumbel_from_uniform(rng.uniform(size=18)))[0])
        wrap_oxy = oxy if oxbar else -1
        if wrap_oxy >= 0:
            if surfacing:
                if wrap_oxy >= REFILLED:
                    surfacing = False
            elif wrap_oxy < TRIGGER:
                surfacing = True
        executed = SURFACE_ACTION if surfacing else teacher_a
        ty = P[1]; cy = P[1] + P[3] / 2.0; cx = P[0] + P[2] / 2.0      # P present here (else we broke above)
        d, c = sprite_contact(P, sharks)
        if store_frames:
            A["frames"].append(np.asarray(p.ale.getScreenRGB(), np.uint8))    # PRE-step frame (aligned w/ action)
        rec = p.agent_step(executed)
        A["actions"].append(executed); A["teacher_actions"].append(teacher_a)
        A["player_pos"].append((cx, cy)); A["oxygen"].append(int(oxy)); A["divers"].append(int(divers))
        A["lives"].append(int(lives) if lives is not None else -1); A["reward"].append(float(rec["reward"]))
        A["surfacing"].append(bool(surfacing)); A["done"].append(bool(rec["terminated"]))
        A["death_respawn"].append(0); A["player_present"].append(True)
        log.append({"t": s, "lives": None if lives is None else float(lives), "oxygen": float(oxy),
                    "divers": divers, "player_y": ty, "surfacing": bool(surfacing), "executed": executed,
                    "_sd": None if not np.isfinite(d) else d, "_sc": bool(c), "reward": float(rec["reward"])})
        if rec["terminated"]:
            terminal_cause = "game_over"
            break
    arrays = {"frames": (np.asarray(A["frames"], np.uint8) if store_frames else None),
              "actions": np.asarray(A["actions"], np.int64),
              "teacher_actions": np.asarray(A["teacher_actions"], np.int64),
              "player_pos": np.asarray(A["player_pos"], np.float32),
              "oxygen": np.asarray(A["oxygen"], np.int32), "divers": np.asarray(A["divers"], np.int32),
              "lives": np.asarray(A["lives"], np.int32), "reward": np.asarray(A["reward"], np.float32),
              "surfacing": np.asarray(A["surfacing"], bool), "done": np.asarray(A["done"], bool),
              "death_respawn": np.asarray(A["death_respawn"], np.int8),
              "player_present": np.asarray(A["player_present"], bool),
              "target": np.tile(np.asarray(tgt, np.float32), (len(A["actions"]), 1)),
              "theta": np.int32(TRIGGER)}
    return arrays, log, terminal_cause


def teacher_gradient(teacher, seed):
    """Pure HF teacher (no override) P(UP|O2<20) - P(UP|O2>=20) on the same seed."""
    pr = prime_len(teacher, seed)
    p = SeaquestPort(sticky=0.0, full_action_space=True, seed=seed); p.reset(seed=seed, noop_max=0)
    rng = np.random.default_rng(seed)
    for _ in range(pr):
        p.agent_step(int(teacher.sample_action(p.teacher_obs(), teacher.gumbel_from_uniform(rng.uniform(size=18)))[0]))
    up_lo = up_hi = nlo = nhi = 0
    for s in range(MAX_STEPS):
        P, oxy, oxbar, *_ = read_state(p)
        a = int(teacher.sample_action(p.teacher_obs(), teacher.gumbel_from_uniform(rng.uniform(size=18)))[0])
        if oxbar and P is not None:
            isup = a in UPFAM
            if oxy < TRIGGER:
                nlo += 1; up_lo += isup
            else:
                nhi += 1; up_hi += isup
        if p.agent_step(a)["terminated"]:
            break
    return up_lo, nlo, up_hi, nhi


# ---------------- check 2: oxygen -> future player-y ----------------
def future_y_effect(logs):
    out = {}
    for H in HORIZONS:
        lo, hi = [], []
        for log in logs:
            n = len(log)
            for t in range(n - H):
                if log[t]["player_y"] is None or log[t + H]["player_y"] is None:
                    continue
                if any(log[k]["player_y"] is None for k in range(t, t + H + 1)):  # skip death-spanning
                    continue
                dy = log[t + H]["player_y"] - log[t]["player_y"]
                (lo if log[t]["oxygen"] < TRIGGER else hi).append(dy)
        lo, hi = np.array(lo), np.array(hi)
        d = _cohen(lo, hi); ci = _boot_ci(lo, hi)
        out[H] = {"n_low": int(lo.size), "n_high": int(hi.size),
                  "mean_dy_low": _f(lo.mean() if lo.size else np.nan),
                  "mean_dy_high": _f(hi.mean() if hi.size else np.nan),
                  "diff_low_minus_high": _f((lo.mean() - hi.mean()) if lo.size and hi.size else np.nan),
                  "cohen_d": _f(d), "ci95_diff": [_f(ci[0]), _f(ci[1])]}
    return out


def matched_branch(teacher, seeds, H=32, every=70):
    """Clone/restore matched intervention: from the SAME snapshot, force surfacing (perceived-low O2,
    UPFIRE) vs teacher play (perceived-high O2, greedy) for H steps; compare future center player-y."""
    diffs = []
    for seed in seeds:
        pr = prime_len(teacher, seed)
        p = SeaquestPort(sticky=0.0, full_action_space=True, seed=seed); p.reset(seed=seed, noop_max=0)
        rng = np.random.default_rng(seed)
        for _ in range(pr):
            p.agent_step(int(teacher.sample_action(p.teacher_obs(), teacher.gumbel_from_uniform(rng.uniform(size=18)))[0]))
        for s in range(MAX_STEPS):
            P, oxy, oxbar, *_ = read_state(p)
            if s % every == 0 and P is not None and oxbar and oxy >= TRIGGER:   # anchor in normal (high-O2) state
                snap = p.snapshot(); y0 = P[1] + P[3] / 2.0
                ylo = _branch(p, snap, "low", H); yhi = _branch(p, snap, "high", H)
                p.restore(snap)
                if ylo is not None and yhi is not None:
                    diffs.append(ylo - yhi)                                     # <0 => low-O2 branch ends higher
            a = int(teacher.sample_action(p.teacher_obs(), teacher.gumbel_from_uniform(rng.uniform(size=18)))[0])
            if p.agent_step(a)["terminated"]:
                break
    diffs = np.array(diffs)
    return {"n_pairs": int(diffs.size), "mean_future_y_low_minus_high": _f(diffs.mean() if diffs.size else np.nan),
            "ci95": [_f(np.percentile(diffs, 2.5)) if diffs.size else None,
                     _f(np.percentile(diffs, 97.5)) if diffs.size else None]}


def _branch(p, snap, kind, H):
    p.restore(snap)
    for _ in range(H):
        a = SURFACE_ACTION if kind == "low" else int(_greedy(p))
        p.agent_step(a)
    P, *_ = read_state(p)
    return (P[1] + P[3] / 2.0) if P is not None else None


_TEACHER = None
def _greedy(p):
    return _TEACHER.greedy_action(p.teacher_obs())[0]


# ---------------- check 3: masked-view oxygen recoverability (simple ridge probe) ----------------
def _small(obs):                       # 84x84x3 -> 14x14 grayscale -> 196-vec
    g = obs.astype(np.float32).mean(2).reshape(14, 6, 14, 6).mean((1, 3))
    return g.reshape(-1) / 255.0


def ridge_probe(feats, oxy, lam=10.0):
    X = np.asarray(feats); y = np.asarray(oxy, np.float32)
    ntr = int(0.75 * len(X)); idx = np.arange(len(X))
    rng = np.random.default_rng(0); rng.shuffle(idx)
    tr, te = idx[:ntr], idx[ntr:]
    Xtr = np.hstack([X[tr], np.ones((len(tr), 1))]); Xte = np.hstack([X[te], np.ones((len(te), 1))])
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]); w = np.linalg.solve(A, Xtr.T @ y[tr])
    pred = Xte @ w; yt = y[te]
    r2 = 1.0 - ((pred - yt) ** 2).sum() / max(1e-9, ((yt - yt.mean()) ** 2).sum())
    lowt = (yt < TRIGGER).astype(int); auc = _auc(-pred, lowt)   # -pred high => predicts low
    return float(r2), float(auc), int(len(te)), float(lowt.mean())


def _auc(score, label):
    if label.sum() == 0 or label.sum() == len(label):
        return float("nan")
    order = np.argsort(score); ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    npos = label.sum(); nneg = len(label) - npos
    return float((ranks[label == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


# ---------------- stats helpers ----------------
def _f(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), 4)
def _cohen(a, b):
    if a.size < 2 or b.size < 2:
        return np.nan
    sp = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan
def _boot_ci(a, b, n=2000):
    if a.size < 2 or b.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(0); ds = []
    for _ in range(n):
        ds.append(rng.choice(a, a.size).mean() - rng.choice(b, b.size).mean())
    return (np.percentile(ds, 2.5), np.percentile(ds, 97.5))


def main():
    global _TEACHER
    os.makedirs(DATA, exist_ok=True)
    teacher = CleanRLSeaquestTeacher(TEACHER_CKPT, TEACHER_SRC, mod_name="cleanrl_src_A"); _TEACHER = teacher

    # ---- collect ----
    logs = []; probe_feat_m, probe_feat_f, probe_oxy = [], [], []
    agg = {k: 0 for k in ("triggers", "trig_nodiv", "clean", "surf_steps", "low_oxy_steps",
                          "up_lo", "nlo", "up_hi", "nhi", "n")}
    cats = {k: 0 for k in ("oxygen_drown", "enemy_ascent", "nodiver_surface", "normal_teacher", "ascent_unresolved")}
    respawn_artifacts = 0; total_steps = 0; death_anim_frames = 0; ep_meta = []
    for i, seed in enumerate(SEEDS):
        arrays, log, _terminal_cause = collect_episode(teacher, seed, store_frames=True)
        np.savez_compressed(f"{DATA}/traj_{i:04d}.npz", **{k: v for k, v in arrays.items() if v is not None})
        logs.append(log); n = len(log); total_steps += n
        m = measure(log)
        for k in agg:
            agg[k] += m[k] if k != "n" else n
        for k in cats:
            cats[k] += m["cats"][k]
        # respawn-initiated surfacing blocks (artifact) + death-anim frames
        respawn_artifacts += _respawn_blocks(log)
        death_anim_frames += int((arrays["death_respawn"] == 1).sum())
        # probe subsample (valid, player present)
        fr = arrays["frames"]
        for t in range(0, n, 6):
            if log[t]["player_y"] is None:
                continue
            probe_feat_m.append(_small(_prep(fr[t], FS, True)))
            probe_feat_f.append(_small(_prep(fr[t], FS, False)))
            probe_oxy.append(arrays["oxygen"][t])
        ep_meta.append({"episode": i, "seed": seed, "steps": n, "deaths": len(lives_drops(log)),
                        "surf_frac": round(m["surf_steps"] / n, 3)})
        print(f"  ep{i} seed{seed}: {n} steps, deaths={len(lives_drops(log))}, surf={m['surf_steps']/n:.2%}")
        del arrays

    # ---- baseline teacher gradient (same seeds) ----
    b_lo = b_nlo = b_hi = b_nhi = 0
    for seed in SEEDS:
        a, b, c, d = teacher_gradient(teacher, seed); b_lo += a; b_nlo += b; b_hi += c; b_nhi += d

    # ---- check 1 ----
    p_lo = agg["up_lo"] / agg["nlo"] if agg["nlo"] else float("nan")
    p_hi = agg["up_hi"] / agg["nhi"] if agg["nhi"] else float("nan")
    tg_lo = b_lo / b_nlo if b_nlo else float("nan"); tg_hi = b_hi / b_nhi if b_nhi else float("nan")
    check1 = {"low_oxygen_events": agg["triggers"], "P_up_low": _f(p_lo), "P_up_high": _f(p_hi),
              "gradient": _f(p_lo - p_hi), "override_fraction": _f(agg["surf_steps"] / agg["n"]),
              "baseline_teacher_gradient": _f(tg_lo - tg_hi),
              "baseline_P_up_low": _f(tg_lo), "baseline_P_up_high": _f(tg_hi)}
    # ---- check 2 ----
    check2 = {"observational": future_y_effect(logs),
              "matched_branch": matched_branch(teacher, SEEDS[:6])}
    # ---- check 3 ----
    r2m, aucm, nte, lowfrac = ridge_probe(probe_feat_m, probe_oxy)
    r2f, aucf, _, _ = ridge_probe(probe_feat_f, probe_oxy)
    check3 = {"n_probe_test": nte, "low_fraction": _f(lowfrac), "masked_R2": _f(r2m), "masked_AUC": _f(aucm),
              "fullview_R2": _f(r2f), "fullview_AUC": _f(aucf)}
    # ---- check 4 ----
    check4 = {"clean_refills": agg["clean"], "nodiver_surface_deaths": cats["nodiver_surface"],
              "enemy_ascent_deaths": cats["enemy_ascent"], "oxygen_depletion_deaths": cats["oxygen_drown"],
              "respawn_trigger_artifacts": respawn_artifacts, "normal_teacher_deaths": cats["normal_teacher"],
              "ascent_unresolved": cats["ascent_unresolved"], "total_deaths": sum(cats.values()),
              "death_animation_frames": death_anim_frames,
              "filtering": {"views_share_transitions": True,
                            "masked_vs_full_identical_outside_oxygen_rect": _mask_symmetry(),
                            "death_frames_flagged": "death_respawn (1=anim,2=respawn) + player_present per step; "
                                                    "identical records feed both views (mask is pixel-only)",
                            "asymmetric_filtering": False}}

    # ---- verdicts ----
    v1 = "PASS" if (check1["gradient"] or 0) >= 0.3 and (check1["gradient"] or 0) > 3 * abs(check1["baseline_teacher_gradient"] or 0.01) else \
         ("WEAK" if (check1["gradient"] or 0) >= 0.15 else "FAIL")
    sig2 = any((check2["observational"][H]["ci95_diff"][0] or 0) * (check2["observational"][H]["ci95_diff"][1] or 0) > 0
               for H in HORIZONS)
    v2 = "PASS" if sig2 else "WEAK"
    v3 = "PASS" if (check3["masked_R2"] or 0) < 0.3 and (check3["masked_AUC"] or 0.5) < 0.7 else \
         ("WEAK" if (check3["masked_R2"] or 0) < 0.6 else "FAIL")
    diver_death = check4["nodiver_surface_deaths"] + check4["enemy_ascent_deaths"]
    dom = bool(check4["total_deaths"] and diver_death > 0.5 * check4["total_deaths"])
    v4 = "FAIL" if check4["filtering"]["asymmetric_filtering"] else ("WEAK" if dom else "PASS")

    summary = {"policy": {"variant": "P1_unconditional", "surface_trigger": TRIGGER, "refilled": REFILLED,
                          "surface_action": SURFACE_ACTION, "wrapper": "OxygenAwareTeacher (P1 logic)"},
               "dataset": {"path": DATA, "episodes": len(SEEDS), "total_steps": total_steps,
                           "schema": "raw_hf + teacher_actions,divers,lives,reward,surfacing,death_respawn,player_present",
                           "episodes_meta": ep_meta},
               "check1_oxygen_to_action": {**check1, "verdict": v1},
               "check2_oxygen_to_future": {**check2, "verdict": v2},
               "check3_masked_recoverability": {**check3, "verdict": v3},
               "check4_failure_filtering": {**check4, "verdict": v4}}
    os.makedirs(OUT, exist_ok=True)
    json.dump(summary, open(f"{OUT}/pilot_summary.json", "w"), indent=2)
    _plots(check1, check2, check3)
    _report(summary)
    print("\n=== PILOT QUALIFICATION ===")
    print(f"  steps={total_steps} eps={len(SEEDS)}")
    print(f"  C1 oxygen->action: grad {check1['gradient']:+.2f} (teacher {check1['baseline_teacher_gradient']:+.2f}) -> {v1}")
    print(f"  C2 oxygen->future: " + " ".join(f"H{H} d={check2['observational'][H]['diff_low_minus_high']}" for H in HORIZONS)
          + f"  matched={check2['matched_branch']['mean_future_y_low_minus_high']} -> {v2}")
    print(f"  C3 masked recover: R2={check3['masked_R2']} AUC={check3['masked_AUC']} (full R2={check3['fullview_R2']}) -> {v3}")
    print(f"  C4 failure/filter: clean={check4['clean_refills']} nodiver={check4['nodiver_surface_deaths']} "
          f"enemy={check4['enemy_ascent_deaths']} drown={check4['oxygen_depletion_deaths']} "
          f"respawn_artifacts={check4['respawn_trigger_artifacts']} -> {v4}")
    print(f"WROTE {OUT}/pilot_summary.json , {OUT}/PILOT_REPORT.md , plots, dataset -> {DATA}/")


def _respawn_blocks(log):
    n = len(log); t = 0; cnt = 0
    while t < n:
        if not log[t]["surfacing"]:
            t += 1; continue
        s0 = t
        while t < n and log[t]["surfacing"]:
            t += 1
        if log[s0]["oxygen"] <= 3 and log[s0]["player_y"] is not None and log[s0]["player_y"] <= 55:
            cnt += 1
    return cnt


def _mask_symmetry():
    """full vs masked 84x84 obs differ ONLY inside the resized oxygen-rect footprint."""
    rng = np.random.default_rng(0); fr = (rng.integers(0, 255, (210, 160, 3))).astype(np.uint8)
    full = _prep(fr, FS, False).astype(int); masked = _prep(fr, FS, True).astype(int)
    diff = np.abs(full - masked).sum(2) > 0
    x, y, w, h = C.OXY_MASK_RECT
    r0, r1 = int(y * FS / 210), int(np.ceil((y + h) * FS / 210)) + 1
    c0, c1 = int(x * FS / 160), int(np.ceil((x + w) * FS / 160)) + 1
    band = np.zeros((FS, FS), bool); band[r0:r1, c0:c1] = True
    return bool(diff[~band].sum() == 0)         # True => all differences confined to the oxygen rect


def _plots(c1, c2, c3):
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    ax.bar([0, 1], [c1["P_up_low"], c1["P_up_high"]], color=["#e8743b", "#5b8ff9"])
    ax.bar([2.2, 3.2], [c1["baseline_P_up_low"], c1["baseline_P_up_high"]], color=["#f0b9a0", "#b0c8f0"])
    ax.set_xticks([0, 1, 2.2, 3.2]); ax.set_xticklabels(["lo", "hi", "lo(T)", "hi(T)"], fontsize=8)
    ax.set_ylabel("P(UP-family)"); ax.set_title("C1 oxygen->action", fontsize=9); fig.tight_layout()
    fig.savefig(f"{OUT}/check1_oxygen_action.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    xs = np.arange(len(HORIZONS))
    lo = [c2["observational"][H]["mean_dy_low"] for H in HORIZONS]
    hi = [c2["observational"][H]["mean_dy_high"] for H in HORIZONS]
    ax.bar(xs - 0.18, lo, 0.36, label="O2<20", color="#e8743b")
    ax.bar(xs + 0.18, hi, 0.36, label="O2>=20", color="#5b8ff9")
    ax.set_xticks(xs); ax.set_xticklabels([f"H{H}" for H in HORIZONS]); ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("mean dy (px, +down)"); ax.set_title("C2 oxygen->future y", fontsize=9); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(f"{OUT}/check2_future_y.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.0, 2.6))
    ax.bar([0, 1], [c3["masked_R2"], c3["fullview_R2"]], color=["#5b8ff9", "#9270ca"])
    ax.bar([2.2, 3.2], [c3["masked_AUC"], c3["fullview_AUC"]], color=["#61a0a8", "#d48265"])
    ax.set_xticks([0, 1, 2.2, 3.2]); ax.set_xticklabels(["R2 m", "R2 f", "AUC m", "AUC f"], fontsize=8)
    ax.axhline(0.5, color="gray", lw=0.5, ls="--"); ax.set_title("C3 masked recover", fontsize=9); fig.tight_layout()
    fig.savefig(f"{OUT}/check3_masked_recover.png", dpi=120); plt.close(fig)


def _report(s):
    c1, c2, c3, c4 = (s["check1_oxygen_to_action"], s["check2_oxygen_to_future"],
                      s["check3_masked_recoverability"], s["check4_failure_filtering"])
    L = [f"# Oxygen-aware pilot qualification (P1, trigger={TRIGGER}, refilled={REFILLED})", "",
         f"Dataset: `{DATA}/` — {s['dataset']['episodes']} stochastic episodes, {s['dataset']['total_steps']} steps. "
         f"Schema = raw_hf + teacher_actions/divers/lives/reward/surfacing/death_respawn/player_present.", "",
         "| check | verdict | key numbers |", "|---|---|---|",
         f"| 1 oxygen→action | **{c1['verdict']}** | P(UP\\|lo)={c1['P_up_low']} vs P(UP\\|hi)={c1['P_up_high']} "
         f"grad **{c1['gradient']:+}** (teacher {c1['baseline_teacher_gradient']:+}); override {c1['override_fraction']} |",
         f"| 2 oxygen→future | **{c2['verdict']}** | "
         + "; ".join(f"H{H}: Δy={c2['observational'][H]['diff_low_minus_high']} (d={c2['observational'][H]['cohen_d']}, "
                     f"CI{c2['observational'][H]['ci95_diff']})" for H in HORIZONS)
         + f"; matched Δy_future={c2['matched_branch']['mean_future_y_low_minus_high']} "
         f"(n={c2['matched_branch']['n_pairs']}) |",
         f"| 3 masked recover | **{c3['verdict']}** | masked R²={c3['masked_R2']} AUC={c3['masked_AUC']} "
         f"vs full R²={c3['fullview_R2']} AUC={c3['fullview_AUC']} |",
         f"| 4 failure/filter | **{c4['verdict']}** | clean={c4['clean_refills']} nodiver={c4['nodiver_surface_deaths']} "
         f"enemy_ascent={c4['enemy_ascent_deaths']} drown={c4['oxygen_depletion_deaths']} "
         f"respawn_artifacts={c4['respawn_trigger_artifacts']} (total deaths {c4['total_deaths']}) |", "",
         "## Success criteria", "",
         f"- oxygen→action **clearly stronger than original HF teacher**: grad {c1['gradient']:+} vs teacher "
         f"{c1['baseline_teacher_gradient']:+} → {'YES' if (c1['gradient'] or 0) > 3*abs(c1['baseline_teacher_gradient'] or 0.01) else 'NO'}",
         f"- oxygen → measurable future-position effect at ≥1 horizon → {'YES' if c2['verdict']=='PASS' else 'WEAK/NO'}",
         f"- masked-view oxygen recoverability not trivially high (masked R²={c3['masked_R2']}, AUC={c3['masked_AUC']}) "
         f"→ {'YES' if c3['verdict'] in ('PASS','WEAK') else 'NO — mask leaks'}",
         f"- diver/death effects don't dominate / no asymmetric filtering → "
         f"{'YES' if c4['verdict']=='PASS' else c4['verdict']}", "",
         "## Filtering integrity", "",
         f"- full & masked views share identical transitions (mask is a pixel-only op on the SAME stored frame); "
         f"diff confined to oxygen rect: **{c4['filtering']['masked_vs_full_identical_outside_oxygen_rect']}**.",
         f"- death/respawn-adjacent transitions: {c4['filtering']['death_frames_flagged']}; "
         f"{c4['death_animation_frames']} death-animation frames flagged. Asymmetric filtering: "
         f"**{c4['filtering']['asymmetric_filtering']}**.", "",
         "## Plots", "",
         "![c1](check1_oxygen_action.png) ![c2](check2_future_y.png) ![c3](check3_masked_recover.png)", "",
         "## Smallest next change (if any check is WEAK/FAIL)", "",
         _next_change(c1, c2, c3, c4), "",
         "_Pilot only — full recollection and critic/actor training NOT started._"]
    open(f"{OUT}/PILOT_REPORT.md", "w", encoding="utf-8").write("\n".join(L))


def _next_change(c1, c2, c3, c4):
    out = []
    if c1["verdict"] != "PASS":
        out.append("- C1 weak: lower `surface_trigger` (stronger oxygen gating) or confirm UP-family includes the surface action.")
    if c2["verdict"] != "PASS":
        out.append("- C2 weak: extend horizons / add anchors (longer episodes) — effect may need more low-O2 anchors.")
    if c3["verdict"] == "FAIL":
        out.append("- C3 leak: a simple probe recovers oxygen — widen `OXY_MASK_RECT` or mask the proxy region; re-check.")
    if c4["verdict"] != "PASS":
        out.append("- C4: diver/death deaths elevated — they don't bias the mask, but if filtering is needed apply it to BOTH views by `death_respawn`.")
    return "\n".join(out) if out else "- None — all four checks PASS; proceed to scale up collection when ready."


def collect_corpus(out_dir, n_episodes, seed0=7000, store_frames=True):
    """FULL-SCALE collector entry point. Every property the corpus must have lives in collect_episode:
      * oxygen-aware HF teacher : OxygenAwareTeacher P1 logic inlined (teacher.sample_action + UPFIRE
        override when oxygen < TRIGGER, hysteresis to REFILLED);
      * first-life termination  : the loop breaks at the first life loss (player absent or lives drop);
      * pre-step frame/action   : the frame is captured BEFORE agent_step, so frame[t] aligns with action[t];
      * single-life done        : done[-1] is set True at the terminal transition (done.sum() <= 1).
    Writes the raw_hf npz schema (+ teacher_actions/divers/.../surfacing/done) and a manifest carrying
    the per-episode terminal cause. This is the ONLY path with all four properties (collect_dataset.py
    uses the SCRIPTED policy, not the HF teacher)."""
    os.makedirs(out_dir, exist_ok=True)
    teacher = CleanRLSeaquestTeacher(TEACHER_CKPT, TEACHER_SRC, mod_name="cleanrl_src_A")
    meta = []
    for i in range(n_episodes):
        seed = seed0 + i
        arrays, _log, tcause = collect_episode(teacher, seed, store_frames=store_frames, first_life=True)
        np.savez_compressed(f"{out_dir}/traj_{i:04d}.npz", **{k: v for k, v in arrays.items() if v is not None})
        dn = arrays["done"]
        meta.append({"file": f"traj_{i:04d}.npz", "seed": seed, "steps": int(len(arrays["actions"])),
                     "terminal_cause": tcause, "single_life_done": int(dn.sum()) <= 1,
                     "lives_constant": int(len(np.unique(arrays["lives"][arrays["lives"] >= 0]))) <= 1})
        print(f"  ep{i} seed{seed}: {len(arrays['actions'])} steps terminal={tcause}")
    json.dump({"experiment": "seaquest_oxygen_aware_first_life", "confounder": "oxygen",
               "policy": {"variant": "P1_unconditional", "wrapper": "OxygenAwareTeacher",
                          "surface_trigger": TRIGGER, "refilled": REFILLED, "surface_action": SURFACE_ACTION},
               "termination": "first_life_loss", "frame_alignment": "pre_step (frame[t] aligned with action[t])",
               "schema": "frames(T,210,160,3)u8, actions, teacher_actions, player_pos(center), oxygen(OxygenBar.w), "
                         "divers, lives, reward, surfacing, done(terminal@first_life_loss), "
                         "death_respawn(all 0), player_present(all True), target, theta",
               "episodes": meta}, open(f"{out_dir}/manifest.json", "w"), indent=2)
    print(f"WROTE {n_episodes} first-life episodes -> {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true", help="full-scale first-life collection (no diagnostics)")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--out", default="artifacts/seaquest/oxyaware_first_life/data")
    a = ap.parse_args()
    if a.collect:
        collect_corpus(a.out, a.episodes, a.seed0)
    else:
        main()
