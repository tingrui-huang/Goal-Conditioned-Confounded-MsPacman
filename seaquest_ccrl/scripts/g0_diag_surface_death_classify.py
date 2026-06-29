"""FINAL per-case classification of the 5 surface-ascent deaths, with calibration against normal
unwrapped-teacher deaths. Measures everything at the TRUE death onset (last valid player frame before
the explosion animation), using a SPRITE-WIDTH-AWARE contact test (OCAtari center-bbox under-reports
collisions, so we treat center_dist <= halfwidth_player+halfwidth_enemy+PAD as contact) and the
closest shark approach over the whole ascent block.

Also runs a CORRECTED no-diver surface-rule test: surface-touches (player_y<=46) bucketed by divers,
with a 30-step life-loss window (the penalty registers ~23 steps after the touch: ~9 freeze + ~14
explosion animation -- the earlier 12-step window missed it).

Categories: 1 ENEMY_SHARK  2 SURFACE_RULE_NODIVER  3 OXYGEN  4 RESET/ARTIFACT  5 UNRESOLVED.
"""
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
from seaquest_ccrl.scripts.g0_diag_surface_death_forensics import objstate, rollout, deaths
from seaquest_stage_s0.teacher_adapter import CleanRLSeaquestTeacher
from seaquest_ccrl.policies.oxygen_aware_teacher import OxygenAwareTeacher

TRIGGER, REFILLED, STEPS = 20, 58, 1300
CASES = [(4000, 372), (4002, 868), (4003, 860), (4004, 852), (4005, 852)]
SURFACE_Y = 48.0
CONTACT_PAD = 3.0


def sprite_contact(P, boxes):
    """center distance and sprite-aware contact (center_dist <= sum of half-extents + PAD)."""
    if P is None or not boxes:
        return (float("inf"), False)
    pc = (P[0] + P[2] / 2, P[1] + P[3] / 2); best = (float("inf"), False)
    for b in boxes:
        bc = (b[0] + b[2] / 2, b[1] + b[3] / 2)
        d = float(np.hypot(pc[0] - bc[0], pc[1] - bc[1]))
        thr = (P[2] + b[2]) / 2 + (P[3] + b[3]) / 2  # generous: half-w + half-h sum
        contact = d <= (P[2] / 2 + b[2] / 2 + CONTACT_PAD) or d <= (P[3] / 2 + b[3] / 2 + CONTACT_PAD)
        if d < best[0]:
            best = (d, contact)
    return best


def death_onset(log, dstep):
    """Last valid player frame before the explosion None-run preceding dstep."""
    k = dstep - 1
    while k >= 0 and log[k]["player_y"] is None:
        k -= 1
    anim = (dstep - 1) - k
    return k, anim


def analyze_block(log, dstep, blockstart):
    """Gather evidence over the ascent block [blockstart..onset] and at the onset frame."""
    onset, anim = death_onset(log, dstep)
    lo = blockstart
    # closest shark/sub approach + ever-contact over the ascent up to the onset
    min_d = float("inf"); ever_contact = False
    for r in log[lo:onset + 1]:
        d, c = r["_shark_d"], r["_shark_contact"]
        if d is not None and d < min_d:
            min_d = d
        ever_contact = ever_contact or c
    o = log[onset]
    # freeze length: constant player_y immediately before onset
    fl = 0; j = onset
    while j > 0 and log[j]["player_y"] is not None and log[j - 1]["player_y"] == log[j]["player_y"]:
        fl += 1; j -= 1
    # oxygen refilling just before onset? (surface refill grants increasing O2)
    oxys = [log[k]["oxygen"] for k in range(max(0, onset - 6), onset + 1) if log[k]["oxygen"] is not None]
    refilling = bool(len(oxys) >= 2 and oxys[-1] > oxys[0])
    min_oxy = min(oxys) if oxys else None
    return {"onset_t": log[onset]["t"], "death_t": log[dstep]["t"], "anim_len": anim, "freeze_len": fl,
            "player_y": o["player_y"], "at_surface": bool(o["player_y"] is not None and o["player_y"] <= SURFACE_Y),
            "divers": o["divers_carried"], "oxy_at_onset": o["oxygen"], "min_oxy": min_oxy,
            "oxy_refilling_before": refilling, "min_shark_dist": None if not np.isfinite(min_d) else round(min_d, 1),
            "ever_sprite_contact": ever_contact,
            "reward_block": sum(log[k]["reward"] for k in range(lo, dstep + 1))}


def classify(ev):
    if ev["min_oxy"] is not None and ev["min_oxy"] <= 0:
        return "3 OXYGEN"
    if ev["at_surface"] and ev["divers"] == 0 and (ev["min_shark_dist"] is None or ev["min_shark_dist"] > 28) \
            and not ev["oxy_refilling_before"]:
        return "2 SURFACE_RULE_NODIVER"
    if ev["ever_sprite_contact"] or (ev["min_shark_dist"] is not None and ev["min_shark_dist"] <= 20):
        return "1 ENEMY_SHARK"
    return "5 UNRESOLVED"


def enrich(log):
    """Attach per-step sprite-aware shark distance/contact (needs object boxes -> recompute via objstate
    is impossible post-hoc; instead this is filled during rollout)."""
    return log


def run_with_objs(teacher, wrap, seed):
    """Rollout that also records sprite-aware shark distance/contact per step."""
    pr = prime_len(teacher, seed)
    p = SeaquestPort(sticky=0.0, full_action_space=True, seed=seed); p.reset(seed=seed, noop_max=0)
    if wrap is not None:
        wrap.reset()
    rng = np.random.default_rng(seed)
    def stoch(obs):
        return int(teacher.sample_action(obs, teacher.gumbel_from_uniform(rng.uniform(size=18)))[0])
    for _ in range(pr):
        p.agent_step(stoch(p.teacher_obs()))
    log = []
    for s in range(STEPS):
        st = objstate(p); obs = p.teacher_obs()
        if wrap is None:
            a, surf = stoch(obs), False
        else:
            a, surf = wrap.act(obs, -1.0 if st["oxy"] is None else st["oxy"],
                               None if st["P"] is None else st["P"][1], mode="stochastic", rng=rng)
        d, c = sprite_contact(st["P"], st["sharks"] + st["subs"])
        rec = p.agent_step(a)
        log.append({"t": s, "lives": st["lives"], "reward": rec["reward"], "oxygen": st["oxy"],
                    "player_y": None if st["P"] is None else st["P"][1], "divers_carried": st["divers"],
                    "surfacing": surf, "_shark_d": None if not np.isfinite(d) else d, "_shark_contact": bool(c)})
    return log


def block_start(log, dstep):
    """Start of the surfacing block that contains/precedes the death."""
    k = dstep
    while k > 0 and log[k - 1]["surfacing"]:
        k -= 1
    return k


def main():
    teacher = CleanRLSeaquestTeacher(TEACHER_CKPT, TEACHER_SRC, mod_name="cleanrl_src_A")
    print("=== FINAL CLASSIFICATION: 5 surface-ascent deaths (trigger=20) ===\n")
    results = []
    for seed, blk in CASES:
        wrap = OxygenAwareTeacher(teacher, surface_trigger=TRIGGER, refilled=REFILLED, surface_action=10)
        log = run_with_objs(teacher, wrap, seed)
        ds = [d for d in deaths(log) if d >= blk - 2]
        dstep = ds[0] if ds else (deaths(log)[0] if deaths(log) else None)
        if dstep is None:
            print(f"seed {seed}: no death"); continue
        bs = block_start(log, dstep)
        ev = analyze_block(log, dstep, bs); cat = classify(ev); ev["seed"] = seed; ev["category"] = cat
        results.append(ev)
        print(f"seed {seed}: death@t={ev['death_t']} (onset t={ev['onset_t']}, freeze={ev['freeze_len']}, anim={ev['anim_len']})")
        print(f"   player_y={ev['player_y']} at_surface={ev['at_surface']} divers={ev['divers']} "
              f"oxy@onset={ev['oxy_at_onset']} min_oxy={ev['min_oxy']} refilling_before={ev['oxy_refilling_before']}")
        print(f"   closest_shark_over_ascent={ev['min_shark_dist']}px sprite_contact={ev['ever_sprite_contact']} "
              f"reward_block={ev['reward_block']}")
        print(f"   ==> {cat}\n")

    # ---- calibration: normal unwrapped-teacher deaths ----
    print("=== CALIBRATION: unwrapped-teacher normal deaths (same measurements) ===\n")
    nd = []
    for seed in (4000, 4001, 4002, 4003):
        log = run_with_objs(teacher, None, seed)
        for dstep in deaths(log):
            bs = max(0, dstep - 30)
            ev = analyze_block(log, dstep, bs); nd.append(ev)
    for ev in nd[:10]:
        print(f"  death@t={ev['death_t']}: player_y={ev['player_y']} at_surface={ev['at_surface']} "
              f"divers={ev['divers']} closest_shark={ev['min_shark_dist']}px sprite_contact={ev['ever_sprite_contact']} "
              f"freeze={ev['freeze_len']} anim={ev['anim_len']} min_oxy={ev['min_oxy']}")

    def summ(rows):
        n = len(rows) or 1
        return (f"n={len(rows)} | at_surface={sum(r['at_surface'] for r in rows)} | divers==0={sum(r['divers']==0 for r in rows)} "
                f"| sprite_contact={sum(r['ever_sprite_contact'] for r in rows)} "
                f"| mean_closest_shark={np.mean([r['min_shark_dist'] for r in rows if r['min_shark_dist'] is not None]):.1f}px "
                f"| mean_anim={np.mean([r['anim_len'] for r in rows]):.1f} | mean_freeze={np.mean([r['freeze_len'] for r in rows]):.1f}")
    print(f"\n  5 surface cases : {summ(results)}")
    print(f"  normal teacher  : {summ(nd)}")
    from collections import Counter
    print("\n  >>> categories: " + str(dict(Counter(r["category"] for r in results))))

    # ---- corrected no-diver surface-rule test (30-step window) ----
    print("\n=== CORRECTED no-diver surface-rule test (30-step life-loss window) ===")
    buck = {"0": [0, 0], "ge1": [0, 0]}
    for seed in range(4000, 4012):
        wrap = OxygenAwareTeacher(teacher, surface_trigger=TRIGGER, refilled=REFILLED, surface_action=10)
        log = run_with_objs(teacher, wrap, seed)
        n = len(log); t = 0
        while t < n:
            if not log[t]["surfacing"]:
                t += 1; continue
            s0 = t
            while t < n and log[t]["surfacing"]:
                t += 1
            blk = log[s0:t]
            touch = next((i for i, r in enumerate(blk) if r["player_y"] is not None and r["player_y"] <= 46), None)
            if touch is None:
                continue
            r0 = blk[touch]; dk = "0" if r0["divers_carried"] == 0 else "ge1"
            # only count touches with NO shark within 25px (isolate from shark deaths)
            if r0["_shark_d"] is not None and r0["_shark_d"] <= 25:
                continue
            after = blk[touch:touch + 30]
            lvs = [r["lives"] for r in after if r["lives"] is not None]
            buck[dk][0] += 1; buck[dk][1] += int(bool(lvs and min(lvs) < lvs[0]))
    for dk, (n, ll) in buck.items():
        lab = "divers==0" if dk == "0" else "divers>=1"
        print(f"  {lab} surface-touch (no shark<25px): n={n}  life_lost_within_30={ll}/{n}"
              + ("  <== penalty" if dk == "0" and n and ll > n / 2 else ""))


if __name__ == "__main__":
    main()
