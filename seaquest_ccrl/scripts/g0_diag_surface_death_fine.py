"""ALE-frame-resolution forensics to resolve the UNRESOLVED surface-ascent deaths, plus a direct
controlled test of the Seaquest no-diver surface rule.

(A) FINE CONTACT: for each death we re-run the exact rollout, but around the death window we wrap
    port.env.step so OCAtari object state is logged on EVERY ale frame (not just post-frameskip).
    We find the first ale frame where the Player object vanishes (explosion) and report the nearest
    shark/sub/enemy-missile contact in the ale frames immediately before it. This catches a torpedo
    or collision that despawns within the 4-frame skip.

(B) SURFACE-RULE TEST: drive the sub UP to the surface from depth in open water (greedy teacher
    suppressed; pure UP) with 0 divers vs >=1 diver, with NO enemy within 40px, and check whether a
    life is lost exactly when player_y reaches the surface -> isolates the no-diver penalty.
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
from seaquest_ccrl.scripts.g0_diag_surface_death_forensics import objstate, nearest, deaths, rollout
from seaquest_stage_s0.teacher_adapter import CleanRLSeaquestTeacher
from seaquest_ccrl.policies.oxygen_aware_teacher import OxygenAwareTeacher

TRIGGER, REFILLED = 20, 58
STEPS = 1300
CASES = [(4000, 372), (4002, 868), (4003, 860), (4004, 852), (4005, 852)]
UP = 2


def fine_window(teacher, seed, dstep, pre=22, post=2):
    """Re-run the exact rollout; log object state on EVERY ale frame in [dstep-pre, dstep+post]."""
    pr = prime_len(teacher, seed)
    wrap = OxygenAwareTeacher(teacher, surface_trigger=TRIGGER, refilled=REFILLED, surface_action=10)
    p = SeaquestPort(sticky=0.0, full_action_space=True, seed=seed); p.reset(seed=seed, noop_max=0)
    wrap.reset(); rng = np.random.default_rng(seed)
    def stoch(obs):
        return int(teacher.sample_action(obs, teacher.gumbel_from_uniform(rng.uniform(size=18)))[0])
    for _ in range(pr):
        p.agent_step(stoch(p.teacher_obs()))
    frames_log = []
    lo, hi = dstep - pre, dstep + post
    for s in range(hi + 1):
        st = objstate(p); obs = p.teacher_obs()
        a, surf = wrap.act(obs, -1.0 if st["oxy"] is None else st["oxy"],
                           None if st["P"] is None else st["P"][1], mode="stochastic", rng=rng)
        if lo <= s <= hi:
            orig = p.env.step; sub = []
            def logged(act, _o=orig, _s=sub):
                out = _o(act); _s.append(objstate(p)); return out
            p.env.step = logged
            p.agent_step(a)
            p.env.step = orig
            for j, fr in enumerate(sub):
                frames_log.append({"agent_t": s, "ale": j, "surf": surf, "exec": a, **fr})
        else:
            p.agent_step(a)
        if s >= hi:
            break
    return frames_log


def resolve(fl):
    """From the ale-frame log, find the explosion frame and the nearest contact just before it."""
    # explosion = first ale frame where Player vanishes (after having been present)
    seen = False; expl = None
    for i, r in enumerate(fl):
        if r["P"] is not None:
            seen = True
        elif seen and r["P"] is None:
            expl = i; break
    idx = expl if expl is not None else len(fl)
    pre = [r for r in fl[max(0, idx - 10):idx] if r["P"] is not None]
    best_e = (np.inf, False); best_m = (np.inf, False); last_py = None; divers = None; oxy = None
    tail = []                                    # last few ale frames: (player_y, nearest shark dist, overlap)
    for r in pre:
        e = nearest(r["P"], r["sharks"] + r["subs"]); m = nearest(r["P"], r["emiss"])
        if e[0] < best_e[0]:
            best_e = (e[0], e[1])
        if m[0] < best_m[0]:
            best_m = (m[0], m[1])
        last_py = r["P"][1]; divers = r["divers"]; oxy = r["oxy"]
        tail.append((round(r["P"][1], 0), None if not np.isfinite(e[0]) else round(e[0], 1), e[1]))
    return {"explosion_ale_idx": expl, "n_ale": len(fl), "last_player_y": last_py,
            "divers_at_explosion": divers, "oxy_at_explosion": oxy,
            "min_enemy_dist": None if not np.isfinite(best_e[0]) else round(best_e[0], 1),
            "enemy_contact": bool(best_e[1]),
            "min_missile_dist": None if not np.isfinite(best_m[0]) else round(best_m[0], 1),
            "missile_contact": bool(best_m[1]), "tail": tail}


def find_death_step(teacher, seed, blk):
    pr = prime_len(teacher, seed)
    wrap = OxygenAwareTeacher(teacher, surface_trigger=TRIGGER, refilled=REFILLED, surface_action=10)
    log, _ = rollout(teacher, wrap, seed, STEPS, pr, keep_frames=False)
    ds = [d for d in deaths(log) if d >= blk - 2]
    return (ds[0] if ds else None), log


def surface_rule_test(teacher):
    """Dive to depth, then force pure-UP ascent in open water; record life loss vs divers carried.
    We collect natural trials from rollouts: ascents that reached the surface, bucketed by divers."""
    print("\n=== (B) SURFACE-RULE TEST: ascents reaching surface, by divers carried ===")
    print("    (does reaching the surface cost a life when divers==0, with no enemy contact?)")
    # bucket every TRUE surface-touch (player_y<=46) by divers carried; isolate "no shark within 40px"
    buckets = {("0", "any"): [0, 0], ("ge1", "any"): [0, 0],
               ("0", "noshark"): [0, 0], ("ge1", "noshark"): [0, 0]}   # [n, life_lost]
    for seed in range(4000, 4012):
        pr = prime_len(teacher, seed)
        wrap = OxygenAwareTeacher(teacher, surface_trigger=TRIGGER, refilled=REFILLED, surface_action=10)
        log, _ = rollout(teacher, wrap, seed, STEPS, pr, keep_frames=False)
        n = len(log); t = 0
        while t < n:
            if not log[t]["surfacing"]:
                t += 1; continue
            s0 = t
            while t < n and log[t]["surfacing"]:
                t += 1
            blk = log[s0:t]
            touch = next((i for i, r in enumerate(blk)
                          if r["player_y"] is not None and r["player_y"] <= 46), None)
            if touch is None:
                continue
            r0 = blk[touch]; dk = "0" if r0["divers_carried"] == 0 else "ge1"
            # life lost from the surface-touch up to +10 steps (the refill hold)?
            after = blk[touch:touch + 12]
            lvs = [r["lives"] for r in after if r["lives"] is not None]
            life_lost = bool(lvs and min(lvs) < lvs[0])
            ned = r0["nearest_enemy_dist"]
            noshark = (ned is None) or (ned > 40)
            buckets[(dk, "any")][0] += 1; buckets[(dk, "any")][1] += life_lost
            if noshark:
                buckets[(dk, "noshark")][0] += 1; buckets[(dk, "noshark")][1] += life_lost
    for (dk, filt), (n, ll) in buckets.items():
        lab = ("divers==0" if dk == "0" else "divers>=1") + (" (no shark<40px)" if filt == "noshark" else "")
        print(f"  {lab:28s}: surface-touches n={n:2d}  life_lost_after_touch={ll}/{n}" if n else f"  {lab:28s}: n=0")


def main():
    teacher = CleanRLSeaquestTeacher(TEACHER_CKPT, TEACHER_SRC, mod_name="cleanrl_src_A")
    print("=== (A) ALE-FRAME CONTACT around each surface-ascent death ===\n")
    for seed, blk in CASES:
        dstep, _ = find_death_step(teacher, seed, blk)
        if dstep is None:
            print(f"seed {seed}: no death"); continue
        fl = fine_window(teacher, seed, dstep)
        r = resolve(fl)
        verdict = ("ENEMY/PROJECTILE" if (r["enemy_contact"] or r["missile_contact"])
                   else ("SURFACE_RULE?" if (r["last_player_y"] is not None and r["last_player_y"] <= 50
                                             and (r["divers_at_explosion"] or 0) == 0)
                         else "NO-CONTACT (cause not an object collision we can see)"))
        print(f"seed {seed} death@t={dstep}: explosion_ale_idx={r['explosion_ale_idx']} "
              f"last_player_y={r['last_player_y']} divers={r['divers_at_explosion']} O2={r['oxy_at_explosion']}")
        print(f"   min_shark/sub={r['min_enemy_dist']}px contact={r['enemy_contact']}  "
              f"min_missile={r['min_missile_dist']}px contact={r['missile_contact']}  => {verdict}")
        print(f"   ale tail (py, sharkdist, overlap) before explosion: {r['tail']}\n")
    surface_rule_test(teacher)


if __name__ == "__main__":
    main()
