"""Categorize the NON-clean surfacing episodes at surface_trigger=20 (the ~24% that the sweep
flagged). Re-runs the identical trigger=20 stochastic rollouts (seeds 4000..4007, 1200 steps) and
classifies every surfacing block by outcome, so we can name the failure mode of each non-clean one.

Failure taxonomy per surfacing block [s0,s1):
  CLEAN                 reached surface, lives constant, exited via O2>=refilled
  ENEMY_DEATH_ASCENT    lives dropped while STILL BELOW the surface (the forced ascent got it killed)
  ENEMY_DEATH_SURFACE   reached the surface, then lives dropped (enemy hit at/near surface while refilling)
  OXY_DROWN             lives dropped with last valid O2<=3 before reaching surface (true drown)
  RESPAWN_REFILL        block began at the surface right after a death (O2~1) -> a post-death refill,
                        not a genuine deep ascent; counted separately so it doesn't masquerade as a failure
"""
import sys
import numpy as np
from seaquest_ccrl.scripts.g0_diag_oxygen_trigger_sweep import (
    rollout, prime_len, deaths_of, last_valid_o2_before, SURFACE_Y, REFILLED, DROWN_O2)
from seaquest_ccrl.scripts.g0_closed_loop_eval import TEACHER_CKPT, TEACHER_SRC
from seaquest_stage_s0.teacher_adapter import CleanRLSeaquestTeacher
from seaquest_ccrl.policies.oxygen_aware_teacher import OxygenAwareTeacher

TRIGGER = 20
SEEDS = [4000, 4001, 4002, 4003, 4004, 4005, 4006, 4007]
STEPS = 1200


def classify(O, LV, PY, SF, seed):
    n = len(O); eps = []
    t = 0
    while t < n:
        if not SF[t]:
            t += 1; continue
        s0 = t
        while t < n and SF[t]:
            t += 1
        s1 = t                                              # block [s0, s1)
        pys = PY[s0:s1]
        fin = np.where(np.isfinite(pys) & (pys <= SURFACE_Y))[0]
        reach_rel = int(fin[0]) if len(fin) else None
        reach_t = (s0 + reach_rel) if reach_rel is not None else None
        # lives drops inside the block
        seg_lv = LV[s0:min(s1 + 1, n)]
        dd = np.diff(seg_lv)
        death_rel = [int(i + 1) for i in np.where((dd < 0) & np.isfinite(dd))[0]]
        death_t = [s0 + r for r in death_rel]
        start_o2 = O[s0]; start_py = PY[s0]
        exit_o2 = O[min(s1, n - 1)]
        respawn_initiated = bool(start_o2 >= 0 and start_o2 <= DROWN_O2
                                 and np.isfinite(start_py) and start_py <= SURFACE_Y + 6)
        # classify
        if death_t:
            d0 = death_t[0]; o2pre = last_valid_o2_before(O, d0)
            if 0 <= o2pre <= DROWN_O2 and (reach_t is None or d0 < reach_t):
                mode = "OXY_DROWN"
            elif reach_t is not None and d0 >= reach_t:
                mode = "ENEMY_DEATH_SURFACE"
            else:
                mode = "ENEMY_DEATH_ASCENT"
        elif respawn_initiated:
            mode = "RESPAWN_REFILL" if (reach_t is not None and exit_o2 >= REFILLED) else "RESPAWN_PARTIAL"
        elif reach_t is not None and exit_o2 >= REFILLED:
            mode = "CLEAN"
        else:
            mode = "NO_REFILL"
        eps.append({"seed": seed, "s0": s0, "len": s1 - s0, "start_py": _f(start_py),
                    "start_o2": _f(start_o2), "reached": reach_t is not None, "min_py": _f(np.nanmin(pys)),
                    "exit_o2": _f(exit_o2), "deaths_in_block": len(death_t),
                    "o2_before_first_death": (round(last_valid_o2_before(O, death_t[0]), 1) if death_t else None),
                    "respawn_initiated": respawn_initiated, "mode": mode})
    return eps


def _f(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 1)


def main():
    teacher = CleanRLSeaquestTeacher(TEACHER_CKPT, TEACHER_SRC, mod_name="cleanrl_src_A")
    all_eps = []
    for sd in SEEDS:
        pr = prime_len(teacher, sd)
        wrap = OxygenAwareTeacher(teacher, surface_trigger=TRIGGER, refilled=REFILLED, surface_action=10)
        O, LV, PY, SF = rollout(teacher, wrap, sd, STEPS, pr, mode="stochastic")
        all_eps += classify(O, LV, PY, SF, sd)

    from collections import Counter
    cnt = Counter(e["mode"] for e in all_eps)
    n = len(all_eps)
    print(f"=== surface_trigger={TRIGGER}: {n} surfacing episodes over {len(SEEDS)} seeds x {STEPS} steps ===\n")
    print("outcome breakdown:")
    for mode, c in cnt.most_common():
        print(f"  {mode:22s} {c:3d}  ({100*c/n:4.1f}%)")
    genuine = [e for e in all_eps if not e["respawn_initiated"]]
    gc = Counter(e["mode"] for e in genuine)
    print(f"\nof the {len(genuine)} GENUINE deep ascents (excluding {n-len(genuine)} post-death respawn refills):")
    for mode, c in gc.most_common():
        print(f"  {mode:22s} {c:3d}  ({100*c/len(genuine):4.1f}%)")

    print("\n--- NON-CLEAN episodes (the failures) ---")
    for e in all_eps:
        if e["mode"] != "CLEAN":
            print(f"  seed{e['seed']} t={e['s0']:4d} len={e['len']:3d} start(py={e['start_py']},O2={e['start_o2']}) "
                  f"min_py={e['min_py']} reached={e['reached']} exit_O2={e['exit_o2']} "
                  f"deaths={e['deaths_in_block']} O2_pre_death={e['o2_before_first_death']} "
                  f"respawn_init={e['respawn_initiated']} -> {e['mode']}")


if __name__ == "__main__":
    main()
