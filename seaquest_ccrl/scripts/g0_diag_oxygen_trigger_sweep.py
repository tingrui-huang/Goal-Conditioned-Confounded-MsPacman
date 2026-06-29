"""Sweep surface_trigger in {15,20,25} (refilled=58, same hysteresis) to pick the LOWEST trigger that:
  (1) reaches the surface before oxygen runs out  -> 0 oxygen-drowns; positive min-O2 margin;
  (2) completes a real refill with lives unchanged -> clean_refills > 0;
  (3) does not increase deaths                     -> deaths <= original-teacher baseline;
  (4) avoids unstable switching / partial ascents  -> few short surfacing blocks, no rapid re-triggers.

Greedy (deterministic) rollouts over several seeds, matching the video. Per surfacing EPISODE (a
contiguous surf=True block) we measure: did it reach the surface, did O2 refill with lives unchanged,
the min O2 seen on the way up (drown margin), and whether a death in the block was an oxygen-drown
(last valid O2 before the death-animation <= 3) or an incidental enemy kill (O2 still high)."""
import argparse
import sys
from unittest.mock import MagicMock

for _m in ("envpool", "gym"):
    sys.modules.setdefault(_m, MagicMock())
import numpy as np
import ocatari.ram.seaquest as _sq
_orig = _sq._detect_objects_ram
_sq._detect_objects_ram = lambda o, r, h: _orig(o, np.asarray(r, np.int64), h)
from seaquest_ccrl.scripts.g0_closed_loop_eval import SeaquestPort, TEACHER_CKPT, TEACHER_SRC
from seaquest_stage_s0.teacher_adapter import CleanRLSeaquestTeacher
from seaquest_ccrl.policies.oxygen_aware_teacher import OxygenAwareTeacher

SURFACE_Y = 52.0
REFILLED = 58
DROWN_O2 = 3.0          # last valid O2 before a death-animation <= this => oxygen drown (not enemy)
SHORT_EP = 6            # surfacing block shorter than this (and not reaching surface) = partial ascent
RAPID_REENTER = 8       # re-entering surfacing within this many steps of exiting = unstable switching


def rollout(teacher, wrap, seed, steps, prime, mode="greedy"):
    p = SeaquestPort(sticky=0.0, full_action_space=True, seed=seed); p.reset(seed=seed, noop_max=0)
    if wrap is not None:
        wrap.reset()
    rng = np.random.default_rng(seed)
    def teacher_a(obs):
        if mode == "greedy":
            return int(teacher.greedy_action(obs)[0])
        return int(teacher.sample_action(obs, teacher.gumbel_from_uniform(rng.uniform(size=18)))[0])
    for _ in range(prime):
        p.agent_step(teacher_a(p.teacher_obs()))
    O, LV, PY, SF = [], [], [], []
    for _ in range(steps):
        f = p.features(); oxy = f.get("oxygen"); oxy = -1.0 if oxy is None else float(oxy)
        lv = f.get("lives"); py = f.get("player_y")
        obs = p.teacher_obs()
        if wrap is None:
            a, surf = teacher_a(obs), False
        else:
            a, surf = wrap.act(obs, oxy, py, mode=mode, rng=rng)
        O.append(oxy); LV.append(np.nan if lv is None else float(lv))
        PY.append(np.nan if py is None else float(py)); SF.append(surf)
        p.agent_step(a)
    return np.array(O), np.array(LV), np.array(PY), np.array(SF, bool)


def prime_len(teacher, seed):
    p = SeaquestPort(sticky=0.0, full_action_space=True, seed=seed); p.reset(seed=seed, noop_max=0)
    for k in range(20):
        ox = p.features().get("oxygen")
        if ox is not None and ox >= REFILLED:
            return k
        p.agent_step(int(teacher.greedy_action(p.teacher_obs())[0]))
    return 20


def deaths_of(LV):
    d = np.diff(LV)
    return [int(i + 1) for i in np.where((d < 0) & np.isfinite(d))[0]]


def last_valid_o2_before(O, t):
    for k in range(t - 1, max(-1, t - 30), -1):
        if O[k] >= 0:
            return O[k]
    return -1.0


def analyze(O, LV, PY, SF):
    n = len(O)
    res = {"episodes": 0, "reached_surface": 0, "clean_refill": 0, "partial_ascent": 0,
           "rapid_reenter": 0, "deaths": 0, "deaths_surf": 0, "oxy_drowns": 0, "enemy_deaths": 0,
           "min_o2_margin": np.inf, "ep_lengths": []}
    dsteps = deaths_of(LV); res["deaths"] = len(dsteps)
    for ds in dsteps:
        surf = bool(SF[ds]) or bool(SF[max(0, ds - 1)])
        o2pre = last_valid_o2_before(O, ds)
        if surf:
            res["deaths_surf"] += 1
            if 0 <= o2pre <= DROWN_O2:
                res["oxy_drowns"] += 1
            else:
                res["enemy_deaths"] += 1
    # surfacing episodes
    t = 0; prev_exit = -10 ** 9
    while t < n:
        if SF[t]:
            s0 = t
            while t < n and SF[t]:
                t += 1
            s1 = t                                   # episode = [s0, s1)
            res["episodes"] += 1; res["ep_lengths"].append(s1 - s0)
            if s0 - prev_exit <= RAPID_REENTER:
                res["rapid_reenter"] += 1
            prev_exit = s1
            pys = PY[s0:s1]; reached = bool(np.isfinite(pys).any() and np.nanmin(pys) <= SURFACE_Y)
            # min O2 on the way up (start -> first reaching surface), among valid readings
            up_end = s1
            fin = np.where(np.isfinite(pys) & (pys <= SURFACE_Y))[0]
            if len(fin):
                up_end = s0 + int(fin[0]) + 1
            seg = O[s0:up_end]; segv = seg[seg >= 0]
            if reached:
                res["reached_surface"] += 1
                if len(segv):
                    res["min_o2_margin"] = min(res["min_o2_margin"], float(segv.min()))
            else:
                if (s1 - s0) < SHORT_EP:
                    res["partial_ascent"] += 1
            # clean refill: reached surface, O2 at/after exit >= refilled, lives unchanged across block
            exit_o2 = O[min(s1, n - 1)]
            lives_const = bool(np.isfinite(LV[s0:min(s1 + 1, n)]).all()
                               and np.nanmax(LV[s0:min(s1 + 1, n)]) == np.nanmin(LV[s0:min(s1 + 1, n)]))
            if reached and exit_o2 >= REFILLED and lives_const:
                res["clean_refill"] += 1
        else:
            t += 1
    if not np.isfinite(res["min_o2_margin"]):
        res["min_o2_margin"] = float("nan")
    return res


def agg(teacher, triggers, seeds, steps, surface_action, mode):
    primes = {sd: prime_len(teacher, sd) for sd in seeds}
    # baseline (original teacher) deaths
    base_deaths = 0
    for sd in seeds:
        O, LV, PY, SF = rollout(teacher, None, sd, steps, primes[sd], mode=mode)
        base_deaths += len(deaths_of(LV))
    print(f"  [baseline] original teacher total deaths over {len(seeds)} seeds x {steps} = {base_deaths}\n")

    out = {}
    for trig in triggers:
        tot = {"episodes": 0, "reached_surface": 0, "clean_refill": 0, "partial_ascent": 0,
               "rapid_reenter": 0, "deaths": 0, "deaths_surf": 0, "oxy_drowns": 0, "enemy_deaths": 0}
        margins, eplens = [], []
        for sd in seeds:
            wrap = OxygenAwareTeacher(teacher, surface_trigger=trig, refilled=REFILLED,
                                      surface_action=surface_action)
            r = analyze(*rollout(teacher, wrap, sd, steps, primes[sd], mode=mode))
            for k in tot:
                tot[k] += r[k]
            if np.isfinite(r["min_o2_margin"]):
                margins.append(r["min_o2_margin"])
            eplens += r["ep_lengths"]
        tot["min_o2_margin"] = float(min(margins)) if margins else float("nan")
        tot["median_ep_len"] = float(np.median(eplens)) if eplens else float("nan")
        tot["base_deaths"] = base_deaths
        out[trig] = tot
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triggers", default="15,20,25")
    ap.add_argument("--seeds", default="3000,3001,3002,3003,3004,3005")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--surface-action", type=int, default=10)
    ap.add_argument("--mode", default="greedy", choices=["greedy", "stochastic"])
    args = ap.parse_args()
    teacher = CleanRLSeaquestTeacher(TEACHER_CKPT, TEACHER_SRC, mod_name="cleanrl_src_A")
    triggers = [int(x) for x in args.triggers.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    print(f"=== surface_trigger sweep ({args.mode}, refilled={REFILLED}, "
          f"surface_action={'UPFIRE' if args.surface_action == 10 else args.surface_action}, "
          f"{len(seeds)} seeds x {args.steps} steps) ===\n")
    out = agg(teacher, triggers, seeds, args.steps, args.surface_action, args.mode)
    print(f"{'trig':>4} {'eps':>4} {'reach':>5} {'clean':>5} {'drown':>5} {'enemy':>5} {'deaths':>6} "
          f"{'d/base':>7} {'minO2':>6} {'partial':>7} {'rapid':>5} {'medLen':>6}")
    for trig, t in out.items():
        print(f"{trig:>4} {t['episodes']:>4} {t['reached_surface']:>5} {t['clean_refill']:>5} "
              f"{t['oxy_drowns']:>5} {t['enemy_deaths']:>5} {t['deaths']:>6} "
              f"{t['deaths']}/{t['base_deaths']:<5} {t['min_o2_margin']:>6.1f} "
              f"{t['partial_ascent']:>7} {t['rapid_reenter']:>5} {t['median_ep_len']:>6.0f}")
    print("\nCriteria: drown=0, clean>0, deaths<=baseline, partial/rapid low. Pick the LOWEST trigger meeting all.")


if __name__ == "__main__":
    main()
