"""Evaluate three low-oxygen policies (diver-aware) on genuinely stochastic episodes, BEFORE any
dataset pipeline change. Each policy wraps the frozen HF teacher and overrides ONLY the low-oxygen
response; the difference is WHEN surfacing is triggered relative to carried divers.

  P1 unconditional : surface whenever oxygen < TRIGGER.
  P2 divers>=1 only : surface only when oxygen < TRIGGER and carried_divers >= 1 (never surface empty).
  P3 staged emergency:
        oxygen < EMERGENCY                      -> force surface (even with 0 divers);
        oxygen < TRIGGER and carried_divers >= 1 -> surface;
        oxygen < TRIGGER and carried_divers == 0 -> keep teacher behavior (seek a diver).
  (all use hysteresis: once surfacing, stay until oxygen >= REFILLED.)

Deaths are classified at the TRUE death onset (last valid player frame before the explosion
animation) with the sprite-width-aware contact test validated in the forensics:
  oxygen_drown      oxy at onset <= 2;
  enemy_ascent      death while surfacing with a shark in sprite-contact range over the ascent;
  nodiver_surface   death while surfacing, at the waterline, carrying 0 divers (Atari penalty);
  normal_teacher    death NOT during surfacing (the teacher's own mortality).

Reports per policy: trigger frequency, %triggers with 0 divers, oxygen-depletion deaths, no-diver
surface deaths, enemy ascent deaths, clean refills, time in low-oxygen mode, override footprint, and
deviation from the pure teacher outside the override.
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
from seaquest_ccrl.scripts.g0_diag_surface_death_forensics import objstate
from seaquest_ccrl.scripts.g0_diag_surface_death_classify import sprite_contact, death_onset
from seaquest_stage_s0.teacher_adapter import CleanRLSeaquestTeacher

TRIGGER, REFILLED, EMERGENCY, SURFACE_ACTION = 20, 58, 8, 10
SEEDS = list(range(5000, 5012))         # 12 genuinely-distinct stochastic episodes
STEPS = 1600
SURFACE_Y = 48.0


class LowOxyPolicy:
    def __init__(self, teacher, mode, trigger=TRIGGER, refilled=REFILLED, emergency=EMERGENCY,
                 surface_action=SURFACE_ACTION):
        self.t, self.mode = teacher, mode
        self.trigger, self.refilled, self.emergency, self.sa = trigger, refilled, emergency, surface_action
        self._surf = False

    def reset(self):
        self._surf = False

    def _enter(self, oxy, divers):
        if self.mode == 1:
            return oxy < self.trigger
        if self.mode == 2:
            return oxy < self.trigger and divers >= 1
        if self.mode == 3:
            if oxy < self.emergency:
                return True
            return oxy < self.trigger and divers >= 1
        raise ValueError(self.mode)

    def act(self, obs, oxy, divers, rng):
        if oxy is not None and oxy >= 0:
            if self._surf:
                if oxy >= self.refilled:
                    self._surf = False
            elif self._enter(oxy, divers):
                self._surf = True
        if self._surf:
            return self.sa, True
        a = int(self.t.sample_action(obs, self.t.gumbel_from_uniform(rng.uniform(size=18)))[0])
        return a, False


def rollout(teacher, policy, seed):
    pr = prime_len(teacher, seed)
    p = SeaquestPort(sticky=0.0, full_action_space=True, seed=seed); p.reset(seed=seed, noop_max=0)
    if policy is not None:
        policy.reset()
    rng = np.random.default_rng(seed)
    for _ in range(pr):
        p.agent_step(int(teacher.sample_action(p.teacher_obs(),
                     teacher.gumbel_from_uniform(rng.uniform(size=18)))[0]))
    log = []
    for s in range(STEPS):
        st = objstate(p); obs = p.teacher_obs()
        oxy = None if st["oxy"] is None else float(st["oxy"]); div = st["divers"]
        if policy is None:
            a = int(teacher.sample_action(obs, teacher.gumbel_from_uniform(rng.uniform(size=18)))[0]); surf = False
        else:
            a, surf = policy.act(obs, -1.0 if oxy is None else oxy, div, rng)
        d, c = sprite_contact(st["P"], st["sharks"] + st["subs"])
        rec = p.agent_step(a)
        log.append({"t": s, "lives": st["lives"], "oxygen": oxy, "divers": div,
                    "player_y": None if st["P"] is None else st["P"][1], "surfacing": surf,
                    "executed": a, "_sd": None if not np.isfinite(d) else d, "_sc": bool(c),
                    "reward": rec["reward"]})
        if rec["terminated"]:
            break
    return log


def lives_drops(log):
    return [i for i in range(1, len(log))
            if log[i - 1]["lives"] is not None and log[i]["lives"] is not None
            and log[i]["lives"] < log[i - 1]["lives"]]


def surf_block(log, dstep):
    k = dstep
    while k > 0 and log[k - 1]["surfacing"]:
        k -= 1
    return k


def classify_death(log, dstep):
    onset, _ = death_onset(log, dstep)
    o = log[onset]; oxy = o["oxygen"]
    if oxy is not None and oxy <= 2:
        return "oxygen_drown"
    if not o["surfacing"]:
        return "normal_teacher"
    bs = surf_block(log, dstep)
    ever_c = any(log[k]["_sc"] for k in range(bs, onset + 1))
    min_sd = min([log[k]["_sd"] for k in range(bs, onset + 1) if log[k]["_sd"] is not None], default=np.inf)
    at_surf = o["player_y"] is not None and o["player_y"] <= SURFACE_Y
    if ever_c or min_sd <= 16:
        return "enemy_ascent"
    if at_surf and o["divers"] == 0:
        return "nodiver_surface"
    return "ascent_unresolved"


def measure(log):
    n = len(log)
    # triggers: oxygen crosses from >=20 to <20
    trig = 0; trig_nodiv = 0
    for i in range(1, n):
        a, b = log[i - 1]["oxygen"], log[i]["oxygen"]
        if a is not None and b is not None and a >= TRIGGER and b < TRIGGER:
            trig += 1
            if log[i]["divers"] == 0:
                trig_nodiv += 1
    # deaths
    cats = {"oxygen_drown": 0, "enemy_ascent": 0, "nodiver_surface": 0, "normal_teacher": 0,
            "ascent_unresolved": 0}
    for d in lives_drops(log):
        cats[classify_death(log, d)] += 1
    # clean refills: surfacing block reaches waterline, refills to >=REFILLED, lives constant
    clean = 0; t = 0; surf_steps = 0
    while t < n:
        if not log[t]["surfacing"]:
            t += 1; continue
        s0 = t
        while t < n and log[t]["surfacing"]:
            t += 1
        blk = log[s0:t]; surf_steps += len(blk)
        reached = any(r["player_y"] is not None and r["player_y"] <= 46 for r in blk)
        exit_oxy = log[min(t, n - 1)]["oxygen"]
        lvs = [r["lives"] for r in blk if r["lives"] is not None]
        lives_const = bool(lvs and max(lvs) == min(lvs))
        if reached and exit_oxy is not None and exit_oxy >= REFILLED and lives_const:
            clean += 1
    low_oxy_steps = sum(1 for r in log if r["oxygen"] is not None and r["oxygen"] < TRIGGER)
    # oxygen->action effect: P(UP | oxy<20) vs P(UP | oxy>=20)
    UP = {2, 6, 7, 10, 14, 15}
    up_lo = up_hi = nlo = nhi = 0
    for r in log:
        if r["oxygen"] is None or r["oxygen"] < 0:
            continue
        isup = r["executed"] in UP
        if r["oxygen"] < TRIGGER:
            nlo += 1; up_lo += isup
        else:
            nhi += 1; up_hi += isup
    return {"n": n, "triggers": trig, "trig_nodiv": trig_nodiv, "cats": cats, "clean": clean,
            "surf_steps": surf_steps, "low_oxy_steps": low_oxy_steps,
            "up_lo": up_lo, "nlo": nlo, "up_hi": up_hi, "nhi": nhi}


def main():
    teacher = CleanRLSeaquestTeacher(TEACHER_CKPT, TEACHER_SRC, mod_name="cleanrl_src_A")
    POLICIES = {"P1 unconditional": 1, "P2 divers>=1 only": 2, "P3 staged emergency": 3}

    # pure-teacher baseline for deviation reference (deaths + steps)
    base_steps = 0; base_deaths = 0
    for sd in SEEDS:
        lg = rollout(teacher, None, sd); base_steps += len(lg); base_deaths += len(lives_drops(lg))
    print(f"[baseline pure teacher] {len(SEEDS)} stochastic eps, total_steps={base_steps}, deaths={base_deaths}\n")

    agg = {}
    for name, mode in POLICIES.items():
        tot = {"n": 0, "triggers": 0, "trig_nodiv": 0, "clean": 0, "surf_steps": 0, "low_oxy_steps": 0,
               "up_lo": 0, "nlo": 0, "up_hi": 0, "nhi": 0,
               "cats": {k: 0 for k in ("oxygen_drown", "enemy_ascent", "nodiver_surface", "normal_teacher", "ascent_unresolved")}}
        for sd in SEEDS:
            policy = LowOxyPolicy(teacher, mode)
            m = measure(rollout(teacher, policy, sd))
            for k in ("n", "triggers", "trig_nodiv", "clean", "surf_steps", "low_oxy_steps",
                      "up_lo", "nlo", "up_hi", "nhi"):
                tot[k] += m[k]
            for k in tot["cats"]:
                tot["cats"][k] += m["cats"][k]
        agg[name] = tot

    print(f"{'policy':22s} {'P(UP|lo)':>9} {'P(UP|hi)':>9} {'grad':>6} {'%0div':>6} {'drown':>6} {'nodiv':>6} "
          f"{'enemy':>6} {'clean':>6} {'lowO2%':>7} {'ovrfp%':>7} {'totD':>5}")
    for name, t in agg.items():
        n = t["n"]
        pct0 = 100 * t["trig_nodiv"] / t["triggers"] if t["triggers"] else 0
        c = t["cats"]; totd = sum(c.values())
        plo = t["up_lo"] / t["nlo"] if t["nlo"] else 0; phi = t["up_hi"] / t["nhi"] if t["nhi"] else 0
        print(f"{name:22s} {plo:9.3f} {phi:9.3f} {plo-phi:+6.2f} {pct0:5.0f}% {c['oxygen_drown']:6d} "
              f"{c['nodiver_surface']:6d} {c['enemy_ascent']:6d} {t['clean']:6d} "
              f"{100*t['low_oxy_steps']/n:6.1f}% {100*t['surf_steps']/n:6.1f}% {totd:5d}")
    print(f"\n  (override footprint = % steps the policy substitutes UPFIRE; outside it the action is the\n"
          f"   teacher's by construction, so deviation-outside-override is 0 -- footprint IS the deviation.)")
    print(f"  totD includes normal_teacher + ascent_unresolved; baseline teacher deaths over same eps = {base_deaths}.")
    for name, t in agg.items():
        c = t["cats"]
        print(f"  {name}: cats={c}")


if __name__ == "__main__":
    main()
