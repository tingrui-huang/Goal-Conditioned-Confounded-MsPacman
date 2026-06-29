"""Phase-C verification of the OxygenAwareTeacher wrapper:
  1. oxygen strongly affects vertical actions  (P(UP | low O2) >> P(UP | high O2));
  2. successful surfacing WITHOUT life loss     (oxygen low->refilled, lives constant, reaches surface);
  3. non-oxygen behavior stays close to the original teacher (action dist / fire-fraction on the
     NON-surfacing steps).
Sweeps the surfacing trigger to pick the lowest one that surfaces reliably (most teacher-like).
"""
import argparse
import sys
from unittest.mock import MagicMock
for _m in ("envpool", "gym"):
    sys.modules.setdefault(_m, MagicMock())
import numpy as np
import ocatari.ram.seaquest as _sq
_orig = _sq._detect_objects_ram
_sq._detect_objects_ram = lambda o, r, h: _orig(o, np.asarray(r, np.int64), h)
from seaquest_ccrl.scripts.g0_closed_loop_eval import SeaquestPort, TEACHER_CKPT, TEACHER_SRC   # noqa: E402
from seaquest_stage_s0.teacher_adapter import CleanRLSeaquestTeacher                            # noqa: E402
from seaquest_ccrl.policies.oxygen_aware_teacher import OxygenAwareTeacher                       # noqa: E402

UP_ACTS = {2, 6, 7, 10, 14, 15}
FIRE_ACTS = {1, 10, 11, 12, 13, 14, 15, 16, 17}
SURFACE_Y = 52.0


def run(teacher, seeds, steps, wrap=None):
    O, LV, PY, ACT, SURF = [], [], [], [], []
    EP = []
    for sd in seeds:
        p = SeaquestPort(sticky=0.0, full_action_space=True, seed=sd); p.reset(seed=sd, noop_max=0)
        if wrap is not None:
            wrap.reset()
        rng = np.random.default_rng(sd); o, lv, py, ac, sf = [], [], [], [], []
        for s in range(steps):
            f = p.features()
            oxy = f.get("oxygen") if f.get("oxygen") is not None else -1.0
            o.append(oxy); lv.append(f.get("lives") if f.get("lives") is not None else np.nan)
            pyv = f.get("player_y"); py.append(np.nan if pyv is None else pyv)
            if wrap is None:
                a = int(teacher.sample_action(p.teacher_obs(), teacher.gumbel_from_uniform(rng.uniform(size=18)))[0]); surf = False
            else:
                a, surf = wrap.act(p.teacher_obs(), oxy, pyv, mode="stochastic", rng=rng)
            ac.append(a); sf.append(surf); p.agent_step(a)
        O.append(np.array(o, float)); LV.append(np.array(lv, float)); PY.append(np.array(py, float))
        ACT.append(np.array(ac)); SURF.append(np.array(sf, bool)); EP.append(len(ac))
    return O, LV, PY, ACT, SURF


def metrics(O, LV, PY, ACT, SURF, trigger):
    o = np.concatenate(O); a = np.concatenate(ACT); sf = np.concatenate(SURF)
    up = np.array([x in UP_ACTS for x in a]); fire = np.array([x in FIRE_ACTS for x in a])
    valid = o >= 0
    p_up_low = float(up[valid & (o < trigger)].mean()) if (valid & (o < trigger)).any() else float("nan")
    p_up_high = float(up[valid & (o >= trigger)].mean()) if (valid & (o >= trigger)).any() else float("nan")
    # deaths + verified surfacings, per episode
    deaths = surfacings = deaths_during_surf = 0
    for Oe, LVe, PYe, SFe in zip(O, LV, PY, SURF):
        dsteps = np.where((np.diff(LVe) < 0) & np.isfinite(np.diff(LVe)))[0]
        deaths += len(dsteps)
        for ds in dsteps:                                  # was the agent surfacing at the death?
            if SFe[max(0, ds - 5):ds + 2].any():
                deaths_during_surf += 1
        n = len(Oe); t = 1
        while t < n:
            if Oe[t] >= 55 and Oe[t - 1] < 55:
                w0 = max(0, t - 60); seg = Oe[w0:t]; vv = np.where(seg >= 0)[0]
                if len(vv):
                    tlo = w0 + int(vv[np.argmin(seg[vv])])
                    if Oe[tlo] <= 15 and tlo > 5:
                        win = slice(max(0, tlo - 3), min(n, t + 3))
                        lv_drop = bool(np.isfinite(LVe[win]).any() and np.nanmin(LVe[win]) < LVe[tlo])
                        reached = bool(np.isfinite(PYe[max(0, tlo - 5):t + 1]).any()
                                       and np.nanmin(PYe[max(0, tlo - 5):t + 1]) <= SURFACE_Y)
                        if (not lv_drop) and reached:
                            surfacings += 1
            t += 1
    return {"p_up_low": p_up_low, "p_up_high": p_up_high, "p_up_gradient": p_up_low - p_up_high,
            "deaths": deaths, "deaths_during_surfacing": deaths_during_surf, "verified_surfacings": surfacings,
            "surfacing_frac": float(sf.mean()), "fire_frac_nonsurf": float(fire[~sf].mean()),
            "fire_frac_all": float(fire.mean()), "n_steps": int(len(a))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triggers", default="25,40,58")
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--surface-action", type=int, default=10, help="2=UP, 10=UPFIRE")
    args = ap.parse_args()
    teacher = CleanRLSeaquestTeacher(TEACHER_CKPT, TEACHER_SRC, mod_name="cleanrl_src_A")
    seeds = list(range(3000, 3000 + args.n_seeds))

    print("=== ORIGINAL HF teacher (baseline) ===")
    base = run(teacher, seeds, args.steps, wrap=None)
    bm = metrics(*base, trigger=30)
    print(f"  deaths={bm['deaths']} verified_surfacings={bm['verified_surfacings']} "
          f"fire_frac={bm['fire_frac_all']:.3f} P(UP|O2<30)={bm['p_up_low']:.3f} P(UP|O2>=30)={bm['p_up_high']:.3f}")

    print("\n=== WRAPPED (oxygen-aware) — trigger sweep ===")
    print(f"  (surface_action={'UPFIRE' if args.surface_action == 10 else 'UP'})")
    for trig in [int(x) for x in args.triggers.split(",")]:
        wrap = OxygenAwareTeacher(teacher, surface_trigger=trig, refilled=58, surface_action=args.surface_action)
        r = run(teacher, seeds, args.steps, wrap=wrap)
        m = metrics(*r, trigger=trig)
        fire_sim = abs(m["fire_frac_nonsurf"] - bm["fire_frac_all"])
        print(f"  trigger={trig:2d}: P(UP|<{trig})={m['p_up_low']:.3f} vs P(UP|>={trig})={m['p_up_high']:.3f} "
              f"(grad {m['p_up_gradient']:+.3f}) | deaths={m['deaths']} (during_surfacing={m['deaths_during_surfacing']}) "
              f"verified_surf={m['verified_surfacings']} | surf_frac={m['surfacing_frac']:.2f} "
              f"| fire(non-surf)={m['fire_frac_nonsurf']:.3f} (|Δ teacher|={fire_sim:.3f})")
    print("\nPick: lowest trigger with deaths low + verified_surfacings>0 + small surfacing_frac + fire_frac close to teacher.")


if __name__ == "__main__":
    main()
