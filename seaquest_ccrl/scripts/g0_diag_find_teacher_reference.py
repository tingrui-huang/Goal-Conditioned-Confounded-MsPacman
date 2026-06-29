"""Find a GOOD teacher gameplay reference by GENERATING FRESH teacher episodes (the 24 stored
"seeds" in episode_actions.npz are one identical death-laden episode copied). Run the actual
teacher policy (stateless, no LSTM) on new seeds and keep one that is: NO life loss and >=1
VERIFIED successful surfacing/refill (oxygen rises while lives stay constant, player reaches the
surface before the refill, no death/respawn). Classified from measured signals, never curve shape.
"""
import argparse
import json
import sys
from unittest.mock import MagicMock
# The teacher source is a full CleanRL training script importing envpool/gym (Linux-only / unused
# here); only its Flax network classes are needed. Stub those modules so the source loads.
for _m in ("envpool", "gym"):
    sys.modules.setdefault(_m, MagicMock())
import numpy as np
import ocatari.ram.seaquest as _sq
_orig = _sq._detect_objects_ram
_sq._detect_objects_ram = lambda o, r, h: _orig(o, np.asarray(r, np.int64), h)
from seaquest_ccrl.scripts.g0_closed_loop_eval import SeaquestPort, TEACHER_CKPT, TEACHER_SRC   # noqa: E402
from seaquest_stage_s0.teacher_adapter import CleanRLSeaquestTeacher                            # noqa: E402

OUT = "artifacts/seaquest/behavioral_diagnosis_compact"
LOW, HIGH, WIN = 15.0, 55.0, 60
SURFACE_Y = 50.0             # player_y <= 50 == at/near the surface waterline (~46)


def generate(teacher, seed, max_steps, mode, rng):
    """Run the teacher policy fresh for one episode; capture per-step signals (+ actions)."""
    port = SeaquestPort(sticky=0.0, full_action_space=True, seed=seed)
    port.reset(seed=seed, noop_max=0)
    O, PY, LV, PR, ACT = [], [], [], [], []
    for s in range(max_steps):
        f = port.features()
        O.append(f.get("oxygen") if f.get("oxygen") is not None else -1.0)
        PY.append(f.get("player_y") if f.get("player_y") is not None else np.nan)
        LV.append(f.get("lives") if f.get("lives") is not None else np.nan)
        PR.append(f.get("player_y") is not None)
        if mode == "greedy":
            a = int(teacher.greedy_action(port.teacher_obs()[None])[0])
        else:
            a = int(teacher.sample_action(port.teacher_obs(), teacher.gumbel_from_uniform(rng.uniform(size=18)))[0])
        ACT.append(a)
        rec = port.agent_step(a)
        if rec["terminated"] or rec["truncated"]:
            break
    return (np.array(O), np.array(PY), np.array(LV), np.array(PR), np.array(ACT, np.int64), len(ACT))


def analyze(O, PY, LV, PR, n):
    deaths = [int(s) for s in range(1, n) if np.isfinite(LV[s]) and np.isfinite(LV[s - 1]) and LV[s] < LV[s - 1]]
    surfacings = []
    t = 1
    while t < n:
        if O[t] >= HIGH and O[t - 1] < HIGH:                       # crossing up into "refilled"
            w0 = max(0, t - WIN); seg = O[w0:t]; valid = np.where(seg >= 0)[0]
            if len(valid):
                tlo = w0 + int(valid[np.argmin(seg[valid])])
                if O[tlo] <= LOW and tlo > 5:                      # genuine low trough (not the start spawn)
                    a, b = tlo, t
                    win = slice(max(0, a - 3), min(n, b + 3))
                    lv_drop = bool(np.isfinite(LV[win]).any() and np.nanmin(LV[win]) < LV[a])
                    absent = bool((~PR[win]).any())
                    reached_surf = bool(np.isfinite(PY[max(0, a - 5):b + 1]).any()
                                        and np.nanmin(PY[max(0, a - 5):b + 1]) <= SURFACE_Y)
                    if (not lv_drop) and (not absent) and reached_surf:
                        surfacings.append({"t_low": a, "t_high": b, "o2": [float(O[a]), float(O[b])],
                                           "lives": float(LV[a]), "player_y_min": float(np.nanmin(PY[max(0, a - 5):b + 1]))})
            t = t  # fallthrough
        t += 1
    return deaths, surfacings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="greedy", choices=["greedy", "stochastic"])
    ap.add_argument("--n-seeds", type=int, default=14)
    ap.add_argument("--seed-base", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=1000)
    args = ap.parse_args()
    teacher = CleanRLSeaquestTeacher(TEACHER_CKPT, TEACHER_SRC, mod_name="cleanrl_src_A")
    rows = []; chosen = None
    for i in range(args.n_seeds):
        seed = args.seed_base + i
        O, PY, LV, PR, ACT, n = generate(teacher, seed, args.max_steps, args.mode, np.random.default_rng(seed))
        deaths, surf = analyze(O, PY, LV, PR, n)
        first_death = deaths[0] if deaths else None
        clip = None
        for s in surf:                                            # a death-free clip that contains a surfacing
            if first_death is None or s["t_high"] + 5 < first_death:
                clip = [int(max(0, s["t_low"] - 40)), int(min(n, s["t_high"] + 40))]; break
        rows.append({"seed": seed, "n_steps": n, "n_deaths": len(deaths), "deaths": deaths,
                     "n_surfacings": len(surf), "surfacings": surf,
                     "death_free_episode": len(deaths) == 0, "clip_with_surfacing": clip})
        print(f"seed {seed}: steps={n} deaths={len(deaths)}{deaths[:3]} surfacings={len(surf)} "
              f"{'[DEATH-FREE]' if not deaths else ''} {'[clip ok]' if clip else ''}")
        # keep the actions of the FIRST fully-valid (death-free + surfacing) episode
        if chosen is None and len(deaths) == 0 and len(surf) >= 1:
            chosen = {"seed": seed, "mode": args.mode, "actions": ACT, "clip": None,
                      "deaths": deaths, "surfacings": surf, "n_steps": n}

    # fallback: a death-free CLIP (episode dies later) if no fully death-free episode exists
    if chosen is None:
        for r in rows:
            if r["clip_with_surfacing"] is not None:
                seed = r["seed"]
                O, PY, LV, PR, ACT, n = generate(teacher, seed, args.max_steps, args.mode, np.random.default_rng(seed))
                chosen = {"seed": seed, "mode": args.mode, "actions": ACT, "clip": r["clip_with_surfacing"],
                          "deaths": r["deaths"], "surfacings": r["surfacings"], "n_steps": n}
                break

    json.dump({"mode": args.mode, "surface_y": SURFACE_Y, "ranking": rows,
               "recommended": ({k: chosen[k] for k in ("seed", "mode", "clip", "deaths", "surfacings", "n_steps")}
                               if chosen else None)},
              open(f"{OUT}/teacher_reference_search.json", "w"), indent=2)
    print("\n=== RECOMMENDED ===")
    if chosen is None:
        print(f"NONE in mode={args.mode}: no fresh episode had a verified death-free surfacing.")
    else:
        np.savez(f"{OUT}/teacher_reference_chosen.npz", actions=chosen["actions"], seed=chosen["seed"],
                 mode=chosen["mode"], clip=np.array(chosen["clip"] if chosen["clip"] else [-1, -1]))
        print(f"seed={chosen['seed']} mode={chosen['mode']} deaths={chosen['deaths']} "
              f"surfacings={len(chosen['surfacings'])} clip={chosen['clip']}")
        print("surfacing evidence:", json.dumps(chosen["surfacings"][:3], indent=2))
        print(f"WROTE {OUT}/teacher_reference_chosen.npz (actions for exact GIF regen)")
    print(f"WROTE {OUT}/teacher_reference_search.json")


if __name__ == "__main__":
    main()
