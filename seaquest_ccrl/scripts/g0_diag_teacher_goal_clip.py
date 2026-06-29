"""Generate a short VERIFIED teacher goal-reaching SUCCESS clip under the same H<=64 anchor
protocol as the critic examples (reset+replay to s_t, then run the teacher B3b for H steps;
success = the submarine enters the 8px goal ball). The teacher is the expert at this short-horizon
task (B3b ~0.92); this replaces the death-laden full-episode 'reference' with competent teacher play.

Selects a clean reach-and-stay (success_at_H) B3b anchor (shortest H, smallest min_dist), reruns the
teacher rollout locally (JAX teacher; envpool/gym stubbed), verifies it reaches <=8px, captures
frames, and saves examples/teacher_goal_reaching_success.npz with the SAME schema as the critic
example clips so the dashboard renders it identically.
"""
import csv
import sys
from unittest.mock import MagicMock
for _m in ("envpool", "gym"):
    sys.modules.setdefault(_m, MagicMock())            # teacher source imports these (unused here)
import numpy as np
import ocatari.ram.seaquest as _sq
_orig = _sq._detect_objects_ram
_sq._detect_objects_ram = lambda o, r, h: _orig(o, np.asarray(r, np.int64), h)
from seaquest_ccrl.scripts.g0_closed_loop_eval import (   # noqa: E402
    run_policy, metrics_for, pcenter, dist, RADIUS, SeaquestPort, TEACHER_CKPT, TEACHER_SRC)
from seaquest_stage_s0.teacher_adapter import CleanRLSeaquestTeacher   # noqa: E402

FV = "artifacts/seaquest/goal_control/full_view/evaluation"
OUT = "artifacts/seaquest/behavioral_diagnosis_compact"


def candidates():
    rows = [r for r in csv.DictReader(open(f"{FV}/per_anchor_results.csv")) if r["policy"] == "B3b"]
    rec = [{"anchor_id": int(r["anchor_id"]), "seed": int(r["seed"]), "t": int(r["t"]), "H": int(r["H"]),
            "min_dist": float(r["min_dist"]), "success_by_H": r["success_by_H"] == "True",
            "success_at_H": r["success_at_H"] == "True"} for r in rows]
    # clean reach-and-stay first, shortest H, closest reach
    rec = [r for r in rec if r["success_at_H"]] or [r for r in rec if r["success_by_H"]]
    return sorted(rec, key=lambda r: (r["H"], r["min_dist"], r["anchor_id"], r["seed"]))


def main():
    teacher = CleanRLSeaquestTeacher(TEACHER_CKPT, TEACHER_SRC, mod_name="cleanrl_src_A")
    ep = np.load(f"{FV}/episode_actions.npz")
    work = SeaquestPort(sticky=0.0, full_action_space=True, seed=1000)
    chosen = None
    for c in candidates()[:25]:
        seed, t, H, aid = c["seed"], c["t"], c["H"], c["anchor_id"]
        acts_full = ep[f"actions_{seed}"]
        work.reset(seed=seed, noop_max=0)
        cpos = [pcenter(work)]
        for s in range(t + H):
            work.agent_step(int(acts_full[s])); cpos.append(pcenter(work))
        g, p0 = cpos[t + H], cpos[t]
        if g is None or p0 is None:
            continue
        traj = run_policy(seed, acts_full, t, H, "B3b", port=work, teacher=teacher, aid=aid)
        m = metrics_for(traj, tuple(g), H)
        print(f"  try anchor {aid} seed{seed} H{H}: rerun min={m['min_dist']:.1f} by_H={m['success_by_H']} at_H={m['success_at_H']}")
        if m["success_by_H"]:                          # VERIFIED teacher reaches the goal
            chosen = (c, g, p0, traj, m); break
    if chosen is None:
        print("NO verified teacher goal-reaching success found in the top candidates."); return
    c, g, p0, traj, m = chosen
    seed, t, H, aid = c["seed"], c["t"], c["H"], c["anchor_id"]
    acts_full = ep[f"actions_{seed}"]
    # frame capture: replay to s_t (anchor frame), then the teacher's chosen actions
    work.reset(seed=seed, noop_max=0)
    for s in range(t):
        work.agent_step(int(acts_full[s]))
    frames = [np.asarray(work.ale.getScreenRGB(), np.uint8)]
    for a in traj["actions"]:
        work.agent_step(int(a)); frames.append(np.asarray(work.ale.getScreenRGB(), np.uint8))
    frames = np.stack(frames)
    pos_arr = np.array([[np.nan, np.nan] if p is None else [float(p[0]), float(p[1])] for p in traj["positions"]], np.float32)
    np.savez_compressed(f"{OUT}/examples/teacher_goal_reaching_success.npz",
                        positions=pos_arr, dists=np.array(m["dists"], np.float32),
                        actions=np.array(traj["actions"], np.int64), frames=frames,
                        goal=np.array(g, np.float32), start=np.array(p0, np.float32),
                        H=H, t=t, seed=seed, anchor_id=aid, category="teacher_goal_reaching_success",
                        min_dist=m["min_dist"], final_dist=m["final_dist"],
                        success_by_H=m["success_by_H"], success_at_H=m["success_at_H"],
                        life_lost=m["life_lost"], terminated=m["terminated"])
    import json
    json.dump({"policy": "B3b_teacher", "anchor_id": aid, "seed": seed, "t": t, "H": H,
               "goal_xy": [float(g[0]), float(g[1])], "start_xy": [float(p0[0]), float(p0[1])],
               "min_dist": m["min_dist"], "final_dist": m["final_dist"], "success_by_H": m["success_by_H"],
               "success_at_H": m["success_at_H"], "n_frames": int(len(frames)), "crn": traj["crn"]},
              open(f"{OUT}/teacher_goal_clip_verification.json", "w"), indent=2)
    print(f"\nSELECTED teacher success: anchor {aid} seed{seed} H{H}  min_dist={m['min_dist']:.1f}px "
          f"by_H={m['success_by_H']} at_H={m['success_at_H']}  frames={len(frames)}")
    print(f"WROTE {OUT}/examples/teacher_goal_reaching_success.npz + teacher_goal_clip_verification.json")


if __name__ == "__main__":
    main()
