"""MINIMAL env rerun for the behavioral-diagnosis package: re-execute ONLY the 4 selected
critic rollouts to recover per-step XY, per-step distance, actions, and rendered frames.

Reuses the eval's OWN run_policy/metrics_for (so the rollout is bit-identical to the formal
evaluation), replicates the center-goal relabeling (g = center reference position at t+H), and
VERIFIES each rerun against the stored raw_rollouts scalars (Gate 3 alignment). Does NOT rerun
the full 1440-rollout evaluation. Full-view critic only (mask_oxygen=False).
"""
import json, os
import numpy as np

from seaquest_ccrl.scripts.g0_closed_loop_eval import (
    run_policy, metrics_for, pcenter, dist, RADIUS, load_critic, SeaquestPort, CKPT)

FV = "artifacts/seaquest/goal_control/full_view/evaluation"
OUT = "artifacts/seaquest/behavioral_diagnosis_compact"


def main():
    sel = json.load(open(f"{OUT}/selected_examples.json"))["selected"]
    ep = np.load(f"{FV}/episode_actions.npz")
    raw = np.load(f"{FV}/raw_rollouts.npz", allow_pickle=True)
    rid = {(int(raw["anchor_id"][i]), int(raw["seed"][i]), int(raw["H"][i])): i
           for i in range(len(raw["anchor_id"])) if raw["policy"].astype(str)[i] == "critic"}
    critic, cfg, oracle = load_critic(CKPT, "cpu")
    assert oracle is True, "expected the full-view (oracle) critic"
    mask_oxygen = False
    # numpy-version workaround: vendored OCAtari Seaquest detection relies on uint8 wraparound
    # (old numpy wrapped silently; this numpy raises OverflowError). Cast RAM to int64 before
    # detection -- the code's explicit %256 preserves the intended wraparound, so object positions
    # are bit-identical; only out-of-range HUD arithmetic (irrelevant to the rollout) is fixed.
    import ocatari.ram.seaquest as _sq
    _orig_detect = _sq._detect_objects_ram
    _sq._detect_objects_ram = lambda objects, ram_state, hud: _orig_detect(
        objects, np.asarray(ram_state, np.int64), hud)
    work = SeaquestPort(sticky=0.0, full_action_space=True, seed=1000)

    gate3 = []
    os.makedirs(f"{OUT}/examples", exist_ok=True)
    for e in sel:
        seed, t, H, aid = e["seed"], e["t"], e["H"], e["anchor_id"]
        acts_full = ep[f"actions_{seed}"]
        # center reference replay -> center goal g = cpos[t+H], start p0 = cpos[t]
        work.reset(seed=seed, noop_max=0)
        cpos = [pcenter(work)]
        for s in range(t + H):
            work.agent_step(int(acts_full[s])); cpos.append(pcenter(work))
        g = cpos[t + H]; p0 = cpos[t]

        # faithful rollout (identical to the formal eval) + metrics
        traj = run_policy(seed, acts_full, t, H, "critic", port=work, critic=critic, cfg=cfg,
                          device="cpu", aid=aid, goal=tuple(g), mask_oxygen=mask_oxygen)
        m = metrics_for(traj, tuple(g), H)

        # Gate 3: verify against stored scalars
        i = rid[(aid, seed, H)]
        st = {k: (float(raw[k][i]) if k in ("min_dist", "final_dist") else bool(raw[k][i]))
              for k in ("min_dist", "final_dist", "success_by_H", "success_at_H")}
        match = (abs(m["min_dist"] - st["min_dist"]) < 1e-6
                 and ((not np.isfinite(st["final_dist"]) and not np.isfinite(m["final_dist"]))
                      or abs(m["final_dist"] - st["final_dist"]) < 1e-6)
                 and m["success_by_H"] == st["success_by_H"] and m["success_at_H"] == st["success_at_H"]
                 and traj["crn"] == traj["crn"])
        gate3.append({"category": e["category"], "anchor_id": aid, "seed": seed, "H": H,
                      "crn": traj["crn"], "rerun": {k: m[k] for k in st}, "stored": st, "match": bool(match)})

        # frame capture: deterministic replay (t ref actions -> anchor) then the H critic actions
        work.reset(seed=seed, noop_max=0)
        for s in range(t):
            work.agent_step(int(acts_full[s]))
        frames = [np.asarray(work.ale.getScreenRGB(), np.uint8)]      # anchor frame (pre-step 0)
        for a in traj["actions"]:
            work.agent_step(int(a)); frames.append(np.asarray(work.ale.getScreenRGB(), np.uint8))
        frames = np.stack(frames)                                     # (1+n_steps, 210,160,3)

        pos_arr = np.array([[np.nan, np.nan] if p is None else [float(p[0]), float(p[1])]
                            for p in traj["positions"]], np.float32)   # None (player absent) -> NaN gap
        np.savez_compressed(f"{OUT}/examples/{e['category']}_a{aid}_s{seed}_H{H}.npz",
                            positions=pos_arr,
                            dists=np.array(m["dists"], np.float32),
                            actions=np.array(traj["actions"], np.int64),
                            frames=frames, goal=np.array(g, np.float32), start=np.array(p0, np.float32),
                            H=H, t=t, seed=seed, anchor_id=aid, category=e["category"],
                            min_dist=m["min_dist"], final_dist=m["final_dist"],
                            success_by_H=m["success_by_H"], success_at_H=m["success_at_H"],
                            life_lost=m["life_lost"], terminated=m["terminated"])
        print(f"[{e['category']}] a{aid} s{seed} H{H}: n_steps={m['n_steps']} min={m['min_dist']:.2f} "
              f"final={m['final_dist']:.2f} by_H={m['success_by_H']} at_H={m['success_at_H']} "
              f"frames={len(frames)} MATCH={match}")

    json.dump({"gate3_alignment": gate3, "ALL_MATCH": all(x["match"] for x in gate3)},
              open(f"{OUT}/gate3_rerun_verification.json", "w"), indent=2)
    print("\nGate 3 ALL_MATCH:", all(x["match"] for x in gate3))


if __name__ == "__main__":
    main()
