"""Formal closed-loop success_by_H for a LEARNED actor, on the IDENTICAL anchor protocol as the
argmax-critic eval (g0_closed_loop_eval): same anchors, same center-goal relabel (g=cpos[t+H]),
same reset+replay to s_t, same metrics_for. Only the per-step policy differs (actor argmax instead
of critic argmax). Apples-to-apples vs critic 0.533 / random 0.394 / teacher 0.922.
"""
import argparse
import json
import numpy as np
import torch

# numpy-version workaround: OCAtari Seaquest detection relies on uint8 wraparound (this numpy raises);
# cast RAM to int64 before detection -> object positions bit-identical (explicit %256 preserved).
import ocatari.ram.seaquest as _sq
_orig_detect = _sq._detect_objects_ram
_sq._detect_objects_ram = lambda objects, ram_state, hud: _orig_detect(objects, np.asarray(ram_state, np.int64), hud)

from seaquest_ccrl.scripts.g0_closed_loop_eval import (
    select_subset, run_policy, metrics_for, pcenter, RADIUS, HORIZONS, boot_ep, SeaquestPort, EVAL)
from seaquest_ccrl.models.actor import GoalConditionedActor
from seaquest_ccrl.training.config import TrainConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actor-ckpt", default="artifacts/seaquest/final_actor.pt")
    ap.add_argument("--per-episode", type=int, default=20)         # 20 -> the same 1440 anchors
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="artifacts/seaquest/goal_control/actor_closed_loop/actor_success.json")
    args = ap.parse_args()
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    ck = torch.load(args.actor_ckpt, map_location=args.device, weights_only=False)
    cfg = TrainConfig(**ck["cfg"])
    actor = GoalConditionedActor(cfg.frame_size, cfg.nb_actions, getattr(cfg, "frame_stack", 1)).to(args.device)
    actor.load_state_dict(ck["state_dict"], strict=True)           # STRICT reload
    actor.eval()
    mask_oxygen = (ck.get("oracle") is False)                      # oracle/full-view -> no mask
    print(f"[actor] {args.actor_ckpt} lam={ck.get('lam')} balance_grad={ck.get('balance_grad')} "
          f"oracle={ck.get('oracle')} -> view={'MASKED' if mask_oxygen else 'FULL_VIEW'} fs={cfg.frame_stack}")

    manifest = json.load(open(f"{EVAL}/anchor_manifest.json"))
    ep_npz = np.load(f"{EVAL}/episode_actions.npz")
    chosen = select_subset(manifest, args.per_episode)
    seeds = manifest["seeds"]
    by_seed = {}
    for H in HORIZONS:
        for a in chosen[H]:
            by_seed.setdefault(a["seed"], {}).setdefault(a["t"], []).append((H, a))

    work = SeaquestPort(sticky=0.0, full_action_space=True, seed=seeds[0])
    rows = []
    for seed in seeds:
        if seed not in by_seed:
            continue
        acts_full = ep_npz[f"actions_{seed}"]
        ts = sorted(by_seed[seed].keys())
        maxneed = max(t + H for t in ts for (H, _) in by_seed[seed][t])
        work.reset(seed=seed, noop_max=0)
        cpos = [pcenter(work)]
        for s in range(maxneed):
            work.agent_step(int(acts_full[s])); cpos.append(pcenter(work))
        for t in ts:
            for (H, a) in by_seed[seed][t]:
                g, p0 = cpos[t + H], cpos[t]
                if g is None or p0 is None:
                    continue
                traj = run_policy(seed, acts_full, t, H, "actor", port=work, actor=actor, cfg=cfg,
                                  device=args.device, aid=a["anchor_id"], goal=tuple(g), mask_oxygen=mask_oxygen)
                m = metrics_for(traj, tuple(g), H)
                d = float(np.hypot(p0[0] - g[0], p0[1] - g[1]))    # center displacement
                from seaquest_ccrl.scripts.g0_closed_loop_eval import difficulty, direction
                rows.append({"anchor_id": a["anchor_id"], "seed": seed, "t": t, "H": H,
                             "difficulty": difficulty(d), "direction": direction(p0[1], g[1]),
                             "success_by_H": bool(m["success_by_H"]), "success_at_H": bool(m["success_at_H"]),
                             "min_dist": m["min_dist"], "final_dist": m["final_dist"]})
        print(f"  seed {seed}: {sum(1 for r in rows if r['seed']==seed)} rollouts  "
              f"(running success_by_H={np.mean([r['success_by_H'] for r in rows]):.3f})")

    sb = [r["success_by_H"] for r in rows]; eps = [r["seed"] for r in rows]
    agg = boot_ep(sb, eps)
    per_h = {H: {"success_by_H": float(np.mean([r["success_by_H"] for r in rows if r["H"] == H])),
                 "success_at_H": float(np.mean([r["success_at_H"] for r in rows if r["H"] == H])),
                 "n": int(sum(1 for r in rows if r["H"] == H))} for H in HORIZONS}
    by_dir = {u: {"success_by_H": float(np.mean([r["success_by_H"] for r in rows if r["direction"] == u])),
                  "n": int(sum(1 for r in rows if r["direction"] == u))} for u in ("up", "down")}
    out = {"actor_ckpt": args.actor_ckpt, "lam": ck.get("lam"), "balance_grad": ck.get("balance_grad"),
           "view": "FULL_VIEW" if not mask_oxygen else "MASKED", "n_rollouts": len(rows),
           "aggregate_success_by_H": agg, "aggregate_success_at_H": float(np.mean([r["success_at_H"] for r in rows])),
           "per_horizon": per_h, "by_direction": by_dir,
           "baselines_same_protocol": {"critic_argmax": 0.533, "random_B1": 0.394, "teacher_B3b": 0.922, "noop_B2": 0.0}}
    json.dump({"summary": out, "rows": rows}, open(args.out, "w"), indent=2)
    print("\n=== ACTOR closed-loop success_by_H (same formal protocol) ===")
    print(f"  aggregate success_by_H = {agg['mean']:.4f}  CI95={[round(x,3) for x in agg['ci95']]}  (n={len(rows)})")
    print(f"  aggregate success_at_H = {out['aggregate_success_at_H']:.4f}")
    print(f"  per-H: " + "  ".join(f"H{H}={per_h[H]['success_by_H']:.3f}" for H in HORIZONS))
    print(f"  by-dir: up={by_dir['up']['success_by_H']:.3f} down={by_dir['down']['success_by_H']:.3f}")
    print(f"  vs baselines: critic-argmax 0.533 | random 0.394 | teacher 0.922 | NOOP 0.0")
    print(f"WROTE {args.out}")


if __name__ == "__main__":
    main()
