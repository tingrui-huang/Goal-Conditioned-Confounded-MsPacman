"""Collect the ghost-confounder offline Ms. Pac-Man dataset (Level-1).

Per step logs: frame (UNMASKED), ghost bboxes (reproducible inpaint + oracle-free),
action, player_pos, done, min_ghost_dist, ghost_near, detour (U->A), life_lost
(U->S'). Ghost inpaint applied at load time. No oxygen mechanic, so deaths are
ghost contact -> clean U isolation; ghosts are dense -> U->S' is frequent.
"""
import os
import json
import argparse

import numpy as np

from mspacman_ccrl import config as C
from mspacman_ccrl.envs.mspacman_gc import MsPacmanGCEnv
from mspacman_ccrl.policies.scripted_behavior import GhostAvoidingPolicy


def _sample_target(rng, cfg):
    x = rng.randint(cfg.target_x_range[0], cfg.target_x_range[1] + 1)
    y = rng.randint(cfg.target_y_range[0], cfg.target_y_range[1] + 1)
    return (float(x), float(y))


def _pad(boxes, max_g):
    arr = np.zeros((max_g, 4), dtype=np.int16)
    n = min(len(boxes), max_g)
    for i in range(n):
        arr[i] = boxes[i]
    return arr, n


def collect(cfg=C.DEFAULT, out_root=None, n_episodes=40, max_steps=2000,
            rho=None, seed=0, verbose=True):
    out_root = out_root or C.DATA_ROOT
    os.makedirs(out_root, exist_ok=True)
    rho = float(C.RHO if rho is None else rho)
    max_g = C.MAX_GHOSTS

    env = MsPacmanGCEnv(cfg)
    policy = GhostAvoidingPolicy(cfg, rho=rho)
    rng = np.random.RandomState(seed)
    manifest = {"game_id": cfg.game_id, "frame_shape": list(C.FRAME_SHAPE),
                "confounder": "ghost", "ghost_category": C.GHOST_CATEGORY,
                "wall_color": list(C.WALL_COLOR), "corridor_color": list(C.CORRIDOR_COLOR),
                "max_ghosts": max_g, "rho": rho, "ghost_contact_px": C.GHOST_CONTACT_PX,
                "eps": cfg.eps, "seed": seed, "episodes": []}

    for ep in range(n_episodes):
        frame, state = env.reset(seed=seed + ep)
        policy.reset()
        target = _sample_target(rng, cfg)
        prev_lives = state.get("lives")

        frames, gboxes, gcounts, actions, positions = [], [], [], [], []
        dones, min_dists, ghost_near, detour, life_lost, targets = [], [], [], [], [], []

        for t in range(max_steps):
            if policy.reached(state, target):
                target = _sample_target(rng, cfg)
            a = policy.act(state, target, frame)
            info = policy.last_info
            box_arr, n_g = _pad(state.get("ghosts") or [], max_g)

            frames.append(frame); gboxes.append(box_arr); gcounts.append(n_g)
            actions.append(a)
            positions.append(state["player_pos"] if state["player_pos"] is not None
                             else (np.nan, np.nan))
            min_dists.append(info.get("min_ghost_dist", np.inf))
            ghost_near.append(bool(info.get("ghost_near", False)))
            detour.append(bool(info.get("detour", False)))
            targets.append(target)

            frame, _, term, trunc, state = env.step(a)
            done = bool(term or trunc)
            cur = state.get("lives")
            lost = (prev_lives is not None and cur is not None and cur < prev_lives)
            life_lost.append(bool(lost or done))
            prev_lives = cur if cur is not None else prev_lives
            dones.append(done)
            if done:
                break

        md = np.asarray(min_dists, dtype=np.float32); md[~np.isfinite(md)] = 9999.0
        ll = np.asarray(life_lost, dtype=np.bool_)
        n_deaths = int(ll.sum())
        n_ghost_deaths = int((ll & (md <= C.GHOST_CONTACT_PX)).sum())

        traj = {
            "frames": np.asarray(frames, dtype=np.uint8),
            "ghost_bboxes": np.asarray(gboxes, dtype=np.int16),
            "n_ghosts": np.asarray(gcounts, dtype=np.int16),
            "actions": np.asarray(actions, dtype=np.int64),
            "player_pos": np.asarray(positions, dtype=np.float32),
            "done": np.asarray(dones, dtype=np.bool_),
            "min_ghost_dist": md,
            "ghost_near": np.asarray(ghost_near, dtype=np.bool_),
            "detour": np.asarray(detour, dtype=np.bool_),
            "life_lost": ll,
            "target": np.asarray(targets, dtype=np.float32),
            "rho": np.float32(rho),
        }
        path = os.path.join(out_root, f"traj_{ep:04d}.npz")
        np.savez_compressed(path, **traj)
        manifest["episodes"].append({"file": os.path.basename(path), "steps": int(len(actions)),
                                     "deaths": n_deaths, "ghost_deaths": n_ghost_deaths,
                                     "detour_frac": float(np.mean(detour)) if detour else 0.0})
        if verbose:
            print(f"[ep {ep:03d}] steps={len(actions):4d} deaths={n_deaths} "
                  f"ghost_deaths={n_ghost_deaths} detour={np.mean(detour):.2f} "
                  f"ghost_near={np.mean(ghost_near):.2f} -> {path}")

    env.close()
    with open(os.path.join(out_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    if verbose:
        tot = sum(e["steps"] for e in manifest["episodes"])
        dd = sum(e["deaths"] for e in manifest["episodes"])
        gd = sum(e["ghost_deaths"] for e in manifest["episodes"])
        print(f"\nDONE: {len(manifest['episodes'])} traj, {tot} steps -> {out_root}")
        print(f"deaths={dd} ghost_deaths={gd} ({100*gd/max(dd,1):.0f}% ghost) [acceptance C wants ~100%]")
    return out_root


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--rho", type=float, default=C.RHO)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=C.DATA_ROOT)
    args = ap.parse_args()
    collect(out_root=args.out, n_episodes=args.episodes, max_steps=args.max_steps,
            rho=args.rho, seed=args.seed)
