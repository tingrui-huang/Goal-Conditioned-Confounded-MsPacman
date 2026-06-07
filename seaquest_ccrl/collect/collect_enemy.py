"""Collect the ENEMY-confounder offline Seaquest dataset (Level-1 v2).

Per episode: sample an underwater target, roll the goal-seeking enemy-avoiding
demonstrator, and log per step:
    frame (UNMASKED), enemy bboxes (for reproducible inpaint + oracle-for-free),
    action, player_pos, oxygen (now VISIBLE/analysis-only), done,
    min_enemy_dist, enemy_near, detour (U->A signal), life_lost (U->S' signal).
Frames are stored UNMASKED; the enemy inpaint mask is applied at load time using
the stored bboxes, so the oracle (unmasked) view is free.

Oxygen is managed by the demonstrator (surfaces in time) so deaths are ~exclusively
enemy contact -- clean isolation of the enemy confounder U (acceptance C).
"""
import os
import json
import argparse

import numpy as np

from seaquest_ccrl import config as C
from seaquest_ccrl.envs.seaquest_gc import SeaquestGCEnv
from seaquest_ccrl.policies.scripted_behavior import EnemyAvoidingPolicy


def _sample_target(rng, cfg):
    x = rng.randint(cfg.target_x_range[0], cfg.target_x_range[1] + 1)
    y = rng.randint(cfg.target_y_range[0], cfg.target_y_range[1] + 1)
    return (float(x), float(y))


def _pad_bboxes(boxes, max_e):
    arr = np.zeros((max_e, 4), dtype=np.int16)
    n = min(len(boxes), max_e)
    for i in range(n):
        arr[i] = boxes[i]
    return arr, n


def collect(cfg=C.DEFAULT, out_root=None, n_episodes=40, max_steps=2000,
            rho=None, seed=0, verbose=True):
    out_root = out_root or C.ENEMY_DATA_ROOT
    os.makedirs(out_root, exist_ok=True)
    rho = float(C.RHO if rho is None else rho)
    max_e = C.MAX_ENEMIES

    env = SeaquestGCEnv(cfg)
    policy = EnemyAvoidingPolicy(cfg, rho=rho)
    rng = np.random.RandomState(seed)

    manifest = {
        "game_id": cfg.game_id, "frame_shape": list(C.FRAME_SHAPE),
        "confounder": "enemy", "hostile_categories": list(C.HOSTILE_CATEGORIES),
        "water_color": list(C.WATER_COLOR), "max_enemies": max_e,
        "rho": rho, "oxy_surface_trigger": C.OXY_SURFACE_TRIGGER,
        "enemy_contact_px": C.ENEMY_CONTACT_PX,
        "eps": cfg.eps, "seed": seed, "frameskip": cfg.frameskip,
        "repeat_action_probability": cfg.repeat_action_probability, "episodes": [],
    }

    for ep in range(n_episodes):
        frame, state = env.reset(seed=seed + ep)
        policy.reset()
        target = _sample_target(rng, cfg)
        prev_lives = state.get("lives")

        frames, ebboxes, ecounts, actions, positions = [], [], [], [], []
        oxygens, dones, min_dists, enemy_near, detour, life_lost, targets = \
            [], [], [], [], [], [], []

        for t in range(max_steps):
            if policy.reached(state, target):
                target = _sample_target(rng, cfg)
            a = policy.act(state, target)
            info = policy.last_info

            box_arr, n_e = _pad_bboxes(state.get("enemies") or [], max_e)
            frames.append(frame)
            ebboxes.append(box_arr)
            ecounts.append(n_e)
            actions.append(a)
            positions.append(state["player_pos"] if state["player_pos"] is not None
                             else (np.nan, np.nan))
            oxygens.append(state["oxygen"] if state["oxygen"] is not None else -1)
            min_dists.append(info.get("min_enemy_dist", np.inf))
            enemy_near.append(bool(info.get("enemy_near", False)))
            detour.append(bool(info.get("detour", False)))
            targets.append(target)

            frame, _, term, trunc, state = env.step(a)
            done = bool(term or trunc)
            # life lost this step? (death event for U->S' analysis)
            cur_lives = state.get("lives")
            lost = (prev_lives is not None and cur_lives is not None
                    and cur_lives < prev_lives)
            life_lost.append(bool(lost or done))
            prev_lives = cur_lives if cur_lives is not None else prev_lives
            dones.append(done)
            if done:
                break

        md = np.asarray(min_dists, dtype=np.float32)
        md[~np.isfinite(md)] = 9999.0
        ll = np.asarray(life_lost, dtype=np.bool_)
        contact = C.ENEMY_CONTACT_PX
        n_deaths = int(ll.sum())
        n_enemy_deaths = int((ll & (md <= contact)).sum())

        traj = {
            "frames": np.asarray(frames, dtype=np.uint8),               # (T,210,160,3)
            "enemy_bboxes": np.asarray(ebboxes, dtype=np.int16),         # (T,MAX_E,4)
            "n_enemies": np.asarray(ecounts, dtype=np.int16),            # (T,)
            "actions": np.asarray(actions, dtype=np.int64),
            "player_pos": np.asarray(positions, dtype=np.float32),       # (T,2)
            "oxygen": np.asarray(oxygens, dtype=np.int32),               # (T,) visible now
            "done": np.asarray(dones, dtype=np.bool_),
            "min_enemy_dist": md,                                        # (T,)
            "enemy_near": np.asarray(enemy_near, dtype=np.bool_),        # (T,)
            "detour": np.asarray(detour, dtype=np.bool_),                # (T,) U->A signal
            "life_lost": ll,                                             # (T,) U->S' signal
            "target": np.asarray(targets, dtype=np.float32),            # (T,2)
            "rho": np.float32(rho),
        }
        path = os.path.join(out_root, f"traj_{ep:04d}.npz")
        np.savez_compressed(path, **traj)
        manifest["episodes"].append({
            "file": os.path.basename(path), "steps": int(len(actions)),
            "deaths": n_deaths, "enemy_deaths": n_enemy_deaths,
            "detour_frac": float(np.mean(detour)) if detour else 0.0,
        })
        if verbose:
            print(f"[ep {ep:03d}] steps={len(actions):4d} deaths={n_deaths} "
                  f"enemy_deaths={n_enemy_deaths} detour={np.mean(detour):.2f} -> {path}")

    env.close()
    with open(os.path.join(out_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    if verbose:
        tot = sum(e["steps"] for e in manifest["episodes"])
        dd = sum(e["deaths"] for e in manifest["episodes"])
        ed = sum(e["enemy_deaths"] for e in manifest["episodes"])
        print(f"\nDONE: {len(manifest['episodes'])} trajectories, {tot} steps -> {out_root}")
        print(f"deaths={dd}  enemy_deaths={ed} ({100*ed/max(dd,1):.0f}% enemy) "
              f"[acceptance C wants ~100%]")
    return out_root


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--rho", type=float, default=C.RHO)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=C.ENEMY_DATA_ROOT)
    args = ap.parse_args()
    collect(out_root=args.out, n_episodes=args.episodes, max_steps=args.max_steps,
            rho=args.rho, seed=args.seed)
