"""Collect a static, by-trajectory offline Seaquest dataset.

Per episode: sample an underwater target, roll the scripted oxygen-aware policy, and
append per-step {frame (UNMASKED), action, player_pos, oxygen, done}. On episode end,
flush the whole trajectory to one .npz file. Time order + episode boundaries are
preserved (Invariant 5). Frames are stored UNMASKED -> the oracle view is free
(Invariant 4); the oxygen mask is applied later, at load time.

NOTHING downstream is produced: no critic, no world model, no worst-case, no training.
"""
import os
import json
import argparse

import numpy as np

from seaquest_ccrl import config as C
from seaquest_ccrl.envs.seaquest_gc import SeaquestGCEnv
from seaquest_ccrl.policies.scripted_behavior import ScriptedBehaviorPolicy


def _sample_target(rng: np.random.RandomState, cfg: C.Config):
    x = rng.randint(cfg.target_x_range[0], cfg.target_x_range[1] + 1)
    y = rng.randint(cfg.target_y_range[0], cfg.target_y_range[1] + 1)
    return (float(x), float(y))


def collect(cfg: C.Config = C.DEFAULT, out_root: str = None, verbose: bool = True):
    out_root = out_root or cfg.data_root
    os.makedirs(out_root, exist_ok=True)

    env = SeaquestGCEnv(cfg)
    policy = ScriptedBehaviorPolicy(cfg)
    rng = np.random.RandomState(cfg.seed)

    manifest = {
        "game_id": cfg.game_id,
        "frame_shape": list(C.FRAME_SHAPE),
        "oxy_mask_rect": list(cfg.oxy_mask_rect),
        "oxy_full_width": cfg.oxy_full_width,
        "theta": cfg.theta,
        "eps": cfg.eps,
        "seed": cfg.seed,
        "frameskip": cfg.frameskip,
        "repeat_action_probability": cfg.repeat_action_probability,
        "depletion_noise": cfg.depletion_noise,
        "episodes": [],
    }

    for ep in range(cfg.n_episodes):
        # Per-episode seed keeps determinism (check F) while varying episodes.
        frame, state = env.reset(seed=cfg.seed + ep)
        target = _sample_target(rng, cfg)

        frames, actions, positions, oxygens, dones, targets = [], [], [], [], [], []

        for t in range(cfg.max_steps_per_ep):
            if policy.reached(state, target):
                target = _sample_target(rng, cfg)  # keep the sub moving
            a = policy.act(state, target)

            # log the CURRENT (pre-action) frame+state with the action taken
            frames.append(frame)
            actions.append(a)
            positions.append(state["player_pos"] if state["player_pos"] is not None
                             else (np.nan, np.nan))
            oxygens.append(state["oxygen"] if state["oxygen"] is not None else -1)
            targets.append(target)

            frame, _, term, trunc, state = env.step(a)
            done = bool(term or trunc)
            dones.append(done)
            if done:
                break

        traj = {
            "frames": np.asarray(frames, dtype=np.uint8),           # (T,210,160,3)
            "actions": np.asarray(actions, dtype=np.int64),          # (T,)
            "player_pos": np.asarray(positions, dtype=np.float32),   # (T,2)
            "oxygen": np.asarray(oxygens, dtype=np.int32),           # (T,)
            "done": np.asarray(dones, dtype=np.bool_),               # (T,)
            "target": np.asarray(targets, dtype=np.float32),         # (T,2)
            "theta": np.int32(cfg.theta),
        }
        path = os.path.join(out_root, f"traj_{ep:04d}.npz")
        np.savez_compressed(path, **traj)
        manifest["episodes"].append({"file": os.path.basename(path),
                                     "steps": int(len(actions))})
        if verbose:
            print(f"[ep {ep:03d}] steps={len(actions):4d} "
                  f"oxy[min/max]={traj['oxygen'].min()}/{traj['oxygen'].max()} -> {path}")

    env.close()
    with open(os.path.join(out_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    if verbose:
        total = sum(e["steps"] for e in manifest["episodes"])
        print(f"\nDONE: {len(manifest['episodes'])} trajectories, {total} steps -> {out_root}")
    return out_root


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=C.N_EPISODES)
    ap.add_argument("--max-steps", type=int, default=C.MAX_STEPS_PER_EP)
    ap.add_argument("--theta", type=int, default=C.THETA)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--out", type=str, default=C.DATA_ROOT)
    args = ap.parse_args()

    cfg = C.Config(n_episodes=args.episodes, max_steps_per_ep=args.max_steps,
                   theta=args.theta, seed=args.seed, data_root=args.out)
    collect(cfg, out_root=args.out)
