"""Loader for the by-trajectory offline Ms. Pac-Man (ghost-confounder) dataset.

Same interface as the Seaquest loader so the contrastive/training/eval pipeline
consumes it unchanged:
    obs           : ghost-inpainted frame (210,160,3) uint8  <- learner observation
    achieved_goal : player_pos (x,y)                          <- goal label
`oracle=True` skips the mask -> unmasked frames (ghosts visible).
"""
import os
import glob
import json
from typing import Iterator, Dict, Any, List

import numpy as np

from mspacman_ccrl import config as C
from mspacman_ccrl.data.masking import apply_ghost_mask


class MsPacmanOfflineDataset:
    def __init__(self, root: str = None, oracle: bool = False):
        self.root = root or C.DATA_ROOT
        self.oracle = oracle
        self.files: List[str] = sorted(glob.glob(os.path.join(self.root, "traj_*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No trajectories under {self.root!r}. "
                                    f"Run collect/collect_ghost.py first.")
        mpath = os.path.join(self.root, "manifest.json")
        self.manifest = json.load(open(mpath)) if os.path.exists(mpath) else None

    def __len__(self) -> int:
        return len(self.files)

    def trajectory(self, idx: int) -> Dict[str, Any]:
        d = np.load(self.files[idx])
        frames = d["frames"]
        if self.oracle:
            obs = frames.copy()
        else:
            bb, ng = d["ghost_bboxes"], d["n_ghosts"]
            obs = np.stack([apply_ghost_mask(frames[t], bb[t][:int(ng[t])])
                            for t in range(len(frames))], axis=0)
        out = {
            "obs": obs,
            "frames_unmasked": frames,
            "action": d["actions"],
            "achieved_goal": d["player_pos"],
            "done": d["done"],
            "target": d["target"],
            "episode_id": idx,
        }
        for k in ("ghost_near", "detour", "life_lost", "min_ghost_dist",
                  "n_ghosts", "ghost_bboxes", "rho"):
            if k in d.files:
                out[k] = d[k]
        return out

    def trajectories(self) -> Iterator[Dict[str, Any]]:
        for i in range(len(self.files)):
            yield self.trajectory(i)

    def steps(self) -> Iterator[Dict[str, Any]]:
        for i in range(len(self.files)):
            traj = self.trajectory(i)
            T = len(traj["action"])
            for t in range(T):
                yield {"episode_id": i, "t": t, "obs": traj["obs"][t],
                       "action": int(traj["action"][t]),
                       "achieved_goal": traj["achieved_goal"][t],
                       "done": bool(traj["done"][t]), "target": traj["target"][t]}
