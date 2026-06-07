"""Loader for the by-trajectory offline Seaquest dataset.

Yields per-step samples:
    obs           : masked pixel frame (210,160,3) uint8   <- the LEARNER observation
    action        : int
    achieved_goal : player_pos (x,y)  <- env-side label, NOT part of obs
    oxygen        : int (confounder; available for analysis, NOT in obs)
    done          : bool
    target        : (x,y) per-episode desired-goal hint (method-side may ignore)

`oracle=True` skips the mask -> unmasked frames (the naive/oracle comparison view),
recovered for free from the same storage. Trajectory order + boundaries are preserved
(Invariant 5): iterate `.trajectories()` for ordered per-episode steps, or `.steps()`
for a flat ordered stream that still carries episode ids + done flags.

This loader does NOT sample hindsight positives or future states -- that is method-side
(out of scope). It only preserves the structure so those are *possible* later.
"""
import os
import glob
import json
from typing import Iterator, Dict, Any, List, Optional

import numpy as np

from seaquest_ccrl import config as C
from seaquest_ccrl.data.masking import apply_oxygen_mask, apply_enemy_mask, oracle as _oracle


class SeaquestOfflineDataset:
    def __init__(self, root: str = None, oracle: bool = False,
                 mask_rect=None):
        self.root = root or C.DATA_ROOT
        self.oracle = oracle
        self.mask_rect = mask_rect if mask_rect is not None else C.OXY_MASK_RECT
        self.files: List[str] = sorted(glob.glob(os.path.join(self.root, "traj_*.npz")))
        if not self.files:
            raise FileNotFoundError(f"No trajectories found under {self.root!r}. "
                                    f"Run collect/collect_dataset.py first.")
        mpath = os.path.join(self.root, "manifest.json")
        self.manifest = json.load(open(mpath)) if os.path.exists(mpath) else None
        # Auto-detect confounder type so the SAME loader serves oxygen (v1) and
        # enemy (v2) datasets with an identical interface.
        self.mode = "oxygen"
        if self.manifest and self.manifest.get("confounder") == "enemy":
            self.mode = "enemy"
        else:
            with np.load(self.files[0]) as d0:
                if "enemy_bboxes" in d0.files:
                    self.mode = "enemy"

    def __len__(self) -> int:
        return len(self.files)

    def _obs(self, frame: np.ndarray) -> np.ndarray:
        return _oracle(frame) if self.oracle else apply_oxygen_mask(frame, self.mask_rect)

    def trajectory(self, idx: int) -> Dict[str, Any]:
        """Load one trajectory; obs frames are masked (or oracle) per dataset mode.

        enemy mode: inpaint each stored enemy bbox to water (naive) or skip (oracle).
        oxygen mode: zero the oxygen-bar rect (naive) or skip (oracle).
        """
        d = np.load(self.files[idx])
        frames = d["frames"]
        if self.mode == "enemy":
            if self.oracle:
                obs = frames.copy()
            else:
                bb, ne = d["enemy_bboxes"], d["n_enemies"]
                obs = np.stack([apply_enemy_mask(frames[t], bb[t][:int(ne[t])])
                                for t in range(len(frames))], axis=0)
        else:
            obs = np.stack([self._obs(f) for f in frames], axis=0)

        out = {
            "obs": obs,                       # (T,210,160,3) masked or oracle
            "frames_unmasked": frames,        # (T,210,160,3) always available
            "action": d["actions"],
            "achieved_goal": d["player_pos"],  # goal label (interface unchanged)
            "oxygen": d["oxygen"],
            "done": d["done"],
            "target": d["target"],
            "episode_id": idx,
        }
        # carry analysis-only fields when present (oxygen v1 or enemy v2)
        for k in ("enemy_near", "detour", "life_lost", "min_enemy_dist",
                  "n_enemies", "enemy_bboxes", "rho"):
            if k in d.files:
                out[k] = d[k]
        out["theta"] = int(d["theta"]) if "theta" in d.files else None
        return out

    def trajectories(self) -> Iterator[Dict[str, Any]]:
        for i in range(len(self.files)):
            yield self.trajectory(i)

    def steps(self) -> Iterator[Dict[str, Any]]:
        """Flat, time-ordered stream; each step keeps episode_id + t + done."""
        for i in range(len(self.files)):
            traj = self.trajectory(i)
            T = len(traj["action"])
            for t in range(T):
                yield {
                    "episode_id": i,
                    "t": t,
                    "obs": traj["obs"][t],
                    "action": int(traj["action"][t]),
                    "achieved_goal": traj["achieved_goal"][t],
                    "oxygen": int(traj["oxygen"][t]),
                    "done": bool(traj["done"][t]),
                    "target": traj["target"][t],
                }
