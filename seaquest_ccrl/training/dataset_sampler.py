"""Batch construction with hindsight (geometric future) relabeling.

Wraps the Level-1 `SeaquestOfflineDataset`. Preloads every trajectory ONCE,
resizing frames to frame_size and keeping per-episode boundaries so future
goals never cross episodes. Passes the `oracle` flag straight through to the
loader -> masked (naive) vs unmasked (oracle) frames, SAME code otherwise
(acceptance check E).

Positive sampling (acceptance check D): for a transition (s_t, a_t) sample a
future offset k ~ Geometric(1 - gamma), set future = min(t + k, T - 1), and use
the achieved_goal (submarine position) at that future step as the positive goal.
Goals are positions, never pixels (check F).
"""
import numpy as np
import torch

from seaquest_ccrl.data.dataset import SeaquestOfflineDataset
from seaquest_ccrl.models.sa_encoder import preprocess_frames


class HindsightSampler:
    def __init__(self, root, oracle: bool, cfg, device="cpu", rng=None):
        self.cfg = cfg
        self.device = device
        self.rng = rng or np.random.default_rng(cfg.seed)
        ds = SeaquestOfflineDataset(root, oracle=oracle)

        self.frames = []        # list of (T,size,size,3) uint8 per episode
        self.actions = []       # list of (T,) int64
        self.goals = []         # list of (T,2) float32 raw pixel positions
        for traj in ds.trajectories():
            small = preprocess_frames(traj["obs"], cfg.frame_size)   # resize once
            self.frames.append(small)
            self.actions.append(np.asarray(traj["action"], dtype=np.int64))
            self.goals.append(np.asarray(traj["achieved_goal"], dtype=np.float32))
        self.lengths = np.array([len(a) for a in self.actions])
        self.n_ep = len(self.actions)
        self.p_geom = 1.0 - cfg.gamma

    def sample(self, B: int):
        """-> (frames uint8 (B,size,size,3), actions (B,), goals_norm (B,2)) tensors."""
        eps = self.rng.integers(0, self.n_ep, size=B)
        f = np.empty((B, self.cfg.frame_size, self.cfg.frame_size, 3), dtype=np.uint8)
        a = np.empty(B, dtype=np.int64)
        g = np.empty((B, 2), dtype=np.float32)
        for i, e in enumerate(eps):
            T = self.lengths[e]
            t = self.rng.integers(0, T)
            k = self.rng.geometric(self.p_geom)          # >= 1
            fut = min(t + k, T - 1)
            f[i] = self.frames[e][t]
            a[i] = self.actions[e][t]
            g[i] = self.goals[e][fut]
        g = self.cfg.normalize_goal(g)
        return (
            torch.from_numpy(f).to(self.device),
            torch.from_numpy(a).to(self.device),
            torch.from_numpy(g).to(self.device),
        )
