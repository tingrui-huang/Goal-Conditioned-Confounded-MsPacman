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

Performance: the whole resized dataset (~473MB at 84x84) is concatenated into
flat tensors that live ON `device`. Sampling is then a vectorized GPU index_select
with no per-step Python loop and no host->device frame copy -- the per-step CPU
overhead that otherwise caps GPU throughput is gone.
"""
import numpy as np
import torch

from seaquest_ccrl.models.sa_encoder import preprocess_frames


class HindsightSampler:
    def __init__(self, game, oracle: bool, cfg, device="cpu", rng=None, root=None):
        self.cfg = cfg
        self.device = device
        self.rng = rng or np.random.default_rng(cfg.seed)
        ds = game.make_dataset(root or game.data_root, oracle)

        frames, actions, goals, lengths = [], [], [], []
        for traj in ds.trajectories():
            frames.append(preprocess_frames(traj["obs"], cfg.frame_size))     # resize once
            actions.append(np.asarray(traj["action"], dtype=np.int64))
            goals.append(np.asarray(traj["achieved_goal"], dtype=np.float32))
            lengths.append(len(actions[-1]))
        self.lengths = np.asarray(lengths, dtype=np.int64)
        # start index of each episode inside the concatenated arrays
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)[:-1]]).astype(np.int64)
        self.n_ep = len(lengths)
        self.p_geom = 1.0 - cfg.gamma
        self.k_stack = max(1, int(getattr(cfg, "frame_stack", 1)))

        # Flat, device-resident tensors (no per-step host->device frame copy).
        self.frames = torch.from_numpy(np.concatenate(frames, axis=0)).to(device)   # (N,H,W,3) uint8
        self.actions = torch.from_numpy(np.concatenate(actions)).to(device)         # (N,) int64
        self.goals = torch.from_numpy(np.concatenate(goals, axis=0)).to(device)     # (N,2) float32 raw px

        # Frame-stack indices: for each global step i, the k frame indices ending at i,
        # CLAMPED to its episode start (no crossing episode boundaries). MASK-then-stack:
        # frames are already masked per-view, so stacking adds only Pac-Man motion.
        N = self.frames.shape[0]
        ep_start = np.repeat(self.offsets, self.lengths)            # (N,) episode start per index
        ar = np.arange(N)
        cols = [np.maximum(ep_start, ar - (self.k_stack - 1) + j) for j in range(self.k_stack)]
        self.stack_idx = torch.from_numpy(np.stack(cols, axis=1)).to(device)  # (N,k) oldest->newest
        lo = np.array([cfg.goal_x_lo, cfg.goal_y_lo], dtype=np.float32)
        hi = np.array([cfg.goal_x_hi, cfg.goal_y_hi], dtype=np.float32)
        self._goal_lo = torch.from_numpy(lo).to(device)
        self._goal_span = torch.from_numpy(hi - lo).to(device)

    def sample(self, B: int):
        """-> (frames uint8 (B,size,size,3), actions (B,), goals_norm (B,2)) on device."""
        ep = self.rng.integers(0, self.n_ep, size=B)
        t = self.rng.integers(0, self.lengths[ep])                 # uniform 0..T-1 per episode
        k = self.rng.geometric(self.p_geom, size=B)                # >= 1 (hindsight offset)
        fut = np.minimum(t + k, self.lengths[ep] - 1)
        gt = torch.from_numpy(self.offsets[ep] + t).to(self.device)   # global index of (s_t, a_t)
        gf = torch.from_numpy(self.offsets[ep] + fut).to(self.device)  # global index of future goal
        if self.k_stack == 1:
            frames = self.frames.index_select(0, gt)                  # (B,H,W,3)
        else:
            idx = self.stack_idx.index_select(0, gt)                  # (B,k)
            st = self.frames[idx]                                     # (B,k,H,W,3)
            B_, k_, H_, W_, C_ = st.shape
            frames = st.permute(0, 2, 3, 1, 4).reshape(B_, H_, W_, k_ * C_)  # (B,H,W,3k)
        actions = self.actions.index_select(0, gt)
        goals = (self.goals.index_select(0, gf) - self._goal_lo) / self._goal_span
        return frames, actions, goals
