"""Level-2 contrastive-RL hyperparameters (Eysenbach et al. 2022, Algorithm 1).

These are METHOD-side knobs, separate from the Level-1 collection config
(`seaquest_ccrl/config.py`). The naive and oracle runs share this config
verbatim — the ONLY difference between them is the `oracle` flag passed to the
dataset loader (masked vs unmasked oxygen bar). See acceptance check E.
"""
from dataclasses import dataclass

from seaquest_ccrl import config as C


@dataclass
class TrainConfig:
    # --- contrastive critic ---------------------------------------------
    repr_dim: int = 256            # d: ϕ and ψ embedding dimension
    frame_size: int = 84           # resize 210x160 -> frame_size x frame_size
    nb_actions: int = C.NB_ACTIONS  # 18 (full Seaquest action set, one-hot)
    frame_stack: int = 1           # k stacked frames (MsPacman=4 to fix pixel-localization
                                   # /action-blindness). MASK-then-STACK: ghosts inpainted in
                                   # every frame for naive, so only Pac-Man motion is visible
                                   # (no ghost-motion leak). 1 => single-frame (Seaquest).

    # --- optimisation ----------------------------------------------------
    batch_size: int = 256          # B positives + (B-1) in-batch negatives each
    lr: float = 3e-4               # Adam
    steps: int = 50000             # training iterations (skill target 100K; CPU-bound)
    gamma: float = 0.99            # discount for geometric future sampling
    goal_radius: float = 0.0       # >0: in-batch goals within this px radius count as
                                   # positives (goal-collision fix). 0 => exact-match (eye).

    # --- logging / eval --------------------------------------------------
    log_every: int = 200
    eval_every: int = 0            # 0 => no in-loop env eval (CPU-cheap); final eval in script
    eval_episodes: int = 50
    ckpt_dir: str = "seaquest_ccrl/checkpoints"
    seed: int = 0

    # --- goal-position normalisation ------------------------------------
    # Map (x, y) pixel position -> ~[0,1] using the Level-1 target-sampling box
    # so g_encoder inputs are well-scaled. Used identically at train and eval.
    goal_x_lo: float = C.TARGET_X_RANGE[0]
    goal_x_hi: float = C.TARGET_X_RANGE[1]
    goal_y_lo: float = C.TARGET_Y_RANGE[0]
    goal_y_hi: float = C.TARGET_Y_RANGE[1]

    def normalize_goal(self, xy):
        """(x, y) pixels -> normalized [0,1]^2 (numpy or torch broadcast-safe)."""
        import numpy as np
        xy = np.asarray(xy, dtype=np.float32)
        lo = np.array([self.goal_x_lo, self.goal_y_lo], dtype=np.float32)
        hi = np.array([self.goal_x_hi, self.goal_y_hi], dtype=np.float32)
        return (xy - lo) / (hi - lo)


DEFAULT = TrainConfig()
