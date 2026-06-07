"""Game registry that makes the Level-2 contrastive pipeline game-agnostic.

Per the ccrl-seaquest-contrastive contract (obs = masked pixel frame, goal = 2D
position, oracle-for-free, eval in the real env) and the enemy-confounder skill's
"interface unchanged" rule, the ONLY game-specific bits are: the dataset loader,
the env, the naive-view mask at eval time, the action count, and the goal-position
normalisation box. A GameSpec bundles those so train_critic / evaluate /
run_naive_vs_oracle work for both Seaquest (oxygen) and Ms. Pac-Man (ghosts).
"""
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


@dataclass
class GameSpec:
    name: str
    make_dataset: Callable        # (root, oracle) -> dataset with .trajectories()
    make_env: Callable            # () -> env with reset()/step()/close()
    mask_obs: Callable            # (frame, state) -> naive (masked) frame
    nb_actions: int
    goal_box: Tuple[float, float, float, float]   # (x_lo, x_hi, y_lo, y_hi)
    target_box: Tuple[Tuple[int, int], Tuple[int, int]]  # (x_range, y_range) for eval
    eps: float
    data_root: str

    def sample_target(self, rng) -> np.ndarray:
        (x0, x1), (y0, y1) = self.target_box
        return np.array([rng.uniform(x0, x1), rng.uniform(y0, y1)], dtype=np.float32)


def _seaquest() -> GameSpec:
    from seaquest_ccrl import config as SC
    from seaquest_ccrl.data.dataset import SeaquestOfflineDataset
    from seaquest_ccrl.envs.seaquest_gc import SeaquestGCEnv
    from seaquest_ccrl.data.masking import apply_oxygen_mask
    return GameSpec(
        name="seaquest",
        make_dataset=lambda root, oracle: SeaquestOfflineDataset(root, oracle=oracle),
        make_env=lambda: SeaquestGCEnv(),
        mask_obs=lambda frame, state: apply_oxygen_mask(frame),
        nb_actions=SC.NB_ACTIONS,
        goal_box=(SC.TARGET_X_RANGE[0], SC.TARGET_X_RANGE[1],
                  SC.TARGET_Y_RANGE[0], SC.TARGET_Y_RANGE[1]),
        target_box=(SC.TARGET_X_RANGE, SC.TARGET_Y_RANGE),
        eps=SC.EPS,
        data_root=SC.DATA_ROOT,
    )


def _mspacman() -> GameSpec:
    from mspacman_ccrl import config as MC
    from mspacman_ccrl.data.dataset import MsPacmanOfflineDataset
    from mspacman_ccrl.envs.mspacman_gc import MsPacmanGCEnv
    from mspacman_ccrl.data.masking import apply_ghost_mask
    return GameSpec(
        name="mspacman",
        make_dataset=lambda root, oracle: MsPacmanOfflineDataset(root, oracle=oracle),
        make_env=lambda: MsPacmanGCEnv(),
        # naive view: inpaint the CURRENT ghost bboxes from the live env state
        mask_obs=lambda frame, state: apply_ghost_mask(frame, state.get("ghosts") or []),
        nb_actions=MC.NB_ACTIONS,
        goal_box=(MC.TARGET_X_RANGE[0], MC.TARGET_X_RANGE[1],
                  MC.TARGET_Y_RANGE[0], MC.TARGET_Y_RANGE[1]),
        target_box=(MC.TARGET_X_RANGE, MC.TARGET_Y_RANGE),
        eps=MC.EPS,
        data_root=MC.DATA_ROOT,
    )


_REGISTRY = {"seaquest": _seaquest, "mspacman": _mspacman}


def get_game(name: str) -> GameSpec:
    if name not in _REGISTRY:
        raise ValueError(f"unknown game {name!r}; choose from {list(_REGISTRY)}")
    return _REGISTRY[name]()
