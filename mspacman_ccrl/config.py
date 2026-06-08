"""Config for the Ms. Pac-Man enemy(ghost)-confounder goal-conditioned build.

Ghost positions are the unobserved confounder U: they decide which corridor is
safe (U->A via the demonstrator's detours) and ghost contact = death (U->S',
frequent and dense -- the reason we moved here from Seaquest). Ghosts are masked
by inpainting their sprite to the local corridor background; Pac-Man, pills,
power-pills and fruit stay visible. Single frame (Markov). Oracle-for-free by
storing unmasked frames + per-step ghost bboxes.
"""
from dataclasses import dataclass
from typing import Tuple

# --- ALE / OCAtari ---------------------------------------------------------
GAME_ID = "ALE/MsPacman-v5"
FRAMESKIP = 4
REPEAT_ACTION_PROBABILITY = 0.0
HUD = True
RENDER_MODE = "rgb_array"
MODE = "ram"
FRAME_SHAPE = (210, 160, 3)
NB_ACTIONS = 9                      # MsPacman: NOOP + 8 directions (no fire)

# --- Ghost confounder / masking -------------------------------------------
GHOST_CATEGORY = "Ghost"
KEEP_VISIBLE = ("Player", "Pill", "PowerPill", "Fruit")  # never masked
# Verified by render: Pac-Man + ghosts move on BLUE; PINK is the wall structure.
WALL_COLOR = (228, 111, 111)       # pink maze walls (NOT walkable)
CORRIDOR_COLOR = (0, 28, 136)      # blue corridor floor (walkable; ghost inpaint ref)
MAX_GHOSTS = 8                     # pad per-step ghost bbox arrays (usually 4)

# --- Demonstrator ----------------------------------------------------------
# RHO = ghost avoidance radius (px) = confounding-strength knob (U->A). Larger RHO
# => more detours => more of the action distribution driven by hidden ghosts.
RHO = 26
MOVE_TOL = 3
WALL_PROBE_PX = 6                  # look-ahead distance to test a direction for a wall
GHOST_CONTACT_PX = 10.0           # player-ghost center distance counted as contact/death

# --- Goal labeling / target sampling --------------------------------------
EPS = 8.0                          # success radius (px)
# Target-SAMPLING box (where eval/collection draw desired goals).
TARGET_X_RANGE = (12, 148)
TARGET_Y_RANGE = (6, 168)
# Goal-NORMALISATION range for the g_encoder = ACTUAL player_pos min/max measured
# from the 100-ep dataset (NOT the target box; the player roams wider). Critic is
# sensitive to this normalisation -- using the target box squashes positions.
GOAL_X_RANGE = (3.5, 162.5)
GOAL_Y_RANGE = (8.0, 164.0)

# --- Collection ------------------------------------------------------------
SEED = 0
N_EPISODES = 40
MAX_STEPS_PER_EP = 2000
DATA_ROOT = "mspacman_ccrl/data/raw"


@dataclass
class Config:
    game_id: str = GAME_ID
    frameskip: int = FRAMESKIP
    repeat_action_probability: float = REPEAT_ACTION_PROBABILITY
    hud: bool = HUD
    render_mode: str = RENDER_MODE
    mode: str = MODE
    rho: float = RHO
    move_tol: int = MOVE_TOL
    wall_probe_px: int = WALL_PROBE_PX
    ghost_contact_px: float = GHOST_CONTACT_PX
    eps: float = EPS
    target_x_range: Tuple[int, int] = TARGET_X_RANGE
    target_y_range: Tuple[int, int] = TARGET_Y_RANGE
    seed: int = SEED
    n_episodes: int = N_EPISODES
    max_steps_per_ep: int = MAX_STEPS_PER_EP
    data_root: str = DATA_ROOT


DEFAULT = Config()
