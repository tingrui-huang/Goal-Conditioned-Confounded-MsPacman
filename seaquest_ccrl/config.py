"""Central config for the Level-1 confounded goal-conditioned Seaquest build.

LOCKED ASSUMPTIONS (from the ccrl-seaquest-build skill; do not silently re-decide):
- Game ALE/Seaquest-v5, OCAtari mode="ram", render_mode="rgb_array", hud=True.
- Learner observation = masked pixel frame (210,160,3) uint8. NEVER an object vector.
- Confounder U = oxygen level, read from the OxygenBar object (filled width).
- Mask the FULL oxygen strip (filled+empty+label), not the filled width (Invariant 3).
- Behavior policy is scripted, oxygen-aware; THETA is the confounding-strength knob.
- Store UNMASKED frames by trajectory, time-ordered; mask is applied at LOAD time.
- ALE determinism: repeat_action_probability=0, fixed frameskip, seeded.
"""
from dataclasses import dataclass, field
from typing import Tuple


# --- ALE / OCAtari ---------------------------------------------------------
GAME_ID = "ALE/Seaquest-v5"
FRAMESKIP = 4
REPEAT_ACTION_PROBABILITY = 0.0   # sticky actions OFF (Invariant: no env-noise leak)
HUD = True
RENDER_MODE = "rgb_array"
MODE = "ram"
FRAME_SHAPE = (210, 160, 3)
NB_ACTIONS = 18                   # full Seaquest action set

# --- Oxygen confounder / mask ---------------------------------------------
# Measured empirically (scripts/explore_oxygen.py): OxygenBar(full) spans
# (x=49,y=170,w=63,h=5); OxygenBar ∪ OxygenBarDepleted union is the same strip.
# OXY_MASK_RECT pads that to cover the whole strip + OXYGEN label margin.
# Format: (x, y, w, h)  ->  rows [y:y+h], cols [x:x+w].
OXY_MASK_RECT: Tuple[int, int, int, int] = (46, 162, 69, 16)
OXY_FULL_WIDTH = 63               # OxygenBar width at full oxygen (for normalizing U)

# --- Behavior policy knobs -------------------------------------------------
# THETA = surfacing threshold in oxygen-width units (0..OXY_FULL_WIDTH).
# This is THE confounding-strength knob (U->A channel). Lower THETA = the policy
# tolerates lower oxygen before surfacing = weaker urgency coupling.
THETA = 20
MOVE_TOL = 4                      # px deadband per axis before issuing a move

# --- Goal labeling (method-side uses these later; Level 1 only stores labels) ---
EPS = 8.0                         # success radius (px); NOT evaluated at collection
# Underwater target-sampling box (px), below the surface line:
TARGET_X_RANGE = (24, 132)
TARGET_Y_RANGE = (52, 168)

# --- Collection ------------------------------------------------------------
SEED = 0
N_EPISODES = 40                   # dataset size (disk ~ N_EP * steps * 100KB/frame)
MAX_STEPS_PER_EP = 2000
DEPLETION_NOISE = False           # locked OFF
DATA_ROOT = "seaquest_ccrl/data/raw"

# === ENEMY CONFOUNDER (Level-1 v2; ccrl-seaquest-enemy-confounder) ==========
# Switch the confounder from oxygen to ENEMY POSITIONS. The enemy version masks
# hostile sprites (inpainted to water, NOT black-boxed) and uses a goal-seeking,
# enemy-avoiding demonstrator. See data/masking.apply_enemy_mask and
# policies/scripted_behavior.EnemyAvoidingPolicy.
ENEMY_DATA_ROOT = "seaquest_ccrl/data/raw_enemy"
# Hostile categories OCAtari exposes for Seaquest (verified by inspection).
HOSTILE_CATEGORIES = ("Shark", "Submarine", "SurfaceSubmarine", "EnemyMissile")
# Background water RGB (sampled from empty mid-water; uniform across the column).
WATER_COLOR = (0, 28, 136)
MAX_ENEMIES = 32                  # pad per-step enemy bbox arrays to this fixed width
# RHO = enemy avoidance radius (px) = THE enemy-confounding-strength knob (U->A).
# Larger RHO => more detours => more of the action distribution driven by hidden
# enemy positions. Sweep for the dose-response.
RHO = 28
# Oxygen mechanic: depletes ~0.12/step at depth but ACCELERATES (~5/step) below ~55,
# while ascent to the surface takes ~30 steps. So surface EARLY and commit to a full
# refill (hysteresis) or the sub drowns mid-ascent. Oxygen is VISIBLE => not confounded.
OXY_SURFACE_TRIGGER = 58          # enter surfacing mode when oxygen drops below this
OXY_REFILLED = 62                 # exit surfacing mode once oxygen back to (near) full
ENEMY_CONTACT_PX = 12.0           # player-enemy center distance counted as "contact"


@dataclass
class Config:
    game_id: str = GAME_ID
    frameskip: int = FRAMESKIP
    repeat_action_probability: float = REPEAT_ACTION_PROBABILITY
    hud: bool = HUD
    render_mode: str = RENDER_MODE
    mode: str = MODE
    oxy_mask_rect: Tuple[int, int, int, int] = OXY_MASK_RECT
    oxy_full_width: int = OXY_FULL_WIDTH
    theta: int = THETA
    move_tol: int = MOVE_TOL
    eps: float = EPS
    target_x_range: Tuple[int, int] = TARGET_X_RANGE
    target_y_range: Tuple[int, int] = TARGET_Y_RANGE
    seed: int = SEED
    n_episodes: int = N_EPISODES
    max_steps_per_ep: int = MAX_STEPS_PER_EP
    depletion_noise: bool = DEPLETION_NOISE
    data_root: str = DATA_ROOT


DEFAULT = Config()
