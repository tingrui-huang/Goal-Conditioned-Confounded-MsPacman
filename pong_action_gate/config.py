"""Frozen configuration for the Pong action-gate phase.

Every teacher-architecture / env field below was read directly out of the teacher
repo and is asserted at load time against the live environment (see load_teacher.py).
Sources (private teacher repo):
  - configs/agent.yaml  : actor_critic { lstm_dim:512, img_channels:3, img_size:${env.train.size},
                                         channels:[32,32,64,64], down:[1,1,1,1] }
  - configs/atari.yaml  : train/test { size:64, num_stack:4, done_on_life_loss:(train True / test False) }
  - envs/env.py         : make_atari_env(full_action_space=False, frameskip=1, frame_skip=4, noop_max=30)
  - envs/atari_preprocessing.py : obs = concat(num_stack grayscale ++ 1 colored RGB) -> (H,W,num_stack+3)
  - TorchEnv._to_tensor : obs normalized /255*2-1, permuted to (B,C,H,W) in [-1,1]
"""
from dataclasses import dataclass, field
from typing import List, Tuple


# --- HuggingFace source for the DIAMOND Atari-100k checkpoint -----------------
HF_REPO_ID = "eloialonso/diamond"
HF_CKPT_TEMPLATE = "atari_100k/models/{game}.pt"   # diamond.py: download(f"atari_100k/models/{name}.pt")

# --- Pong environment ---------------------------------------------------------
GAME = "Pong"
ENV_ID = "PongNoFrameskip-v4"
FULL_ACTION_SPACE = False                          # reduced set; asserted to give n==6
N_ACTIONS = 6                                      # NOOP,FIRE,RIGHT,LEFT,RIGHTFIRE,LEFTFIRE (verified live)
ACTION_MEANINGS = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]

NOOP_MAX = 30
FRAME_SKIP = 4                                     # AtariPreprocessing frame_skip
NUM_STACK = 4                                      # grayscale frames stacked in the obs
IMG_SIZE = 64                                      # env.train.size; teacher img_size
N_OBS_CHANNELS = NUM_STACK + 3                     # 4 grayscale + 3 colored RGB = 7

# Channel slice the DIAMOND teacher consumes from the (B,7,H,W) obs: the 3 colored
# RGB channels (verified: gen_saliency.py:97 / student/dqn.py:255 use obs[:, 4:, :, :]).
TEACHER_OBS_SLICE = slice(NUM_STACK, N_OBS_CHANNELS)   # == [:, 4:7, :, :]

# --- DIAMOND ActorCritic architecture (built directly; bypasses hydra) --------
LSTM_DIM = 512
TEACHER_IMG_CHANNELS = 3
TEACHER_CHANNELS: List[int] = field(default_factory=lambda: [32, 32, 64, 64])
TEACHER_DOWN: List[int] = field(default_factory=lambda: [1, 1, 1, 1])


@dataclass(frozen=True)
class TeacherArch:
    lstm_dim: int = LSTM_DIM
    img_channels: int = TEACHER_IMG_CHANNELS
    img_size: int = IMG_SIZE
    channels: Tuple[int, ...] = (32, 32, 64, 64)
    down: Tuple[int, ...] = (1, 1, 1, 1)
    num_actions: int = N_ACTIONS


@dataclass(frozen=True)
class M1Config:
    game: str = GAME
    env_id: str = ENV_ID
    seed: int = 0
    # test-mode env (no done-on-life-loss) is the evaluation/rollout config
    done_on_life_loss: bool = False
    max_episode_steps: int = 0          # 0 -> natural termination (None passed to env)
    safety_step_cap: int = 20000        # hard loop cap so a non-terminating ep can't hang
    device: str = "cpu"
    arch: TeacherArch = field(default_factory=TeacherArch)


# Final-task goal definition (recorded now; not exercised until later milestones).
GOAL_STAR = 15        # d_t = agent_score - opponent_score; success if max_t d_t >= +15
