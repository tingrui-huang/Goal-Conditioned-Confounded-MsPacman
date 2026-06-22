"""Minimal Pong policy evaluation -> (metrics dict, rollout video frames). NO TensorBoard here.

Runs the DIAMOND teacher in Pong for N complete games at a given sampling temperature and reports
the headline metrics. Goal = lead the opponent by GOAL_LEAD (+15) points (the deploy goal).
Returns plain numpy/dict + an optional (T,210,160,3) uint8 frame array for video; the caller
(notebook) does all TensorBoard plumbing. Temperature: 1.0 = raw policy, 2.0 = behavior policy,
<=0 = greedy argmax.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch

from .. import config as C
from ..teacher.load_teacher import TeacherPolicy, load_teacher, make_env

GOAL_LEAD = 15
ACTION_NAMES = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]
N_ACTIONS = len(ACTION_NAMES)


def _act(model, obs, h, temperature):
    with torch.no_grad():
        lg, _, h2 = model.predict_act_value(obs[:, C.TEACHER_OBS_SLICE], h)
    if temperature <= 0:
        a = int(lg.argmax(-1).item())                                  # greedy
    else:
        a = int(torch.distributions.Categorical(logits=lg / temperature).sample().item())
    return a, h2


def evaluate(n_episodes: int = 10, temperature: float = 1.0, seed: int = 40000,
             max_steps: int = 20000, video_max_frames: int = 900,
             device: str = "cpu", verbose: bool = True) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    model, _ = load_teacher(device=device)
    teacher = TeacherPolicy(model, device=device)
    env = make_env(replace(C.M1Config(), device=device), num_envs=1)

    returns, lengths, max_diffs, success, distances = [], [], [], [], []
    action_counts = np.zeros(N_ACTIONS, np.int64)
    video = None
    for ep in range(n_episodes):
        torch.manual_seed(seed + ep)
        obs, _ = env.reset(seed=[seed + ep]); h = teacher.initial_state(1)
        ag = op = t = 0; mx = 0; frames = []
        while True:
            if ep == 0 and len(frames) < video_max_frames:
                frames.append(np.asarray(env.env.render(), np.uint8))      # record the 1st game
            a, h = _act(model, obs, h, temperature); action_counts[a] += 1
            obs, rew, end, trunc, _ = env.step(a); r = float(rew.item())
            ag += int(r > 0); op += int(r < 0); mx = max(mx, ag - op); t += 1
            if bool((end | trunc).item()) or t >= max_steps:
                break
        returns.append(ag - op); lengths.append(t); max_diffs.append(mx)
        success.append(int(mx >= GOAL_LEAD)); distances.append(float(max(0, GOAL_LEAD - mx)))
        if ep == 0 and frames:
            video = np.stack(frames)
        if verbose:
            print(f"  [T={temperature}] ep {ep:2d}: return={ag - op:+d} len={t} maxlead={mx} "
                  f"goal={'reached' if mx >= GOAL_LEAD else 'no'}")
    env.close()

    ret = np.asarray(returns, float)
    metrics = {
        "episode_return": float(ret.mean()),
        "score_difference": float(ret.mean()),               # == episode_return in Pong (sum of ±1)
        "cumulative_reward": float(ret.mean()),              # mean total reward / game (same in Pong)
        "episode_length": float(np.mean(lengths)),
        "goal_success": float(np.mean(success)),             # fraction of games that reached +15 lead
        "goal_distance": float(np.mean(distances)),          # mean (15 - max_lead), 0 if reached
        "action_freq": (action_counts / max(action_counts.sum(), 1)).round(4).tolist(),
        "action_counts": action_counts.tolist(),
        "action_names": ACTION_NAMES,
        "n_episodes": n_episodes, "temperature": temperature, "goal_lead": GOAL_LEAD,
        "returns_per_ep": [int(x) for x in returns],
        "lengths_per_ep": [int(x) for x in lengths],
    }
    return metrics, video
