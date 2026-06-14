"""M7 Phase A.6-7 — ALE emulator clone/restore branch validation + clustered bootstrap.

At selected anchor states we snapshot the FULL resumable state (ALE system state +
AtariPreprocessing frame buffers + teacher LSTM hidden + scores + the critic inputs),
then for each forced first action run N continuations of the teacher to the next scoring
event, estimate the empirical outcome per first action, and compare the EMPIRICAL action
ranking to the critic's ranking. Bootstrap is clustered by cloned state.

This is evaluation only (kept separate from any counterfactual-training method).
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .. import config as C
from ..objects import PongObjects, extract_pong_objects
from ..teacher.load_teacher import TeacherPolicy, load_teacher, make_env
from . import dataset as D
from .train_critic import TrainConfig, load_ckpt, make_critic, resolve_device, run_dir


# --------------------------------------------------------------------------- #
# ALE + wrapper snapshot / restore
# --------------------------------------------------------------------------- #
def _ale_clone(ale):
    try:
        return ("system", ale.cloneSystemState())
    except Exception:
        return ("basic", ale.cloneState())


def _ale_restore(ale, snap):
    kind, st = snap
    (ale.restoreSystemState if kind == "system" else ale.restoreState)(st)


def snapshot_env(env, hx, cx, agent, opp, anchor_obs, gray_buf, cur_obj, prev_obj):
    w = env.env  # AtariPreprocessing
    return {
        "ale": _ale_clone(w.ale),
        "frames": copy.deepcopy(w.frames),
        "sebulba_frames": copy.deepcopy(w.sebulba_frames),
        "obs_buffer": [b.copy() for b in w.obs_buffer],
        "lives": w.lives, "game_over": w.game_over,
        "hx": hx.clone(), "cx": cx.clone(),
        "agent": int(agent), "opp": int(opp),
        "anchor_obs": anchor_obs.clone(),
        "gray_stack": np.stack(list(gray_buf)).astype(np.float32) / 255.0,  # (k,84,84)
        "cur_obj": cur_obj, "prev_obj": prev_obj,
        "score_diff_pre": int(agent - opp),
    }


def restore_env(env, snap):
    w = env.env
    _ale_restore(w.ale, snap["ale"])
    w.frames = copy.deepcopy(snap["frames"])
    w.sebulba_frames = copy.deepcopy(snap["sebulba_frames"])
    w.obs_buffer = [b.copy() for b in snap["obs_buffer"]]
    w.lives = snap["lives"]; w.game_over = snap["game_over"]
    return snap["hx"].clone(), snap["cx"].clone(), snap["agent"], snap["opp"]


# --------------------------------------------------------------------------- #
# State features for a single anchor (mirrors dataset.build_state_features)
# --------------------------------------------------------------------------- #
def _feat_single(cur: PongObjects, prev: Optional[PongObjects], score_diff_pre: int) -> np.ndarray:
    bx = cur.ball_x or 0.0; by = cur.ball_y or 0.0
    py = cur.player_y or 0.0; oy = cur.opp_y or 0.0
    bp = 1.0 if cur.ball_present else 0.0
    op = 1.0 if cur.opp_y is not None else 0.0

    def mot(c, p, ok):
        return float(np.clip(((c - p) if ok else 0.0) / 20.0, -1, 1))
    bdx = mot(bx, prev.ball_x if (prev and prev.ball_present and cur.ball_present) else bx,
              bool(prev and prev.ball_present and cur.ball_present))
    bdy = mot(by, prev.ball_y if (prev and prev.ball_present and cur.ball_present) else by,
              bool(prev and prev.ball_present and cur.ball_present))
    pdy = mot(py, prev.player_y if (prev and prev.player_y is not None) else py, bool(prev and prev.player_y is not None))
    ody = mot(oy, prev.opp_y if (prev and prev.opp_y is not None and cur.opp_y is not None) else oy,
              bool(prev and prev.opp_y is not None and cur.opp_y is not None))
    return np.array([bx/160.0, by/210.0, bp, bdx, bdy, py/210.0, pdy, oy/210.0, op, ody, score_diff_pre/21.0],
                    np.float32)


# --------------------------------------------------------------------------- #
# Continuation rollout from a restored state
# --------------------------------------------------------------------------- #
def _continue(env, teacher, snap, first_action: int, T: float, horizon: int, seed: int) -> int:
    """Force first_action, then teacher continues; return outcome in {-1,0,+1} (next scorer)."""
    torch.manual_seed(seed)
    hx, cx, ag, op = restore_env(env, snap)
    with torch.no_grad():
        _, _, (hx, cx) = teacher.model.predict_act_value(snap["anchor_obs"], (hx, cx))  # advance hidden (action-indep)
    obs, rew, end, trunc, info = env.step(int(first_action))
    r = float(rew.item())
    if r != 0:
        return int(np.sign(r))
    for _ in range(horizon):
        with torch.no_grad():
            logits, _, (hx, cx) = teacher.model.predict_act_value(obs[:, C.TEACHER_OBS_SLICE, :, :], (hx, cx))
            a = torch.distributions.Categorical(logits=logits / T).sample()
        obs, rew, end, trunc, info = env.step(a)
        r = float(rew.item())
        if r != 0:
            return int(np.sign(r))
        if bool((end | trunc).item()):
            break
    return 0


def clone_restore_repro_test(env, teacher, snap, T: float, horizon: int) -> Dict[str, Any]:
    """Restore + fixed seed + same forced action must reproduce the outcome bit-for-bit."""
    o1 = _continue(env, teacher, snap, first_action=2, T=T, horizon=horizon, seed=123)
    o2 = _continue(env, teacher, snap, first_action=2, T=T, horizon=horizon, seed=123)
    return {"outcome_run1": o1, "outcome_run2": o2, "reproducible": o1 == o2}


# --------------------------------------------------------------------------- #
# Collect branch states from a teacher rollout
# --------------------------------------------------------------------------- #
def collect_states(teacher, env, cfg_seed: int, n_states: int, stride: int, k: int = 4) -> List[Dict]:
    from collections import deque
    import cv2
    torch.manual_seed(cfg_seed)
    obs, info = env.reset(seed=[cfg_seed])
    ale = env.env.ale
    hx, cx = teacher.initial_state(1)
    gray_buf = deque(maxlen=k)

    def push_gray(info):
        raw = np.asarray(info["original_obs"]); raw = raw[0] if raw.ndim == 4 else raw
        g = cv2.resize(cv2.cvtColor(raw.astype(np.uint8), cv2.COLOR_RGB2GRAY), (84, 84), interpolation=cv2.INTER_AREA)
        gray_buf.append(g)
    push_gray(info)
    while len(gray_buf) < k:
        gray_buf.append(gray_buf[-1])

    snaps = []
    prev_obj = None
    ag = op = 0
    t = 0
    while len(snaps) < n_states:
        obs_slice = obs[:, C.TEACHER_OBS_SLICE, :, :]
        cur_obj = extract_pong_objects(ale.getRAM())
        if t > 0 and t % stride == 0:
            snaps.append(snapshot_env(env, hx, cx, ag, op, obs_slice, gray_buf, cur_obj, prev_obj))
        with torch.no_grad():
            logits, _, (hx, cx) = teacher.model.predict_act_value(obs_slice, (hx, cx))
            a = torch.distributions.Categorical(logits=logits / C.BEHAVIOR_TEMPERATURE).sample()
        prev_obj = cur_obj
        obs, rew, end, trunc, info = env.step(a)
        push_gray(info)
        r = float(rew.item())
        if r > 0: ag += 1
        elif r < 0: op += 1
        t += 1
        if bool((end | trunc).item()):
            obs, info = env.reset(seed=[cfg_seed + 1000 + t]); hx, cx = teacher.initial_state(1)
            gray_buf.clear(); push_gray(info)
            while len(gray_buf) < k: gray_buf.append(gray_buf[-1])
            prev_obj = None; ag = op = 0
    return snaps


# --------------------------------------------------------------------------- #
# Critic ranking + comparison
# --------------------------------------------------------------------------- #
def _critic_scores(critic, cfg, snap) -> np.ndarray:
    goal = snap["score_diff_pre"] + 1            # "agent scores next" goal
    goaln = torch.as_tensor(D.norm_goal(np.array([goal], np.float32)))
    if cfg.critic == "state":
        feat = _feat_single(snap["cur_obj"], snap["prev_obj"], snap["score_diff_pre"])
        obs = torch.as_tensor(feat[None])
    else:
        obs = torch.as_tensor(snap["gray_stack"][None])
    with torch.no_grad():
        return critic.scores_all_actions(obs, goaln)[0].numpy()


def _spearman(a, b):
    ar = np.argsort(np.argsort(a)); br = np.argsort(np.argsort(b))
    if ar.std() == 0 or br.std() == 0:
        return 0.0
    return float(np.corrcoef(ar, br)[0, 1])


def run(cfg: TrainConfig, n_states: int, n_cont: int, horizon: int, seed: int) -> Dict[str, Any]:
    device = resolve_device(cfg.device)
    d = run_dir(cfg)
    selected = json.loads((d / "selected.json").read_text())
    critic = make_critic(cfg).to(device)
    critic.load_state_dict(load_ckpt(d / selected["ckpt"], device)["model"]); critic.eval()

    teacher_model, _ = load_teacher(device=device)
    teacher = TeacherPolicy(teacher_model, device=device)
    env = make_env(replace(C.M1Config(), device=device), num_envs=1)

    snaps = collect_states(teacher, env, seed, n_states, stride=37)
    repro = clone_restore_repro_test(env, teacher, snaps[0], C.BEHAVIOR_TEMPERATURE, horizon)

    per_state = []
    for snap in snaps:
        emp = np.zeros(6)
        for a in range(6):
            outs = [_continue(env, teacher, snap, a, C.BEHAVIOR_TEMPERATURE, horizon, seed=1000 + a * 10 + c)
                    for c in range(n_cont)]
            emp[a] = float(np.mean(outs))
        cs = _critic_scores(critic, cfg, snap)
        top1 = int(np.argmax(cs) == np.argmax(emp))
        regret = float(emp.max() - emp[int(np.argmax(cs))])
        per_state.append({"empirical_outcome": emp.tolist(), "critic_scores": cs.tolist(),
                          "top1_agree": top1, "spearman": _spearman(cs, emp), "regret": regret})

    # clustered bootstrap by cloned state
    rng = np.random.default_rng(seed + 5)
    def boot(key):
        vals = np.array([p[key] for p in per_state])
        bs = [float(vals[rng.integers(len(vals), size=len(vals))].mean()) for _ in range(1000)]
        return {"point": float(vals.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}

    env.close()
    out = {
        "milestone": "M7-emulator-branch",
        "frozen_checkpoint": selected["ckpt"], "device": device,
        "n_states": len(per_state), "continuations_per_action": n_cont, "horizon": horizon,
        "ale_clone_kind": snaps[0]["ale"][0],
        "clone_restore_reproducibility": repro,
        "top1_agreement": boot("top1_agree"),
        "spearman_rank_correlation": boot("spearman"),
        "critic_action_regret": boot("regret"),
        "random_baseline_top1": 1.0 / 6,
        "per_state": per_state,
        "NOTE": "Evaluation only; dry-run metrics are NOT to be interpreted scientifically.",
    }
    (d / "emulator_branch.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="M7 ALE emulator-branch validation + clustered bootstrap.")
    ap.add_argument("--critic", type=str, default="state")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--n-states", type=int, default=4)
    ap.add_argument("--n-cont", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=300)
    args = ap.parse_args()
    cfg = TrainConfig(critic=args.critic, seed=args.seed, device=args.device)
    out = run(cfg, args.n_states, args.n_cont, args.horizon, args.seed)
    print(json.dumps({k: out[k] for k in
                      ["clone_restore_reproducibility", "ale_clone_kind", "top1_agreement",
                       "spearman_rank_correlation", "critic_action_regret", "random_baseline_top1"]}, indent=2))


if __name__ == "__main__":
    main()
