"""Small on-policy T=2.0 sanity check (NOT a full M3 study).

Samples a few representative frames from actual T=2.0 rollouts, generates saliency
heatmaps, and repeats the opponent-removal probability ablation on those on-policy
snapshots — to confirm ball / paddles / opponent information remain behaviourally
relevant under the locked behaviour policy before M4 data collection.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from .. import config as C
from ..objects import extract_pong_objects
from ..teacher.load_teacher import TeacherPolicy, load_teacher, make_env
from .m3 import m3a_saliency
from .m36 import sampled_opp_dependence

ART = Path("artifacts/pong_action_gate/sanity_t2")


def collect_t2_snaps(policy, cfg, seeds: List[int], T: float,
                     snap_every: int = 9, max_snaps: int = 90) -> List[Dict[str, Any]]:
    model = policy.model
    snaps: List[Dict[str, Any]] = []
    for s in seeds:
        torch.manual_seed(s)
        env = make_env(replace(cfg, seed=s), num_envs=1)
        ale = env.env.ale
        obs, info = env.reset(seed=[s])
        hx, cx = policy.initial_state(1)
        t = 0
        while True:
            obs_slice = obs[:, C.TEACHER_OBS_SLICE, :, :]
            hx_in, cx_in = hx, cx
            with torch.no_grad():
                logits, _, (hx, cx) = model.predict_act_value(obs_slice, (hx_in, cx_in))
            action = torch.distributions.Categorical(logits=logits / T).sample()
            o = extract_pong_objects(ale.getRAM())
            if (t % snap_every == 0) and (len(snaps) < max_snaps):
                snaps.append({"obs_rgb": obs_slice.clone(), "hx": hx_in.clone(),
                              "cx": cx_in.clone(), "objects": o, "logits": logits.clone(),
                              "phase": "rally" if o.ball_present else "serve",
                              "seed": s, "t": t})
            obs, rew, end, trunc, info = env.step(action)
            t += 1
            if bool((end | trunc).item()) or t >= cfg.safety_step_cap:
                break
        env.close()
    return snaps


def run(seeds: List[int], T: float, n_frames: int, device: str = "cpu") -> Dict[str, Any]:
    model, meta = load_teacher(device=device)
    policy = TeacherPolicy(model, device=device)
    cfg = replace(C.M1Config(), device=device)
    ART.mkdir(parents=True, exist_ok=True)

    snaps = collect_t2_snaps(policy, cfg, seeds, T)
    sal = m3a_saliency(policy, snaps, n_frames=n_frames, save_dir=ART / "saliency", n_save=n_frames)
    oppdep = sampled_opp_dependence(policy, snaps, T)

    reg = sal["mean_saliency_by_region"]
    bg = reg.get("background") or 0.0
    confirm = {
        "ball_relevant": reg.get("ball") is not None and reg["ball"] > bg,
        "own_paddle_relevant": reg.get("own_paddle") is not None and reg["own_paddle"] > bg,
        "opp_paddle_relevant": reg.get("opp_paddle") is not None and reg["opp_paddle"] > bg,
        "opponent_removal_changes_behavior": (oppdep["coupled_action_disagree_rate"] or 0) > 0
                                             and (oppdep["mean_kl_bits"] or 0) > 0,
    }
    report = {
        "milestone": "sanity_t2 (on-policy)",
        "temperature": T,
        "seeds": seeds,
        "n_snapshots": len(snaps),
        "n_heatmaps": n_frames,
        "saliency_by_region": reg,
        "on_policy_opponent_removal_ablation": oppdep,
        "confirmations": confirm,
        "all_confirmed": all(confirm.values()),
        "teacher_meta": {k: meta[k] for k in ["ckpt_sha256", "teacher_source"]},
        "NOTE": "Small sanity check, not a full M3 study.",
    }
    with open(ART / "sanity_t2_report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="On-policy T=2.0 sanity check.")
    ap.add_argument("--base-seed", type=int, default=5000)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--temp", type=float, default=C.BEHAVIOR_TEMPERATURE)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    seeds = [args.base_seed + i for i in range(args.episodes)]
    report = run(seeds, args.temp, args.frames, device=args.device)
    print(json.dumps({k: report[k] for k in
                      ["n_snapshots", "saliency_by_region", "on_policy_opponent_removal_ablation",
                       "confirmations", "all_confirmed"]}, indent=2))


if __name__ == "__main__":
    main()
