"""Stage-1B cross-evaluation: 2x2 (trained-on) x (evaluated-on) per seed. NO retraining.

Stage-1B trained its 6 state critics in-memory and did not persist them. Training is fully
deterministic (fixed seed -> bit-identical weights), so on first run we REPRODUCE the
identical checkpoints once and save them; thereafter this script only LOADS and evaluates.

For each seed, both the uniform-trained and decision-focused-trained critic are evaluated on
both the uniform and the decision-focused anchor subsets built from the SAME held-out
episodes. All diagnostics are identical across the four cells.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from ..data.goal_sampler import duplicate_corrected_targets
from . import dataset as D
from .critics import StateSACritic, nce_loss
from .stage1b import _state_make_batch, filter_anchors, train_state

ART = Path("artifacts/pong_action_gate/stage1b")
CKPT_DIR = ART / "ckpts"


def load_or_reproduce(sampler: str, seed: int, all_eps, feats, n_episodes, steps, batch):
    """Load the deterministic Stage-1B critic; reproduce+save once if absent (NOT a new experiment)."""
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = CKPT_DIR / f"{sampler}_seed{seed}.pt"
    critic = StateSACritic(D.STATE_DIM, 6)
    if path.exists():
        critic.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        critic.eval()
        return critic, json.loads((CKPT_DIR / f"{sampler}_seed{seed}.json").read_text()), "loaded"
    decision = (sampler == "decision_focused")
    train_ids, _ = D.split_episodes(n_episodes, 0.2, seed)
    tr = filter_anchors(D.build_anchor_pool(all_eps, train_ids), all_eps, decision)
    val_ids = D.split_episodes(n_episodes, 0.2, seed)[1]
    va = filter_anchors(D.build_anchor_pool(all_eps, val_ids), all_eps, decision)
    critic, sel = train_state(tr, va, feats, seed, steps, batch)
    torch.save(critic.state_dict(), path)
    (CKPT_DIR / f"{sampler}_seed{seed}.json").write_text(json.dumps(sel))
    return critic, sel, "reproduced+saved"


def _ep_metrics(critic, ep_anchor_idx: np.ndarray, anchors, feats, B, rng):
    idx = rng.choice(ep_anchor_idx, size=min(B, len(ep_anchor_idx)), replace=False)
    obs, action, goaln, goals = _state_make_batch(anchors, idx, feats)
    tgt = torch.as_tensor(duplicate_corrected_targets(goals))
    with torch.no_grad():
        correct = float(nce_loss(critic.logits_matrix(obs, action, goaln), tgt))
        perm = torch.as_tensor(rng.permutation(len(action)))
        shuffled = float(nce_loss(critic.logits_matrix(obs, action[perm], goaln), tgt))
        randa = torch.as_tensor(rng.integers(6, size=len(action)).astype(np.int64))
        replaced = float(nce_loss(critic.logits_matrix(obs, randa, goaln), tgt))
        sa = critic.scores_all_actions(obs, goaln)
    return {"correct": correct, "shuffle_minus_correct": shuffled - correct,
            "replace_minus_correct": replaced - correct,
            "same_state_all_action_std": float(sa.std(1).mean())}


def evaluate(critic, eval_anchors, feats, seed, B=256, n_boot=2000, min_ep=8):
    by_ep: Dict[int, List[int]] = {}
    for i, a in enumerate(eval_anchors):
        by_ep.setdefault(a.ep, []).append(i)
    rng = np.random.default_rng(seed)
    per_ep = [_ep_metrics(critic, np.array(ix), eval_anchors, feats, B, rng)
              for ix in by_ep.values() if len(ix) >= min_ep]

    def ci(key):
        vals = np.array([m[key] for m in per_ep])
        br = np.random.default_rng(seed + 1)
        boots = [float(vals[br.integers(len(vals), size=len(vals))].mean()) for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"point": float(vals.mean()), "ci95": [float(lo), float(hi)], "ci_excludes_zero_pos": bool(lo > 0)}

    return {"n_episodes": len(per_ep), "n_anchors": len(eval_anchors),
            "correct_loss_mean": float(np.mean([m["correct"] for m in per_ep])),
            "shuffle_minus_correct": ci("shuffle_minus_correct"),
            "replace_minus_correct": ci("replace_minus_correct"),
            "same_state_all_action_std": ci("same_state_all_action_std")}


def run(n_episodes: int, seeds: List[int], steps: int, batch: int) -> Dict[str, Any]:
    all_eps = D.load_subset("full", list(range(n_episodes)), with_pixels=False)
    feats = [D.build_state_features(e) for e in all_eps]

    per_seed = {}
    materialized = {}
    for seed in seeds:
        # held-out episodes (matched per seed); build BOTH eval subsets from the SAME episodes
        val_ids = D.split_episodes(n_episodes, 0.2, seed)[1]
        eval_sets = {
            "uniform": filter_anchors(D.build_anchor_pool(all_eps, val_ids), all_eps, False),
            "decision_focused": filter_anchors(D.build_anchor_pool(all_eps, val_ids), all_eps, True)}
        critics = {}
        for sampler in ["uniform", "decision_focused"]:
            critics[sampler], _, how = load_or_reproduce(sampler, seed, all_eps, feats, n_episodes, steps, batch)
            materialized[f"{sampler}_seed{seed}"] = how
        cells = {}
        for trained in ["uniform", "decision_focused"]:
            for eval_on in ["uniform", "decision_focused"]:
                cells[f"trained={trained}|eval={eval_on}"] = evaluate(
                    critics[trained], eval_sets[eval_on], feats, seed)
        per_seed[f"seed{seed}"] = cells

    # interpretation flags (aggregate the qualitative reading; per-seed numbers stay separate)
    def cell(seed, tr, ev, key="shuffle_minus_correct"):
        return per_seed[f"seed{seed}"][f"trained={tr}|eval={ev}"][key]
    state_localized = all(cell(s, tr, "decision_focused")["point"] > cell(s, tr, "uniform")["point"]
                          for s in seeds for tr in ["uniform", "decision_focused"])
    focused_helps_on_focused = all(
        cell(s, "decision_focused", "decision_focused")["point"] >= cell(s, "uniform", "decision_focused")["point"]
        for s in seeds)
    focused_generalizes = all(
        cell(s, "decision_focused", "uniform")["point"] >= cell(s, "uniform", "uniform")["point"]
        for s in seeds)

    interpretation = []
    if state_localized:
        interpretation.append("Action relevance is STATE-LOCALIZED: both models are more action-sensitive "
                              "on decision-focused evaluation than on uniform evaluation (regardless of how "
                              "they were trained).")
    if focused_helps_on_focused:
        interpretation.append("Focused TRAINING genuinely improves learning of the sparse signal: the "
                              "decision-trained critic >= the uniform-trained critic on the SAME "
                              "decision-focused evaluation set.")
    if focused_generalizes:
        interpretation.append("Focused training GENERALIZES: it also improves action sensitivity on uniform "
                              "evaluation.")
    if not interpretation:
        interpretation.append("No consistent ordering across seeds; see per-seed cells.")

    out = {"milestone": "Stage-1B-cross-eval", "n_episodes": n_episodes, "seeds": seeds,
           "checkpoints": materialized,
           "note": "2x2 per seed = (trained-on) x (evaluated-on); both eval subsets from the SAME held-out "
                   "episodes; identical diagnostics; NO retraining beyond one-time deterministic reproduction.",
           "per_seed": per_seed,
           "interpretation_flags": {"state_localized": state_localized,
                                    "focused_training_helps_on_focused_eval": focused_helps_on_focused,
                                    "focused_training_generalizes_to_uniform_eval": focused_generalizes},
           "interpretation": interpretation}
    (ART / "stage1b_cross_eval.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-1B 2x2 checkpoint cross-evaluation (no retraining).")
    ap.add_argument("--n-episodes", type=int, default=80)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    out = run(args.n_episodes, args.seeds, args.steps, args.batch)
    print(json.dumps({"checkpoints": out["checkpoints"],
                      "interpretation_flags": out["interpretation_flags"],
                      "interpretation": out["interpretation"]}, indent=2))


if __name__ == "__main__":
    main()
