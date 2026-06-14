"""M6.1 — pixel-overfit diagnostic (no M7, no full run, no new objective).

(1) Clarifies B×B sigmoid-BCE loss scaling so the ~40-170 numbers are interpretable.
(2) Fixed deterministic tiny-batch memorization test (SAME batch every step) for state
    and pixel critics, with proper ranking metrics (multi-positive aware).
(3) If pixel still fails, diagnoses the pixel pathway + a controlled synthetic fixture.
(4) Predeclares M7 checkpoint-selection discipline (recorded, not executed).

No auxiliary losses, reconstruction, reward prediction, or future information added.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from ..data.goal_sampler import duplicate_corrected_targets
from . import dataset as D
from .critics import PixelSACritic, StateSACritic, nce_loss

ART = Path("artifacts/pong_action_gate/m6_1")
DEVICE = "cpu"
THRESH = 0.0   # declared decision threshold on the logit (sigmoid 0.5)


def _t(x):
    return torch.as_tensor(x, device=DEVICE)


# --------------------------------------------------------------------------- #
# (1) loss scaling
# --------------------------------------------------------------------------- #
def loss_scaling(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    pos = targets > 0
    B = targets.shape[0]
    return {
        "batch_B": int(B),
        "target_positive_pair_fraction": float(targets.mean()),
        "loss_reduction_convention": "BCE per pair -> sum over candidate columns (dim=1) -> mean over anchor rows",
        "per_row_summed_BCE (the objective)": float(bce.sum(1).mean()),
        "per_pair_mean_BCE": float(bce.mean()),
        "mean_positive_pair_BCE": float(bce[pos].mean()),
        "mean_negative_pair_BCE": float(bce[~pos].mean()),
        "note": "per_row_summed_BCE == per_pair_mean_BCE * B (column sum).",
    }


# --------------------------------------------------------------------------- #
# (2) ranking metrics (multi-positive aware)
# --------------------------------------------------------------------------- #
def _quants(x):
    return {q: float(np.quantile(x, v)) for q, v in
            {"p05": .05, "p25": .25, "p50": .5, "p75": .75, "p95": .95}.items()}


def pair_metrics(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, Any]:
    L = logits.detach().numpy(); Tg = targets.detach().numpy()
    lab = Tg.reshape(-1).astype(int); sc = L.reshape(-1)
    pos = sc[lab == 1]; neg = sc[lab == 0]
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none").detach().numpy()
    # per-row ranking accuracy (each anchor: its target-positive cols vs its negatives)
    row_acc = []
    for i in range(L.shape[0]):
        p = L[i][Tg[i] == 1]; n = L[i][Tg[i] == 0]
        if len(p) and len(n):
            row_acc.append(float((p[:, None] > n[None, :]).mean()))
    return {
        "n_positive_pairs": int((lab == 1).sum()), "n_negative_pairs": int((lab == 0).sum()),
        "positive_logit": {"mean": float(pos.mean()), "median": float(np.median(pos)), **_quants(pos)},
        "negative_logit": {"mean": float(neg.mean()), "median": float(np.median(neg)), **_quants(neg)},
        "pos_minus_neg_gap": float(pos.mean() - neg.mean()),
        "AUROC": float(roc_auc_score(lab, sc)) if lab.min() != lab.max() else None,
        "AUPRC": float(average_precision_score(lab, sc)) if lab.min() != lab.max() else None,
        "positive_recall@thr0": float((pos > THRESH).mean()),
        "negative_specificity@thr0": float((neg <= THRESH).mean()),
        "per_positive_BCE": float(bce[Tg == 1].mean()),
        "per_negative_BCE": float(bce[Tg == 0].mean()),
        "per_row_ranking_accuracy": float(np.mean(row_acc)) if row_acc else None,
    }


def memorize(critic, obs, action, goaln, targets, steps: int, lr: float = 1e-3,
             eval_every: int = 250, plateau_tol: float = 1e-3) -> Dict[str, Any]:
    opt = torch.optim.Adam(critic.parameters(), lr=lr)
    curve = []
    last = None; plateau_at = None
    for s in range(steps + 1):
        logits = critic.logits_matrix(obs, action, goaln)
        loss = nce_loss(logits, targets)
        if s % eval_every == 0:
            curve.append({"step": s, "loss": float(loss.detach())})
            if last is not None and plateau_at is None and abs(last - float(loss.detach())) < plateau_tol:
                plateau_at = s
            last = float(loss.detach())
        if s < steps:
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        final_logits = critic.logits_matrix(obs, action, goaln)
    return {"learning_curve": curve, "plateau_step": plateau_at,
            "final_loss": float(nce_loss(final_logits, targets)),
            "metrics": pair_metrics(final_logits, targets)}


# --------------------------------------------------------------------------- #
# (3) pixel pathway diagnosis
# --------------------------------------------------------------------------- #
def pixel_diagnosis(critic: PixelSACritic, obs, action, goaln, targets) -> Dict[str, Any]:
    acts = {}
    hooks = []
    relu_idx = [i for i, m in enumerate(critic.conv) if isinstance(m, torch.nn.ReLU)]
    for i in relu_idx:
        def mk(i):
            def h(_m, _in, out): acts[i] = out.detach()
            return h
        hooks.append(critic.conv[i].register_forward_hook(mk(i)))

    critic.zero_grad(set_to_none=True)
    logits = critic.logits_matrix(obs, action, goaln)
    loss = nce_loss(logits, targets)
    loss.backward()
    for h in hooks:
        h.remove()

    conv_layers = [i for i, m in enumerate(critic.conv) if isinstance(m, torch.nn.Conv2d)]
    a_emb = critic.action_embed(F.one_hot(action.long(), critic.n_actions).float())
    g_emb = critic.psi(goaln.view(-1, 1))
    pos = targets > 0
    return {
        "input_4frame": {"min": float(obs.min()), "max": float(obs.max()),
                         "mean": float(obs.mean()), "std": float(obs.std())},
        "relu_activation_variance": {f"relu_{i}": float(acts[i].var()) for i in relu_idx},
        "dead_relu_fraction": {f"relu_{i}": float((acts[i] == 0).float().mean()) for i in relu_idx},
        "conv_grad_norms": {f"conv_{i}": float(critic.conv[i].weight.grad.norm()) for i in conv_layers},
        "action_embed_norm_mean": float(a_emb.norm(dim=1).mean()),
        "goal_embed_norm_mean": float(g_emb.norm(dim=1).mean()),
        "logit_distribution": {"min": float(logits.min()), "max": float(logits.max()),
                               "mean": float(logits.mean()), "std": float(logits.std())},
        "all_negative_collapse": {
            "frac_positive_targets_predicted_negative": float((logits[pos] < THRESH).float().mean()),
            "frac_all_logits_below_thr": float((logits < THRESH).float().mean()),
        },
    }


def synthetic_fixture(K: int = 8, steps: int = 2000) -> Dict[str, Any]:
    """K distinct random pixel tensors + distinct actions + distinct scalar goals.
    The architecture MUST be able to memorize this trivially separable fixture."""
    g = torch.Generator().manual_seed(0)
    frames = torch.rand(K, 4, 84, 84, generator=g)
    actions = _t(np.arange(K) % 6)
    goals = np.arange(K).astype(np.float32)            # distinct -> identity targets
    goaln = _t(((goals + 2) / 23).astype(np.float32))
    targets = _t(duplicate_corrected_targets(goals))
    critic = PixelSACritic(4, 6)
    res = memorize(critic, frames, actions, goaln, targets, steps, lr=1e-3, eval_every=400)
    return {"K": K, "final_loss": res["final_loss"], "metrics": res["metrics"],
            "learning_curve": res["learning_curve"],
            "verdict": "architecture CAN fit pixel->goal mapping"
            if res["metrics"]["per_row_ranking_accuracy"] and res["metrics"]["per_row_ranking_accuracy"] > 0.95
            else "architecture FAILS even on trivially separable fixture (investigate code/training)"}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_fixed_batch(tag: str, n: int, seed: int):
    ep_ids = list(range(8))
    eps = D.load_subset(tag, ep_ids, with_pixels=True)
    feats = [D.build_state_features(e) for e in eps]
    anchors = D.build_anchor_pool(eps, list(range(len(eps))))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(anchors), size=n, replace=False)
    s, a, g = D.state_batch(anchors, idx, feats)
    f, _, _ = D.pixel_batch(anchors, idx, eps)
    goaln = D.norm_goal(g)
    targets = duplicate_corrected_targets(g.astype(int))
    return {
        "state": (_t(s), _t(a.astype(np.int64)), _t(goaln)),
        "pixel": (_t(f), _t(a.astype(np.int64)), _t(goaln)),
        "targets": _t(targets), "goals_int": g.astype(int),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="M6.1 pixel-overfit diagnostic.")
    ap.add_argument("--tag", type=str, default="full")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    ART.mkdir(parents=True, exist_ok=True)

    fb = build_fixed_batch(args.tag, args.batch, args.seed)
    sobs, saction, sgoaln = fb["state"]
    pobs, paction, pgoaln = fb["pixel"]
    targets = fb["targets"]

    # (1) loss scaling (using an untrained state critic's logits just to show pair structure)
    sc = StateSACritic(D.STATE_DIM, 6); sc._init_args = (D.STATE_DIM, 6)
    pc = PixelSACritic(4, 6); pc._init_args = (4, 6)
    with torch.no_grad():
        scale = loss_scaling(sc.logits_matrix(sobs, saction, sgoaln), targets)

    # (2) fixed-batch memorization
    state_mem = memorize(StateSACritic(D.STATE_DIM, 6), sobs, saction, sgoaln, targets, args.steps)
    pixel_mem = memorize(PixelSACritic(4, 6), pobs, paction, pgoaln, targets, args.steps)

    report: Dict[str, Any] = {
        "milestone": "M6.1",
        "M6_conclusion_revision": (
            "M6 is a PARTIAL pass: data/sampler/split/targets/wiring and action-path GRADIENTS are "
            "validated, and the state-critic smoke is acceptable. Both action PATHWAYS are wired and "
            "produce only PRELIMINARY action-dependent changes; meaningful action sensitivity remains an "
            "M7 question requiring confidence intervals and emulator-branch validation. Pixel positive-goal "
            "discrimination was NOT sufficiently verified by the earlier tiny-overfit, which M6.1 addresses."),
        "fixed_batch": {"n": args.batch, "seed": args.seed, "steps": args.steps,
                        "same_batch_every_step": True, "resampling": False, "augmentation": False},
        "loss_scaling": scale,
        "state_critic_memorization": state_mem,
        "pixel_critic_memorization": pixel_mem,
        "checkpoint_selection_discipline_for_M7": {
            "rule": "Predeclare BEFORE M7: select the checkpoint by EITHER a fixed training-step budget OR "
                    "validation loss with a fixed early-stopping rule. Freeze it, THEN run action diagnostics "
                    "ONCE. Never select a checkpoint because its action-shuffle delta turned positive.",
        },
        "NOTE": "Diagnostic only. No M7, no emulator branch, no full pixel run. No auxiliary objective.",
    }

    # (3) pixel pathway diagnosis if pixel did not strongly overfit
    pix_ok = (pixel_mem["metrics"]["per_row_ranking_accuracy"] or 0) > 0.95
    report["pixel_strongly_overfit"] = bool(pix_ok)
    if not pix_ok:
        diag_critic = PixelSACritic(4, 6)
        # quick partial train so activations/grads reflect a learning model, not init
        memorize(diag_critic, pobs, paction, pgoaln, targets, steps=300, eval_every=300)
        report["pixel_pathway_diagnosis"] = pixel_diagnosis(diag_critic, pobs, paction, pgoaln, targets)
        report["synthetic_fixture"] = synthetic_fixture()

    with open(ART / "m6_1_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
