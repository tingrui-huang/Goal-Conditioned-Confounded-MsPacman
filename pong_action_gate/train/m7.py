"""M7 orchestration: local CPU dry run + per-seed / aggregate reporting.

Phase B (dry run) ONLY. Does NOT launch the full multi-seed experiment. Dry-run action
metrics are NOT to be interpreted scientifically — the dry run verifies plumbing.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from . import dataset as D
from . import emulator_branch, gate_diag
from .train_critic import (CKPT_ROOT, TrainConfig, latest_ckpt, load_ckpt,
                           resolve_device, run_dir, train)


def dry_run(seed: int, device: str) -> Dict[str, Any]:
    cfg = TrainConfig(critic="state", seed=seed, n_episodes=12, val_frac=0.25,
                      steps=300, batch=128, eval_every=100, ckpt_every=100,
                      val_batches=2, device=device)
    # start the resume test from a clean run dir (avoid stale checkpoints confounding it)
    import shutil
    shutil.rmtree(run_dir(cfg), ignore_errors=True)
    checks: Dict[str, Any] = {}

    # 1. device selection (CPU + CUDA)
    checks["device_selection"] = {
        "requested": device, "resolved_cpu": resolve_device("cpu"),
        "resolved_cuda": resolve_device("cuda"), "cuda_available": bool(torch.cuda.is_available())}

    # 2. deterministic episode splits
    s1 = D.split_episodes(cfg.n_episodes, cfg.val_frac, cfg.seed)
    s2 = D.split_episodes(cfg.n_episodes, cfg.val_frac, cfg.seed)
    checks["deterministic_split"] = {"equal_across_calls": s1 == s2, "train": s1[0], "val": s1[1]}

    # 3. train + predeclared val-loss checkpoint selection (no action diagnostics)
    tr = train(cfg)
    checks["checkpoint_selection"] = {
        "rule": tr["selected"]["rule"], "selected_step": tr["selected"]["selected_step"],
        "consulted_action_diagnostics": False, "val_curve_points": len(tr["val_curve"])}

    # 4. checkpoint save/load + resume
    d = run_dir(cfg)
    before = load_ckpt(latest_ckpt(d))["step"]
    tr2 = train(replace(cfg, steps=cfg.steps + 200), resume=True)
    after = load_ckpt(latest_ckpt(d))["step"]
    checks["resume"] = {"step_before_resume": int(before), "step_after_resume": int(after),
                        "resumed_and_advanced": after > before,
                        "reselected_step": tr2["selected"]["selected_step"]}

    # 5+6. frozen-checkpoint action diagnostics + episode-level bootstrap
    gd = gate_diag.run(cfg, n_boot=500, per_ep_batch=128, seed=seed)
    checks["frozen_action_diagnostics"] = {
        "frozen_checkpoint": gd["frozen_checkpoint"],
        "n_val_episodes": gd["n_val_episodes_used"],
        "bootstrap_ran": all("ci95" in v for v in gd["episode_level_bootstrap_ci"].values()),
        "shuffle_ci95": gd["episode_level_bootstrap_ci"]["shuffle_minus_correct"]["ci95"]}

    # 7+8. ALE clone/restore branch + clustered bootstrap
    eb = emulator_branch.run(cfg, n_states=4, n_cont=2, horizon=300, seed=seed)
    checks["emulator_branch"] = {
        "ale_clone_kind": eb["ale_clone_kind"],
        "clone_restore_reproducible": eb["clone_restore_reproducibility"]["reproducible"],
        "n_states": eb["n_states"], "top1_ci95": eb["top1_agreement"]["ci95"]}

    # 9. artifact + restart safety
    arts = ["selected.json", "val_curve.json", "gate_diagnostics.json", "emulator_branch.json"]
    checks["artifacts_written"] = {a: (d / a).exists() for a in arts}

    all_ok = (checks["deterministic_split"]["equal_across_calls"]
              and checks["resume"]["resumed_and_advanced"]
              and checks["frozen_action_diagnostics"]["bootstrap_ran"]
              and checks["emulator_branch"]["clone_restore_reproducible"]
              and all(checks["artifacts_written"].values()))

    return {"milestone": "M7-dry-run", "seed": seed, "config": asdict(cfg),
            "checks": checks, "all_plumbing_ok": bool(all_ok),
            "WARNING": "Dry-run action metrics are NOT scientific; verifies plumbing only."}


def aggregate(seeds: List[int], critic: str = "state") -> Dict[str, Any]:
    """Per-seed + aggregate reporting from already-produced artifacts (Phase A.8)."""
    per_seed = {}
    for s in seeds:
        d = CKPT_ROOT / f"{critic}_seed{s}"
        gd = json.loads((d / "gate_diagnostics.json").read_text()) if (d / "gate_diagnostics.json").exists() else None
        eb = json.loads((d / "emulator_branch.json").read_text()) if (d / "emulator_branch.json").exists() else None
        per_seed[s] = {
            "shuffle_minus_correct": gd["episode_level_bootstrap_ci"]["shuffle_minus_correct"] if gd else None,
            "top1_agreement": eb["top1_agreement"] if eb else None}
    pts = [v["shuffle_minus_correct"]["point"] for v in per_seed.values() if v["shuffle_minus_correct"]]
    return {"per_seed": per_seed,
            "aggregate_shuffle_minus_correct": {
                "mean": float(np.mean(pts)) if pts else None,
                "across_seed_std": float(np.std(pts)) if len(pts) > 1 else None,
                "n_seeds": len(pts)}}


def main() -> None:
    ap = argparse.ArgumentParser(description="M7 local dry run + aggregation.")
    ap.add_argument("--mode", choices=["dry-run", "aggregate"], default="dry-run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()
    CKPT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.mode == "dry-run":
        out = dry_run(args.seed, args.device)
        (CKPT_ROOT / "dry_run_report.json").write_text(json.dumps(out, indent=2))
    else:
        out = aggregate(args.seeds)
        (CKPT_ROOT / "aggregate_report.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
