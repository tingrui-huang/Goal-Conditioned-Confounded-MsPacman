"""Stage-G0 step 4 — full-view vs masked config diff + FULL_VIEW_CONFIG_DRIFT gate.

Both configs are derived from the SAME frozen code (get_game + TrainConfig + run_hf_4frame
defaults); the full-view critic differs from the masked four-frame critic ONLY by the
`oracle` flag (oxygen mask OFF) and the output checkpoint path. Any OTHER training-field
difference is config drift and stops the stage.

NOTE: the LOCAL artifacts/seaquest/oxygen_4frame/naive_critic/resolved_config.json is a
500-step smoke (ckpt_dir=_smoke4f); the authoritative masked critic trained 50000 steps in
Colab. The diff is therefore anchored on the DECLARED base config (contract + run_hf_4frame
defaults), and the stale smoke artifact is reported as an explicit note, not used as truth.
"""
import os, json, argparse
from dataclasses import asdict

from seaquest_ccrl.games import get_game
from seaquest_ccrl.training.config import TrainConfig

OUT = "artifacts/seaquest/goal_control/full_view"
MASKED_CKPT = "seaquest_ccrl/checkpoints/hf_4frame_seed0"
FULLVIEW_CKPT = "seaquest_ccrl/checkpoints/g0_full_view_seed0"
# the contract's declared base configuration (Stage-G0 section 3)
CONTRACT_BASE = {"frame_stack": 4, "frame_size": 84, "nb_actions": 18, "repr_dim": 256,
                 "batch_size": 256, "lr": 3e-4, "steps": 50000, "gamma": 0.99,
                 "goal_radius": 8.0, "seed": 0}


def base_cfg(ckpt_dir):
    """The frozen four-frame config exactly as run_hf_4frame builds it (steps=50000)."""
    game = get_game("seaquest")
    gx0, gx1, gy0, gy1 = game.goal_box
    return TrainConfig(steps=50000, seed=0, nb_actions=game.nb_actions,
                       goal_x_lo=gx0, goal_x_hi=gx1, goal_y_lo=gy0, goal_y_hi=gy1,
                       goal_radius=game.eps, frame_stack=4, ckpt_dir=ckpt_dir), game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg_m, game = base_cfg(MASKED_CKPT)
    cfg_f, _ = base_cfg(FULLVIEW_CKPT)
    dm, df = asdict(cfg_m), asdict(cfg_f)

    # field-by-field diff of the two training configs (oracle flag is NOT a TrainConfig field)
    OUTPUT_PATH_FIELDS = {"ckpt_dir"}
    training_diffs, output_diffs = {}, {}
    for k in dm:
        if dm[k] != df[k]:
            (output_diffs if k in OUTPUT_PATH_FIELDS else training_diffs)[k] = [dm[k], df[k]]
    observation_diffs = ["oxygen_mask ON->OFF (oracle False->True): masked critic loads "
                         "oracle=False (zeroed OXY_MASK_RECT); full-view loads oracle=True (unmasked)"]

    # the masked canonical config must equal the contract's declared base config
    base_check, base_mismatch = {}, []
    for k, exp in CONTRACT_BASE.items():
        act = dm.get(k)
        match = (abs(act - exp) < 1e-9) if isinstance(exp, float) else (act == exp)
        base_check[k] = {"expected": exp, "actual": act, "match": bool(match)}
        if not match:
            base_mismatch.append(k)

    # local smoke note
    local = "artifacts/seaquest/oxygen_4frame/naive_critic/resolved_config.json"
    note = {"exists": os.path.exists(local)}
    if note["exists"]:
        rc = json.load(open(local))
        note.update({"resolved_config_steps": rc.get("steps"), "ckpt_dir": rc.get("ckpt_dir"),
                     "is_smoke": rc.get("steps") != 50000,
                     "comment": "local naive_critic resolved_config is a smoke; authoritative masked "
                                "critic = 50000 steps (Colab). Diff anchored on declared base config."})

    unexpected = list(training_diffs.keys()) + base_mismatch
    outcome = "FULL_VIEW_CONFIG_DRIFT" if unexpected else "CONFIG_CLEAN"
    report = {
        "outcome": outcome,
        "masked": {"oracle": False, "oxygen_mask": "ON", "cfg": dm},
        "full_view": {"oracle": True, "oxygen_mask": "OFF", "cfg": df},
        "training_field_diffs_unexpected": training_diffs,
        "observation_masking_diffs": observation_diffs,
        "output_path_diffs": output_diffs,
        "contract_base_config_check": base_check,
        "contract_base_mismatch": base_mismatch,
        "local_naive_artifact_note": note,
        "intended_only_diffs": "observation masking (oxygen_mask OFF) + output checkpoint path",
    }
    # resolved full-view config (used by the trainer + audit)
    resolved = {**df, "oracle": True, "oxygen_mask": "OFF", "view": "full_view",
                "game": "seaquest", "goal_box": list(game.goal_box), "eps": game.eps}
    json.dump(report, open(f"{args.out_dir}/config_diff_vs_masked.json", "w"), indent=2)
    json.dump(resolved, open(f"{args.out_dir}/resolved_config.json", "w"), indent=2)
    print(json.dumps({"outcome": outcome, "unexpected_training_diffs": training_diffs,
                      "base_mismatch": base_mismatch, "output_path_diffs": output_diffs,
                      "observation_diffs": observation_diffs,
                      "local_smoke_note": note.get("is_smoke")}, indent=2))
    print(f"WROTE {args.out_dir}/config_diff_vs_masked.json + resolved_config.json")
    if outcome == "FULL_VIEW_CONFIG_DRIFT":
        raise SystemExit(4)


if __name__ == "__main__":
    main()
