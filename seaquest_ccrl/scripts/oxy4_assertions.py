"""Phase-2 pre-probe assertions (run + save BEFORE any probe training).
Confirms: masked frame input (B,84,84,12); all four frames masked; visible frames used
only as the leakage control; action at t aligns with the stack ending at t; future targets
do not cross episode boundaries; future oxygen is NOT a target; and the exact split episode
IDs (seed 2606) that every Phase-2 experiment shares.
"""
import os, json, argparse
import torch

from seaquest_ccrl.probes.oxy4_data import Phase2Data, run_assertions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="seaquest_ccrl/data/raw_hf")
    ap.add_argument("--out", default="artifacts/seaquest/oxygen_4frame/naive_critic/phase2_assertions.json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = Phase2Data(args.root, load_visible=True, device=device)   # visible loaded so assertion 3 runs
    res = run_assertions(data, out=args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(data.manifest(), open(os.path.join(os.path.dirname(args.out), "phase2_split_manifest.json"), "w"), indent=2)
    print(json.dumps({k: res[k] for k in ["frame_shape_masked", "masked_oxygen_region_mean",
                                          "action_aligns_with_stack_end", "future_targets_no_boundary_cross",
                                          "future_targets", "split_seed", "assertions_pass"]}, indent=2))
    print(f"WROTE {args.out} + phase2_split_manifest.json")


if __name__ == "__main__":
    main()
