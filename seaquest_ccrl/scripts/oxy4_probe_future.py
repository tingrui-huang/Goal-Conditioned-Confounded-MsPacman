"""Phase 2.3 — conditional oxygen -> future.
Given the MASKED four-frame state and current action, does oxygen add information about
FUTURE outcomes (NOT future oxygen) beyond state+action? Matched pairs (only difference =
appended oxygen scalar):
  F-SA : future ~ masked state + action
  F-SAU: future ~ masked state + action + oxygen
Horizons H = 16, 32, 64 agent decisions. Continuous targets (future player x/y, displacement
x/y, distance-toward-surface) + binary events (termination-before-H, refill-before-H, S0.5
def). Per-target held-out improvement + episode-bootstrap CI. Samples crossing an episode
boundary at +H are excluded.
"""
import os, json, argparse
import numpy as np
import torch

from seaquest_ccrl.probes.oxy4_data import Phase2Data, run_assertions
from seaquest_ccrl.probes import oxy4_train as T

HORIZONS = [16, 32, 64]
CONT = ["future_player_x", "future_player_y", "displacement_x", "displacement_y", "distance_toward_surface"]
BINARY = ["termination_before_H", "refill_before_H"]


def onehot(a, n=18):
    o = np.zeros((len(a), n), dtype=np.float32); o[np.arange(len(a)), a.astype(int)] = 1.0
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="seaquest_ccrl/data/raw_hf")
    ap.add_argument("--out-dir", default="artifacts/seaquest/oxygen_4frame/oxygen_to_future")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizons", default="16,32,64")
    ap.add_argument("--targets", default="all")  # 'all' or comma list (smoke)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(f"{args.out_dir}/predictions", exist_ok=True)
    os.makedirs(f"{args.out_dir}/figures", exist_ok=True)
    Hs = [int(x) for x in args.horizons.split(",")]
    cont = CONT if args.targets == "all" else [t for t in CONT if t in args.targets.split(",")]
    binr = BINARY if args.targets == "all" else [t for t in BINARY if t in args.targets.split(",")]

    data = Phase2Data(args.root, load_visible=False, device=device)
    run_assertions(data, out=f"{args.out_dir}/../naive_critic/phase2_assertions.json")

    for H in Hs:
        # valid (in-episode) rows per split
        def valid_split(s):
            gi = data.split_indices(s); _, ok = data.future_index(gi, H); return gi[ok]
        gi_tr, gi_va, gi_te = valid_split("train"), valid_split("val"), valid_split("test")
        Ttr, _ = data.future_targets(data.split_indices("train"), H)
        Tva, _ = data.future_targets(data.split_indices("val"), H)
        Tte, _ = data.future_targets(data.split_indices("test"), H)
        sa = lambda gi: onehot(data.actions[gi])                      # F-SA extra = action one-hot
        sau = lambda gi: np.concatenate([onehot(data.actions[gi]), data.oxygen[gi][:, None].astype(np.float32)], 1)
        out = {"H": H, "n_test": int(len(gi_te)),
               "matched_pair_only_diff": "oxygen scalar appended to extras (state+action input identical)",
               "continuous": {}, "binary": {}}
        rawp = {}

        if cont:
            Yc_tr = np.stack([Ttr[k] for k in cont], 1).astype(np.float32)
            Yc_va = np.stack([Tva[k] for k in cont], 1).astype(np.float32)
            Yc_te = np.stack([Tte[k] for k in cont], 1).astype(np.float32)
            rS = T.train_probe(data, "masked", gi_tr, gi_va, gi_te, sa, Yc_tr, Yc_va, Yc_te, "reg", len(cont),
                               epochs=args.epochs, seed=args.seed, device=device)
            rSU = T.train_probe(data, "masked", gi_tr, gi_va, gi_te, sau, Yc_tr, Yc_va, Yc_te, "reg", len(cont),
                                epochs=args.epochs, seed=args.seed, device=device)
            ep = rS["episode"]
            for j, k in enumerate(cont):
                eS = (rS["pred"][:, j] - Yc_te[:, j]) ** 2; eSU = (rSU["pred"][:, j] - Yc_te[:, j]) ** 2
                sstot = ((Yc_te[:, j] - Yc_te[:, j].mean()) ** 2).mean() + 1e-9
                out["continuous"][k] = {
                    "r2_SA": float(1 - eS.mean() / sstot), "r2_SAU": float(1 - eSU.mean() / sstot),
                    "mse_SA": float(eS.mean()), "mse_SAU": float(eSU.mean()),
                    "oxygen_improvement_mse_reduction": T.boot_ci(eS - eSU, ep)}
            rawp["cont_pred_SA"] = rS["pred"]; rawp["cont_pred_SAU"] = rSU["pred"]; rawp["cont_y"] = Yc_te

        for k in binr:
            ytr = Ttr[k].astype(np.int64); yva = Tva[k].astype(np.int64); yte = Tte[k].astype(np.int64)
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                out["binary"][k] = {"skipped": "degenerate labels"}; continue
            rS = T.train_probe(data, "masked", gi_tr, gi_va, gi_te, sa, ytr, yva, yte, "clf", 2,
                               epochs=args.epochs, seed=args.seed, device=device)
            rSU = T.train_probe(data, "masked", gi_tr, gi_va, gi_te, sau, ytr, yva, yte, "clf", 2,
                                epochs=args.epochs, seed=args.seed, device=device)
            out["binary"][k] = {"SA": rS["metrics"], "SAU": rSU["metrics"],
                                "oxygen_logloss_improvement": T.paired_improvement_ci(rS["per_row_loss"], rSU["per_row_loss"], rS["episode"])}
            rawp[f"bin_{k}_pred_SA"] = rS["pred"]; rawp[f"bin_{k}_pred_SAU"] = rSU["pred"]; rawp[f"bin_{k}_y"] = yte

        np.savez_compressed(f"{args.out_dir}/predictions/H{H}.npz", episode_test=data.episode_of[gi_te], **rawp)
        json.dump(out, open(f"{args.out_dir}/metrics_H{H}.json", "w"), indent=2)
        msg = " ".join(f"{k}:dR2={out['continuous'][k]['r2_SAU']-out['continuous'][k]['r2_SA']:+.3f}" for k in cont)
        print(f"[2.3 U->future H={H}] n={len(gi_te)} {msg}")
        for k in binr:
            b = out["binary"][k]
            if "oxygen_logloss_improvement" in b:
                print(f"  {k}: logloss_improvement={b['oxygen_logloss_improvement']['mean']:+.4f} "
                      f"CI{b['oxygen_logloss_improvement']['ci95']} excl0={b['oxygen_logloss_improvement']['ci_excludes_0']}")
        print(f"  WROTE metrics_H{H}.json")


if __name__ == "__main__":
    main()
