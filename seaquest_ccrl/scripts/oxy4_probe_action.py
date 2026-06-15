"""Phase 2.2 — conditional oxygen -> action.
Given the MASKED four-frame state, does current oxygen add information about the expert
action beyond the state? Matched pairs (the ONLY difference is the appended oxygen scalar):
  A-S : action ~ masked four-frame state
  A-SU: action ~ masked four-frame state + oxygen
Primary target = exact 18-action ID; secondary = S0.5 semantic category. Held-out log-loss
improvement + episode-bootstrap CI; stratified by player_y bin and oxygen bin (eval only).
"""
import os, json, argparse
import numpy as np
import torch

from seaquest_ccrl.probes.oxy4_data import Phase2Data, oxygen_class, ACTION_TO_CAT, CAT_NAMES, run_assertions
from seaquest_ccrl.probes import oxy4_train as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="seaquest_ccrl/data/raw_hf")
    ap.add_argument("--out-dir", default="artifacts/seaquest/oxygen_4frame/oxygen_to_action")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(f"{args.out_dir}/figures", exist_ok=True)

    data = Phase2Data(args.root, load_visible=False, device=device)
    run_assertions(data, out=f"{args.out_dir}/../naive_critic/phase2_assertions.json")
    gi_tr, gi_va, gi_te = (data.split_indices(s) for s in ("train", "val", "test"))
    oxy_extra = lambda gi: data.oxygen[gi][:, None].astype(np.float32)   # the ONLY added input

    def y_exact(gi): return data.actions[gi].astype(np.int64)
    def y_cat(gi): return np.array([ACTION_TO_CAT[int(a)] for a in data.actions[gi]], dtype=np.int64)

    res = {"n_test": int(len(gi_te)),
           "matched_pair_only_diff": "oxygen scalar appended to extras (state input identical)"}
    raw = {}
    for tgt_name, yf, nclass in [("exact_action", y_exact, 18), ("semantic_category", y_cat, 12)]:
        rS = T.train_probe(data, "masked", gi_tr, gi_va, gi_te, None,
                           yf(gi_tr), yf(gi_va), yf(gi_te), "clf", nclass,
                           epochs=args.epochs, seed=args.seed, device=device)
        rSU = T.train_probe(data, "masked", gi_tr, gi_va, gi_te, oxy_extra,
                            yf(gi_tr), yf(gi_va), yf(gi_te), "clf", nclass,
                            epochs=args.epochs, seed=args.seed, device=device)
        imp = T.paired_improvement_ci(rS["per_row_loss"], rSU["per_row_loss"], rS["episode"])
        res[tgt_name] = {
            "A_S": rS["metrics"], "A_SU": rSU["metrics"],
            "oxygen_logloss_improvement": imp,   # mean>0 & CI excludes 0 => oxygen helps
            "S_extra_dim": rS["extra_dim"], "SU_extra_dim": rSU["extra_dim"],
        }
        raw[f"{tgt_name}_S_pred"] = rS["pred"]; raw[f"{tgt_name}_SU_pred"] = rSU["pred"]
        raw[f"{tgt_name}_y"] = yf(gi_te)
        if tgt_name == "exact_action":
            # stratified log-loss improvement (eval only): player_y bin x oxygen bin
            py = data.player_pos[gi_te][:, 1]; oxy = data.oxygen[gi_te]
            py_bins = np.digitize(py, [60, 90, 120, 150]); ox_bins = oxygen_class(oxy)
            strat = {}
            for pb in range(5):
                for ob in range(3):
                    m = (py_bins == pb) & (ox_bins == ob)
                    if m.sum() >= 30:
                        d = (rS["per_row_loss"][m] - rSU["per_row_loss"][m]).mean()
                        strat[f"player_y_bin{pb}_oxy{ob}"] = {"n": int(m.sum()), "logloss_improvement": float(d)}
            res[tgt_name]["stratified_improvement"] = strat

    np.savez_compressed(f"{args.out_dir}/predictions.npz",
                        episode_test=data.episode_of[gi_te], oxygen_test=data.oxygen[gi_te],
                        player_y_test=data.player_pos[gi_te][:, 1], **raw)
    json.dump(res, open(f"{args.out_dir}/metrics.json", "w"), indent=2)
    ex = res["exact_action"]
    print(f"[2.2 U->A] exact: A-S logloss={ex['A_S']['log_loss']:.4f} acc={ex['A_S']['accuracy']:.3f} "
          f"top3={ex['A_S'].get('top3_acc')} | A-SU logloss={ex['A_SU']['log_loss']:.4f}")
    print(f"  oxygen logloss improvement={ex['oxygen_logloss_improvement']['mean']:+.4f} "
          f"CI{ex['oxygen_logloss_improvement']['ci95']} excludes0={ex['oxygen_logloss_improvement']['ci_excludes_0']}")
    print(f"WROTE {args.out_dir}/metrics.json + predictions.npz")


if __name__ == "__main__":
    main()
