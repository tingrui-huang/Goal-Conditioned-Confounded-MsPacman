"""Phase 2.1 — four-frame oxygen leakage probe.
How much oxygen is inferable from the MASKED four-frame state vs the VISIBLE state?
Matched small CNNs (masked / visible), continuous oxygen + low/med/high class, plus a
trivial no-image baseline (train mean / class prior). Episode-bootstrap CIs. Saves raw
predictions. Visible frames are used ONLY here, as the leakage control.
"""
import os, json, argparse
import numpy as np
import torch

from seaquest_ccrl.probes.oxy4_data import Phase2Data, oxygen_class, run_assertions
from seaquest_ccrl.probes import oxy4_train as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="seaquest_ccrl/data/raw_hf")
    ap.add_argument("--out-dir", default="artifacts/seaquest/oxygen_4frame/leakage")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(f"{args.out_dir}/figures", exist_ok=True)

    data = Phase2Data(args.root, load_visible=True, device=device)
    run_assertions(data, out=f"{args.out_dir}/../naive_critic/phase2_assertions.json")
    json.dump(data.manifest(), open(f"{args.out_dir}/split_manifest.json", "w"), indent=2)
    gi_tr, gi_va, gi_te = (data.split_indices(s) for s in ("train", "val", "test"))

    def yreg(gi): return data.oxygen[gi][:, None].astype(np.float32)
    def ycls(gi): return oxygen_class(data.oxygen[gi])

    res = {"n_test": int(len(gi_te))}
    raw = {}
    for view in ["masked", "visible"]:
        r = T.train_probe(data, view, gi_tr, gi_va, gi_te, None,
                          yreg(gi_tr), yreg(gi_va), yreg(gi_te), "reg", 1,
                          epochs=args.epochs, seed=args.seed, device=device)
        c = T.train_probe(data, view, gi_tr, gi_va, gi_te, None,
                          ycls(gi_tr), ycls(gi_va), ycls(gi_te), "clf", 3,
                          epochs=args.epochs, seed=args.seed, device=device)
        res[view] = {"regression": r["metrics"], "classification": c["metrics"],
                     "reg_mae_ci": T.boot_ci(np.abs(r["pred"][:, 0] - yreg(gi_te)[:, 0]), r["episode"]),
                     "clf_acc_ci": T.boot_ci((c["pred"].argmax(1) == ycls(gi_te)).astype(float), c["episode"])}
        raw[f"{view}_reg_pred"] = r["pred"]; raw[f"{view}_clf_pred"] = c["pred"]

    # trivial no-image baselines (train statistics)
    ytr = data.oxygen[gi_tr]; yte = data.oxygen[gi_te]
    base_mae = float(np.abs(yte - ytr.mean()).mean())
    base_r2 = 0.0
    prior = np.bincount(oxygen_class(ytr), minlength=3) / len(ytr)
    base_acc = float((oxygen_class(yte) == prior.argmax()).mean())
    base_ll = float(-np.log(np.clip(prior[oxygen_class(yte)], 1e-9, 1)).mean())
    res["trivial_baseline"] = {"reg_mae": base_mae, "reg_r2": base_r2,
                               "clf_acc": base_acc, "clf_log_loss": base_ll, "class_prior": prior.tolist()}

    # interpretation: visible-minus-masked, masked-minus-baseline, recoverable fraction
    mr2 = res["masked"]["regression"]["r2"]; vr2 = res["visible"]["regression"]["r2"]
    macc = res["masked"]["classification"]["accuracy"]; vacc = res["visible"]["classification"]["accuracy"]
    res["interpretation"] = {
        "visible_minus_masked_r2": vr2 - mr2,
        "masked_minus_baseline_r2": mr2 - base_r2,
        "recoverable_fraction_r2": (mr2 - base_r2) / (vr2 - base_r2) if (vr2 - base_r2) > 1e-6 else None,
        "visible_minus_masked_acc": vacc - macc,
        "masked_minus_baseline_acc": macc - base_acc,
        "recoverable_fraction_acc": (macc - base_acc) / (vacc - base_acc) if (vacc - base_acc) > 1e-6 else None,
    }
    np.savez_compressed(f"{args.out_dir}/predictions.npz",
                        oxygen_test=yte, oxygen_class_test=oxygen_class(yte),
                        episode_test=data.episode_of[gi_te], **raw)
    json.dump(res, open(f"{args.out_dir}/metrics.json", "w"), indent=2)
    print(f"[2.1 leakage] masked R2={mr2:.3f} acc={macc:.3f} | visible R2={vr2:.3f} acc={vacc:.3f} | "
          f"baseline mae={base_mae:.2f} acc={base_acc:.3f}")
    print(f"  recoverable fraction (R2)={res['interpretation']['recoverable_fraction_r2']} "
          f"(acc)={res['interpretation']['recoverable_fraction_acc']}")
    print(f"WROTE {args.out_dir}/metrics.json + predictions.npz")


if __name__ == "__main__":
    main()
