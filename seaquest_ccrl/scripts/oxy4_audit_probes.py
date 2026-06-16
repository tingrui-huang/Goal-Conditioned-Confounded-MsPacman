"""Phase 2.1b steps 4-7 — matched visual probes (V1-V4) + non-image proxy baselines
(P1-P3) + sanity controls, all on the FROZEN seed-2606 episode split with matched
optimizer/steps/seed/metrics/bootstrap. Only the visual input differs across V1-V4.

V1 newest masked frame (3ch)   V2 four masked frames (12ch, reproduces Phase 2.1)
V3 four frames, bottom-HUD masked   V4 four-frame gameplay-only crop
visible four masked->unmasked frames (recoverable-fraction reference)

Saves per-sample predictions/targets/episode/timestep/split + full metrics + bootstrap CIs.
"""
import os, json, argparse
import numpy as np
import torch
from sklearn.linear_model import Ridge, LogisticRegression

from seaquest_ccrl.probes.oxy4_data import oxygen_class, OXY_LOW_HI
from seaquest_ccrl.probes.oxy4_audit_data import AuditData, train_audit_probe, stack_from_bank
from seaquest_ccrl.probes.oxy4_train import boot_ci

OUTDIR = "artifacts/seaquest/oxygen_4frame/leakage/source_audit"
# V1 uses the masked bank with a single newest frame; others use 4-frame stacks.
VISUAL = {  # name -> (bank_variant, k, in_ch)
    "V1_newest_oxybar_masked": ("oxybar_masked", 1, 3),
    "V2_four_oxybar_masked":   ("oxybar_masked", 4, 12),
    "V3_four_bottomhud_masked":("bottomhud_masked", 4, 12),
    "V4_four_gameplay_crop":   ("gameplay_crop", 4, 12),
    "visible_four":            ("visible", 4, 12),
}


def boot_scalar(y, pred, episodes, kind, n_boot=2000, seed=0):
    """Episode-bootstrap CI for a metric recomputed per resample (r2 / acc / balanced_acc)."""
    uniq = np.unique(episodes)
    idx_by = {e: np.where(episodes == e)[0] for e in uniq}
    rng = np.random.RandomState(seed); vals = []

    def metric(ix):
        yy, pp = y[ix], pred[ix]
        if kind == "r2":
            ss = ((yy - pp) ** 2).sum(); tot = ((yy - yy.mean()) ** 2).sum() + 1e-9
            return 1 - ss / tot
        if kind == "acc":
            return (pp == yy).mean()
        if kind == "balanced_acc":
            cs = [yy == c for c in np.unique(yy)]
            return float(np.mean([(pp[c] == yy[c]).mean() for c in cs if c.any()]))
    for _ in range(n_boot):
        pk = rng.choice(uniq, size=len(uniq), replace=True)
        ix = np.concatenate([idx_by[e] for e in pk])
        vals.append(metric(ix))
    point = metric(np.arange(len(y)))
    return {"mean": float(point), "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}


def clf_extra(y, pred_lab, episodes):
    """balanced accuracy, macro-F1, confusion matrix (3x3)."""
    K = 3
    cm = np.zeros((K, K), int)
    for a, b in zip(y, pred_lab):
        cm[int(a), int(b)] += 1
    recalls = [cm[c, c] / max(cm[c].sum(), 1) for c in range(K)]
    f1 = []
    for c in range(K):
        tp = cm[c, c]; fp = cm[:, c].sum() - tp; fn = cm[c].sum() - tp
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1.append(2 * prec * rec / max(prec + rec, 1e-9))
    return {"balanced_accuracy": float(np.mean(recalls)), "macro_f1": float(np.mean(f1)),
            "confusion_matrix": cm.tolist()}


def reg_metrics(y, pred):
    err = pred - y
    ss = (err ** 2).sum(); tot = ((y - y.mean()) ** 2).sum() + 1e-9
    return {"mae": float(np.abs(err).mean()), "rmse": float(np.sqrt((err ** 2).mean())),
            "r2": float(1 - ss / tot)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="seaquest_ccrl/data/raw_hf")
    ap.add_argument("--out-dir", default=OUTDIR)
    ap.add_argument("--variants", default="all", help="comma list of VISUAL keys or 'all'")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-cnn", action="store_true", help="proxies + sanity-on-data only")
    ap.add_argument("--max-train", type=int, default=0, help="subsample train rows (smoke); 0=all")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(f"{args.out_dir}/figures", exist_ok=True)
    os.makedirs(f"{args.out_dir}/predictions", exist_ok=True)

    data = AuditData(args.root)
    gi_tr0, gi_va, gi_te = (data.split_indices(s) for s in ("train", "val", "test"))
    gi_tr = gi_tr0 if not args.max_train else gi_tr0[np.linspace(0, len(gi_tr0) - 1, args.max_train).astype(int)]
    yox_tr = data.oxygen[gi_tr][:, None].astype(np.float32)
    yox_te = data.oxygen[gi_te].astype(np.float32)
    ycl_tr = oxygen_class(data.oxygen[gi_tr]); ycl_te = oxygen_class(data.oxygen[gi_te])
    ep_te = data.episode_of[gi_te]; t_te = data.t_in_ep[gi_te]
    res = {"n_train": int(len(gi_tr)), "n_test": int(len(gi_te)), "device": device,
           "epochs": args.epochs, "oxy_low_hi": list(OXY_LOW_HI)}

    # --- trivial no-image baseline ---
    base_mae = float(np.abs(yox_te - data.oxygen[gi_tr].mean()).mean())
    prior = np.bincount(ycl_tr, minlength=3) / len(ycl_tr)
    base_acc = float((ycl_te == prior.argmax()).mean())
    pred_const = np.full(len(ycl_te), prior.argmax())
    res["trivial_baseline"] = {"reg_mae": base_mae, "reg_r2": 0.0, "clf_acc": base_acc,
                               "clf_balanced_acc": clf_extra(ycl_te, pred_const, ep_te)["balanced_accuracy"],
                               "class_prior": prior.tolist()}

    # --- non-image proxy baselines P1-P3 (same split) ---
    tnorm = (data.t_in_ep / np.repeat(data.lengths, data.lengths)).astype(np.float32)  # normalized episode time
    py = data.player_pos[:, 1].astype(np.float32)
    feats = {"P1_timestep": tnorm[:, None], "P2_player_y": py[:, None],
             "P3_timestep_player_y": np.stack([tnorm, py], 1)}
    res["proxy_baselines"] = {}
    for name, F in feats.items():
        Xtr, Xte = F[gi_tr], F[gi_te]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        rr = Ridge(alpha=1.0).fit(Xtr, data.oxygen[gi_tr]); pr = rr.predict(Xte)
        lr = LogisticRegression(max_iter=400).fit(Xtr, ycl_tr)
        pc = lr.predict(Xte); pp = lr.predict_proba(Xte)
        ll = float(-np.log(np.clip(pp[np.arange(len(ycl_te)), ycl_te], 1e-9, 1)).mean())
        res["proxy_baselines"][name] = {
            "regression": {**reg_metrics(yox_te, pr),
                           "r2_ci": boot_scalar(yox_te, pr, ep_te, "r2"),
                           "mae_ci": boot_ci(np.abs(pr - yox_te), ep_te)},
            "classification": {"accuracy": float((pc == ycl_te).mean()), "log_loss": ll, **clf_extra(ycl_te, pc, ep_te),
                               "acc_ci": boot_scalar(ycl_te, pc, ep_te, "acc"),
                               "balanced_acc_ci": boot_scalar(ycl_te, pc, ep_te, "balanced_acc")}}
        np.savez_compressed(f"{args.out_dir}/predictions/{name}.npz",
                            pred_reg=pr, pred_clf=pc, y_oxygen=yox_te, y_class=ycl_te,
                            episode=ep_te, timestep=t_te, split=np.array(["test"] * len(ep_te)))

    # --- visual probes V1-V4 + visible (+ sanity controls on the V2 masked bank) ---
    want = list(VISUAL) if args.variants == "all" else args.variants.split(",")
    res["visual_probes"] = {}; res["sanity"] = {}
    if not args.skip_cnn:
        built = {}
        for vname in want:
            variant, k, in_ch = VISUAL[vname]
            bank = built.get(variant)
            if bank is None:
                bank = data.build_bank(variant); built = {variant: bank}  # keep only current bank
            r = train_audit_probe(data, bank, k, gi_tr, gi_va, gi_te, yox_tr, yox_te,
                                  "reg", 1, in_ch, epochs=args.epochs, seed=args.seed, device=device)
            c = train_audit_probe(data, bank, k, gi_tr, gi_va, gi_te, ycl_tr, ycl_te,
                                  "clf", 3, in_ch, epochs=args.epochs, seed=args.seed, device=device)
            pr = r["pred"][:, 0]; pc = c["pred"].argmax(1)
            res["visual_probes"][vname] = {
                "regression": {**reg_metrics(yox_te, pr), "r2_ci": boot_scalar(yox_te, pr, ep_te, "r2"),
                               "mae_ci": boot_ci(np.abs(pr - yox_te), ep_te)},
                "classification": {"accuracy": float((pc == ycl_te).mean()),
                                   "log_loss": float(c["metrics"]["log_loss"]), **clf_extra(ycl_te, pc, ep_te),
                                   "acc_ci": boot_scalar(ycl_te, pc, ep_te, "acc"),
                                   "balanced_acc_ci": boot_scalar(ycl_te, pc, ep_te, "balanced_acc")}}
            np.savez_compressed(f"{args.out_dir}/predictions/{vname}.npz",
                                pred_reg=pr, pred_clf=pc, y_oxygen=yox_te, y_class=ycl_te,
                                episode=ep_te, timestep=t_te, split=np.array(["test"] * len(ep_te)))
            print(f"[{vname}] R2={res['visual_probes'][vname]['regression']['r2']:.3f} "
                  f"acc={res['visual_probes'][vname]['classification']['accuracy']:.3f}")

            # sanity controls run on the V2 masked bank (already resident)
            if vname == "V2_four_oxybar_masked":
                # (1) label-permutation control: shuffle TRAIN oxygen, retrain reg
                rngp = np.random.RandomState(args.seed)
                yperm = yox_tr[rngp.permutation(len(yox_tr))]
                rp = train_audit_probe(data, bank, 4, gi_tr, gi_va, gi_te, yperm, yox_te,
                                       "reg", 1, 12, epochs=args.epochs, seed=args.seed, device=device)
                # (2) random-init untrained CNN
                rr0 = train_audit_probe(data, bank, 4, gi_tr, gi_va, gi_te, yox_tr, yox_te,
                                        "reg", 1, 12, epochs=args.epochs, seed=args.seed, device=device,
                                        random_init=True)
                res["sanity"]["label_permutation_r2"] = reg_metrics(yox_te, rp["pred"][:, 0])["r2"]
                res["sanity"]["random_init_cnn_r2"] = reg_metrics(yox_te, rr0["pred"][:, 0])["r2"]
                # (3) duplicate newest-frame count across splits + (4) nearest-image audit
                newest_tr = bank[gi_tr].reshape(len(gi_tr), -1)
                newest_te = bank[gi_te].reshape(len(gi_te), -1)
                htr = {hash(row.tobytes()) for row in newest_tr}
                dup = int(sum(hash(row.tobytes()) in htr for row in newest_te))
                samp = newest_te[np.linspace(0, len(newest_te) - 1, min(200, len(newest_te))).astype(int)].astype(np.int16)
                tr_s = newest_tr[np.linspace(0, len(newest_tr) - 1, min(2000, len(newest_tr))).astype(int)].astype(np.int16)
                nn = [int(np.abs(tr_s - s).sum(1).min()) for s in samp]
                res["sanity"]["exact_duplicate_frames_train_test"] = dup
                res["sanity"]["nearest_train_L1_median"] = float(np.median(nn))
                res["sanity"]["nearest_train_L1_min"] = float(np.min(nn))

    # --- differences (R2 and accuracy) ---
    vp = res["visual_probes"]
    def g(n, fld, key): return vp.get(n, {}).get(fld, {}).get(key)
    res["differences"] = {}
    for label, a, b in [("V2_minus_V1_temporal_history", "V2_four_oxybar_masked", "V1_newest_oxybar_masked"),
                        ("V2_minus_V3_rest_of_bottom_hud", "V2_four_oxybar_masked", "V3_four_bottomhud_masked"),
                        ("V3_minus_V4_top_hud", "V3_four_bottomhud_masked", "V4_four_gameplay_crop"),
                        ("visible_minus_V4_total_direct_hud", "visible_four", "V4_four_gameplay_crop")]:
        if a in vp and b in vp:
            res["differences"][label] = {"dr2": g(a, "regression", "r2") - g(b, "regression", "r2"),
                                         "dacc": g(a, "classification", "accuracy") - g(b, "classification", "accuracy")}

    json.dump(res, open(f"{args.out_dir}/audit_probe_metrics.json", "w"), indent=2)
    print("PROXY P3 R2 =", res["proxy_baselines"]["P3_timestep_player_y"]["regression"]["r2"])
    print(f"WROTE {args.out_dir}/audit_probe_metrics.json + predictions/*.npz")


if __name__ == "__main__":
    main()
