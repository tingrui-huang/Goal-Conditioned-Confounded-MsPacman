"""Probe A — hiddenness (Section 13). Colab side.

Matched probes on identical train/test rows:
  P0  = train-prior / non-image baseline (closed form)
  PV  = original VISIBLE four-frame pixels
  PM  = hostile-REMOVED four-frame pixels
Targets (per enemy/missile component): 3x3 occupancy (multilabel), presence,
clipped-count, and nearest dx/dy (presence rows only, regression).

recovery_fraction = (loss_prior - loss_masked) / (loss_prior - loss_visible),
computed only when the visible probe beats the prior. Returns the structured dict
consumed by h0_qualify.decide(), and saves raw predictions / per-row losses / ids.
"""
import os, json
import numpy as np

from seaquest_ccrl.hostile import probe_runner as PR


def _recovery(loss_prior, loss_visible, loss_masked, episodes, n_boot=2000, seed=0):
    """Per-row recovery fraction with episode bootstrap; only valid if visible<prior."""
    denom = loss_prior - loss_visible
    visible_better = float(np.mean(denom)) > 0
    # per-row recovery (guard tiny denom); bootstrap at episode level on the RATIO of means
    rng = np.random.RandomState(seed)
    uniq = np.unique(episodes)
    idx_by = {e: np.where(episodes == e)[0] for e in uniq}
    fr = []
    for _ in range(n_boot):
        pk = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by[e] for e in pk])
        num = np.mean(loss_prior[rows] - loss_masked[rows])
        den = np.mean(loss_prior[rows] - loss_visible[rows])
        if abs(den) > 1e-9:
            fr.append(num / den)
    point = (np.mean(loss_prior - loss_masked)) / (np.mean(denom) + 1e-12)
    lo, hi = (float(np.percentile(fr, 2.5)), float(np.percentile(fr, 97.5))) if fr else (None, None)
    return {"recovery_point": float(point), "recovery_ci": [lo, hi],
            "visible_better_than_prior": bool(visible_better),
            "loss_prior": float(np.mean(loss_prior)), "loss_visible": float(np.mean(loss_visible)),
            "loss_masked": float(np.mean(loss_masked))}


def run(data, out_dir, components=("enemy", "missile"), epochs=12, seed=0, device="cpu",
        save_raw=True):
    assert data.frames_visible is not None, "load HostileH0Data(load_visible=True) for Probe A"
    os.makedirs(out_dir, exist_ok=True)
    gi_tr = data.split_indices("train"); gi_te = data.split_indices("test")
    result = {}

    for comp in components:
        ht_tr = data.hidden_targets(gi_tr, comp)
        ht_te = data.hidden_targets(gi_te, comp)
        comp_res = {"targets": {}}

        # --- primary: 3x3 multilabel grid ---
        yv_tr, yv_te = ht_tr["grid"], ht_te["grid"]
        pm = PR.train_probe(data, "removed", gi_tr, gi_te, yv_tr, yv_te, "multilabel", 9,
                            epochs=epochs, seed=seed, device=device)
        pv = PR.train_probe(data, "visible", gi_tr, gi_te, yv_tr, yv_te, "multilabel", 9,
                            epochs=epochs, seed=seed, device=device)
        p0_row, p0_m = PR.prior_loss("multilabel", yv_tr, yv_te, 9)
        rec_grid = _recovery(p0_row, pv["per_row_loss"], pm["per_row_loss"], pm["episode"], seed=seed)
        comp_res["targets"]["grid_multilabel"] = {
            "PM_metrics": pm["metrics"], "PV_metrics": pv["metrics"], "P0": p0_m, **rec_grid}

        # --- presence (binary clf) ---
        pm_p = PR.train_probe(data, "removed", gi_tr, gi_te, ht_tr["presence"], ht_te["presence"],
                              "clf", 2, epochs=epochs, seed=seed, device=device)
        pv_p = PR.train_probe(data, "visible", gi_tr, gi_te, ht_tr["presence"], ht_te["presence"],
                              "clf", 2, epochs=epochs, seed=seed, device=device)
        p0p_row, p0p_m = PR.prior_loss("clf", ht_tr["presence"], ht_te["presence"], 2)
        comp_res["targets"]["presence"] = {
            "PM_metrics": pm_p["metrics"], "PV_metrics": pv_p["metrics"], "P0": p0p_m,
            **_recovery(p0p_row, pv_p["per_row_loss"], pm_p["per_row_loss"], pm_p["episode"], seed=seed)}

        # --- clipped count (multiclass) ---
        pm_c = PR.train_probe(data, "removed", gi_tr, gi_te, ht_tr["count"], ht_te["count"],
                              "clf", 4, epochs=epochs, seed=seed, device=device)
        pv_c = PR.train_probe(data, "visible", gi_tr, gi_te, ht_tr["count"], ht_te["count"],
                              "clf", 4, epochs=epochs, seed=seed, device=device)
        p0c_row, p0c_m = PR.prior_loss("clf", ht_tr["count"], ht_te["count"], 4)
        comp_res["targets"]["count"] = {
            "PM_metrics": pm_c["metrics"], "PV_metrics": pv_c["metrics"], "P0": p0c_m,
            **_recovery(p0c_row, pv_c["per_row_loss"], pm_c["per_row_loss"], pm_c["episode"], seed=seed)}

        # --- secondary: nearest dx/dy on PRESENCE rows only (regression) ---
        pres_tr = ht_tr["presence"] == 1; pres_te = ht_te["presence"] == 1
        masked_r2 = None
        if pres_tr.sum() > 50 and pres_te.sum() > 50:
            ydx_tr = np.stack([ht_tr["nearest_dx"], ht_tr["nearest_dy"]], 1)[pres_tr]
            ydx_te = np.stack([ht_te["nearest_dx"], ht_te["nearest_dy"]], 1)[pres_te]
            pm_n = PR.train_probe(data, "removed", gi_tr[pres_tr], gi_te[pres_te],
                                  ydx_tr, ydx_te, "reg", 2, epochs=epochs, seed=seed, device=device)
            pv_n = PR.train_probe(data, "visible", gi_tr[pres_tr], gi_te[pres_te],
                                  ydx_tr, ydx_te, "reg", 2, epochs=epochs, seed=seed, device=device)
            masked_r2 = pm_n["metrics"]["r2"]
            comp_res["targets"]["nearest_offset"] = {
                "PM_metrics": pm_n["metrics"], "PV_metrics": pv_n["metrics"],
                "n_presence_test": int(pres_te.sum())}

        # primary recovery for the gate = grid multilabel
        comp_res["recovery_ci"] = rec_grid["recovery_ci"]
        comp_res["visible_better_than_prior"] = rec_grid["visible_better_than_prior"]
        comp_res["masked_nearest_r2"] = masked_r2
        comp_res["adequate_support"] = bool(pres_te.sum() > 200)
        result[comp] = comp_res

        if save_raw:
            np.savez_compressed(os.path.join(out_dir, f"hiddenness_raw_{comp}.npz"),
                                grid_PM_pred=pm["pred"], grid_PM_loss=pm["per_row_loss"],
                                grid_PV_loss=pv["per_row_loss"], grid_P0_loss=p0_row,
                                grid_y=pm["y"], episode=pm["episode"], gi_test=gi_te)

    json.dump(result, open(os.path.join(out_dir, "hiddenness_summary.json"), "w"), indent=2)
    return result
