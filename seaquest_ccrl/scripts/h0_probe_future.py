"""Probe C — conditional U -> future (Section 15). Colab side.

Matched models (extra = action one-hot ++ U-slot of fixed width):
  F-SA          : removed state + action one-hot + ZERO U
  F-SAU         : removed state + action one-hot + REAL U
  F-SAU-shuffled: removed state + action one-hot + episode-shuffled U
Horizons H in {4,16,32,64}; rows whose future crosses an episode boundary are excluded.
Primary continuous targets: future_player_x/y, displacement_x/y (NOT future enemy/missile
position). Per-target MSE reduction with episode-bootstrap CI + shuffled control.
"""
import os, json
import numpy as np

from seaquest_ccrl.hostile import probe_runner as PR
from seaquest_ccrl.hostile import data as D

HORIZONS = (4, 16, 32, 64)
PRIMARY = ["future_player_x", "future_player_y", "displacement_x", "displacement_y"]


def _onehot(a, n):
    o = np.zeros((len(a), n), dtype=np.float32)
    o[np.arange(len(a)), a.astype(int)] = 1.0
    return o


def _per_target_reduction(pred_sa, pred_su, y, episodes, seed=0):
    out = {}
    for k, name in enumerate(PRIMARY):
        err_sa = (pred_sa[:, k] - y[:, k]) ** 2
        err_su = (pred_su[:, k] - y[:, k]) ** 2
        ci = PR.boot_ci(err_sa - err_su, episodes, seed=seed)
        out[name] = {"mse_red_mean": ci["mean"], "mse_red_ci": ci["ci95"],
                     "mse_sa": float(err_sa.mean()), "mse_su": float(err_su.mean())}
    return out


def run(data, out_dir, components=("enemy", "missile", "joint"), horizons=HORIZONS,
        epochs=12, seed=0, device="cpu", save_raw=True):
    os.makedirs(out_dir, exist_ok=True)
    gi_tr_all = data.split_indices("train"); gi_te_all = data.split_indices("test")
    result = {}

    for comp in components:
        per_horizon = {}
        shuffled_reproduces = False
        for H in horizons:
            ftr, ok_tr = data.future_targets(gi_tr_all, H)
            fte, ok_te = data.future_targets(gi_te_all, H)
            gi_tr = gi_tr_all[ok_tr]; gi_te = gi_te_all[ok_te]
            if len(gi_tr) < 100 or len(gi_te) < 100:
                per_horizon[str(H)] = {"skipped": "insufficient in-episode rows",
                                       "n_train": int(len(gi_tr)), "n_test": int(len(gi_te))}
                continue
            y_tr = np.stack([ftr[k] for k in PRIMARY], 1).astype(np.float32)
            y_te = np.stack([fte[k] for k in PRIMARY], 1).astype(np.float32)
            ep_te = data.episode_of[gi_te]

            a_tr = _onehot(data.actions[gi_tr], D.N_ACTIONS)
            a_te = _onehot(data.actions[gi_te], D.N_ACTIONS)
            U_tr = data.U(gi_tr, comp); U_te = data.U(gi_te, comp)
            Z_tr = np.zeros_like(U_tr); Z_te = np.zeros_like(U_te)
            Ush_tr = D.shuffle_U_within_split(U_tr, data.episode_of[gi_tr], seed=seed)
            Ush_te = D.shuffle_U_within_split(U_te, ep_te, seed=seed)

            sa = PR.train_probe(data, "removed", gi_tr, gi_te, y_tr, y_te, "reg", len(PRIMARY),
                                extra_tr=np.concatenate([a_tr, Z_tr], 1),
                                extra_te=np.concatenate([a_te, Z_te], 1),
                                epochs=epochs, seed=seed, device=device)
            su = PR.train_probe(data, "removed", gi_tr, gi_te, y_tr, y_te, "reg", len(PRIMARY),
                                extra_tr=np.concatenate([a_tr, U_tr], 1),
                                extra_te=np.concatenate([a_te, U_te], 1),
                                epochs=epochs, seed=seed, device=device)
            sh = PR.train_probe(data, "removed", gi_tr, gi_te, y_tr, y_te, "reg", len(PRIMARY),
                                extra_tr=np.concatenate([a_tr, Ush_tr], 1),
                                extra_te=np.concatenate([a_te, Ush_te], 1),
                                epochs=epochs, seed=seed, device=device)

            red = _per_target_reduction(sa["pred"], su["pred"], y_te, ep_te, seed=seed)
            red_sh = _per_target_reduction(sa["pred"], sh["pred"], y_te, ep_te, seed=seed)
            # best primary target at this horizon (max mean reduction with lower CI > 0)
            best = max(red.items(), key=lambda kv: (kv[1]["mse_red_ci"][0] > 0, kv[1]["mse_red_mean"]))
            per_horizon[str(H)] = {"target": best[0], **best[1],
                                   "all_targets": red, "shuffled_targets": red_sh,
                                   "n_test": int(len(gi_te))}
            # shuffled reproduces if any shuffled target has a comparable positive lower CI
            if any(v["mse_red_ci"][0] > 0 and v["mse_red_mean"] >= 0.5 * best[1]["mse_red_mean"]
                   for v in red_sh.values()) and best[1]["mse_red_mean"] > 0:
                shuffled_reproduces = True
            if save_raw:
                np.savez_compressed(os.path.join(out_dir, f"future_raw_{comp}_H{H}.npz"),
                                    pred_SA=sa["pred"], pred_SAU=su["pred"], pred_SUsh=sh["pred"],
                                    y=y_te, episode=ep_te, gi_test=gi_te)

        result[comp] = {"per_horizon": per_horizon, "shuffled_reproduces": shuffled_reproduces,
                        "primary_targets": PRIMARY}

    json.dump(result, open(os.path.join(out_dir, "future_summary.json"), "w"), indent=2)
    return result
