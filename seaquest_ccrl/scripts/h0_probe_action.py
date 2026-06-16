"""Probe B — conditional U -> action (Section 14). Colab side.

Identical removed four-frame states + identical-capacity nets. The U slot is the SAME
width in every condition:
  A-S          : removed state + ZERO U vector
  A-SU         : removed state + REAL U vector
  A-SU-shuffled: removed state + episode-shuffled U vector
Run separately for U_enemy_stack / U_missile_stack / U_joint_stack.

Primary target = exact 18-action id; primary metric = held-out exact-action log-loss
improvement loss(A-S) - loss(A-SU), with episode-bootstrap CI. Secondary = 12-cat
semantic action. Reports natural + hostile-active subsets.
"""
import os, json
import numpy as np

from seaquest_ccrl.hostile import probe_runner as PR
from seaquest_ccrl.hostile import data as D


def _improvement(loss_s, loss_su, episodes, seed=0):
    ci = PR.boot_ci(loss_s - loss_su, episodes, seed=seed)
    return {"improvement_mean": ci["mean"], "improvement_ci": ci["ci95"],
            "one_episode": PR.single_episode_driven(loss_s - loss_su, episodes)}


def run(data, out_dir, components=("enemy", "missile", "joint"), epochs=12, seed=0,
        device="cpu", save_raw=True):
    os.makedirs(out_dir, exist_ok=True)
    gi_tr = data.split_indices("train"); gi_te = data.split_indices("test")
    y_tr = data.action_targets(gi_tr)["action18"]
    y_te = data.action_targets(gi_te)["action18"]
    ep_te = data.episode_of[gi_te]
    result = {}

    for comp in components:
        dim = data.U_dim(comp)
        U_tr = data.U(gi_tr, comp); U_te = data.U(gi_te, comp)
        Z_tr = np.zeros_like(U_tr); Z_te = np.zeros_like(U_te)
        Ush_tr = D.shuffle_U_within_split(U_tr, data.episode_of[gi_tr], seed=seed)
        Ush_te = D.shuffle_U_within_split(U_te, ep_te, seed=seed)

        a_s = PR.train_probe(data, "removed", gi_tr, gi_te, y_tr, y_te, "clf", D.N_ACTIONS,
                             extra_tr=Z_tr, extra_te=Z_te, epochs=epochs, seed=seed, device=device)
        a_su = PR.train_probe(data, "removed", gi_tr, gi_te, y_tr, y_te, "clf", D.N_ACTIONS,
                              extra_tr=U_tr, extra_te=U_te, epochs=epochs, seed=seed, device=device)
        a_sh = PR.train_probe(data, "removed", gi_tr, gi_te, y_tr, y_te, "clf", D.N_ACTIONS,
                              extra_tr=Ush_tr, extra_te=Ush_te, epochs=epochs, seed=seed, device=device)

        imp = _improvement(a_s["per_row_loss"], a_su["per_row_loss"], ep_te, seed=seed)
        sh = PR.boot_ci(a_s["per_row_loss"] - a_sh["per_row_loss"], ep_te, seed=seed)

        # hostile-active subset
        active = data.active_mask(comp)[gi_te] if comp in ("enemy", "missile") else (
            (data.hostile_count_by_kind("enemy")[gi_te] > 0) |
            (data.hostile_count_by_kind("missile")[gi_te] > 0))
        act_imp = _improvement(a_s["per_row_loss"][active], a_su["per_row_loss"][active],
                               ep_te[active], seed=seed) if active.sum() > 50 else None

        result[comp] = {
            "U_dim": dim,
            "improvement_mean": imp["improvement_mean"], "improvement_ci": imp["improvement_ci"],
            "one_episode": imp["one_episode"],
            "shuffled_mean": sh["mean"], "shuffled_ci": sh["ci95"],
            "A_S_metrics": a_s["metrics"], "A_SU_metrics": a_su["metrics"],
            "A_SU_shuffled_metrics": a_sh["metrics"],
            "active_subset": act_imp, "n_active_test": int(active.sum()),
            "natural_test_n": int(len(gi_te)),
        }
        if save_raw:
            np.savez_compressed(os.path.join(out_dir, f"action_raw_{comp}.npz"),
                                loss_S=a_s["per_row_loss"], loss_SU=a_su["per_row_loss"],
                                loss_SUsh=a_sh["per_row_loss"], episode=ep_te,
                                y=y_te, gi_test=gi_te)

    json.dump(result, open(os.path.join(out_dir, "action_summary.json"), "w"), indent=2)
    return result
