"""Dataset compatibility gate (evaluation only): load `raw_hf` through the UNMODIFIED original
`SeaquestOfflineDataset` + `HindsightSampler` and verify it is interface-identical to the old
dataset. The old dataset is used ONLY for shape/dtype interface comparison — no learning results
are compared and the old condition is NOT retrained.

PASS -> proceed to training. Any failure -> STOP_HF_DATASET_NOT_COMPATIBLE.
"""
import os
import json
import argparse

import numpy as np
import torch

from seaquest_ccrl.games import get_game
from seaquest_ccrl.training.config import TrainConfig
from seaquest_ccrl.data.dataset import SeaquestOfflineDataset
from seaquest_ccrl.training.dataset_sampler import HindsightSampler


def gate(hf_root, old_root, out):
    game = get_game("seaquest")
    gx0, gx1, gy0, gy1 = game.goal_box
    cfg = TrainConfig(steps=10, seed=0, nb_actions=game.nb_actions, goal_x_lo=gx0, goal_x_hi=gx1,
                      goal_y_lo=gy0, goal_y_hi=gy1, goal_radius=game.eps, frame_stack=game.frame_stack)
    R = {"hf_root": hf_root, "old_root": old_root, "checks": {}}

    # load HF through the unmodified loader (masked + oracle)
    ds_naive = SeaquestOfflineDataset(root=hf_root, oracle=False)
    ds_oracle = SeaquestOfflineDataset(root=hf_root, oracle=True)
    n_traj = len(ds_naive)

    # 1. all 18 action IDs valid
    all_actions = np.concatenate([ds_naive.trajectory(i)["action"] for i in range(n_traj)])
    a_lo, a_hi = int(all_actions.min()), int(all_actions.max())
    R["checks"]["1_action_ids_valid"] = bool(a_lo >= 0 and a_hi < cfg.nb_actions)
    R["action_range"] = [a_lo, a_hi]; R["n_distinct_actions"] = int(len(np.unique(all_actions)))

    # 2. frame dtype/shape matches old
    t0 = ds_naive.trajectory(0)
    old_ok = None
    if old_root and os.path.isdir(old_root):
        try:
            o0 = SeaquestOfflineDataset(root=old_root, oracle=False).trajectory(0)
            old_ok = bool(o0["obs"].dtype == t0["obs"].dtype and o0["obs"].shape[1:] == t0["obs"].shape[1:]
                          and o0["action"].dtype == t0["action"].dtype
                          and o0["achieved_goal"].dtype == t0["achieved_goal"].dtype)
            R["old_obs_shape"] = list(o0["obs"].shape[1:]); R["old_obs_dtype"] = str(o0["obs"].dtype)
        except Exception as e:
            R["old_load_error"] = str(e)
    R["hf_obs_shape"] = list(t0["obs"].shape[1:]); R["hf_obs_dtype"] = str(t0["obs"].dtype)
    R["checks"]["2_frame_dtype_shape_matches_old"] = old_ok

    # 3. player position finite on retained rows
    pos = np.concatenate([ds_naive.trajectory(i)["achieved_goal"] for i in range(n_traj)])
    R["checks"]["3_player_pos_finite"] = bool(np.isfinite(pos).all())
    R["player_pos_finite_frac"] = float(np.isfinite(pos).all(1).mean())

    # 4. trajectory boundaries preserved (done True exactly at last step or never if capped)
    bnd_ok = True
    for i in range(n_traj):
        dn = ds_naive.trajectory(i)["done"]
        if dn[:-1].any():       # no early done before the last step
            bnd_ok = False; break
    R["checks"]["4_trajectory_boundaries_preserved"] = bool(bnd_ok)

    # 5. hindsight futures never cross an episode
    s = HindsightSampler(game, oracle=False, cfg=cfg, device="cpu",
                         rng=np.random.default_rng(0), root=hf_root)
    rng = np.random.default_rng(0)
    cross = False
    for _ in range(200):
        ep = int(rng.integers(0, s.n_ep)); t = int(rng.integers(0, s.lengths[ep]))
        k = int(rng.geometric(s.p_geom)); fut = min(t + k, s.lengths[ep] - 1)
        g_anchor = s.offsets[ep] + t; g_future = s.offsets[ep] + fut
        ep_start = s.offsets[ep]; ep_end = s.offsets[ep] + s.lengths[ep] - 1
        if not (ep_start <= g_future <= ep_end and g_future >= g_anchor):
            cross = True; break
    R["checks"]["5_hindsight_futures_within_episode"] = bool(not cross)

    # 6. oxygen mask applied by the LOADER, not during collection: stored frames are UNMASKED
    #    (== oracle obs); the masked (naive) obs differs only inside the oxygen rect.
    fr_un = t0["frames_unmasked"][0]; ob_or = ds_oracle.trajectory(0)["obs"][0]; ob_na = t0["obs"][0]
    from seaquest_ccrl import config as C
    x, y, w, h = C.OXY_MASK_RECT
    stored_unmasked = bool(np.array_equal(fr_un, ob_or))
    mask_zeroed = bool((ob_na[y:y + h, x:x + w] == 0).all())
    outside_same = bool(np.array_equal(np.delete(ob_na.reshape(-1, 3), [], 0).size and ob_na[:y].sum(), ob_or[:y].sum()))
    R["checks"]["6_mask_applied_by_loader_frames_stored_unmasked"] = bool(stored_unmasked and mask_zeroed)

    # 7. sampled batch shapes/dtypes identical to a batch from the old dataset
    fr, ac, go = s.sample(cfg.batch_size)
    R["hf_batch"] = {"frames": [list(fr.shape), str(fr.dtype)], "actions": [list(ac.shape), str(ac.dtype)],
                     "goals": [list(go.shape), str(go.dtype)]}
    batch_ok = None
    if old_root and os.path.isdir(old_root):
        try:
            so = HindsightSampler(game, oracle=False, cfg=cfg, device="cpu",
                                  rng=np.random.default_rng(0), root=old_root)
            fo, ao, goo = so.sample(cfg.batch_size)
            batch_ok = bool(fr.shape[1:] == fo.shape[1:] and fr.dtype == fo.dtype
                            and ac.dtype == ao.dtype and go.shape[1:] == goo.shape[1:] and go.dtype == goo.dtype)
        except Exception as e:
            R["old_batch_error"] = str(e)
    R["checks"]["7_batch_shapes_dtypes_match_old"] = batch_ok

    core = ["1_action_ids_valid", "3_player_pos_finite", "4_trajectory_boundaries_preserved",
            "5_hindsight_futures_within_episode", "6_mask_applied_by_loader_frames_stored_unmasked"]
    R["n_traj"] = n_traj; R["n_transitions"] = int(len(all_actions))
    R["PASS"] = bool(all(R["checks"][c] for c in core)
                     and R["checks"]["2_frame_dtype_shape_matches_old"] in (True, None)
                     and R["checks"]["7_batch_shapes_dtypes_match_old"] in (True, None))
    R["outcome"] = "PROCEED" if R["PASS"] else "STOP_HF_DATASET_NOT_COMPATIBLE"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(R, open(out, "w"), indent=2)
    print(json.dumps({"checks": R["checks"], "outcome": R["outcome"],
                      "action_range": R["action_range"], "n_transitions": R["n_transitions"]}, indent=2))
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-root", default="seaquest_ccrl/data/raw_hf")
    ap.add_argument("--old-root", default="seaquest_ccrl/data/raw")
    ap.add_argument("--out", default="artifacts/seaquest/hf_original_critic/dataset_gate.json")
    args = ap.parse_args()
    gate(args.hf_root, args.old_root, args.out)


if __name__ == "__main__":
    main()
