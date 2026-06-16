"""Stage-G0 steps 6-7 — train the FULL-VIEW four-frame critic (oracle=True, oxygen mask OFF)
with the FROZEN pipeline, package the contract artifacts, then run the action-use sanity gate.

Fresh initialization (no masked checkpoint loaded). Identical to the masked four-frame run
except oracle=True + output path. Checkpoint = final step (frozen train_critic has no
val-loss selection; never selected on rollout success). Stops with
FULL_VIEW_CRITIC_DOES_NOT_USE_ACTION if the critic completely fails action-use.
"""
import os, json, shutil, hashlib, argparse
import numpy as np
import torch

from seaquest_ccrl.games import get_game
from seaquest_ccrl.training.config import TrainConfig
from seaquest_ccrl.training.train_critic import train, load_critic
from seaquest_ccrl.training.dataset_sampler import HindsightSampler
from seaquest_ccrl.models.contrastive_critic import ContrastiveCritic
from seaquest_ccrl.scripts.run_hf_4frame import hard_assertions, assert_checkpoint
from seaquest_ccrl.scripts.eval_hf_action_use import build_fixed_tuples, run_diagnostic

OUT = "artifacts/seaquest/goal_control/full_view"


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="seaquest_ccrl/data/raw_hf")
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--ckpt-dir", default="seaquest_ccrl/checkpoints/g0_full_view_seed0")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-tuples", type=int, default=4096)
    ap.add_argument("--tuple-seed", type=int, default=777)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    game = get_game("seaquest")
    gx0, gx1, gy0, gy1 = game.goal_box
    cfg = TrainConfig(steps=args.steps, seed=args.seed, nb_actions=game.nb_actions,
                      goal_x_lo=gx0, goal_x_hi=gx1, goal_y_lo=gy0, goal_y_hi=gy1,
                      goal_radius=game.eps, frame_stack=4, ckpt_dir=args.ckpt_dir)
    hard_assertions(game, cfg, args.ckpt_dir, args.root)
    assert "mspacman" not in args.ckpt_dir.lower()

    # ---- step 6: train full-view (oracle=True), fresh init ----
    path = train(oracle=True, cfg=cfg, game=game, root=args.root, device=device, verbose=True)
    assert_checkpoint(path)                       # in_channels=12, frame_stack=4, nb_actions=18
    sd = torch.load(path, map_location="cpu")
    assert sd.get("oracle") is True, "full-view checkpoint must have oracle=True"

    ckpt_out = f"{args.out_dir}/critic_full_view.pt"
    shutil.copy(path, ckpt_out)
    hist_src = os.path.join(args.ckpt_dir, "history_oracle.json")
    if os.path.exists(hist_src):
        shutil.copy(hist_src, f"{args.out_dir}/history.json")
    hist = json.load(open(f"{args.out_dir}/history.json")) if os.path.exists(f"{args.out_dir}/history.json") else {}
    loss = [r[1] for r in hist.get("loss", []) if isinstance(r, (list, tuple)) and len(r) >= 2]
    optimizes = bool(loss) and bool(np.isfinite(loss).all()) and loss[-1] < loss[0]
    json.dump({"first_conv_in_channels": 12, "frame_stack": 4, "nb_actions": 18, "oracle": True,
               "view": "full_view", "checkpoint_sha256": sha(ckpt_out),
               "loss_first": loss[0] if loss else None, "loss_last": loss[-1] if loss else None},
              open(f"{args.out_dir}/checkpoint_hash.json", "w"), indent=2)

    # ---- step 7: action-use sanity (oracle view auto-read from the checkpoint) ----
    critic, ccfg, oracle = load_critic(ckpt_out, device)
    assert oracle is True
    sampler = HindsightSampler(game, oracle=True, cfg=ccfg, device=device,
                               rng=np.random.default_rng(ccfg.seed), root=args.root)
    anchor, future, ep = build_fixed_tuples(sampler, args.n_tuples, args.tuple_seed)
    trained = run_diagnostic(critic, sampler, anchor, future, ep, ccfg, device, "trained")
    torch.manual_seed(0)
    rand = ContrastiveCritic(ccfg.repr_dim, ccfg.frame_size, ccfg.nb_actions, ccfg.frame_stack).to(device).eval()
    control = run_diagnostic(rand, sampler, anchor, future, ep, ccfg, device, "random_init_control")
    diag = {"ckpt": ckpt_out, "oracle_view": True, "n_tuples": args.n_tuples,
            "tuple_seed": args.tuple_seed, "nb_actions": int(ccfg.nb_actions),
            "trained": trained, "random_init_control": control}
    json.dump(diag, open(f"{args.out_dir}/action_use_diag.json", "w"), indent=2)
    np.savez(f"{args.out_dir}/action_use_diag_tuples.npz", anchor=anchor, future=future, episode=ep)

    d = trained["delta_shuffle"]
    tr_spread = trained["all_action_spread"]["mean_score_std_across_actions"]
    ct_spread = control["all_action_spread"]["mean_score_std_across_actions"]
    gate = {"shuffle_delta_gt_0": bool(d["mean"] > 0), "ci_lower_bound_gt_0": bool(d["ci_lower_above_0"]),
            "trained_spread_gt_control": bool(tr_spread > ct_spread), "optimizes_normally": optimizes}
    uses_action = all(gate.values())
    # "completely fails" => no action sensitivity at all (delta not >0 AND spread not > control)
    complete_fail = (not gate["shuffle_delta_gt_0"]) and (not gate["trained_spread_gt_control"])
    outcome = "FULL_VIEW_CRITIC_DOES_NOT_USE_ACTION" if complete_fail else (
        "FULL_VIEW_CRITIC_USES_ACTION" if uses_action else "FULL_VIEW_CRITIC_ACTION_USE_WEAK")
    json.dump({"gate": gate, "uses_action": uses_action, "complete_fail": complete_fail,
               "outcome": outcome, "delta_shuffle": d, "trained_spread": tr_spread,
               "control_spread": ct_spread, "optimizes_normally": optimizes,
               "note": "sanity check only; the PRIMARY Stage-G0 gate is closed-loop goal reaching"},
              open(f"{args.out_dir}/action_use_gate.json", "w"), indent=2)
    print(f"[G0 train] saved {ckpt_out} | loss {loss[0] if loss else '?'}->{loss[-1] if loss else '?'}")
    print(json.dumps({"gate": gate, "outcome": outcome,
                      "delta_shuffle_mean": d["mean"], "ci95": d["ci95"]}, indent=2))
    print(f"WROTE {args.out_dir}/critic_full_view.pt + history.json + checkpoint_hash.json "
          f"+ action_use_diag.json + action_use_gate.json")
    if complete_fail:
        raise SystemExit(7)


if __name__ == "__main__":
    main()
