"""Oxygen 4-Frame Study — explicit FOUR-FRAME experiment runner.

Identical to run_hf_expert_critic EXCEPT the single scientific change the oxygen
study requires: frame_stack = 1 -> 4 (an EXPLICIT experiment config; the global/game
defaults are NOT modified). Everything else (critic, sampler, loss, encoders, goal
rule, gamma, frame_size, repr_dim, batch, optimizer, steps, seed) is the unchanged
committed pipeline; this duplicates NO training logic and calls the frozen train().

naive: oxygen bar masked in all four frames (oracle=False, default).
oracle: original unmasked frames (oracle=True).
Default output: seaquest_ccrl/checkpoints/hf_4frame_seed0/ (does NOT overwrite the
single-frame hf_seed0 checkpoint).
"""
import argparse
import torch

from seaquest_ccrl.games import get_game
from seaquest_ccrl.training.config import TrainConfig
from seaquest_ccrl.training.train_critic import train
from seaquest_ccrl.models.sa_encoder import SAEncoder

FRAME_STACK = 4  # the ONLY scientific change vs the single-frame HF run


def hard_assertions(game, cfg, ckpt_dir, root):
    """Refuse any MsPacman / ambiguous / non-18-action / non-4-frame configuration."""
    for label, p in [("ckpt_dir", ckpt_dir), ("root", root)]:
        low = str(p).lower()
        if "mspacman" in low or "pacman" in low:
            raise AssertionError(f"refusing MsPacman path in {label}: {p}")
    assert game.name == "seaquest" if hasattr(game, "name") else True
    assert game.nb_actions == 18, f"Seaquest nb_actions must be 18, got {game.nb_actions} (9=MsPacman -> refuse)"
    assert cfg.nb_actions == 18, f"cfg.nb_actions must be 18, got {cfg.nb_actions}"
    assert cfg.frame_stack == 4, f"frame_stack must be 4, got {cfg.frame_stack}"
    # first-conv input channels must be 12 (4 RGB frames)
    enc = SAEncoder(cfg.repr_dim, cfg.frame_size, cfg.nb_actions, frame_stack=cfg.frame_stack)
    in_ch = enc.conv[0].weight.shape[1]
    assert in_ch == 12, f"first-conv in_channels must be 12 (4 RGB), got {in_ch}"
    print(f"  [assert] PASS nb_actions=18 frame_stack=4 first_conv_in_channels=12 path-not-mspacman")


def assert_checkpoint(path):
    """Post-train hard check on the saved checkpoint."""
    sd = torch.load(path, map_location="cpu")
    in_ch = sd["state_dict"]["sa_encoder.conv.0.weight"].shape[1]
    cfg = sd.get("cfg", {})
    low = str(path).lower()
    assert "mspacman" not in low and "pacman" not in low, f"refusing MsPacman checkpoint path: {path}"
    assert in_ch == 12, f"saved checkpoint first-conv in_channels must be 12, got {in_ch}"
    assert cfg.get("frame_stack") == 4, f"saved cfg.frame_stack must be 4, got {cfg.get('frame_stack')}"
    assert cfg.get("nb_actions") == 18, f"saved cfg.nb_actions must be 18, got {cfg.get('nb_actions')}"
    print(f"  [assert] saved checkpoint OK: in_channels=12 frame_stack=4 nb_actions=18")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="seaquest_ccrl/data/raw_hf")
    ap.add_argument("--oracle", action="store_true",
                    help="unmasked four-frame view; default is masked (naive)")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-dir", default="seaquest_ccrl/checkpoints/hf_4frame_seed0")
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    game = get_game("seaquest")
    gx0, gx1, gy0, gy1 = game.goal_box
    cfg = TrainConfig(steps=args.steps, seed=args.seed, nb_actions=game.nb_actions,
                      goal_x_lo=gx0, goal_x_hi=gx1, goal_y_lo=gy0, goal_y_hi=gy1,
                      goal_radius=game.eps, frame_stack=FRAME_STACK,   # <-- explicit 1 -> 4
                      ckpt_dir=args.ckpt_dir)
    tag = "oracle" if args.oracle else "naive"
    print(f"[HF 4-frame critic] view={tag} root={args.root} device={device}")
    print(f"  cfg: steps={cfg.steps} seed={cfg.seed} nb_actions={cfg.nb_actions} batch={cfg.batch_size} "
          f"lr={cfg.lr} gamma={cfg.gamma} frame_size={cfg.frame_size} FRAME_STACK={cfg.frame_stack} "
          f"repr_dim={cfg.repr_dim} goal_radius={cfg.goal_radius} goal_box=({gx0},{gx1},{gy0},{gy1})")
    hard_assertions(game, cfg, args.ckpt_dir, args.root)
    path = train(oracle=args.oracle, cfg=cfg, game=game, root=args.root, device=device, verbose=True)
    assert_checkpoint(path)
    print(f"[HF 4-frame critic] DONE -> {path}")


if __name__ == "__main__":
    main()
