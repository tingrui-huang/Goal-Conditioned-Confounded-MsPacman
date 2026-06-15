"""Phase-1 pre-training audit: four-frame stack assertions + masking check + visual grid.

Self-contained committed script (no notebook inline logic) so Colab just calls it.
Hard-asserts: Seaquest nb_actions=18, frame_stack=4, first-conv in_channels=12; refuses
MsPacman/ambiguous roots. Verifies the 6 stack properties on the REAL dataset, saves a
visual grid PNG and a JSON, and confirms raw_hf is unchanged on disk (load-time masking).
"""
import os, argparse, glob, hashlib, json
import numpy as np
import torch

from seaquest_ccrl.games import get_game
from seaquest_ccrl.training.config import TrainConfig
from seaquest_ccrl.training.dataset_sampler import HindsightSampler
from seaquest_ccrl.models.sa_encoder import SAEncoder
from seaquest_ccrl.data.dataset import SeaquestOfflineDataset
from seaquest_ccrl import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="seaquest_ccrl/data/raw_hf")
    ap.add_argument("--out-dir", default="artifacts/seaquest/oxygen_4frame/naive_critic")
    args = ap.parse_args()
    low = args.root.lower()
    assert "mspacman" not in low and "pacman" not in low, f"refusing MsPacman root: {args.root}"
    os.makedirs(args.out_dir, exist_ok=True)

    game = get_game("seaquest"); gx0, gx1, gy0, gy1 = game.goal_box
    assert game.nb_actions == 18, f"Seaquest nb_actions must be 18 (9=MsPacman), got {game.nb_actions}"
    cfg = TrainConfig(nb_actions=18, frame_stack=4, goal_x_lo=gx0, goal_x_hi=gx1,
                      goal_y_lo=gy0, goal_y_hi=gy1, goal_radius=game.eps)
    assert cfg.frame_stack == 4
    enc = SAEncoder(cfg.repr_dim, cfg.frame_size, 18, frame_stack=4)
    assert enc.conv[0].weight.shape[1] == 12, f"first-conv in_channels must be 12, got {enc.conv[0].weight.shape[1]}"
    print("[guard] nb_actions=18 frame_stack=4 first_conv_in_channels=12 OK")

    files = sorted(glob.glob(args.root + "/traj_*.npz"))[:3]
    raw_before = {f: hashlib.sha256(open(f, "rb").read()).hexdigest() for f in files}

    # masking (definitive, raw frames): rect zeroed; differs from oracle ONLY in rect
    dn = SeaquestOfflineDataset(args.root, oracle=False)
    do = SeaquestOfflineDataset(args.root, oracle=True)
    tn = dn.trajectory(0); to = do.trajectory(0); x, y, w, h = C.OXY_MASK_RECT
    T = tn["obs"].shape[0]
    for t in [0, 1, 2, T // 2, T - 1]:
        assert int(tn["obs"][t][y:y + h, x:x + w, :].sum()) == 0, f"oxygen rect not zeroed at t={t} (assertion 5)"
    diff = (tn["obs"][5] != to["obs"][5]); outside = diff.copy(); outside[y:y + h, x:x + w, :] = False
    assert not outside.any(), "masked vs oracle differ OUTSIDE oxygen rect"
    assert to["obs"][5][y:y + h, x:x + w, :].sum() > 0, "oracle should retain oxygen-bar pixels"
    print(f"[mask] OXY_MASK_RECT={C.OXY_MASK_RECT} zeroed in all frames; differs from oracle ONLY in rect OK")

    smp = HindsightSampler(game, oracle=False, cfg=cfg, device="cpu", root=args.root)
    offs, lens = smp.offsets, smp.lengths
    anchors = []
    for ei in range(min(4, smp.n_ep)):
        s = int(offs[ei]); L = int(lens[ei]); anchors += [s + 0, s + 1, s + 2, s + L // 2, s + L - 1]
    anchors = np.array(anchors[:12])
    idx = smp.stack_idx.index_select(0, torch.as_tensor(anchors)).numpy()
    ep_start = np.repeat(offs, lens)[anchors]
    assert (idx[:, -1] == anchors).all(), "newest frame must be the pre-action frame t (assertions 2,3)"
    assert (idx >= ep_start[:, None]).all(), "stack crossed an episode boundary (assertion 4)"
    assert (np.diff(idx, axis=1) >= 0).all(), "not oldest->newest (assertion 2)"
    st = smp.frames[torch.as_tensor(idx)]
    stacked = st.permute(0, 2, 3, 1, 4).reshape(len(anchors), 84, 84, 12)
    for j in range(4):
        assert torch.equal(stacked[..., 3 * j:3 * j + 3], smp.frames[torch.as_tensor(idx[:, j])]), \
            f"channel triplet {j} != masked frame at stack col {j} (assertion 1)"
    raw_after = {f: hashlib.sha256(open(f, "rb").read()).hexdigest() for f in files}
    assert raw_before == raw_after, "raw_hf changed on disk (masking must be load-time, assertion 6)"
    print("ALL 4-FRAME STACK ASSERTIONS PASS (channels=4 RGB, oldest->newest, ends at t, no boundary cross, masked, raw unchanged)")

    # visual grid
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        show = anchors[:6]; sidx = smp.stack_idx.index_select(0, torch.as_tensor(show)).numpy()
        fig, ax = plt.subplots(len(show), 4, figsize=(10, 2.2 * len(show)))
        for r, a in enumerate(show):
            for c4 in range(4):
                ax[r][c4].imshow(smp.frames[int(sidx[r, c4])].numpy()); ax[r][c4].axis("off")
                if r == 0: ax[r][c4].set_title(["oldest", "", "", "newest=t"][c4], fontsize=8)
        plt.tight_layout(); plt.savefig(f"{args.out_dir}/stack_visual_audit.png", dpi=110); plt.close()
        grid = "stack_visual_audit.png"
    except Exception as e:
        grid = f"skipped ({e})"

    out = {"pass": True, "nb_actions": 18, "frame_stack": 4, "first_conv_in_channels": 12,
           "oxy_mask_rect": list(C.OXY_MASK_RECT), "n_episodes": int(smp.n_ep),
           "episode_start_stack_clamped_example": smp.stack_idx[int(offs[0])].numpy().tolist(),
           "raw_hf_sample_sha256": raw_after, "visual_grid": grid,
           "assertions": ["channels=4 consecutive RGB frames", "oldest->newest",
                          "action at t <-> stack ends at frame t", "no cross-episode frame",
                          "oxygen masked in all 4 frames", "raw unchanged on disk"]}
    json.dump(out, open(f"{args.out_dir}/stack_audit.json", "w"), indent=2)
    print(f"WROTE {args.out_dir}/stack_audit.json + {grid}")


if __name__ == "__main__":
    main()
