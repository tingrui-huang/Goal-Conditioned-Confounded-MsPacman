"""Two-stage gate for the k=4 frame-stacked MsPacman critic.

STAGE 1 (cheap): train one critic (k=4) for `--stage1-steps`, then measure
action-sensitivity (does f depend on the action now?). The single-frame critic was
action-BLIND (action/goal sensitivity ratio 0.03); the state critic was 0.58. If
k=4 fixed the pixel-localization blocker, this ratio should clear the gate
(`--gate-ratio`, default 0.15). If it does NOT, frame-stacking didn't fix it -> stop
and report (don't waste GPU on the full sweep).

STAGE 2 (only if Stage 1 passes): run the real naive-vs-oracle seed sweep + 500-ep
gap. The script prints the exact commands (uses the existing, tested tools).

Run (Colab GPU):
  python -m seaquest_ccrl.scripts.gate_mspacman --stage1-steps 10000 --gate-ratio 0.15
"""
import argparse
import numpy as np
import torch

from seaquest_ccrl.games import get_game
from seaquest_ccrl.training.config import TrainConfig
from seaquest_ccrl.training.train_critic import train, load_critic
from seaquest_ccrl.training.dataset_sampler import HindsightSampler


def _positions(game):
    P = []
    for tr in game.make_dataset(game.data_root, oracle=True).trajectories():
        p = np.asarray(tr["achieved_goal"], dtype=np.float32)
        for i in range(1, len(p)):
            if not np.isfinite(p[i, 0]):
                p[i] = p[i-1]
        P.append(p)
    return np.concatenate(P, axis=0)


def action_sensitivity(critic, cfg, game, device="cpu", n=120):
    """Return (action_std_mean, goal_std_mean, ratio, dir_acc_far) for the stacked critic."""
    rng = np.random.default_rng(0)
    samp = HindsightSampler(game, oracle=True, cfg=cfg, device=device, rng=rng)
    P = _positions(game)
    N = samp.frames.shape[0]; k = samp.k_stack
    lo = np.array([cfg.goal_x_lo, cfg.goal_y_lo]); span = np.array([cfg.goal_x_hi-cfg.goal_x_lo, cfg.goal_y_hi-cfg.goal_y_lo])
    def gn(p): return ((np.asarray(p, np.float32)-lo)/span).astype(np.float32)
    CARD = {1: (0, -1), 2: (1, 0), 3: (-1, 0), 4: (0, 1)}
    def card_to(p, g):
        dx, dy = g[0]-p[0], g[1]-p[1]
        return (2 if dx > 0 else 3) if abs(dx) >= abs(dy) else (4 if dy > 0 else 1)
    a_std, g_std = [], []; dfar = [0, 0]
    for _ in range(n):
        i = int(rng.integers(0, N - 140))
        if not np.isfinite(P[i, 0]):
            continue
        idx = samp.stack_idx[i]
        st = samp.frames[idx]                                  # (k,H,W,3)
        obs = st.permute(1, 2, 0, 3).reshape(st.shape[1], st.shape[2], k*3).cpu().numpy().astype(np.uint8)
        for kind, kk in [("near", int(rng.integers(2, 7))), ("far", int(rng.integers(60, 130)))]:
            g = P[i + kk]
            sc = critic.score_all_actions(obs, gn(g), device)
            a_std.append(float(np.std(sc)))
            if kind == "far":
                cid = [1, 2, 3, 4]; best = cid[int(np.argmax([sc[c] for c in cid]))]
                dfar[1] += 1; dfar[0] += (best == card_to(P[i], g))
        # goal sensitivity: fix obs+action=0, vary goal
        with torch.no_grad():
            ot = torch.from_numpy(obs[None]).to(device)
            sa = critic.sa_encoder(ot, torch.zeros(1, dtype=torch.long, device=device))
            gg = torch.rand(12, 2, device=device)
            gv = (critic.g_encoder(gg) @ sa.T).squeeze(1).cpu().numpy()
        g_std.append(float(np.std(gv)))
    am, gm = float(np.mean(a_std)), float(np.mean(g_std))
    return am, gm, am/max(gm, 1e-6), dfar[0]/max(dfar[1], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1-steps", type=int, default=10000)
    ap.add_argument("--gate-ratio", type=float, default=0.15)
    ap.add_argument("--steps", type=int, default=50000, help="Stage-2 full train steps")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--eval-episodes", type=int, default=500)
    ap.add_argument("--ckpt-dir", default="mspacman_ccrl/checkpoints")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    game = get_game("mspacman")
    gx0, gx1, gy0, gy1 = game.goal_box
    cfg = TrainConfig(steps=args.stage1_steps, seed=0, nb_actions=game.nb_actions,
                      goal_x_lo=gx0, goal_x_hi=gx1, goal_y_lo=gy0, goal_y_hi=gy1,
                      goal_radius=game.eps, frame_stack=game.frame_stack,
                      ckpt_dir=f"{args.ckpt_dir}/_gate_stage1")

    print("=" * 60)
    print(f"  STAGE 1: train k={game.frame_stack} critic {args.stage1_steps} steps, probe action-sensitivity")
    print("=" * 60)
    path = train(oracle=True, cfg=cfg, game=game, device=device, verbose=True)
    critic, ccfg, _ = load_critic(path, device)
    am, gm, ratio, dfar = action_sensitivity(critic, ccfg, game, device)
    print(f"\n  action-std {am:.3f}  goal-std {gm:.3f}  RATIO {ratio:.3f}  far-dir-acc {dfar:.2f}")
    print(f"  (single-frame was 0.03 [BLIND]; state-critic 0.58; gate = {args.gate_ratio})")
    if ratio < args.gate_ratio:
        print(f"\n  >>> STAGE 1 FAILED (ratio {ratio:.3f} < {args.gate_ratio}). "
              f"k={game.frame_stack} stacking did NOT un-blind the critic. STOP -- do not run the sweep.")
        return
    print(f"\n  >>> STAGE 1 PASSED (ratio {ratio:.3f} >= {args.gate_ratio}). The k=4 critic uses the action.")
    print("\n" + "=" * 60)
    print("  STAGE 2: run the real naive-vs-oracle sweep + 500-ep gap")
    print("=" * 60)
    seeds = " ".join(map(str, args.seeds))
    print("Run these (Colab GPU):\n")
    print(f"  for s in {seeds}; do python -m seaquest_ccrl.scripts.run_naive_vs_oracle \\")
    print(f"     --game mspacman --steps {args.steps} --seed $s --max-steps 400 \\")
    print(f'     --eval-episodes 50 --ckpt-dir "$WORK/ckpt_mspacman_k4" \\')
    print(f'     --out "$WORK/results/k4_seed$s.json"; done\n')
    print(f"  python -m seaquest_ccrl.scripts.eval_parallel --game mspacman \\")
    print(f'     --ckpt-base "$WORK/ckpt_mspacman_k4" --seeds {seeds} \\')
    print(f"     --episodes {args.eval_episodes} --max-steps 400 \\")
    print(f'     --out "$WORK/results/k4_eval{args.eval_episodes}.json"')


if __name__ == "__main__":
    main()
