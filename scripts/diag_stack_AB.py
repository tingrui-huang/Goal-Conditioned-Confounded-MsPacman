"""READ-ONLY diagnostic: is the k=4 frame-stack Stage-1 failure
   (A) frame-stacking not actually working (a bug), or
   (B) frame-stacking working but the motion washed out by downsampling?

Measures only. Does NOT modify/fix any pipeline/model/training/eval code, does NOT
retrain, needs NO GPU and NO critic checkpoint (all three tests are about
data/preprocessing). Stops at a printed verdict (A / B / neither).

NOTE on this repo's ACTUAL implementation (the script measures the real thing):
  - stack = k=4 RGB frames (12 channels), CONSECUTIVE stored frames (stack skip = 1),
    built by HindsightSampler.stack_idx + .frames -- the SAME path the gate probe fed.
  - each stored frame is already 4 game-frames apart (env frameskip=4).
  - preprocess resizes native (160x210) -> 84x84; conv bottleneck is 11x11
    (84 ->42 ->21 ->11 via three stride-2 convs).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from seaquest_ccrl.games import get_game
from seaquest_ccrl.training.config import TrainConfig
from seaquest_ccrl.training.dataset_sampler import HindsightSampler

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_out")
os.makedirs(OUT, exist_ok=True)
SX, SY = 84.0 / 160.0, 84.0 / 210.0    # native (col,row) -> 84px mapping
C2 = 11.0 / 84.0                        # 84px -> 11x11 cell


def positions(game):
    P = []
    for tr in game.make_dataset(game.data_root, oracle=True).trajectories():
        p = np.asarray(tr["achieved_goal"], dtype=np.float32)
        for i in range(1, len(p)):
            if not np.isfinite(p[i, 0]):
                p[i] = p[i - 1]
        P.append(p)
    return np.concatenate(P, axis=0)


def main():
    game = get_game("mspacman")
    gx0, gx1, gy0, gy1 = game.goal_box
    cfg = TrainConfig(nb_actions=game.nb_actions, goal_x_lo=gx0, goal_x_hi=gx1,
                      goal_y_lo=gy0, goal_y_hi=gy1, frame_stack=game.frame_stack)
    k = cfg.frame_stack
    print(f"frame_stack k={k}  (oracle=True, consecutive stored frames; env frameskip=4)\n")

    samp = HindsightSampler(game, oracle=True, cfg=cfg, device="cpu",
                            rng=np.random.default_rng(0))
    P = positions(game)
    N = samp.frames.shape[0]
    assert len(P) == N, f"position/frame misalignment {len(P)} vs {N}"
    ep_start = np.repeat(samp.offsets, samp.lengths)         # (N,) episode start per index

    # mid-episode: exclude samples whose stack would hit padding (i-(k-1) < ep_start)
    rng = np.random.default_rng(1)
    valid = np.where(np.arange(N) - (k - 1) >= ep_start)[0]
    sel = rng.choice(valid, size=min(512, len(valid)), replace=False)
    print(f"N={N} steps, {len(valid)} mid-episode (non-padded), using {len(sel)} samples\n")

    # ---- confirm the probe/sampler build the SAME stack ----
    i0 = int(sel[0])
    idx0 = samp.stack_idx[i0]
    st = samp.frames[idx0]                                   # (k,H,W,3)
    obs = st.permute(1, 2, 0, 3).reshape(st.shape[1], st.shape[2], k * 3).numpy()  # gate-probe path
    ok_path = all(np.array_equal(obs[:, :, 3*j:3*j+3], samp.frames[idx0[j]].numpy())
                  for j in range(k))
    print(f"[path check] gate-probe stack == sampler.frames[stack_idx] frames: {ok_path}")
    print(f"[path check] stack_idx[{i0}] = {idx0.tolist()} (must be {k} distinct consecutive indices)\n")

    d_all = []          # (M, k-1) adjacent-frame mean-abs-diff in [0,1]
    near_identical = 0
    fr_stats = []
    disp_nat, disp84, disp_cell, adj_nat = [], [], [], []
    snr84, snr11 = [], []
    fig_saved = 0

    for n, i in enumerate(sel):
        i = int(i)
        idx = samp.stack_idx[i].numpy()
        fk = samp.frames[idx].numpy().astype(np.float32) / 255.0    # (k,84,84,3) in [0,1]

        # ---- TEST 1: are the k frames different? ----
        di = [float(np.mean(np.abs(fk[j] - fk[j + 1]))) for j in range(k - 1)]
        d_all.append(di)
        if all(x < 1e-4 for x in di):
            near_identical += 1
        if n < 200:
            for j in range(k):
                fr_stats.append((fk[j].min(), fk[j].mean(), fk[j].max()))

        # ---- TEST 2: how much did Pac-Man move across the stack? ----
        pa, pb = P[idx[0]], P[idx[-1]]                              # native (col,row) oldest,newest
        dn = float(np.hypot(pb[0] - pa[0], pb[1] - pa[1]))
        pa84 = (pa[0] * SX, pa[1] * SY); pb84 = (pb[0] * SX, pb[1] * SY)
        d84 = float(np.hypot(pb84[0] - pa84[0], pb84[1] - pa84[1]))
        disp_nat.append(dn); disp84.append(d84); disp_cell.append(d84 * C2)
        pm = P[idx[-2]]
        adj_nat.append(float(np.hypot(pb[0] - pm[0], pb[1] - pm[1])))

        # ---- TEST 3: does the motion survive 84 -> 11 downsampling? ----
        ga = fk[0].mean(axis=2); gb = fk[-1].mean(axis=2)          # grayscale 84x84
        diff84 = np.abs(ga - gb)
        d11 = F.adaptive_avg_pool2d(torch.from_numpy(diff84)[None, None], 11)[0, 0].numpy()
        cx84, cy84 = int(round(pb[0] * SX)), int(round(pb[1] * SY))
        cx84 = min(83, max(0, cx84)); cy84 = min(83, max(0, cy84))
        win = diff84[max(0, cy84-1):cy84+2, max(0, cx84-1):cx84+2]
        sig84 = float(win.max()) if win.size else 0.0
        yy, xx = np.mgrid[0:84, 0:84]
        farm = (np.hypot(xx - cx84, yy - cy84) > 20)
        noi84 = float(np.median(diff84[farm])) if farm.any() else 1e-6
        cx11, cy11 = int(pb[0] * 11 / 160), int(pb[1] * 11 / 210)
        cx11 = min(10, max(0, cx11)); cy11 = min(10, max(0, cy11))
        sig11 = float(d11[cy11, cx11])
        yy2, xx2 = np.mgrid[0:11, 0:11]
        farm11 = (np.maximum(np.abs(xx2 - cx11), np.abs(yy2 - cy11)) > 3)
        noi11 = float(np.median(d11[farm11])) if farm11.any() else 1e-6
        snr84.append(sig84 / (noi84 + 1e-6)); snr11.append(sig11 / (noi11 + 1e-6))

        if fig_saved < 3:
            _save_fig(diff84, d11, (cx84, cy84), (cx11, cy11), fig_saved); fig_saved += 1

    d_all = np.array(d_all); fr_stats = np.array(fr_stats)
    disp_nat = np.array(disp_nat); disp84 = np.array(disp84); disp_cell = np.array(disp_cell)
    snr84 = np.array(snr84); snr11 = np.array(snr11)
    frac_ident = near_identical / len(sel)

    def ms(x): return f"mean {np.mean(x):.3f} median {np.median(x):.3f}"
    def q(x): return (f"mean {np.mean(x):.2f} median {np.median(x):.2f} "
                      f"p10 {np.percentile(x,10):.2f} p90 {np.percentile(x,90):.2f}")

    print("=" * 64)
    print("TEST 1 — are the k stacked frames actually different?  (checks A)")
    print("=" * 64)
    for j in range(k - 1):
        print(f"  adj-frame diff d{j} (|f{j}-f{j+1}|, [0,1] scale): {ms(d_all[:, j])}")
    print(f"  fraction of samples with ALL d_i < 1e-4 (frames ~identical): {frac_ident:.3f}")
    print(f"  per-frame pixel stats: min {fr_stats[:,0].mean():.3f}  mean {fr_stats[:,1].mean():.3f}"
          f"  max {fr_stats[:,2].mean():.3f}  (not a constant/black image)")

    print("\n" + "=" * 64)
    print("TEST 2 — how far did Pac-Man move across the stack?  (B physical qty)")
    print("=" * 64)
    print(f"  stack-span disp (t-{k-1} -> t)  native px : {q(disp_nat)}")
    print(f"  stack-span disp                 84px      : {q(disp84)}")
    print(f"  stack-span disp                 11x11 cells: {q(disp_cell)}")
    print(f"  adjacent-frame disp             native px : {q(adj_nat)}")

    print("\n" + "=" * 64)
    print("TEST 3 — does that motion survive 84->11 downsampling?  (decides B)")
    print("=" * 64)
    print(f"  SNR_84 (signal@PacMan / noise far)  : {ms(snr84)}")
    print(f"  SNR_11 (after avg-pool to 11x11)    : {ms(snr11)}")
    print(f"  saved {fig_saved} side-by-side [diff84 | diff11] figures -> {OUT}")

    # ---------------- verdict ----------------
    print("\n" + "=" * 64); print("  VERDICT"); print("=" * 64)
    di_mean = float(d_all.mean())
    cell_med = float(np.median(disp_cell)); snr11_med = float(np.median(snr11))
    snr84_med = float(np.median(snr84))
    if frac_ident > 0.20 or di_mean < 1e-3 or not ok_path:
        print("  >>> A: FRAME-STACKING NOT WORKING.")
        if not ok_path:
            print("      cause: probe/sampler stack construction MISMATCH (different path).")
        elif (samp.stack_idx[sel].numpy()[:, 0] == samp.stack_idx[sel].numpy()[:, -1]).mean() > 0.2:
            print("      cause: stack indices collapse to the same frame.")
        else:
            print("      cause: frames are (near-)identical despite distinct indices "
                  "(check grayscale-constant / preprocessing).")
    elif cell_med < 0.5 and snr84_med > 1.5 and snr11_med < 1.3:
        print("  >>> B: STACKING WORKS, BUT MOTION IS WASHED OUT BY DOWNSAMPLING.")
        print(f"      Pac-Man moves ~{np.median(disp84):.1f}px on 84 (~{cell_med:.2f} of an 11-cell);")
        print(f"      SNR_84 {snr84_med:.2f} (>1, motion present) but SNR_11 {snr11_med:.2f} "
              f"(~1, gone after pooling).")
    elif snr11_med > 1.5:
        print("  >>> NEITHER A NOR B: motion DOES survive to 11x11 (SNR_11 "
              f"{snr11_med:.2f}) -> info is there, the critic just didn't use it.")
    else:
        print("  >>> INCONCLUSIVE — numbers above don't cleanly fit A/B; reporting as-is.")
        print(f"      frac_ident {frac_ident:.2f}  cell_med {cell_med:.2f}  "
              f"SNR_84 {snr84_med:.2f}  SNR_11 {snr11_med:.2f}")


def _save_fig(diff84, diff11, pac84, pac11, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.4))
    ax[0].imshow(diff84, cmap="magma"); ax[0].set_title("diff84 (|f_t-3 - f_t|)")
    ax[0].plot(pac84[0], pac84[1], "c+", ms=12, mew=2)
    ax[1].imshow(diff11, cmap="magma"); ax[1].set_title("diff11 (avg-pool 11x11)")
    ax[1].plot(pac11[0], pac11[1], "c+", ms=14, mew=2)
    for a in ax:
        a.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, f"sample_{n}.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
