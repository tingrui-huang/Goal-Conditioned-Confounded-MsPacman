"""Visualize what the Seaquest game is actually doing re: oxygen — for intuition / advisor.
Produces, for one raw_hf episode:
  * oxygen_sawtooth.png  : oxygen(t) + depth(t) over time (the dive/refill cycle);
  * oxygen_filmstrip.png : key frames across one cycle, annotated with O2 + depth + action;
  * oxygen_episode.gif   : annotated animation of one full dive->deplete->refill cycle.
Read-only; no training/eval touched.
"""
import os, argparse
import numpy as np
from PIL import Image, ImageDraw

ALE = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT',
       'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']
OUT = "artifacts/seaquest/oxygen_4frame/viz"


def annot(fr, o2, py, a, t, sc=2):
    img = Image.fromarray(fr).resize((160 * sc, 210 * sc), Image.NEAREST)
    panel = 64
    cv = Image.new("RGB", (160 * sc, 210 * sc + panel), (18, 18, 18))
    cv.paste(img, (0, 0)); dr = ImageDraw.Draw(cv); y0 = 210 * sc + 6
    frac = max(0.0, min(1.0, o2 / 63.0))
    col = (int(235 * (1 - frac)) + 20, int(200 * frac) + 20, 40)     # red(low) -> green(high)
    dr.rectangle([8, y0, 160 * sc - 8, y0 + 14], outline=(110, 110, 110))
    dr.rectangle([9, y0 + 1, 9 + int((160 * sc - 18) * frac), y0 + 13], fill=col)
    asc = "  ^ASCENDING" if a in (2, 6, 7, 10, 14, 15) else ""
    dr.text((9, y0 + 20), f"t={t}  O2={int(o2)}/63  depth={int(py)}  {ALE[a]}{asc}", fill=(240, 240, 240))
    return cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="seaquest_ccrl/data/raw_hf")
    ap.add_argument("--traj", default="traj_0000.npz")
    ap.add_argument("--t0", type=int, default=0)
    ap.add_argument("--t1", type=int, default=640)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--out-dir", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    d = np.load(os.path.join(args.root, args.traj))
    fr, ox, py, ac = d["frames"], d["oxygen"].astype(int), d["player_pos"][:, 1].astype(int), d["actions"].astype(int)
    t1 = min(args.t1, len(fr))

    # 1) sawtooth plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(13, 4))
    T = np.arange(len(ox))
    ax1.plot(T, ox, color="#1f77b4", lw=1.4, label="oxygen (0-63)")
    ax1.set_xlabel("step"); ax1.set_ylabel("oxygen", color="#1f77b4"); ax1.set_ylim(-2, 66)
    ax2 = ax1.twinx(); ax2.plot(T, py, color="#d62728", lw=1.0, alpha=.7, label="depth (player_y)")
    ax2.set_ylabel("depth player_y (small=surface)", color="#d62728"); ax2.invert_yaxis()
    ax1.set_title(f"{args.traj}: oxygen sawtooth vs depth — refill at surface, deplete ~0.12/step while deep")
    ax1.axvspan(args.t0, t1, color="gold", alpha=.08)
    fig.tight_layout(); fig.savefig(f"{args.out_dir}/oxygen_sawtooth.png", dpi=110); plt.close(fig)

    # 2) filmstrip: 8 frames spanning [t0,t1]
    ts = np.linspace(args.t0, t1 - 1, 8).astype(int)
    tiles = [annot(fr[t], ox[t], py[t], ac[t], t) for t in ts]
    w, h = tiles[0].size
    strip = Image.new("RGB", (w * 4 + 3 * 6, h * 2 + 6), (10, 10, 10))
    for i, im in enumerate(tiles):
        strip.paste(im, ((i % 4) * (w + 6), (i // 4) * (h + 6)))
    strip.save(f"{args.out_dir}/oxygen_filmstrip.png")

    # 3) animated gif over [t0,t1]
    gif = [annot(fr[t], ox[t], py[t], ac[t], t) for t in range(args.t0, t1, args.stride)]
    gif[0].save(f"{args.out_dir}/oxygen_episode.gif", save_all=True, append_images=gif[1:],
                duration=70, loop=0, optimize=True)
    print(f"WROTE {args.out_dir}/oxygen_sawtooth.png, oxygen_filmstrip.png, oxygen_episode.gif "
          f"({len(gif)} frames, t={args.t0}..{t1})")


if __name__ == "__main__":
    main()
