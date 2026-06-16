"""Phase 2.1b step 3 — define & visualize the RAW-frame regions a human can verify.

Renders, for the SAME raw frame: original | oxygen-bar masked | bottom-HUD masked |
gameplay-only crop | pixel-difference map, with the region rectangles drawn on the original.
Saves the raw-frame coordinates + a per-row temporal-variance profile (the empirical basis
for the boundaries). Coordinates are NOT chosen from probe performance.
"""
import os, json, argparse
import numpy as np
from PIL import Image, ImageDraw

from seaquest_ccrl.probes import oxy4_regions as R

OUTDIR = "artifacts/seaquest/oxygen_4frame/leakage/source_audit"


def _up(a, s=3):
    return Image.fromarray(a).resize((a.shape[1] * s, a.shape[0] * s), Image.NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="seaquest_ccrl/data/raw_hf")
    ap.add_argument("--traj", default="traj_0000.npz")
    ap.add_argument("--out-dir", default=OUTDIR)
    args = ap.parse_args()
    os.makedirs(f"{args.out_dir}/figures", exist_ok=True)
    d = np.load(os.path.join(args.root, args.traj))
    frames, ox = d["frames"], d["oxygen"]
    t = int(np.argsort(ox)[len(ox) // 2])                 # a median-oxygen frame
    raw = frames[t]

    panels = {
        "original": raw,
        "oxygen_bar_masked": R.transform_raw(raw, "oxybar_masked"),
        "bottom_hud_masked": R.transform_raw(raw, "bottomhud_masked"),
        "gameplay_crop": R.transform_raw(raw, "gameplay_crop"),
    }
    # pixel-difference map: which pixels the gameplay-only path removes (top+bottom HUD)
    diff = np.abs(raw.astype(np.int16) - R.transform_raw(raw, "top_and_bottom_masked").astype(np.int16))
    diff = (diff.sum(-1).clip(0, 255)).astype(np.uint8)
    diff = np.stack([diff] * 3, -1)

    # draw region rectangles on a copy of the original
    annot = _up(raw, 3).convert("RGB")
    dr = ImageDraw.Draw(annot)
    colors = {"oxy_bar_rect": (255, 0, 0), "bottom_hud_rect": (0, 255, 0),
              "top_hud_rect": (255, 255, 0), "gameplay_crop": (0, 200, 255)}
    for key, col in colors.items():
        x, y, w, h = getattr(R, key.upper())
        dr.rectangle([x * 3, y * 3, (x + w) * 3 - 1, (y + h) * 3 - 1], outline=col, width=2)
    annot.save(f"{args.out_dir}/figures/regions_annotated.png")

    # compose the 5-panel strip (pad the crop back to raw height for alignment)
    def to_raw_canvas(a):
        if a.shape[:2] != (R.RAW_H, R.RAW_W):
            cv = np.zeros((R.RAW_H, R.RAW_W, 3), np.uint8)
            x, y, w, h = R.GAMEPLAY_CROP
            cv[y:y + h, x:x + w] = a
            return cv
        return a
    order = ["original", "oxygen_bar_masked", "bottom_hud_masked", "gameplay_crop"]
    imgs = [_up(to_raw_canvas(panels[k])) for k in order] + [_up(diff)]
    W = sum(im.width for im in imgs) + 4 * 8
    strip = Image.new("RGB", (W, imgs[0].height), (30, 30, 30))
    xo = 0
    for im in imgs:
        strip.paste(im, (xo, 0)); xo += im.width + 8
    strip.save(f"{args.out_dir}/figures/regions_5panel.png")

    # temporal-variance row profile (empirical basis for the row boundaries)
    std = frames.astype(np.float32).std(0).mean(-1)        # (210,160)
    rowstd = std.mean(1).tolist()

    man = R.regions_manifest()
    man["figure_frame"] = {"traj": args.traj, "t": t, "oxygen": int(ox[t])}
    man["row_temporal_std"] = rowstd
    man["figures"] = ["figures/regions_5panel.png", "figures/regions_annotated.png"]
    json.dump(man, open(f"{args.out_dir}/regions.json", "w"), indent=2)
    print(json.dumps({k: man[k] for k in ["oxy_bar_rect", "bottom_hud_rect",
                                          "top_hud_rect", "gameplay_crop"]}, indent=2))
    print(f"WROTE {args.out_dir}/regions.json + figures/regions_5panel.png + regions_annotated.png")


if __name__ == "__main__":
    main()
