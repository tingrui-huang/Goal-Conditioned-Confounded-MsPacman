"""Build the reachable-target pool for eval.

Uniformly sampling goals in the bounding box puts most targets inside maze WALLS
(unreachable) -> eval success is capped low for ANY policy, masking the
naive-vs-oracle gap. Instead, sample eval goals from positions Pac-Man ACTUALLY
visited in the dataset (guaranteed on-corridor and reachable). We grid visited
player positions to a small cell size and keep one representative per cell for
roughly-uniform spatial coverage of the reachable maze.

Output: mspacman_ccrl/data/reachable_targets.npy  (N,2) float32  (committed; small)
"""
import os
import glob
import argparse

import numpy as np

from mspacman_ccrl import config as C


def build(root: str = None, grid: int = 4, out: str = None) -> str:
    root = root or C.DATA_ROOT
    out = out or os.path.join(root, "reachable_targets.npy")
    pos = []
    for f in sorted(glob.glob(os.path.join(root, "traj_*.npz"))):
        p = np.load(f)["player_pos"]
        pos.append(p[np.isfinite(p[:, 0])])
    pos = np.concatenate(pos, axis=0)
    # one representative position per grid cell (mean of in-cell visited points)
    keys = (pos // grid).astype(np.int64)
    cells = {}
    for k, p in zip(map(tuple, keys), pos):
        cells.setdefault(k, []).append(p)
    reps = np.array([np.mean(v, axis=0) for v in cells.values()], dtype=np.float32)
    np.save(out, reps)
    print(f"reachable targets: {len(reps)} positions (grid={grid}px) -> {out}")
    print(f"  x {reps[:,0].min():.1f}..{reps[:,0].max():.1f}  "
          f"y {reps[:,1].min():.1f}..{reps[:,1].max():.1f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    build(args.root, args.grid, args.out)
