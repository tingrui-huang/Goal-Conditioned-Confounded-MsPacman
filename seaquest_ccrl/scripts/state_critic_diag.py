"""STEP 4 diagnostic: train the SAME contrastive critic on STATE observations
(OCAtari Pac-Man position) instead of pixels, on the SAME expert data, then run the
SAME action-sensitivity probe. Separates cause (1) data/objective from (4) pixels.

- state = Pac-Man (x,y) (normalized). action = demonstrated action (one-hot).
  goal = future (x,y) (normalized, geometric hindsight). f = phi(state,a) . psi(g).
- Same NCE loss (sum-BCE over candidates, goal_radius soft target).
- If the STATE critic becomes action-SENSITIVE (directional-acc high, action-std
  comparable to goal-std) -> the pixel critic's blindness was (4) pixel localization.
- If the STATE critic is ALSO action-blind -> it's (1): the expert-data + contrastive
  objective simply don't teach action-values, pixels irrelevant.

Tiny MLP -> runs on CPU in minutes. No retraining of anything else.
"""
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mspacman_ccrl import config as C

DEVICE = "cpu"
GX = (C.GOAL_X_RANGE[0], C.GOAL_X_RANGE[1]); GY = (C.GOAL_Y_RANGE[0], C.GOAL_Y_RANGE[1])
LO = np.array([GX[0], GY[0]], np.float32); SPAN = np.array([GX[1]-GX[0], GY[1]-GY[0]], np.float32)
NB = C.NB_ACTIONS
GAMMA = 0.99
GOAL_RADIUS = C.EPS   # 8px soft target, same as pixel critic


def load():
    pos, act, lens = [], [], []
    for f in sorted(glob.glob("mspacman_ccrl/data/raw/traj_*.npz")):
        d = np.load(f); p = d["player_pos"].astype(np.float32); a = d["actions"].astype(np.int64)
        m = np.isfinite(p[:, 0])
        # keep contiguous; fill non-finite by forward fill (rare)
        for i in range(1, len(p)):
            if not np.isfinite(p[i, 0]):
                p[i] = p[i-1]
        pos.append(p); act.append(a); lens.append(len(a))
    return pos, act, np.array(lens)


class StateCritic(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.sa = nn.Sequential(nn.Linear(2 + NB, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, d))
        self.g = nn.Sequential(nn.Linear(2, 256), nn.ReLU(), nn.Linear(256, d))

    def saf(self, s, a):
        oh = F.one_hot(a, NB).float()
        return self.sa(torch.cat([s, oh], 1))

    def forward(self, s, a, g):
        return torch.einsum("ik,jk->ij", self.saf(s, a), self.g(g))

    def scores_all_actions(self, s_norm, g_norm):
        s = torch.tensor(s_norm, dtype=torch.float32)[None].repeat(NB, 1)
        a = torch.arange(NB)
        g = torch.tensor(g_norm, dtype=torch.float32)[None]
        with torch.no_grad():
            return (self.saf(s, a) @ self.g(g).T).squeeze(1).numpy()


def train(pos, act, lens, steps=20000, B=256, seed=0):
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    offs = np.concatenate([[0], np.cumsum(lens)[:-1]])
    P = np.concatenate(pos); A = np.concatenate(act)
    Pn = (P - LO) / SPAN
    Pn_t = torch.tensor(Pn, dtype=torch.float32); A_t = torch.tensor(A)
    crit = StateCritic(); opt = torch.optim.Adam(crit.parameters(), 3e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for step in range(1, steps + 1):
        ep = rng.integers(0, len(lens), B)
        t = (rng.random(B) * lens[ep]).astype(int)
        k = rng.geometric(1 - GAMMA, B)
        fut = np.minimum(t + k, lens[ep] - 1)
        gi = offs[ep] + t; gf = offs[ep] + fut
        s = Pn_t[gi]; a = A_t[gi]; g = Pn_t[gf]
        logits = crit(s, a, g)
        raw = g * torch.tensor(SPAN) + torch.tensor(LO)
        D = torch.cdist(raw, raw)
        labels = (D <= GOAL_RADIUS).float()
        loss = bce(logits, labels).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 4000 == 0:
            with torch.no_grad():
                pick = logits.argmax(1)
                acc = (torch.norm(raw[pick]-raw, dim=1) <= GOAL_RADIUS).float().mean().item()
            print(f"  step {step}/{steps} loss {loss.item():.3f} soft-diag-acc {acc:.3f}")
    return crit


def probe(crit, pos, lens):
    CARD = {1: (0, -1), 2: (1, 0), 3: (-1, 0), 4: (0, 1)}
    def card_to(p, g):
        dx, dy = g[0]-p[0], g[1]-p[1]
        return (2 if dx > 0 else 3) if abs(dx) >= abs(dy) else (4 if dy > 0 else 1)
    rng = np.random.default_rng(1)
    res = {"near": ([], 0, 0), "far": ([], 0, 0)}
    n_std = {"near": [], "far": []}; n_dir = {"near": [0, 0], "far": [0, 0]}
    goal_std = []
    for ei in rng.integers(0, len(lens), 200):
        p = pos[ei]; T = len(p)
        if T < 160:
            continue
        t = int(rng.integers(0, T - 150))
        for kind, k in [("near", int(rng.integers(2, 7))), ("far", int(rng.integers(60, 140)))]:
            g = p[t + k]
            sc = crit.scores_all_actions((p[t] - LO) / SPAN, (g - LO) / SPAN)
            n_std[kind].append(float(np.std(sc)))
            cid = [1, 2, 3, 4]; best = cid[int(np.argmax([sc[c] for c in cid]))]
            n_dir[kind][1] += 1; n_dir[kind][0] += (best == card_to(p[t], g))
        # goal sensitivity (vary goal, fix state+action)
        s = (p[t] - LO) / SPAN
        gg = rng.random((9, 2)).astype(np.float32)
        with torch.no_grad():
            sa = crit.saf(torch.tensor(s, dtype=torch.float32)[None], torch.zeros(1, dtype=torch.long))
            gv = (crit.g(torch.tensor(gg)) @ sa.T).squeeze(1).numpy()
        goal_std.append(float(np.std(gv)))
    print("\n=== STATE-CRITIC action-sensitivity probe (compare to pixel critic) ===")
    for kind in ["near", "far"]:
        print(f"  {kind:4s} goal: action-std {np.mean(n_std[kind]):.3f}  directional-acc {n_dir[kind][0]/n_dir[kind][1]:.2f} (chance .25)")
    print(f"  goal-std (vary goal): {np.mean(goal_std):.3f}")
    a_all = np.mean(n_std['near'] + n_std['far'])
    print(f"  action/goal sensitivity ratio: {a_all/np.mean(goal_std):.3f}  (pixel critic was 0.03)")
    print("\n  PIXEL critic was: action-std ~0.11, dir-acc ~0.37, ratio 0.03 (action-BLIND)")
    print("  => if STATE dir-acc >> 0.37 (e.g. >0.7) and ratio >> 0.03: pixel(4) was the blocker.")
    print("  => if STATE also ~0.37 / ratio ~0.03: it's (1) data/objective, pixels irrelevant.")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=20000)
    args = ap.parse_args()
    print("loading positions..."); pos, act, lens = load()
    print(f"{len(lens)} episodes, {lens.sum()} steps. training STATE critic ({args.steps} steps)...")
    crit = train(pos, act, lens, steps=args.steps)
    probe(crit, pos, lens)
