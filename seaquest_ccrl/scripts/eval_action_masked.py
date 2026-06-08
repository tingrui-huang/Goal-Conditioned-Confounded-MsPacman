"""Cheapest extraction diagnostic: critic-argmax eval, but restrict candidates to
LEGAL cardinal moves (the 4 directions the demonstrator used, minus into-wall ones).

No actor, no lambda, no discrete-reparametrization -- isolates "is the critic usable
for navigation once we stop it from picking OOD/illegal actions?" If this lifts the
success rate (esp. oracle), the critic is fine and plain argmax-over-all-9-actions
was the problem; the wall-banging was an extraction artifact, not a critic failure.

Walls are read from the (always-visible) env frame via the demonstrator's pixel
wall-probe; critic scores use the policy's own view (masked for naive).
"""
import argparse
from collections import Counter

import numpy as np

from seaquest_ccrl.training.train_critic import load_critic
from seaquest_ccrl.games import get_game
from mspacman_ccrl.policies.scripted_behavior import CARDINAL  # {UP:..,RIGHT:..,LEFT:..,DOWN:..}
from mspacman_ccrl import config as MC


def _blocked(frame, pos, d, wall, probe_px=6):
    H, W = frame.shape[:2]
    px = int(round(pos[0] + probe_px * d[0])); py = int(round(pos[1] + probe_px * d[1]))
    if px < 0 or px >= W or py < 0 or py >= H:
        return True
    return bool(np.abs(frame[py, px].astype(np.int16) - wall).sum() < 60)


def run(ckpt, game_name="mspacman", n_episodes=40, max_steps=400, device="cpu", seed=0):
    g = get_game(game_name)
    critic, cfg, oracle = load_critic(ckpt, device)
    env = g.make_env(); rng = np.random.default_rng(seed); eps = g.eps
    wall = np.asarray(MC.WALL_COLOR, dtype=np.int16)
    outcomes = Counter(); stuck_fracs = []; min_ds = []
    try:
        for ep in range(n_episodes):
            target = g.sample_target(rng)
            frame, st = env.reset(seed=int(rng.integers(1 << 30)))
            reached = False; ended = None; prev = None; stuck = 0; n = 0; min_d = np.inf
            for t in range(max_steps):
                obs = frame if oracle else g.mask_obs(frame, st)
                scores = critic.score_all_actions(obs, cfg.normalize_goal(target), device)
                p = st.get("player_pos")
                if p is not None:
                    legal = [a for a, d in CARDINAL.items() if not _blocked(frame, p, d, wall)]
                    if legal:
                        a = legal[int(np.argmax([scores[k] for k in legal]))]
                    else:
                        a = int(np.argmax(scores))
                else:
                    a = int(np.argmax(scores))
                frame, _, term, trunc, st = env.step(a); n += 1
                p2 = st.get("player_pos")
                if p2 is not None:
                    d = float(np.hypot(p2[0]-target[0], p2[1]-target[1])); min_d = min(min_d, d)
                    if p is not None and np.hypot(p2[0]-p[0], p2[1]-p[1]) < 0.5: stuck += 1
                    if d <= eps:
                        reached = True; ended = "REACHED"; break
                if term or trunc:
                    ended = "DIED"; break
            if ended is None:
                ended = "TIMEOUT"
            outcomes[ended] += 1; stuck_fracs.append(stuck/max(n, 1)); min_ds.append(min_d)
    finally:
        env.close()
    tag = "oracle" if oracle else "naive"
    succ = outcomes.get("REACHED", 0)
    print(f"ckpt={ckpt}  view={tag}")
    print(f"  ACTION-MASKED success: {succ/n_episodes:.3f} ({succ}/{n_episodes})")
    print(f"  outcomes: REACHED {outcomes.get('REACHED',0)}  DIED {outcomes.get('DIED',0)}  "
          f"TIMEOUT {outcomes.get('TIMEOUT',0)}")
    print(f"  mean stuck frac {np.mean(stuck_fracs):.2f} (was ~0.6 with free argmax)  "
          f"mean closest {np.mean(min_ds):.1f}px")
    return succ / n_episodes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--game", default="mspacman")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--max-steps", type=int, default=400)
    args = ap.parse_args()
    run(args.ckpt, args.game, args.episodes, args.max_steps)
