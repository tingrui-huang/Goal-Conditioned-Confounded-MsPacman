"""Classify WHY each goal-reaching eval episode ends, on a trained checkpoint.

Outcomes per episode:
  - REACHED   : got within eps of the goal (success)
  - DIED      : env terminated (Pac-Man lost all lives = ghost contact) before reaching
  - TIMEOUT   : hit max_steps, still alive, never reached. Sub-typed:
      * STUCK     : barely moved (wall-thrashing) -- stuck_frac high
      * NEAR_MISS : closest approach within 1.5*eps but never sealed it
      * WANDER    : moved around but never got close

Also reports: mean deaths-per-episode (life drops, incl. respawns), mean closest
distance, mean stuck fraction. Tells you whether the ~50% non-success is ghosts
(DIED) or navigation (TIMEOUT/STUCK), which is the thing to act on.

Run:
  python -m seaquest_ccrl.scripts.eval_failure_modes --ckpt PATH --game mspacman --episodes 50
"""
import argparse
from collections import Counter

import numpy as np

from seaquest_ccrl.training.train_critic import load_critic
from seaquest_ccrl.games import get_game
from seaquest_ccrl.evaluation.policy import ContrastiveGCPolicy, ActorGCPolicy


def run(ckpt, game_name, n_episodes=50, max_steps=400, device="cpu", seed=0,
        actor_ckpt=None):
    g = get_game(game_name)
    if actor_ckpt:
        from seaquest_ccrl.training.train_actor import load_actor
        actor, cfg, oracle = load_actor(actor_ckpt, device)
        pol = ActorGCPolicy(actor, cfg, device=device)
        kind = "ACTOR"
    else:
        critic, cfg, oracle = load_critic(ckpt, device)
        pol = ContrastiveGCPolicy(critic, cfg, device=device)
        kind = "argmax-critic"
    env = g.make_env()
    rng = np.random.default_rng(seed)
    eps = g.eps

    outcomes = Counter()
    deaths_per_ep, min_dists, stuck_fracs, steps_used = [], [], [], []
    ghostdist_at_death = []
    try:
        for ep in range(n_episodes):
            target = g.sample_target(rng)
            frame, st = env.reset(seed=int(rng.integers(1 << 30)))
            lives0 = st.get("lives")
            min_d = np.inf; reached = False; ended = None
            prev = None; stuck = 0; nsteps = 0; deaths = 0; prev_lives = lives0
            last_ghost_d = np.inf
            for t in range(max_steps):
                obs = frame if oracle else g.mask_obs(frame, st)
                a = pol.act(obs, target)
                frame, _, term, trunc, st = env.step(a)
                nsteps += 1
                p = st.get("player_pos")
                gc = st.get("ghost_centers") or []
                if p is not None:
                    d = float(np.hypot(p[0] - target[0], p[1] - target[1]))
                    min_d = min(min_d, d)
                    if gc:
                        last_ghost_d = min(np.hypot(g0[0]-p[0], g0[1]-p[1]) for g0 in gc)
                    if prev is not None and np.hypot(p[0]-prev[0], p[1]-prev[1]) < 0.5:
                        stuck += 1
                    prev = p
                    if d <= eps:
                        reached = True; ended = "REACHED"; break
                cur_lives = st.get("lives")
                if prev_lives is not None and cur_lives is not None and cur_lives < prev_lives:
                    deaths += 1
                prev_lives = cur_lives if cur_lives is not None else prev_lives
                if term or trunc:
                    ended = "DIED"; ghostdist_at_death.append(last_ghost_d); break
            if ended is None:
                ended = "TIMEOUT"
            # sub-type timeouts
            sf = stuck / max(nsteps, 1)
            if ended == "TIMEOUT":
                if sf > 0.5:
                    ended = "TIMEOUT/STUCK"
                elif min_d <= 1.5 * eps:
                    ended = "TIMEOUT/NEAR_MISS"
                else:
                    ended = "TIMEOUT/WANDER"
            outcomes[ended] += 1
            deaths_per_ep.append(deaths); min_dists.append(min_d)
            stuck_fracs.append(sf); steps_used.append(nsteps)
    finally:
        env.close()

    tag = "oracle" if oracle else "naive"
    succ = outcomes.get("REACHED", 0)
    print(f"\nckpt={actor_ckpt or ckpt}")
    print(f"  policy={kind}  view={tag}  episodes={n_episodes}  max_steps={max_steps}  eps={eps:.0f}px")
    print(f"  SUCCESS rate: {succ/n_episodes:.2f} ({succ}/{n_episodes})")
    print("  outcome breakdown:")
    for k in ["REACHED", "DIED", "TIMEOUT/STUCK", "TIMEOUT/NEAR_MISS", "TIMEOUT/WANDER"]:
        v = outcomes.get(k, 0)
        print(f"    {k:18s}: {v:3d}  ({100*v/n_episodes:.0f}%)")
    print(f"  mean deaths/episode (life drops): {np.mean(deaths_per_ep):.2f}")
    print(f"  mean closest dist to goal: {np.mean(min_dists):.1f}px  (eps={eps:.0f})")
    print(f"  mean stuck fraction: {np.mean(stuck_fracs):.2f}  mean steps used: {np.mean(steps_used):.0f}")
    if ghostdist_at_death:
        print(f"  ghost dist at death: median {np.median(ghostdist_at_death):.1f}px "
              f"(confirms ghost contact if small)")
    return outcomes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="critic ckpt (argmax-critic policy)")
    ap.add_argument("--actor-ckpt", default=None, help="actor ckpt (BC/GCBC actor policy)")
    ap.add_argument("--game", default="mspacman")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=400)
    args = ap.parse_args()
    run(args.ckpt, args.game, args.episodes, args.max_steps, actor_ckpt=args.actor_ckpt)
