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
        actor_ckpt=None, action_mask=False):
    g = get_game(game_name)
    if actor_ckpt:
        from seaquest_ccrl.training.train_actor import load_actor
        actor, cfg, oracle = load_actor(actor_ckpt, device)
        pol = ActorGCPolicy(actor, cfg, device=device)
        kind = "ACTOR"
        if action_mask:
            print("(--action-mask ignored: only applies to the critic-argmax policy)")
            action_mask = False
    else:
        critic, cfg, oracle = load_critic(ckpt, device)
        pol = ContrastiveGCPolicy(critic, cfg, device=device)
        kind = "argmax-critic" + ("+mask" if action_mask else "")

    # Action-mask: restrict critic-argmax to LEGAL cardinal moves only, reusing the
    # demonstrator's EXACT pixel wall-probe (no reimplementation, identical legality).
    probe = CARD = None
    fallback_count = 0
    if action_mask:
        from mspacman_ccrl.policies.scripted_behavior import GhostAvoidingPolicy, CARDINAL
        from mspacman_ccrl import config as MC
        probe = GhostAvoidingPolicy(MC.DEFAULT)   # only its _blocked() wall-probe is used
        CARD = CARDINAL                            # {UP,RIGHT,LEFT,DOWN} -> (dx,dy); excludes diag/NOOP

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
                if action_mask:
                    # CONFOUNDING-SAFETY: legality uses ONLY Pac-Man position + wall
                    # geometry, probed on the GHOST-MASKED frame (ghosts inpainted out),
                    # so it reads NO ghost/masked-region info. Walls are identical in
                    # naive & oracle views and ghosts never read as pink walls -> the
                    # legal set leaks nothing about the confounder U.
                    p0 = st.get("player_pos")
                    scores = critic.score_all_actions(obs, cfg.normalize_goal(target), device)
                    legal_frame = g.mask_obs(frame, st)
                    legal = ([act for act, dvec in CARD.items()
                              if not probe._blocked(legal_frame, p0, dvec)]
                             if p0 is not None else [])
                    if legal:
                        a = int(legal[int(np.argmax([scores[k] for k in legal]))])
                    else:
                        a = int(np.argmax(scores)); fallback_count += 1   # empty corridor (shouldn't happen)
                else:
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
    if action_mask:
        print(f"  empty-candidate fallback steps (expected ~0): {fallback_count}")
    gdod = float(np.median(ghostdist_at_death)) if ghostdist_at_death else float("nan")
    if ghostdist_at_death:
        print(f"  ghost dist at death: median {gdod:.1f}px (confirms ghost contact if small)")
    return {
        "view": tag, "masked": bool(action_mask), "success": succ / n_episodes,
        "REACHED": outcomes.get("REACHED", 0), "DIED": outcomes.get("DIED", 0),
        "STUCK": outcomes.get("TIMEOUT/STUCK", 0),
        "NEAR_MISS": outcomes.get("TIMEOUT/NEAR_MISS", 0),
        "WANDER": outcomes.get("TIMEOUT/WANDER", 0),
        "stuck_frac": float(np.mean(stuck_fracs)), "closest": float(np.mean(min_dists)),
        "deaths_ep": float(np.mean(deaths_per_ep)), "fallback": fallback_count,
        "ghostdist_at_death": gdod, "n": n_episodes,
    }


def _compare(naive_ckpt, oracle_ckpt, game, episodes, max_steps):
    """Run masked + un-masked critic-argmax for BOTH views; print side-by-side."""
    rows = []
    for ck, view in [(naive_ckpt, "naive"), (oracle_ckpt, "oracle")]:
        for am in (False, True):
            rows.append(run(ck, game, episodes, max_steps, action_mask=am))
    hdr = (f"{'view':7s}{'mask':5s}{'success':9s}{'REACH':6s}{'DIED':5s}{'STUCK':6s}"
           f"{'NEARM':6s}{'WAND':5s}{'stuckfr':8s}{'closest':8s}{'deaths/ep':10s}{'fallbk':6s}")
    print("\n" + "=" * len(hdr))
    print("  MASKED vs UN-MASKED  (critic-argmax, legal-cardinal restriction)")
    print("=" * len(hdr)); print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['view']:7s}{('on' if r['masked'] else 'off'):5s}{r['success']:<9.3f}"
              f"{r['REACHED']:<6d}{r['DIED']:<5d}{r['STUCK']:<6d}{r['NEAR_MISS']:<6d}"
              f"{r['WANDER']:<5d}{r['stuck_frac']:<8.2f}{r['closest']:<8.1f}"
              f"{r['deaths_ep']:<10.2f}{r['fallback']:<6d}")
    print("=" * len(hdr))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="critic ckpt (single run)")
    ap.add_argument("--actor-ckpt", default=None, help="actor ckpt (BC/GCBC actor policy)")
    ap.add_argument("--naive-ckpt", default=None, help="naive critic ckpt (for masked-vs-unmasked table)")
    ap.add_argument("--oracle-ckpt", default=None, help="oracle critic ckpt (for masked-vs-unmasked table)")
    ap.add_argument("--game", default="mspacman")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--action-mask", action="store_true",
                    help="restrict critic-argmax to legal cardinal moves (eval-time only)")
    args = ap.parse_args()
    if args.naive_ckpt and args.oracle_ckpt:
        _compare(args.naive_ckpt, args.oracle_ckpt, args.game, args.episodes, args.max_steps)
    elif args.actor_ckpt:
        run(args.ckpt, args.game, args.episodes, args.max_steps, actor_ckpt=args.actor_ckpt)
    else:
        run(args.ckpt, args.game, args.episodes, args.max_steps, action_mask=args.action_mask)
