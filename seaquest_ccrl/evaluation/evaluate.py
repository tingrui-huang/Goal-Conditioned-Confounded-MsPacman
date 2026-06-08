"""Goal-reaching evaluation in the real OCAtari Seaquest env (acceptance check G).

Runs the trained policy online: sample a target position, reset the env, and at
each step feed the policy the MASKED frame (naive critic) or UNMASKED frame
(oracle critic) -- matching the view its critic was trained on -- plus the goal.
Success = submarine comes within EPS of the target before the episode ends.

The naive-vs-oracle success-rate gap is the confounding signal.
"""
import numpy as np

from seaquest_ccrl.evaluation.policy import ContrastiveGCPolicy


def evaluate(critic, cfg, game, oracle: bool, n_episodes: int = 50,
             max_steps: int = 600, eps: float = None, device: str = "cpu",
             temperature: float = 0.0, seed: int = 0, verbose: bool = True,
             policy=None) -> dict:
    """Online goal-reaching eval in `game`'s real env. Naive critics see the
    game's masked view (game.mask_obs), oracle critics see the unmasked frame.
    `policy` overrides the default argmax-over-critic policy (e.g. an ActorGCPolicy)."""
    eps = game.eps if eps is None else eps
    rng = np.random.default_rng(seed)
    env = game.make_env()
    if policy is None:
        policy = ContrastiveGCPolicy(critic, cfg, device=device, temperature=temperature)

    successes = 0
    steps_to_goal = []
    min_dists = []
    tag = "oracle" if oracle else "naive"
    try:
        for ep in range(n_episodes):
            target = game.sample_target(rng)
            frame, state = env.reset(seed=int(rng.integers(1 << 30)))
            reached = False
            min_d = np.inf
            for t in range(max_steps):
                obs = frame if oracle else game.mask_obs(frame, state)
                action = policy.act(obs, target)
                frame, _, term, trunc, state = env.step(action)
                pos = state["player_pos"]
                if pos is not None:
                    d = float(np.hypot(pos[0] - target[0], pos[1] - target[1]))
                    min_d = min(min_d, d)
                    if d <= eps:
                        reached = True
                        steps_to_goal.append(t + 1)
                        break
                if term or trunc:
                    break
            successes += int(reached)
            min_dists.append(min_d)
            if verbose:
                print(f"[{tag}] ep {ep+1:3d}/{n_episodes}  "
                      f"target=({target[0]:.0f},{target[1]:.0f})  "
                      f"reached={reached}  min_dist={min_d:.1f}")
    finally:
        env.close()

    rate = successes / n_episodes
    res = {
        "oracle": oracle,
        "success_rate": rate,
        "successes": successes,
        "n_episodes": n_episodes,
        "mean_min_dist": float(np.mean(min_dists)),
        "median_steps_to_goal": float(np.median(steps_to_goal)) if steps_to_goal else None,
    }
    if verbose:
        print(f"[{tag}] success_rate={rate:.3f} ({successes}/{n_episodes})  "
              f"mean_min_dist={res['mean_min_dist']:.1f}")
    return res


if __name__ == "__main__":
    import argparse
    from seaquest_ccrl.training.train_critic import load_critic
    from seaquest_ccrl.games import get_game
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--game", default="seaquest")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=600)
    args = ap.parse_args()
    critic, cfg, oracle = load_critic(args.ckpt)
    evaluate(critic, cfg, get_game(args.game), oracle,
             n_episodes=args.episodes, max_steps=args.max_steps)
