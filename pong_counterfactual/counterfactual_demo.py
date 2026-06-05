"""The four-step counterfactual warm-up for Pong.

Run:  python -m pong_counterfactual.counterfactual_demo

  STEP 1  Load Pong, print the object state -> see that "state" is just 4 numbers.
  STEP 2  reset(seed=0), run a fixed action sequence, record the trajectory.
  STEP 3  reset(seed=0), run the SAME actions -> confirm the two runs are identical
          (this is determinism, seen with your own eyes).
  STEP 4  reset(seed=0), change ONLY the action at step k -> watch the trajectory
          fork from that moment on. That fork is the counterfactual.

Why this matters: a counterfactual asks "what WOULD have happened if, all else
equal up to step k, I had acted differently at step k?". Determinism (steps 2-3)
is what makes "all else equal" literally true here -- so any divergence in step 4
is caused ONLY by the changed action, nothing else.
"""
import numpy as np

from pong_counterfactual.env import PongEnv, ACTIONS


def rollout(env, actions, seed=0):
    """Run a fixed action sequence from a seeded reset; return list of states."""
    s = env.reset(seed=seed)
    traj = [s]
    for a in actions:
        s, _, done = env.step(a)
        traj.append(s)
        if done:
            break
    return traj


def state_str(s):
    ball = s["ball"]
    ball = "    none    " if ball is None else f"({ball[0]:6.1f},{ball[1]:6.1f})"
    py = "  -" if s["player_y"] is None else f"{s['player_y']:5.1f}"
    ey = "  -" if s["enemy_y"] is None else f"{s['enemy_y']:5.1f}"
    return f"ball={ball}  player_y={py}  enemy_y={ey}"


def states_equal(a, b):
    return (a["ball"] == b["ball"]
            and a["player_y"] == b["player_y"]
            and a["enemy_y"] == b["enemy_y"])


def first_divergence(t1, t2):
    for i in range(min(len(t1), len(t2))):
        if not states_equal(t1[i], t2[i]):
            return i
    return None


def main():
    env = PongEnv()

    # ---- STEP 1: the "state" is just a few numbers --------------------------
    print("=" * 72)
    print("STEP 1  Load Pong and print the object state")
    print("=" * 72)
    s0 = env.reset(seed=0)
    print("  right after reset(seed=0):")
    print("   ", state_str(s0))
    # take a few steps so the ball is served and visible
    for _ in range(20):
        s0, _, _ = env.step(1)  # FIRE serves the ball
    print("  after serving the ball:")
    print("   ", state_str(s0))
    print("  => the entire state is 4 numbers: ball_x, ball_y, player_y, enemy_y\n")

    # A fixed, reproducible action sequence (no RNG: deterministic by construction).
    # Mix of UP / DOWN / NOOP so the paddle and ball actually move around.
    rng = np.random.default_rng(0)
    actions = [1] * 20 + list(rng.integers(0, 6, size=300))  # serve, then 300 steps
    actions = [int(a) for a in actions]

    # ---- STEP 2 & 3: determinism -------------------------------------------
    print("=" * 72)
    print("STEP 2 + 3  Same seed + same actions twice -> identical trajectory")
    print("=" * 72)
    traj_a = rollout(env, actions, seed=0)
    traj_b = rollout(env, actions, seed=0)
    div = first_divergence(traj_a, traj_b)
    if div is None and len(traj_a) == len(traj_b):
        print(f"  ran {len(actions)} actions twice from reset(seed=0)")
        print(f"  trajectories are IDENTICAL at every one of {len(traj_a)} states.")
        print("  => deterministic. 'all else equal' is literally true here.\n")
    else:
        print(f"  !! diverged at step {div} -- determinism assumption broken.\n")

    # ---- STEP 4: the counterfactual ----------------------------------------
    print("=" * 72)
    print("STEP 4  Change ONLY the action at one step k -> watch it fork")
    print("=" * 72)
    traj_factual = rollout(env, actions, seed=0)

    def ball_divergence(traj1, traj2):
        """Max |Δx|+|Δy| between two ball paths (only where both have a ball)."""
        g = 0.0
        for a, b in zip(traj1, traj2):
            if a["ball"] and b["ball"]:
                g = max(g, abs(a["ball"][0] - b["ball"][0]) + abs(a["ball"][1] - b["ball"][1]))
        return g

    # The k-th action only moves the paddle; it forks the BALL only if the paddle
    # was about to touch it. So we scan candidate k's and keep the single-step
    # flip whose downstream ball-path divergence is largest -- the "interesting"
    # counterfactual. (Still a single-action change, exactly as asked.)
    best = None  # (ball_gap, k, cf_a, traj_cf, fork)
    for K in range(20, len(actions) - 1):
        orig_a = actions[K]
        cf_a = 3 if orig_a != 3 else 2  # flip to a clearly different move (UP/DOWN)
        cf_actions = list(actions)
        cf_actions[K] = cf_a
        traj_cf = rollout(env, cf_actions, seed=0)
        gap = ball_divergence(traj_factual, traj_cf)
        fork = first_divergence(traj_factual, traj_cf)
        if best is None or gap > best[0]:
            best = (gap, K, cf_a, traj_cf, fork)

    _, K, cf_a, traj_cf, fork = best
    orig_a = actions[K]
    print(f"  scanned every step; the most impactful single-action change is at k={K}")
    print(f"  factual   action[{K}] = {orig_a} ({ACTIONS[orig_a]})")
    print(f"  counterfac action[{K}] = {cf_a} ({ACTIONS[cf_a]})\n")
    print(f"  trajectories identical for steps 0..{fork-1}, then fork at step {fork}")
    print(f"  (we changed step {K}; the state reflects it one step later)\n")
    print("  step | factual                                         | counterfactual")
    print("  -----+-------------------------------------------------+----------------")
    lo, hi = max(0, fork - 1), min(len(traj_factual), fork + 8)
    for i in range(lo, hi):
        mark = "  <-- fork" if i == fork else ""
        print(f"  {i:4d} | {state_str(traj_factual[i])} | {state_str(traj_cf[i])}{mark}")

    # the cascade: the paddle forks at step k+1, but the BALL only forks once the
    # (now differently-placed) paddle actually touches it -- possibly much later.
    ball_fork = None
    max_gap = 0.0
    for i, (fa, cf) in enumerate(zip(traj_factual, traj_cf)):
        if fa["ball"] and cf["ball"]:
            gap = abs(fa["ball"][0] - cf["ball"][0]) + abs(fa["ball"][1] - cf["ball"][1])
            if gap > 0 and ball_fork is None:
                ball_fork = i
            max_gap = max(max_gap, gap)
    print()
    print(f"  paddle forks at step {fork}; the BALL forks later at step {ball_fork}")
    print("  -> the changed paddle position propagated through a bounce into the ball:")
    if ball_fork is not None:
        for i in (ball_fork - 1, ball_fork, ball_fork + 1):
            if 0 <= i < min(len(traj_factual), len(traj_cf)):
                mark = "  <-- ball forks" if i == ball_fork else ""
                print(f"  {i:4d} | {state_str(traj_factual[i])} | {state_str(traj_cf[i])}{mark}")
    print(f"\n  max ball |dx|+|dy| between the two worlds: {max_gap:.1f} px")
    print("  => one different action at step k, everything else equal, branching futures.")
    print("     That branch is exactly what a counterfactual model has to predict.")

    env.close()


if __name__ == "__main__":
    main()
