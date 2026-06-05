"""Counterfactual vs intervention, separated by a recordable "wind".

Run:  python -m pong_counterfactual.wind_counterfactual

The point of the previous demo (counterfactual_demo.py) was: change one action,
watch the trajectory fork. But that env is fully deterministic -- there is no
randomness to hold fixed, so "counterfactual" and "intervention" mean the same
thing there. This script adds an EXOGENOUS noise source we control -- a "wind" --
so the two come apart:

  WIND  = a per-step array of bools, sampled from OUR OWN rng (independent of the
          env seed). True at step t means "the wind interrupts you this step":
          your intended action is dropped and last step's action repeats (sticky).
          Because we generate it ourselves, we can SAVE it and REPLAY it.

  factual (F)         : intended actions, wind W0
  counterfactual (CF) : change action at step k, REPLAY the SAME wind W0
  intervention (IV)   : change action at step k, but a FRESH wind W1

The experiment is CF vs IV, run at two noise levels:
  p=0     -> no wind -> W0 == W1 == all-False -> CF and IV are identical (div = 0).
  p=0.25  -> wind matters -> replaying W0 (CF) differs from re-rolling W1 (IV).

That gap is the whole idea of a counterfactual: it isn't just "a different action",
it's "a different action with the same luck". Re-rolling the luck (IV) is a
different question, and the gap between them measures how much the noise -- not
the action -- drives the outcome.
"""
import numpy as np

from pong_counterfactual.env import PongEnv


# ---- 1. make a wind: a recordable, replayable array -----------------------
def make_wind(T, p, wind_seed):
    rng = np.random.default_rng(wind_seed)   # our own rng, independent of env seed
    return rng.random(T) < p                 # one bool per step: True = interrupted


# ---- 2. apply the wind to intended actions -> executed actions ------------
def apply_wind(intended, wind):
    executed = list(intended)
    for t in range(1, len(executed)):
        if wind[t]:
            executed[t] = executed[t - 1]    # interrupted = repeat last action (sticky)
    return executed


# ---- 3. rollout: executed actions -> state trajectory ---------------------
def rollout(env, executed, env_seed=0):
    s = env.reset(seed=env_seed)
    traj = [s]
    for a in executed:
        s, _, done = env.step(int(a))
        traj.append(s)
        if done:
            break
    return traj


def max_div(a, b):
    """Max ball |dx|+|dy| between two trajectories (where both have a ball)."""
    g = 0.0
    for s1, s2 in zip(a, b):
        if s1["ball"] and s2["ball"]:
            g = max(g, abs(s1["ball"][0] - s2["ball"][0]) + abs(s1["ball"][1] - s2["ball"][1]))
    return g


def run_at(env, intended, k, a_prime, p, T, n_iv_winds=8):
    # 4. factual: intended actions under this run's wind W0 (saved)
    W0 = make_wind(T, p, wind_seed=100)
    exec_F = apply_wind(intended, W0)
    traj_F = rollout(env, exec_F, env_seed=0)

    # 5. counterfactual: change step k, REPLAY the same wind W0
    intended_cf = list(intended)
    intended_cf[k] = a_prime
    exec_CF = apply_wind(intended_cf, W0)
    traj_CF = rollout(env, exec_CF, env_seed=0)

    # 6. intervention: same changed action, but a FRESH wind W1
    #    (also average over several fresh winds so the number isn't one noisy draw)
    W1 = make_wind(T, p, wind_seed=101)
    exec_IV = apply_wind(intended_cf, W1)
    traj_IV = rollout(env, exec_IV, env_seed=0)
    cf_vs_iv_single = max_div(traj_CF, traj_IV)

    iv_divs = []
    for s in range(101, 101 + n_iv_winds):
        Ws = make_wind(T, p, wind_seed=s)
        traj = rollout(env, apply_wind(intended_cf, Ws), env_seed=0)
        iv_divs.append(max_div(traj_CF, traj))
    cf_vs_iv_avg = float(np.mean(iv_divs))

    # 7. compare
    return {
        "p": p,
        "interrupts_W0": int(W0.sum()),
        "wind_hit_k": bool(W0[k]),               # did the wind swallow our intervention?
        "F_vs_CF": max_div(traj_F, traj_CF),     # the usual single-action fork
        "CF_vs_IV": cf_vs_iv_single,             # <-- the punchline (single W1)
        "CF_vs_IV_avg": cf_vs_iv_avg,            # <-- punchline, averaged over winds
    }


def main():
    env = PongEnv()

    # intended action sequence: serve, then a fixed pseudo-random plan
    rng = np.random.default_rng(0)
    intended = [1] * 20 + [int(a) for a in rng.integers(0, 6, size=300)]
    T = len(intended)

    # the single-action change we study. k=47 is impactful AND not swallowed by
    # the p=0.25 wind (W0[46] happens to be True; W0[47] is False), so the
    # F-vs-CF side column stays meaningful at both noise levels.
    k, a_prime = 47, 2  # at step 47, do UP instead of the intended action

    print("=" * 72)
    print("Wind = recordable exogenous noise. CF replays it; IV re-rolls it.")
    print(f"intended length T={T}, intervene at k={k} (action -> {a_prime}=UP)")
    print("=" * 72)
    print(f"{'p':>6} | {'#interrupts':>11} | {'wind hit k?':>11} | "
          f"{'F vs CF':>8} | {'CF vs IV':>8} | {'CF vs IV (avg)':>14}")
    print("-" * 72)
    for p in (0.0, 0.25):
        r = run_at(env, intended, k, a_prime, p, T)
        print(f"{r['p']:>6.2f} | {r['interrupts_W0']:>11d} | {str(r['wind_hit_k']):>11} | "
              f"{r['F_vs_CF']:>8.1f} | {r['CF_vs_IV']:>8.1f} | {r['CF_vs_IV_avg']:>14.1f}")

    print("-" * 72)
    print("Read the CF-vs-IV columns (the punchline):")
    print("  p=0.00 : no wind -> every IV wind equals W0 -> CF and IV identical -> 0.0.")
    print("           Without noise there is nothing to 'hold fixed': counterfactual")
    print("           and intervention collapse into the same thing.")
    print("  p=0.25 : wind bites. CF (replay W0) and IV (fresh wind) take the SAME")
    print("           action plan but DIFFERENT luck -> they diverge (>0).")
    print("           That gap is the slice of the outcome driven by noise, not by")
    print("           your action. A counterfactual freezes the luck; an")
    print("           intervention re-rolls it. (F-vs-CF stays ~the action fork at")
    print("           both p because we chose a k the wind does not swallow.)")
    env.close()


if __name__ == "__main__":
    main()
