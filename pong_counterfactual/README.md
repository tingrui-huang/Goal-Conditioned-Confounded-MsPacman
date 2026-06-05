# pong_counterfactual

Second project: counterfactuals, starting from Pong. This folder is the warm-up —
get a feel for object state + determinism + a single-action intervention before
building anything heavier.

## Run

```
python -m pong_counterfactual.counterfactual_demo
```

(uses the repo `.venv`, which has OCAtari installed editable from `OC_Atari/`)

## What it does — the four steps

1. **Object state is just numbers.** Load `ALE/Pong-v5` via OCAtari and print the
   state. It's 4 numbers: `ball_x, ball_y, player_y (right paddle), enemy_y (left paddle)`.
2. **Record a trajectory.** `reset(seed=0)`, run a fixed action sequence, log the states.
3. **Determinism, seen not asserted.** `reset(seed=0)` and run the *same* actions again
   — the two trajectories are byte-identical at every step. This is what makes
   "all else equal" literally true.
4. **The counterfactual.** `reset(seed=0)`, change **only** the action at one step `k`,
   re-run. The trajectory is identical up to `k`, then forks. The demo scans every `k`
   and shows the most impactful single-action flip: the paddle forks immediately, and
   ~13 steps later the (now differently-placed) paddle hits-or-misses the ball, so the
   ball itself forks — branching futures from one changed action.

## Experiment 2 — wind: counterfactual vs intervention

```
python -m pong_counterfactual.wind_counterfactual
```

The first demo's env is fully deterministic, so "counterfactual" and "intervention"
are the same thing. This adds an **exogenous noise source we control** — a "wind":
a per-step boolean array, sampled from *our own* rng (independent of the env seed),
where `True` means "this step's action is interrupted and the previous action repeats"
(sticky). Because we generate it, we can **save and replay** it.

- **factual (F)**     : intended actions, wind `W0`
- **counterfactual (CF)** : change action at step `k`, **replay the same** `W0`
- **intervention (IV)**   : change action at step `k`, but a **fresh** wind `W1`

The punchline is `CF vs IV`, run at two noise levels:

| p    | F vs CF | CF vs IV | CF vs IV (avg over winds) |
|------|---------|----------|---------------------------|
| 0.00 | ~234    | **0**    | **0**                     |
| 0.25 | ~249    | **234**  | **~245**                  |

At `p=0` there is no luck to hold fixed, so CF and IV collapse into one run (div 0).
At `p=0.25` they take the *same action plan* but *different luck*, so they diverge —
and that gap is the slice of the outcome driven by **noise, not by your action**. A
proper counterfactual freezes the luck; an intervention re-rolls it.

> Note: the wind can land on step `k` itself and swallow the intervention
> (`executed[k]=executed[k-1]`). We pick a `k` the wind doesn't hit so the `F vs CF`
> column stays meaningful; the script prints `wind hit k?` so you can see when this happens.

## Files

- `env.py` — thin deterministic OCAtari Pong wrapper; `reset`/`step` return the 4-number state.
- `counterfactual_demo.py` — the four-step demo (object state, determinism, single-action fork).
- `wind_counterfactual.py` — the wind experiment (counterfactual vs intervention).
- `cjepa_rung1/` — **Rung 1**: the first *learned* Gumbel-Max abduction model (see its README).

## Rungs

- **Rung 0 (this folder):** oracle / god-mode counterfactuals — determinism, wind,
  counterfactual-vs-intervention by replaying vs re-rolling the wind.
- **Rung 1 (`cjepa_rung1/`):** a learned transition model that *infers* the wind from
  observations (abduction) and beats a no-abduction baseline against the oracle.
  Run `python -m pong_counterfactual.cjepa_rung1.eval_rung1`.

## Why determinism matters here

A counterfactual asks: *given everything identical up to step k, what if I had acted
differently at k?* With sticky actions off (`repeat_action_probability=0`), fixed
frameskip, and a seeded reset, "everything identical up to k" is exact — so any
divergence after k is caused **only** by the changed action. That clean isolation is
the whole reason to start in a deterministic sandbox like this.
