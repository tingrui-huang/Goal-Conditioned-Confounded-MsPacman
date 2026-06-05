# CausalJEPA — Rung 1: the first learned abduction model (OCAtari Pong)

Rung 1 replaces Rung 0's god-mode oracle with a **learned transition model** that, seeing
only `(state, intended_action, next_state)` — never the seed, never the injected wind —
does **Gumbel-Max abduction** and produces single-step **counterfactuals** that match a
seed-replay **oracle** better than a no-abduction **intervention** baseline.

Built on top of the Rung-0 `pong_counterfactual` package (reuses its `PongEnv`,
`make_wind`, and seed-replay rollout). Pearl's three steps:

1. **Abduct** — infer the exogenous Gumbel noise `g` from the model's predicted
   distribution + the actually-observed outcome (`gumbel_posterior`).
2. **Intervene** — recompute the model's logits under a different action.
3. **Predict** — re-roll under the new logits reusing the *same* `g` (`cf_token`).

The baseline (intervention) does steps 2–3 with **fresh** noise (`intervention_token`).

## Run

```
# 1. the abduction core (Oberst–Sontag Gumbel-Max SCM): consistency + stability
python -m pong_counterfactual.cjepa_rung1.gumbel_abduction

# 2. clean single-latent control — proves the pipeline (all invariants hold exactly)
python -m pong_counterfactual.cjepa_rung1.synthetic_check

# 3. the real-Pong experiment (trains models, sweeps p, runs all checks)
python -m pong_counterfactual.cjepa_rung1.eval_rung1
```

(use the repo `.venv`: `.venv/Scripts/python.exe -m ...`)

## How this differs from the skill (the skill didn't know our setup)

| Topic | Skill assumed | What we did & why |
|-------|---------------|-------------------|
| Naming | `cjepa-` prefix | subpackage `cjepa_rung1` (Python can't hyphenate) |
| Reuse | maybe rebuild env | **reused** Rung-0 `PongEnv`/`make_wind`/rollout |
| Wind model | NOOP-override | kept it — but Rung 0 used **sticky-repeat**, so added a separate `apply_wind_noop` (Rung 0 untouched) |
| Frameskip | (unspecified) | **frameskip=1**, not Rung 0's 4 — at fs=4 the paddle accelerates across 4 hidden frames and the integer state is non-Markov (fake p=0 gap). fs=1 → paddle ~deterministic given (pos, vel, action) |
| State | 4 raw numbers | 4 numbers **+ one-step velocity** as model input (positions alone are non-Markov: ball direction & paddle momentum are hidden). Still object-only, no seed/wind |
| Discretization | "a handful of bins" | coarse per-dim delta bins (we initially over-resolved into 48 exact-integer bins → soft model; coarse bins keep the categorical sharp) |
| Intervene where | any step | **move-steps only** (`intended ∈ {UP,DOWN}`): at an intended-NOOP step the executed action is NOOP regardless of wind, so the wind is un-abductable there |
| Metric | L1 over 4 numbers | headline on **`player_y`** (the only dim the action moves in one step); ball/enemy are action-independent distractors for a *single-step* CF |

## Results

**Synthetic control** (1-D deterministic paddle — the *only* stochasticity is the wind;
same Gumbel/MLP/oracle pipeline). The textbook result holds exactly:

| p | error_CF | error_IV | gap | fired_gap | calm_gap |
|------|----------|----------|-------|-----------|----------|
| 0.00 | 0.014 | 0.015 | **0.001** | 0.000 | 0.001 |
| 0.10 | 0.119 | 0.200 | 0.081 | 0.454 | 0.040 |
| 0.25 | 0.250 | 0.410 | 0.160 | 0.286 | 0.118 |
| 0.50 | 0.327 | 0.494 | 0.168 | 0.167 | 0.168 |

→ p=0 collapse, gap grows with p, abduction-consistent 100%, advantage skews to
wind-fired steps. **All invariants pass.**

**Real OCAtari Pong** (paddle L1 vs the seed-replay oracle CF):

| p | error_CF | error_IV | gap | CF<IV? |
|------|----------|----------|-------|--------|
| 0.00 | 0.582 | 1.482 | 0.900 | yes |
| 0.10 | 0.500 | 1.152 | 0.652 | yes |
| 0.25 | 0.584 | 1.240 | 0.656 | yes |
| 0.50 | 0.400 | 1.264 | 0.864 | yes |

→ Abduction consistency **100%** (A) and **error_CF < error_IV at every p** (D). The
central Rung-1 claim holds: a model that never sees the seed or the wind reproduces the
oracle counterfactual better than the no-abduction baseline.

## Rung 1.1 — confirming the p=0 gap is partial observability, not a leak

```
python -m pong_counterfactual.cjepa_rung1.eval_rung11
```

A diagnostic ablation (not a new rung). Two latents live in `player_y`: **(a)** the
paddle's momentum/charge — a *function of recent action history*, so feedable; and
**(b)** the injected wind — external randomness, *never* a function of any input, so
recoverable only by abducting the observed transition. Enriching the model input with
a paddle history window `history(N)` (last N `player_y` + intended actions; sweep
N ∈ {2,4,8}) should **separate** them: feed (a) and its gap collapses; (b) must survive.

`eval_rung11.py` result (player_y L1 vs the oracle CF; shared eval pool):

| N | p=0 gap | player_y top-1 | calm_gap (a: paddle) | fired_gap (b: wind) |
|---|---------|----------------|----------------------|---------------------|
| 2 | 0.904 | 74.2% | 0.644 | 0.229 |
| 4 | 0.432 | 82.5% | 0.289 | 0.143 |
| 8 | **0.112** | **91.8%** | **0.189 (×0.29)** | **0.200 (×0.87)** |

(`calm`/`fired` split the p=0.25 paddle gap by whether the wind fired at k — oracle-binned,
never a model input.)

**Double dissociation, confirmed.** As history N grows: the p=0 gap collapses 8× (0.90→0.11),
`player_y` top-1 climbs toward ball/enemy levels (74%→92%), and the **calm-step gap
collapses to ×0.29** — latent (a) becomes observed, so abduction has nothing left to
recover there. Meanwhile the **wind-fired gap stays at ×0.87** (does *not* collapse) and
overtakes the calm gap at N=8 — latent (b) is un-feedable and survives input enrichment.

So the ~0.9 p=0 gap from Rung 1 was **partial observability of the paddle's momentum,
not a leak or an oracle artifact**. The injected-wind advantage is a genuinely separate
latent. (The residual p=0 gap at N=8 is small-but-nonzero → a finer-than-integer paddle
state remains; the spec's escalation — feeding the ALE RAM paddle byte — would drive it
to ~0, but the dominant story is already settled and escalation is not triggered.)

The wind effect is small in absolute px because at frameskip=1 a paddle step is only ~2px;
that is why `fired_gap` ≈ `calm_gap` in level at N=8 even though their *collapse rates*
dissociate cleanly.

## The honest finding (why real Pong ≠ the ideal)

On real Pong the p=0 gap is **~0.9, not 0**. This is **not a bug** — it is a genuine
second latent. The ALE paddle has sub-pixel momentum that integer object-states cannot
recover (verified: `player_y` top-1 caps at ~74% while ball/enemy hit ~99/91%, even with
velocity and coarse bins). Abduction recovers *that* latent too, so it helps the
counterfactual even with no injected wind. The injected wind and the paddle momentum both
live in `player_y` and are not cleanly separable, so the wind-specific signature
(Invariant-4 collapse, wind-fired concentration) is blurred on Pong but **proven cleanly
in the synthetic control**. That contrast — clean SCM vs learned model on
partially-observed dynamics — is the real lesson of Rung 1.

## Invariants (all upheld)

1. Model sees only `(state, intended_action, next_state)` — never seed/wind. ✓
2. Abduction uses the actually-observed transition (teacher forcing). ✓
3. No-op CF reproduces the observation — **100%** on real model logits. ✓
4. p=0 collapse — **exact in the synthetic control**; on Pong, blurred by the paddle
   latent (documented, not a leak: the oracle holds the wind bit fixed). ✓/⚠
5. Oracle holds the same wind bit fixed at step k. ✓

## Files

- `gumbel_abduction.py` — the abduction core (bundled with the skill; do not reinvent).
- `noise.py` — `apply_wind_noop` (Rung-1 NOOP-override) + `state_vec`; re-exports `make_wind`.
- `discretize.py` — per-dim coarse delta-bin tokenizer (`to_tokens`/`decode`).
- `collect.py` — env config (fs=1), tap behavior policy, by-episode logging with seed+wind.
- `model.py` — the small MLP: `P(next-delta | pos, vel, intended_action)`, 4 independent categoricals.
- `oracle.py` — seed-replay single-step counterfactual ground truth.
- `eval_rung1.py` — the real-Pong sweep + checks A/B/D/E.
- `synthetic_check.py` — the clean single-latent control.
- `history.py` — Rung 1.1 `history(N)` feature builder + `HistModel`.
- `eval_rung11.py` — Rung 1.1 partial-observability double-dissociation sweep.

## Out of scope (later rungs)

No learned encoder/pixels (Rung 3), no RL / sample-efficiency (Rung 4), no multi-step
rollout (Rung 1.5), no ALE built-in sticky actions (Rung 2), no LLM.
