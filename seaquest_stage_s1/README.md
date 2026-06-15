# Seaquest Stage-S1 — Vanilla State Critic (Colab training + forced-branch alignment)

Answers: **can a vanilla Eysenbach-style state critic actually learn and USE the action
input on the validated Seaquest observational data?** No oxygen-confounder investigation
in this stage. No pixels, no actor, no causal/robust losses, no data regeneration in Colab.

## Two strictly separated parts

### Part A — local (this repo, validated Docker only)
`export_colab_pack.py` runs in the `seaquest-s0:ocatari` image with the **frozen
Stage-S0 teacher** to produce a portable, self-contained data pack — it does NOT train.
- Regenerates the corrected **O-Sampled** rollout (seed 104, frozen `teacher.sample_action`).
- Builds the observational dataset: learner **state vector** at t, one-hot action, and
  **H=16 future goals** in the `no_player` and `world_only` views (reused EXACTLY from
  corrected S0.5).
- Builds the **branch pack**: per-anchor, per locally-supported-action H=16 futures from
  forced-first-action branches (frozen teacher continuation) + valid mask + support counts.
- Episode-level **70/15/15 split (seed 1601)**; normalization fit on **train episodes only**.
- Packs everything into `artifacts/seaquest/stage_s1/seaquest_s1_colab_pack.zip` with a
  manifest (git commit, S0.5 hashes, schema, dims, censoring rules, per-file SHA256).

`validate_colab_pack.py` fails on hash mismatch, wrong dims, NaN/Inf, action ∉ [0,17],
episode leakage across split, invalid H=16 future, or schema/column mismatch.

Run (local):
```
docker run --rm -v <repo>:/work -w /work -e PYTHONPATH=/work/OC_Atari \
  seaquest-s0:ocatari python seaquest_stage_s1/export_colab_pack.py
python seaquest_stage_s1/validate_colab_pack.py
```

### Part B — Google Colab (PyTorch only)
`notebooks/Seaquest_Stage_S1_Vanilla_State_Critic.ipynb` is self-contained: it loads the
frozen pack (upload or Drive), trains the critic, evaluates, applies the gates, and
exports. It **must not** install/use EnvPool, OCAtari, ALE, ROMs, JAX/Flax, or the CleanRL
teacher, and must **never regenerate data**.

## Model (frozen architecture)
`f(s,a,g) = phi(s,a)·psi(g)/sqrt(128)`. `phi`: state⊕one-hot(18) → 256→256→128.
`psi`: goal → 256→256→128. Separate weights. Plus a matched-capacity **no-action baseline**
`f(s,g)`. Eysenbach in-batch **sigmoid-BCE NCE** (identity labels). Adam 3e-4, batch 256,
≤60 epochs, patience 8, grad-clip 10, **checkpoint on lowest val NCE only**.

## Predeclared seed-0 models & gates
Model A (action, no_player), Model B (no-action, no_player), Model C (action, world_only).
Gates C1 optimization, C2 action-sensitivity (global+local shuffle, zero-action), C3 action
beats no-action, C4 forced-branch alignment (no_player), C5 world-only alignment. Seeds
[1,2] are exposed but only run manually after seed-0 passes.

Files: `critic.py`, `losses.py`, `evaluation.py` mirror the notebook's inline code (kept in
sync by the notebook builder) and support local unit tests. All Stage-S1 work stays
**uncommitted** until the notebook + seed-0 results are human-reviewed.
