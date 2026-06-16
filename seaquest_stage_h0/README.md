# Seaquest Stage-H0 — HF Hostile-Field Candidate Qualification

Qualifies the Seaquest **hostile field** (enemies + enemy missiles) as a candidate
**unobserved confounder** for the validated HF-expert four-frame goal-conditioned
contrastive RL experiment. **No critic is trained in this stage**; the pipeline stops
after the qualification report.

Candidate hidden variable `U`:
- `U_enemy` = Shark + Submarine + SurfaceSubmarine
- `U_missile` = EnemyMissile
- `U_joint` = `U_enemy` + `U_missile`

The learner keeps observing Player, PlayerMissile, Diver, oxygen/HUD, and all other
ordinary pixels. The hostile field qualifies only if it has support, can be removed
without revealing its location / deleting visible objects, is hard to recover from the
hostile-removed four-frame state, and incrementally predicts both the HF action and
future player outcomes after conditioning on that removed state.

## This is the ISOLATED, validated path — NOT the old enemy build
Strictly unused here (see `artifacts/seaquest/hostile_h0/repo_audit.json`):
`collect_enemy.py`, `EnemyAvoidingPolicy`, `apply_enemy_mask`, `raw_enemy`, RHO/detour.
The behavior policy is the **frozen HF CleanRL Seaquest expert** (Stage-S0 teacher).
Metadata is stored **separately** from `raw_hf` (writing `enemy_bboxes` into `raw_hf`
would silently flip the legacy `SeaquestOfflineDataset` into its whole-bbox `enemy`
mode). Removal is **class-aware pixel removal**, never a whole-bbox fill.

## Layout
```
seaquest_stage_h0/
  collect_hf_hostile_metadata.py   # Part A: exact HF recollection + metadata + parity (Docker)
  validate_recollection_parity.py  # standalone pack re-validation (anywhere)
  export_hostile_h0_pack.py        # bundle metadata + checksums for Colab
  build_hostile_h0_notebook.py     # regenerate the thin notebook
seaquest_ccrl/hostile/             # schema, extraction, removal, features, data, probe_runner
seaquest_ccrl/scripts/             # h0_audit_schema, h0_audit_removal, h0_probe_*, h0_qualify
tests/                             # deterministic unit tests (no teacher/ALE needed)
notebooks/Seaquest_Hostile_H0_Qualification.ipynb   # THIN — every cell calls a module
```

## Local / Colab separation (Section 17)
**Local validated Docker only** (`seaquest-s0:ocatari`): HF recollection, OCAtari RAM
extraction, raw_hf parity, object overlays, metadata creation, removal audit, pack export.
**Colab / PyTorch only**: load frozen raw_hf + metadata, verify every raw SHA256,
construct removed four-frame states, run hiddenness / U→action / U→future probes,
write the qualification report. **Do not** install/run ALE, ROMs, OCAtari, EnvPool, JAX,
Flax, or the HF teacher in Colab; do not regenerate data in Colab.

## Run order
### Local (Docker `seaquest-s0:ocatari`, mounted at `/work`)
The recollection's CONFIG SOURCE OF TRUTH is `raw_hf/manifest.json` (base_seed,
max_steps, n_episodes). CLI flags are assertion-only — omit them, or pass the exact
manifest values, or the run aborts. The frozen teacher ckpt/adapter/port SHA256 are
asserted against the original manifest.
```bash
DOCKER="docker run --rm -v <repo>:/work -w /work -e PYTHONPATH=/work/OC_Atari seaquest-s0:ocatari"
# 1. exact HF recollection + hostile metadata + per-episode base-array parity
$DOCKER python seaquest_stage_h0/collect_hf_hostile_metadata.py     # -> HOSTILE_RECOLLECTION_ALIGNED
# 2. object-identity audit: observe RGB -> resolved_palette.json -> verify (no blind tolerance)
$DOCKER python seaquest_ccrl/scripts/h0_audit_schema.py             # -> HOSTILE_OBJECT_SCHEMA_ALIGNED
# 3. removal audit (uses resolved palette; invariants + grids)
$DOCKER python seaquest_ccrl/scripts/h0_audit_removal.py            # -> HOSTILE_REMOVAL_ALIGNED
# 4. support report (metadata-only; frozen thresholds)
$DOCKER python seaquest_ccrl/scripts/h0_support_report.py
# 5. standalone parity re-validation + pack export for Colab
$DOCKER python seaquest_stage_h0/validate_recollection_parity.py
$DOCKER python seaquest_stage_h0/export_hostile_h0_pack.py          # -> HOSTILE_EXPORT_OK
```
Steps 2–5 are pure numpy/cv2 over raw frames + metadata (no teacher); they may also run
on the host. The recollection (step 1) requires the Docker jax teacher.
### Colab (PyTorch only; upload raw_hf + the exported pack to Drive)
Open `notebooks/Seaquest_Hostile_H0_Qualification.ipynb`. It verifies every raw SHA,
builds removed four-frame states, runs the three probes and writes
`artifacts/seaquest/hostile_h0/hostile_qualification.{json,md}`.

## Stop conditions (hard, never silently skipped)
`HOSTILE_RECOLLECTION_NOT_IDENTICAL`, `HOSTILE_OBJECT_SCHEMA_INVALID`,
`HOSTILE_REMOVAL_INVALID`, plus the scientific outcomes in `h0_qualify.ALLOWED`.

The smoke test (`build_hostile_h0_notebook.py --smoke` / the notebook's smoke cell)
is an engineering check only and never emits a qualification claim.
```
