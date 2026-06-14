# M7 Colab readiness — staged, resumable, disconnection-safe

The M7 pipeline is split into independently-runnable stages so a Colab disconnect never
loses a run. Checkpoints and artifacts are written frequently; every stage reads the
previous stage's artifacts from disk. **Do not run a monolithic job.**

## Required Python packages
```
torch  numpy  gymnasium  ale-py  opencv-python  scipy  scikit-learn  huggingface_hub  omegaconf
```
(`omegaconf` is needed by the teacher import; `wandb` is auto-stubbed if absent. `pandas`/`matplotlib`
are only needed for earlier-milestone plotting, not M7.)

## Environment / path configuration
| what | how |
|---|---|
| Teacher repo (PRIVATE, never committed) | `export CONF_AGENT_TEACHER_PATH=/content/Confounded-Agent-Distillation-main/Confounded-Agent-Distillation-main` (the inner dir containing `teacher/` and `envs/`) |
| Dataset | the M4 dataset must live at `artifacts/pong_action_gate/m4/full/episodes/*.npz` (relative to repo root, i.e. the CWD when launching). It is gitignored — upload it or regenerate with `python -m pong_action_gate.data.collect --base-seed 6000 --episodes 100 --tag full`. |
| HF checkpoint | downloaded automatically to the HF cache on first teacher load (set `HF_HOME` to a Drive path to persist across sessions). |
| M7 checkpoints + artifacts | `artifacts/pong_action_gate/m7/<critic>_seed<seed>/` — point this at a mounted Drive folder to survive disconnects (e.g. symlink `artifacts/` to Drive). |

## Stage 1 — GPU critic training + frozen-checkpoint action diagnostics
Run **per training seed** (e.g. seeds 0, 1, 2). Training saves a checkpoint every `--ckpt-every`
steps and is resumable; the checkpoint is selected by validation loss BEFORE any diagnostic.

```bash
# train (GPU). Repeat for --critic pixel.
python -m pong_action_gate.train.train_critic \
    --critic state --seed 0 --n-episodes 100 --val-frac 0.2 \
    --steps 50000 --batch 256 --lr 3e-4 \
    --eval-every 500 --ckpt-every 500 --device cuda

# RESUME after a disconnect (same command + --resume):
python -m pong_action_gate.train.train_critic --critic state --seed 0 \
    --n-episodes 100 --val-frac 0.2 --steps 50000 --device cuda --resume

# frozen-checkpoint action diagnostics + episode-level bootstrap CI (GPU or CPU)
python -m pong_action_gate.train.gate_diag \
    --critic state --seed 0 --n-episodes 100 --val-frac 0.2 \
    --device cuda --n-boot 5000 --per-ep-batch 256
```
Outputs per seed: `ckpt_*.pt`, `val_curve.json`, `selected.json`, `gate_diagnostics.json`.

## Stage 2 — CPU emulator-branch rollout (uses the already-selected checkpoint)
ALE stepping is CPU-bound; run this on a CPU runtime after Stage 1 selected a checkpoint.
```bash
python -m pong_action_gate.train.emulator_branch \
    --critic state --seed 0 --device cpu \
    --n-states 200 --n-cont 8 --horizon 400
```
Output per seed: `emulator_branch.json` (clone/restore reproducibility, top-1 agreement,
Spearman, regret, all with clustered-by-state bootstrap CIs).

## Stage 3 — Final aggregation from separately produced artifacts
```bash
python -m pong_action_gate.train.m7 --mode aggregate --seeds 0 1 2
```
Reads each seed's `gate_diagnostics.json` + `emulator_branch.json` and writes
`artifacts/pong_action_gate/m7/aggregate_report.json` (per-seed + across-seed).

## Expected GPU vs CPU stages
| stage | device | why |
|---|---|---|
| 1 critic training | **GPU** | CNN/MLP minibatch training |
| 1 action diagnostics | GPU or CPU | a few forward passes on the frozen net |
| 2 emulator branch | **CPU** | ALE clone/restore + teacher rollouts are CPU-bound |
| 3 aggregation | CPU | reads JSON artifacts only |

## Disconnection safety checklist
- Mount Drive and place `artifacts/` (or at least `artifacts/pong_action_gate/m7/`) on it.
- Always pass `--resume` when relaunching Stage 1; it picks up the latest `ckpt_*.pt`.
- Stages 2 and 3 only read finalized artifacts, so they are safe to re-run idempotently.
- Persist the HF cache (`HF_HOME`) on Drive to avoid re-downloading the teacher checkpoint.

## Predeclared checkpoint discipline (M7)
The checkpoint is selected by **minimum validation loss over the fixed step budget** (or a
fixed early-stopping rule) — `selected.json` records this. Action diagnostics are run ONCE on
the frozen, already-selected checkpoint. A checkpoint is never chosen because its
action-shuffle delta became positive.
```
