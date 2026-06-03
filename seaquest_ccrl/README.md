# Seaquest — Level-1 Confounded Goal-Conditioned Offline Dataset

Built with the `ccrl-seaquest-build` skill. This package **builds the env and
collects a static offline dataset only**. The contrastive critic, world model,
worst-case / Manski operator, hindsight positive-sampling, and any training are
**out of scope** (Level 2/3).

## The task in one paragraph

OCAtari reads ground-truth Seaquest RAM **env-side** to (a) label the achieved goal =
submarine position and (b) read the confounder **U = oxygen level**. The learner's
observation is *always* the masked pixel frame `(210,160,3)` — never an object vector.
A scripted behavior policy navigates toward a per-episode target and **surfaces when
oxygen `< THETA`**, so the unobserved oxygen confounds the data (U→A). Oxygen depletion
can also end the episode (U→S′, drown). The oxygen bar is masked out of the pixels, so
the confounder is unidentifiable from the observation — exactly the setup for causal
contrastive / pessimistic RL downstream.

## Layout

| File | Role |
|------|------|
| `config.py` | Locked assumptions, measured `OXY_MASK_RECT=(46,162,69,16)`, `THETA`, `EPS`, seeds |
| `envs/seaquest_gc.py` | OCAtari wrapper. `reset()/step()` → **unmasked** frame + `state{player_pos, oxygen, done}` |
| `policies/scripted_behavior.py` | Oxygen-aware greedy navigator; `THETA` = confounding-strength knob |
| `collect/collect_dataset.py` | By-trajectory rollout logging → one `.npz` per episode |
| `data/masking.py` | `apply_oxygen_mask()` zeroes the full bar strip (oracle = identity) |
| `data/dataset.py` | Loader; mask applied **at load**; `oracle=True` skips it |
| `scripts/explore_oxygen.py` | How `OXY_MASK_RECT` was measured |
| `scripts/validate.py` | Acceptance checks A–F |

## Invariants enforced

1. **Observation = masked pixels.** Object positions/oxygen are env-side metadata only.
2. **Causal structure holds.** U→A (surfacing depends on oxygen), U→S′ (drown ends episode), U unobserved (masked frame has no oxygen info).
3. **Full strip masked**, not the filled width — the filled/empty boundary is the leak; the fixed rect covers the whole strip + label.
4. **Oracle-for-free.** Unmasked frames stored → oracle view recoverable without re-collecting.
5. **By-trajectory, time-ordered.** One episode = one trajectory; boundaries preserved.
6. **No frame generation.** Real emulator frames only.
7. **Single oxygen→action channel.** Only the surfacing rule reads oxygen.

## Reproduce

```bash
# 1. measure the mask rect (already encoded in config.py)
python -m seaquest_ccrl.scripts.explore_oxygen

# 2. collect the dataset (per-trajectory .npz under data/raw/)
python -m seaquest_ccrl.collect.collect_dataset --episodes 40

# 3. acceptance checks A–F (all must pass)
python -m seaquest_ccrl.scripts.validate
```

## Consume the dataset

```python
from seaquest_ccrl.data.dataset import SeaquestOfflineDataset

ds = SeaquestOfflineDataset("seaquest_ccrl/data/raw", oracle=False)  # masked = learner view
for traj in ds.trajectories():
    obs  = traj["obs"]            # (T,210,160,3) masked frames  <- learner observation
    ag   = traj["achieved_goal"]  # (T,2) submarine pos          <- goal label (NOT in obs)
    oxy  = traj["oxygen"]         # (T,) confounder (analysis only)
    done = traj["done"]
# oracle view (oxygen visible) for the naive-vs-causal comparison:
ds_oracle = SeaquestOfflineDataset("seaquest_ccrl/data/raw", oracle=True)
```

## Stored fields per trajectory (`traj_XXXX.npz`)

`frames (T,210,160,3 uint8, unmasked)`, `actions (T,)`, `player_pos (T,2)`,
`oxygen (T,)`, `done (T,)`, `target (T,2)`, `theta`. A `manifest.json` records the
config + per-episode step counts.

## Config knobs

- `THETA` (default 20): surfacing threshold = confounding strength. Sweep later for a dose-response curve (`--theta`).
- `EPS` (default 8.0): success radius — stored for method-side evaluation; collection does **not** compute success.
- `OXY_MASK_RECT`: fixed; re-measure with `explore_oxygen.py` if the ROM/HUD changes.
