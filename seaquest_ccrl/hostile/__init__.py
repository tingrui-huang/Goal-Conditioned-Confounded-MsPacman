"""Stage-H0 hostile-field qualification library (ISOLATED from the old enemy build).

This package is the NEW, validated scientific path for the Seaquest hostile-field
confounder qualification (Stage-H0). It deliberately does NOT import or reuse:

  * seaquest_ccrl.collect.collect_enemy / EnemyAvoidingPolicy   (scripted demonstrator)
  * seaquest_ccrl.data.masking.apply_enemy_mask                 (whole-bbox water fill)
  * raw_enemy / RHO / detour logic

The behavior policy is the FROZEN HF CleanRL Seaquest expert (Stage-S0 teacher). Hostile
metadata is collected by exact recollection of the frozen raw_hf path and stored in a
SEPARATE directory (never written into raw_hf, which would silently flip
SeaquestOfflineDataset into its legacy 'enemy' mode). Removal is class-aware PIXEL removal,
never a whole-bbox fill.

Modules
-------
schema      : frozen class IDs, padded-metadata encode/decode + validation
extraction  : OCAtari objects -> per-timestep padded metadata (Docker side)
removal      : class-aware pixel removal + changed-mask audit (Docker side)
features    : deterministic fixed four-frame U feature representation (Colab side)
data        : raw_hf + metadata loader, removed four-frame state builder, split (Colab side)
"""
