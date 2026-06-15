"""Seaquest Stage-S0: feasibility and instrumentation audit (isolated package).

This package is intentionally separate from `seaquest_ccrl/` (historical, read-only).
It trains NOTHING and implements NO causal correction. It only audits teachers,
the environment, timing, clone/restore, forced-first-action branch effects, and
observational oxygen relationships, to decide whether to proceed to dataset
collection and vanilla contrastive-critic experiments.

Do NOT inherit assumptions from seaquest_ccrl (oxygen-as-confounder, 18 actions,
mask coordinates, scripted policy, reward=0, etc.). Everything is re-audited here.
"""
__all__ = []
