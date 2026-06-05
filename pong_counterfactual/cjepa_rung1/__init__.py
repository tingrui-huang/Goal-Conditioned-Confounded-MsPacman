"""CausalJEPA Rung 1 on OCAtari Pong.

The first LEARNED transition model that abducts exogenous noise (Gumbel-Max SCM)
from observations only, and reproduces the seed-replay oracle counterfactual more
accurately than a no-abduction intervention baseline.

Built ON TOP of the Rung-0 `pong_counterfactual` package (env, make_wind,
seed-replay rollout). The only conflict with Rung 0 is the noise model: Rung 0 uses
sticky-repeat wind; Rung 1 uses NOOP-override wind (see noise.apply_wind_noop and
the README). Everything else is reused.
"""
