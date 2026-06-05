"""
Gumbel-Max SCM abduction for counterfactual world models.

Core idea (Oberst & Sontag 2019; the Gumbel-Max SCM):
  A categorical sample is modelled as   token = argmax_i ( logits_i + g_i ),
  where  g ~ Gumbel(0,1) i.i.d.  is the EXOGENOUS NOISE U of the SCM.

This module does the ONE piece the counterfactual-llms repo does NOT give you:
ABDUCTION FROM AN EXTERNALLY-OBSERVED OUTCOME. In our setting the factual token
comes from the ENVIRONMENT, not from our own sampler, so we cannot "store the
RNG state". Instead we infer the posterior over g, given that a particular
token was observed to win (top-down / truncated-Gumbel sampling).

Pearl's three steps:
  1. ABDUCT    : g  = gumbel_posterior(logits_factual, observed_token)
  2. INTERVENE : recompute logits under the new action  ->  logits_cf
  3. PREDICT   : token_cf = cf_token(logits_cf, g)
"""

import numpy as np


def gumbel_posterior(logits, obs_idx, rng):
    """
    Sample the posterior over the Gumbel noise vector g, given that category
    `obs_idx` was the observed argmax of (logits + g).

    logits  : (K,) array, the model's predicted log-scores for P(next | s, a)
              (need NOT be normalised).
    obs_idx : int, the category actually observed to occur.
    rng     : np.random.Generator
    returns : g, a (K,) Gumbel-noise vector consistent with the observation.
    """
    logits = np.asarray(logits, dtype=np.float64)
    K = logits.shape[0]
    T = np.empty(K)

    # 1. The WINNING perturbed score is the max over all categories.
    #    Marginally that max ~ Gumbel(location = logsumexp(logits)).
    lse = _logsumexp(logits)
    M = lse - np.log(-np.log(_u(rng)))     # one Gumbel(lse) draw = the max value
    T[obs_idx] = M                         # the winner sits exactly at the max

    # 2. Every LOSER's perturbed score is a Gumbel(logit_i) TRUNCATED below M.
    for i in range(K):
        if i == obs_idx:
            continue
        v = _u(rng)
        # truncated-Gumbel inverse-CDF, conditioned on being < M:
        T[i] = logits[i] - np.log(np.exp(-(M - logits[i])) - np.log(v))

    # 3. Recover the noise:  g_i = (perturbed score) - logit_i
    return T - logits


def cf_token(logits, g):
    """Re-roll with NEW logits but the SAME abducted noise g -> counterfactual token."""
    return int(np.argmax(np.asarray(logits, dtype=np.float64) + g))


def intervention_token(logits, rng):
    """Baseline: change the action but draw FRESH noise (no abduction) -> intervention token."""
    logits = np.asarray(logits, dtype=np.float64)
    g_fresh = -np.log(-np.log(rng.random(logits.shape[0]).clip(1e-9, 1 - 1e-9)))
    return int(np.argmax(logits + g_fresh))


# ---- numerically-safe helpers ----
def _u(rng):
    return float(np.clip(rng.random(), 1e-9, 1 - 1e-9))   # uniform in (0,1), away from 0/1

def _logsumexp(x):
    m = x.max()
    return m + np.log(np.sum(np.exp(x - m)))


# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # The model's predicted distribution for the FACTUAL (state, action):
    logits_factual = np.log(np.array([0.1, 0.6, 0.3]))   # K = 3
    observed = 2                                          # env happened to give token 2

    # TEST 1 (the key correctness check): abduction must REPRODUCE the
    # observation when we re-roll under the SAME logits (a no-op counterfactual
    # has to equal the factual). If this ever fails, your abduction is wrong.
    for _ in range(2000):
        g = gumbel_posterior(logits_factual, observed, rng)
        assert cf_token(logits_factual, g) == observed
    print("TEST 1 passed: a no-op counterfactual always reproduces the observed token.")

    # TEST 2: counterfactual (reuse abducted noise) vs intervention (fresh noise),
    # under a DIFFERENT action's predicted distribution.
    logits_cf = np.log(np.array([0.5, 0.3, 0.2]))        # new action -> new distribution
    n = 40000
    cf_hits = np.zeros(3)
    iv_hits = np.zeros(3)
    for _ in range(n):
        g = gumbel_posterior(logits_factual, observed, rng)   # abduct from the SAME observation
        cf_hits[cf_token(logits_cf, g)] += 1                  # counterfactual: reuse g
        iv_hits[intervention_token(logits_cf, rng)] += 1      # intervention: fresh g

    print("intervention  P(token | do a')                          :", np.round(iv_hits / n, 3),
          " <- just the new action's marginal [0.5, 0.3, 0.2]")
    print("counterfactual P(token | do a', GIVEN we saw token 2)   :", np.round(cf_hits / n, 3),
          " <- stickier toward token 2 (counterfactual stability)")
