"""M5 — scoring-event goal sampler (no critic training).

Achieved goal = POST-event score difference d_{j+1} = agent_score[j] - opp_score[j]
at a future scoring event j within the SAME episode.

Convention (see schema.TRANSITION_CONVENTION): for an anchor transition t, action a_t
produces reward r_{t+1}=reward[t]; if t is itself a scoring event, that event (index t,
horizon 0) is the "immediately resulting event from a_t". Whether it is eligible as the
positive goal is controlled by `include_immediate` (default True — this is the most
action-relevant signal; the alternative requires horizon >= 1).

Modes:
  * next_score_event              : first event j >= t (or > t if include_immediate=False).
  * multi_event_future_score_diff : sample among future events j >= t, either
                                    'uniform' over events or 'stratified_by_direction'
                                    (50/50 agent/opponent) — the stratified target is
                                    reported separately and never silently substituted.

Negatives (for NCE) are drawn from events in OTHER episodes; a false negative is a
negative whose scalar goal value equals the positive goal value (duplicate collision).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ART_ROOT = Path("artifacts/pong_action_gate/m4")


@dataclass
class EpisodeIndex:
    event_idx: np.ndarray      # (E,) indices of scoring events
    goal_value: np.ndarray     # (E,) post-event score diff at each event
    direction: np.ndarray      # (E,) +1 agent-scored, -1 opponent-scored
    length: int


def build_episode_index(ep: Dict[str, Any]) -> EpisodeIndex:
    ise = np.asarray(ep["is_scoring_event"]).astype(bool)
    reward = np.asarray(ep["reward"])
    post = np.asarray(ep["agent_score"]).astype(np.int64) - np.asarray(ep["opp_score"]).astype(np.int64)
    ev = np.where(ise)[0]
    return EpisodeIndex(event_idx=ev, goal_value=post[ev],
                        direction=np.sign(reward[ev]).astype(np.int64), length=len(ise))


def duplicate_corrected_targets(goal_values) -> np.ndarray:
    """B×B NCE target: target[i,j] = 1 iff goal_values[j] == goal_values[i] (EXACT equality).

    This is the PRIMARY valid objective. The discrete score-difference goal repeats across
    anchors (~5% pairwise collision at B=256), so a diagonal-only target would label many
    identical goal inputs as negatives -> contradictory supervision. Exact equality only;
    NO goal radius, NO label smoothing. Diagonal is always 1.
    """
    g = np.asarray(goal_values).reshape(-1)
    return (g[:, None] == g[None, :]).astype(np.float32)


def diagonal_targets(n: int) -> np.ndarray:
    """Diagonal-only target. DIAGNOSTIC BASELINE ONLY: it mislabels duplicate-goal columns
    as negatives and is NOT the valid objective. Use duplicate_corrected_targets for training."""
    return np.eye(n, dtype=np.float32)


def next_event_for(eidx: EpisodeIndex, t: int, include_immediate: bool = True
                   ) -> Optional[Tuple[int, int, int, int]]:
    """Return (event_index, goal_value, direction, horizon) for the next event, or None.

    Indexing: the anchor is (o_t, a_t). A scoring event STORED at transition index j occurs
    AFTER the action, i.e. at time j+1 (reward[j] = r_{j+1}). `horizon = j - t` counts the
    intervening transitions; horizon=0 means the event is stored at index t itself (the direct
    outcome of a_t, realized at t+1) — it does NOT mean the goal is observed before the action.
    """
    lo = t if include_immediate else t + 1
    mask = eidx.event_idx >= lo
    if not mask.any():
        return None
    k = np.argmax(mask)   # first True
    j = int(eidx.event_idx[k])
    return j, int(eidx.goal_value[k]), int(eidx.direction[k]), j - t


def future_events_for(eidx: EpisodeIndex, t: int, include_immediate: bool = True) -> np.ndarray:
    lo = t if include_immediate else t + 1
    return np.where(eidx.event_idx >= lo)[0]   # positions into eidx arrays


class ScoringGoalSampler:
    def __init__(self, episodes: List[Dict[str, Any]]):
        self.eps = episodes
        self.index = [build_episode_index(e) for e in episodes]
        # flat pool of ALL event goal values (for negatives + collision stats)
        self.all_goal_values = np.concatenate([ix.goal_value for ix in self.index]) \
            if self.index else np.array([], dtype=np.int64)

    # --- positive goal sampling -------------------------------------------- #
    def sample_positive(self, ep_i: int, t: int, mode: str, rng: np.random.Generator,
                        include_immediate: bool = True, weighting: str = "uniform"
                        ) -> Optional[Dict[str, Any]]:
        eidx = self.index[ep_i]
        if mode == "next_score_event":
            res = next_event_for(eidx, t, include_immediate)
            if res is None:
                return None
            j, g, d, h = res
        elif mode == "multi_event_future_score_diff":
            pos = future_events_for(eidx, t, include_immediate)
            if len(pos) == 0:
                return None
            if weighting == "uniform":
                k = pos[rng.integers(len(pos))]
            elif weighting == "stratified_by_direction":
                dirs = eidx.direction[pos]
                # choose a direction class present, 50/50, then uniform within it
                classes = [c for c in (1, -1) if (dirs == c).any()]
                c = classes[rng.integers(len(classes))]
                cand = pos[dirs == c]
                k = cand[rng.integers(len(cand))]
            else:
                raise ValueError(f"unknown weighting {weighting}")
            j = int(eidx.event_idx[k]); g = int(eidx.goal_value[k])
            d = int(eidx.direction[k]); h = j - t
        else:
            raise ValueError(f"unknown mode {mode}")
        return {"ep": ep_i, "t": t, "event_index": j, "goal_value": g,
                "direction": d, "horizon": h}

    def valid_anchors(self, ep_i: int, mode: str, include_immediate: bool = True) -> np.ndarray:
        """Transitions t that have at least one eligible future event (no cross-episode)."""
        eidx = self.index[ep_i]
        if len(eidx.event_idx) == 0:
            return np.array([], dtype=np.int64)
        last_event = int(eidx.event_idx[-1])
        hi = last_event + 1 if include_immediate else last_event   # t must have an event >= lo
        return np.arange(0, hi, dtype=np.int64)

    # --- batch with negatives + false-negative detection ------------------- #
    def sample_batch(self, B: int, mode: str, rng: np.random.Generator,
                     include_immediate: bool = True, weighting: str = "uniform") -> Dict[str, Any]:
        anchors = []
        for ep_i in range(len(self.eps)):
            va = self.valid_anchors(ep_i, mode, include_immediate)
            if len(va):
                anchors.append((ep_i, va))
        rows = []
        while len(rows) < B:
            ep_i, va = anchors[rng.integers(len(anchors))]
            t = int(va[rng.integers(len(va))])
            pos = self.sample_positive(ep_i, t, mode, rng, include_immediate, weighting)
            if pos is not None:
                rows.append(pos)
        pos_vals = np.array([r["goal_value"] for r in rows])
        # negatives: uniform over the global event-goal pool (in-batch / cross-episode)
        neg_vals = self.all_goal_values[rng.integers(len(self.all_goal_values), size=B)]
        false_neg = (pos_vals == neg_vals)
        return {
            "rows": rows, "positive_goal_values": pos_vals, "negative_goal_values": neg_vals,
            "false_negative_mask": false_neg,
            "false_negative_rate": float(false_neg.mean()),
        }


def _dist(values: np.ndarray) -> Dict[str, Any]:
    vals, counts = np.unique(values, return_counts=True)
    pmf = counts / counts.sum()
    return {"marginal": {int(v): int(c) for v, c in zip(vals, counts)},
            "collision_prob": float((pmf ** 2).sum()),
            "min": int(values.min()), "max": int(values.max()), "n": int(len(values))}


def empirical_report(sampler: ScoringGoalSampler, n: int = 20000, seed: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    out: Dict[str, Any] = {}
    for mode in ["next_score_event", "multi_event_future_score_diff"]:
        b = sampler.sample_batch(n, mode, rng, include_immediate=True,
                                 weighting="uniform")
        rows = b["rows"]
        horizons = np.array([r["horizon"] for r in rows])
        dirs = np.array([r["direction"] for r in rows])
        gvals = b["positive_goal_values"]
        out[mode] = {
            "horizon": {"mean": float(horizons.mean()), "p50": int(np.median(horizons)),
                        "p95": int(np.quantile(horizons, .95)), "max": int(horizons.max())},
            "goal_value_distribution": _dist(gvals),
            "event_direction": {"agent_frac": float((dirs > 0).mean()),
                                "opponent_frac": float((dirs < 0).mean())},
            "false_negative_rate_uniform_negatives": b["false_negative_rate"],
        }
    # uniform vs stratified for multi_event (report BOTH target distributions explicitly)
    u = sampler.sample_batch(n, "multi_event_future_score_diff", rng, weighting="uniform")
    s = sampler.sample_batch(n, "multi_event_future_score_diff", rng, weighting="stratified_by_direction")
    out["uniform_vs_stratified_multi_event"] = {
        "uniform": {"agent_frac": float((np.array([r["direction"] for r in u["rows"]]) > 0).mean()),
                    "goal_value_distribution": _dist(u["positive_goal_values"])},
        "stratified_by_direction": {"agent_frac": float((np.array([r["direction"] for r in s["rows"]]) > 0).mean()),
                                    "goal_value_distribution": _dist(s["positive_goal_values"])},
        "note": "Stratified balances agent/opponent events 50/50 and therefore CHANGES the goal-value "
                "target distribution vs uniform-over-future-events. Reported side by side; uniform is "
                "the default and is never silently replaced.",
    }
    return out


def from_dataset(tag: str = "full") -> ScoringGoalSampler:
    paths = sorted((ART_ROOT / tag / "episodes").glob("*.npz"))
    eps = []
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            eps.append({k: z[k] for k in
                        ["is_scoring_event", "reward", "agent_score", "opp_score", "score_diff"]})
    return ScoringGoalSampler(eps)


def main() -> None:
    import argparse
    import json
    ap = argparse.ArgumentParser(description="M5 scoring-event goal sampler — empirical report.")
    ap.add_argument("--tag", type=str, default="full")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sampler = from_dataset(args.tag)
    rep = empirical_report(sampler, n=args.n, seed=args.seed)
    out = {"milestone": "M5", "tag": args.tag, "n_samples": args.n,
           "n_episodes": len(sampler.eps),
           "total_events": int(len(sampler.all_goal_values)), "report": rep}
    outdir = ART_ROOT / args.tag
    with open(outdir / "m5_goal_sampler_report.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
