"""M5 goal-sampler unit tests over hand-built toy episodes.

Run: python -m pytest pong_action_gate/tests/test_m5.py -v
"""
import numpy as np
import pytest

from pong_action_gate.data.goal_sampler import (
    ScoringGoalSampler, build_episode_index, diagonal_targets,
    duplicate_corrected_targets, future_events_for, next_event_for)


def make_episode(rewards):
    """Build an episode dict from a list of per-step rewards in {-1,0,+1}."""
    r = np.array(rewards, np.float32)
    agent = np.cumsum(r > 0).astype(np.int32)
    opp = np.cumsum(r < 0).astype(np.int32)
    post = agent.astype(np.int64) - opp.astype(np.int64)
    pre = (post - r.astype(np.int64)).astype(np.int32)
    return {"reward": r, "agent_score": agent, "opp_score": opp,
            "score_diff": pre, "is_scoring_event": (r != 0)}


# rewards: events at idx 2(+1), 4(-1), 6(+1); post = [0,0,1,1,0,0,1]
EP = make_episode([0, 0, 1, 0, -1, 0, 1])


def test_index_events_goals_directions():
    ix = build_episode_index(EP)
    assert list(ix.event_idx) == [2, 4, 6]
    assert list(ix.goal_value) == [1, 0, 1]        # post-event score diffs
    assert list(ix.direction) == [1, -1, 1]


def test_next_event_basic_and_horizon():
    ix = build_episode_index(EP)
    assert next_event_for(ix, 0, True) == (2, 1, 1, 2)   # (event_idx, goal, dir, horizon)
    assert next_event_for(ix, 5, True) == (6, 1, 1, 1)


def test_immediate_event_eligibility():
    ix = build_episode_index(EP)
    # t=2 is itself a scoring event
    assert next_event_for(ix, 2, include_immediate=True) == (2, 1, 1, 0)
    assert next_event_for(ix, 2, include_immediate=False) == (4, 0, -1, 2)


def test_opponent_event_direction_and_postminus1():
    ix = build_episode_index(EP)
    # event at idx 4 is opponent-scored: post = pre - 1
    res = next_event_for(ix, 3, True)
    assert res == (4, 0, -1, 1)
    pre4 = int(EP["score_diff"][4]); post4 = int(EP["agent_score"][4] - EP["opp_score"][4])
    assert post4 == pre4 - 1


def test_no_future_event_returns_none():
    ix = build_episode_index(EP)
    assert next_event_for(ix, 6, include_immediate=False) is None   # idx6 is last event
    empty = build_episode_index(make_episode([0, 0, 0]))
    assert next_event_for(empty, 0, True) is None


def test_episode_ending_immediately_after_score():
    ix = build_episode_index(make_episode([0, 1]))   # event at last index
    assert next_event_for(ix, 0, True) == (1, 1, 1, 1)
    assert next_event_for(ix, 1, True) == (1, 1, 1, 0)
    assert next_event_for(ix, 1, False) is None


def test_no_cross_episode_sampling():
    e0 = make_episode([0, 0, 1, 0, 1])     # length 5, events 2,4
    e1 = make_episode([0, -1, 0, 1])       # length 4, events 1,3
    s = ScoringGoalSampler([e0, e1])
    rng = np.random.default_rng(0)
    for _ in range(200):
        for ep_i, L in [(0, 5), (1, 4)]:
            va = s.valid_anchors(ep_i, "multi_event_future_score_diff")
            t = int(va[rng.integers(len(va))])
            pos = s.sample_positive(ep_i, t, "multi_event_future_score_diff", rng)
            assert pos is None or (t <= pos["event_index"] < L)   # within THIS episode only


def test_multi_event_only_future_within_episode():
    ix = build_episode_index(EP)
    for t in range(7):
        fe = future_events_for(ix, t, True)
        assert all(ix.event_idx[k] >= t for k in fe)


def test_false_negative_detection_on_duplicates():
    # NOTE: goal value = cumulative POST score-diff (not per-event reward).
    # Three identical single-event episodes -> all goal values == 1 -> every uniform
    # negative collides with every positive (false-negative rate 1.0).
    s = ScoringGoalSampler([make_episode([0, 1]) for _ in range(3)])
    rng = np.random.default_rng(1)
    b = s.sample_batch(500, "next_score_event", rng)
    assert list(np.unique(s.all_goal_values)) == [1]
    assert b["false_negative_rate"] == 1.0
    # mask is exactly elementwise value equality
    assert np.array_equal(b["false_negative_mask"],
                          b["positive_goal_values"] == b["negative_goal_values"])


def test_false_negative_rate_partial_and_mask():
    # score-diff oscillates 1,0,1,0 -> duplicate goal values, partial collision
    s = ScoringGoalSampler([make_episode([0, 1, -1, 1, -1])])
    rng = np.random.default_rng(2)
    b = s.sample_batch(2000, "multi_event_future_score_diff", rng)
    assert 0.0 < b["false_negative_rate"] < 1.0
    assert np.array_equal(b["false_negative_mask"],
                          b["positive_goal_values"] == b["negative_goal_values"])


# --------------------------------------------------------------------------- #
# Duplicate-goal NCE target correction (next_score_event)
# --------------------------------------------------------------------------- #
def test_corrected_target_all_same_goal_columns_positive():
    # (1) every column whose scalar goal equals the row's positive goal is labelled 1
    goals = np.array([3, 7, 3, 5, 7, 3])
    T = duplicate_corrected_targets(goals)
    for i in range(len(goals)):
        same = goals == goals[i]
        assert (T[i, same] == 1).all()
        assert (T[i, ~same] == 0).all()


def test_corrected_target_no_conflicting_labels_in_row():
    # (2) within a row, identical goal inputs never receive conflicting labels
    goals = np.array([1, 1, 2, 2, 2, 9])
    T = duplicate_corrected_targets(goals)
    for i in range(len(goals)):
        for v in np.unique(goals):
            cols = goals == v
            assert len(np.unique(T[i, cols])) == 1   # all same label for identical goals


def test_corrected_target_all_identical_is_all_ones():
    # (3) all identical goals -> all-positive target matrix
    T = duplicate_corrected_targets(np.array([4, 4, 4, 4]))
    assert (T == 1).all()


def test_corrected_target_partial_duplicate_blocks():
    # (4) partial duplicates -> expected multi-positive blocks
    goals = np.array([1, 2, 1])
    T = duplicate_corrected_targets(goals)
    expected = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], np.float32)
    assert np.array_equal(T, expected)
    assert np.array_equal(np.diag(T), np.ones(3))    # diagonal always positive


def test_diagonal_vs_corrected_differ_only_at_duplicates():
    # (5) diagonal-only and corrected differ EXACTLY at off-diagonal duplicate positions
    goals = np.array([5, 5, 8, 5, 8])
    corrected = duplicate_corrected_targets(goals)
    diag = diagonal_targets(len(goals))
    diff = corrected - diag
    off_dup = (goals[:, None] == goals[None, :]) & ~np.eye(len(goals), dtype=bool)
    assert np.array_equal(diff > 0, off_dup)
    assert (diff[~off_dup] == 0).all()


def test_valid_anchors_respect_include_immediate():
    ix_ep = make_episode([0, 0, 1, 0, 1])      # events at 2,4; length 5
    s = ScoringGoalSampler([ix_ep])
    inc = s.valid_anchors(0, "next_score_event", include_immediate=True)
    exc = s.valid_anchors(0, "next_score_event", include_immediate=False)
    assert inc.max() == 4    # t up to last event index (event >= t includes t=4)
    assert exc.max() == 3    # need event > t, so last valid t is 3
