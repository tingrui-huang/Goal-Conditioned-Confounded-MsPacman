"""Tests for robust checkpoint discovery (latest_ckpt) under Drive-style duplicate names.

Run: python -m pytest pong_action_gate/tests/test_ckpt_discovery.py -v
"""
import warnings

import pytest

from pong_action_gate.train.train_critic import latest_ckpt


def _touch(d, *names):
    for n in names:
        (d / n).write_bytes(b"")   # contents irrelevant; latest_ckpt inspects names only


def test_normal_checkpoint_names(tmp_path):
    _touch(tmp_path, "ckpt_0.pt", "ckpt_100.pt", "ckpt_200.pt")
    assert latest_ckpt(tmp_path).name == "ckpt_200.pt"


def test_out_of_order_names_use_integer_not_lexicographic(tmp_path):
    # lexicographically "ckpt_1000" < "ckpt_50"; integer step must win
    _touch(tmp_path, "ckpt_200.pt", "ckpt_50.pt", "ckpt_1000.pt")
    assert latest_ckpt(tmp_path).name == "ckpt_1000.pt"


def test_duplicate_drive_style_name_is_ignored_with_warning(tmp_path):
    _touch(tmp_path, "ckpt_1000.pt", "ckpt_1800.pt", "ckpt_1800 (1).pt")
    with pytest.warns(UserWarning, match=r"ckpt_1800 \(1\)\.pt"):
        chosen = latest_ckpt(tmp_path)
    assert chosen.name == "ckpt_1800.pt"   # does not crash; valid max wins


def test_unrelated_pt_files_are_skipped(tmp_path):
    _touch(tmp_path, "ckpt_100.pt", "model.pt", "optimizer.pt", "best.pt")
    with warnings.catch_warnings():
        warnings.simplefilter("error")     # unrelated .pt must NOT warn
        assert latest_ckpt(tmp_path).name == "ckpt_100.pt"


def test_no_valid_checkpoints_returns_none(tmp_path):
    assert latest_ckpt(tmp_path) is None                      # empty dir
    _touch(tmp_path, "model.pt", "ckpt_.pt", "ckptX_5.pt")    # none match exactly
    assert latest_ckpt(tmp_path) is None


def test_malformed_ckpt_names_warn_but_do_not_crash(tmp_path):
    _touch(tmp_path, "ckpt_300.pt", "ckpt_abc.pt", "ckpt_12 copy.pt")
    with pytest.warns(UserWarning):
        assert latest_ckpt(tmp_path).name == "ckpt_300.pt"
