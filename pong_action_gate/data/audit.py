"""M4 transition-alignment audit (read-only; does NOT recollect the dataset).

Proves the pre/post-action convention and that all per-index modalities correspond
to the same pre-action observation. Records dtype/shape/compressed/uncompressed size.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from .. import config as C
from . import schema
from .schema import load_episode, score_diff_post, score_diff_pre

ART_ROOT = Path("artifacts/pong_action_gate/m4")
IMAGE_FIELDS = ["raw_rgb", "teacher_obs", "gray_learner"]


def _convention_tests(ep: Dict[str, Any]) -> List[str]:
    issues = []
    pre = score_diff_pre(ep)
    post = score_diff_post(ep)
    rew = ep["reward"]
    ise = ep["is_scoring_event"]

    # 1. post = agent - opp  (definitional, but assert the stored arrays agree)
    if not np.array_equal(post, ep["agent_score"].astype(np.int64) - ep["opp_score"].astype(np.int64)):
        issues.append("T1: score_diff_post != agent_score - opp_score")
    # 2. reward[t] = post[t] - pre[t]
    if not np.array_equal(rew.astype(np.int64), post - pre):
        issues.append("T2: reward != post - pre")
    # 3. non-scoring transitions: pre == post
    ns = ~ise
    if not np.array_equal(pre[ns], post[ns]):
        issues.append("T3: non-scoring transition has pre != post")
    # 4. agent scoring events: post = pre + 1
    ag = rew > 0
    if not np.array_equal(post[ag], pre[ag] + 1):
        issues.append("T4: agent event post != pre + 1")
    # 5. opponent scoring events: post = pre - 1
    op = rew < 0
    if not np.array_equal(post[op], pre[op] - 1):
        issues.append("T5: opponent event post != pre - 1")
    # 6. is_scoring_event == (pre != post)
    if not np.array_equal(ise, pre != post):
        issues.append("T6: is_scoring_event != (pre != post)")
    # 7. no post-event score shifted into t+1: pre[t+1] == post[t] within the episode
    if len(pre) > 1 and not np.array_equal(pre[1:], post[:-1]):
        issues.append("T7: pre[t+1] != post[t] (score leaked across transitions)")
    return issues


def _alignment_tests(ep: Dict[str, Any], n_sample: int = 25) -> Dict[str, Any]:
    """Recompute gray_learner / teacher-RGB from raw_rgb[t] and confirm equality; ball pixel bright."""
    T = len(ep["action"])
    rng = np.random.default_rng(0)
    idxs = rng.choice(T, size=min(n_sample, T), replace=False)
    gray_ok = teacher_rgb_ok = ball_ok = ball_checked = 0
    for t in idxs:
        raw = ep["raw_rgb"][t]
        # gray_learner[t] must be exactly grayscale->resize(84) of raw_rgb[t]
        g = cv2.resize(cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY), (84, 84), interpolation=cv2.INTER_AREA)
        gray_ok += int(np.array_equal(g, ep["gray_learner"][t]))
        # teacher_obs[t] colored channels [4:7] must match resize(64) of raw_rgb[t] (±1 rounding)
        col = cv2.resize(raw, (C.IMG_SIZE, C.IMG_SIZE), interpolation=cv2.INTER_AREA)  # (64,64,3)
        tcol = np.transpose(ep["teacher_obs"][t][4:7], (1, 2, 0))                       # (64,64,3)
        teacher_rgb_ok += int(np.abs(col.astype(int) - tcol.astype(int)).max() <= 1)
        # ball coords -> bright pixel in raw_rgb[t]
        if ep["ball_present"][t]:
            bx, by = int(ep["ball_x"][t]), int(ep["ball_y"][t])
            if 0 <= by < 210 and 0 <= bx < 160:
                ball_checked += 1
                ball_ok += int(raw[max(0, by-4):by+5, max(0, bx-4):bx+5].max() > 200)
    n = len(idxs)
    return {
        "sampled": int(n),
        "gray_learner_matches_raw": f"{gray_ok}/{n}",
        "teacher_rgb_matches_raw": f"{teacher_rgb_ok}/{n}",
        "ball_pixel_bright": f"{ball_ok}/{ball_checked}",
        "ok": gray_ok == n and teacher_rgb_ok == n and (ball_checked == 0 or ball_ok == ball_checked),
    }


def _size_audit(path: Path, ep: Dict[str, Any]) -> Dict[str, Any]:
    fields = {}
    uncompressed = 0
    for k in schema.ARRAY_FIELDS:
        a = ep[k]
        fields[k] = {"dtype": str(a.dtype), "shape": list(a.shape), "bytes": int(a.nbytes)}
        uncompressed += a.nbytes
    comp = int(path.stat().st_size)
    image_dtypes_uint8 = all(str(ep[k].dtype) == "uint8" for k in IMAGE_FIELDS)
    return {
        "fields": fields,
        "image_arrays_uint8": image_dtypes_uint8,
        "uncompressed_bytes": int(uncompressed),
        "compressed_bytes": comp,
        "compression_ratio": round(uncompressed / max(comp, 1), 1),
    }


def run_audit(tag: str, n_align: int = 25) -> Dict[str, Any]:
    outdir = ART_ROOT / tag
    paths = sorted((outdir / "episodes").glob("*.npz"))
    conv_fail, align_fail = [], []
    align_summary = {"gray": 0, "teacher_rgb": 0, "ball": 0, "ball_checked": 0, "n": 0}
    size_example = None
    dtypes_consistent = True
    ref_dtypes = None
    image_uint8_all = True
    tot_uncomp = tot_comp = 0

    for p in paths:
        ep = load_episode(p)
        ci = _convention_tests(ep)
        if ci:
            conv_fail.append({"path": p.name, "issues": ci})
        ai = _alignment_tests(ep, n_align)
        if not ai["ok"]:
            align_fail.append({"path": p.name, **ai})
        g, n = map(int, ai["gray_learner_matches_raw"].split("/"))
        tr, _ = map(int, ai["teacher_rgb_matches_raw"].split("/"))
        bo, bc = map(int, ai["ball_pixel_bright"].split("/"))
        align_summary["gray"] += g; align_summary["teacher_rgb"] += tr
        align_summary["ball"] += bo; align_summary["ball_checked"] += bc; align_summary["n"] += n

        sz = _size_audit(p, ep)
        tot_uncomp += sz["uncompressed_bytes"]; tot_comp += sz["compressed_bytes"]
        image_uint8_all = image_uint8_all and sz["image_arrays_uint8"]
        dt = {k: sz["fields"][k]["dtype"] for k in schema.ARRAY_FIELDS}
        if ref_dtypes is None:
            ref_dtypes = dt; size_example = sz
        elif dt != ref_dtypes:
            dtypes_consistent = False

    report = {
        "milestone": "M4-audit",
        "tag": tag,
        "n_episodes": len(paths),
        "transition_convention": schema.TRANSITION_CONVENTION,
        "convention_tests_1_to_7": {"all_passed": len(conv_fail) == 0, "failures": conv_fail},
        "alignment_tests": {
            "all_passed": len(align_fail) == 0,
            "totals": {
                "gray_learner_matches_raw": f"{align_summary['gray']}/{align_summary['n']}",
                "teacher_rgb_matches_raw": f"{align_summary['teacher_rgb']}/{align_summary['n']}",
                "ball_pixel_bright": f"{align_summary['ball']}/{align_summary['ball_checked']}",
            },
            "failures": align_fail,
            "note": f"{n_align} frames/episode recomputed from raw_rgb[t]",
        },
        "size_audit": {
            "image_arrays_uint8": image_uint8_all,
            "dtypes_consistent_across_episodes": dtypes_consistent,
            "per_field_example": size_example["fields"] if size_example else None,
            "total_uncompressed_bytes": tot_uncomp,
            "total_compressed_bytes": tot_comp,
            "overall_compression_ratio": round(tot_uncomp / max(tot_comp, 1), 1),
            "total_uncompressed_MB": round(tot_uncomp / 1024 / 1024, 1),
            "total_compressed_MB": round(tot_comp / 1024 / 1024, 1),
        },
        "all_passed": len(conv_fail) == 0 and len(align_fail) == 0,
        "misalignment_detected": len(conv_fail) > 0 or len(align_fail) > 0,
    }
    with open(outdir / "transition_audit.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="M4 transition-alignment audit (read-only).")
    ap.add_argument("--tag", type=str, default="full")
    ap.add_argument("--n-align", type=int, default=25)
    args = ap.parse_args()
    rep = run_audit(args.tag, args.n_align)
    print(json.dumps({k: rep[k] for k in
                      ["convention_tests_1_to_7", "alignment_tests", "size_audit",
                       "all_passed", "misalignment_detected"]}, indent=2))


if __name__ == "__main__":
    main()
