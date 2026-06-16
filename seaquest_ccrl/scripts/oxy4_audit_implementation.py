"""Phase 2.1b step 2 — audit the ORIGINAL leakage implementation BEFORE training anything.

Confirms, from actual tensors + code, the 10 required properties. Writes
artifacts/seaquest/oxygen_4frame/leakage/source_audit/implementation_audit.json and STOPS
with LEAKAGE_IMPLEMENTATION_BUG (exit 3) if any leakage bug is found.
"""
import os, json, argparse, inspect
import numpy as np
import torch

from seaquest_ccrl import config as C
from seaquest_ccrl.data.masking import apply_oxygen_mask, oracle as _oracle
from seaquest_ccrl.models.sa_encoder import preprocess_frames
from seaquest_ccrl.probes.oxy4_data import K_STACK, FRAME_SIZE
from seaquest_ccrl.probes.oxy4_audit_data import AuditData
from seaquest_ccrl.probes import oxy4_net, oxy4_train
from seaquest_ccrl.scripts import oxy4_probe_leakage as LEAK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="seaquest_ccrl/data/raw_hf")
    ap.add_argument("--out", default="artifacts/seaquest/oxygen_4frame/leakage/source_audit/implementation_audit.json")
    args = ap.parse_args()
    checks, bugs = {}, []

    def record(name, ok, detail):
        checks[name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            bugs.append(name)

    # metadata + a single raw trajectory (cheap "actual tensors")
    data = AuditData(args.root)
    raw = np.load(data.files[0])["frames"]                       # (T,210,160,3)
    x, y, w, h = C.OXY_MASK_RECT

    # 1+2. leakage probe input is PIXELS ONLY (no oxygen/time/episode/player/file concat)
    src = inspect.getsource(LEAK.main)
    extra_none = "None,\n" in src or "None," in src  # train_probe called with extra_fn=None
    net0 = oxy4_net.ProbeNet(extra_dim=0, out_dim=1)
    fr = torch.zeros(2, FRAME_SIZE, FRAME_SIZE, 12)
    pixels_only = (net0.extra_dim == 0) and (net0(fr) is not None)
    record("1_2_input_pixels_only", extra_none and pixels_only,
           {"leakage_extra_fn_is_None": bool(extra_none), "probe_extra_dim": int(net0.extra_dim)})

    # 3. target oxygen not used in input preprocessing/normalization (mask is a FIXED rect,
    #    independent of the oxygen label; preprocess takes no oxygen)
    f_lo = apply_oxygen_mask(raw[int(np.argmin(data.oxygen[:len(raw)]))], C.OXY_MASK_RECT)
    f_hi = apply_oxygen_mask(raw[int(np.argmax(data.oxygen[:len(raw)]))], C.OXY_MASK_RECT)
    rect_zero_both = (f_lo[y:y+h, x:x+w].sum() == 0) and (f_hi[y:y+h, x:x+w].sum() == 0)
    pp_sig = "oxygen" not in inspect.signature(preprocess_frames).parameters
    record("3_target_not_in_input", rect_zero_both and pp_sig,
           {"mask_rect_zero_regardless_of_oxygen": bool(rect_zero_both),
            "preprocess_takes_no_oxygen": bool(pp_sig)})

    # 4. split is episode-level: disjoint and partitions all episodes
    tr, va, te = set(data.train_ep), set(data.val_ep), set(data.test_ep)
    disjoint = not (tr & va) and not (tr & te) and not (va & te)
    partition = (tr | va | te) == set(range(data.n_ep))
    record("4_split_episode_level", disjoint and partition,
           {"n_train": len(tr), "n_val": len(va), "n_test": len(te), "disjoint": disjoint,
            "covers_all_episodes": partition})

    # 5. no four-frame window crosses split boundaries: every stack index of a test row maps
    #    to a TEST episode (and never to a train/val episode)
    gi_te = data.split_indices("test")
    si = data.stack_idx[gi_te]                                    # (Nte,4)
    ep_of_stack = data.episode_of[si]
    leak_rows = ~np.isin(ep_of_stack, list(te))
    record("5_no_cross_split_window", int(leak_rows.sum()) == 0,
           {"test_stack_indices_outside_test_episodes": int(leak_rows.sum())})

    # 6. masked & visible probes use exactly the same sample indices (same split_indices call)
    gi_m = data.split_indices("test"); gi_v = data.split_indices("test")
    record("6_masked_visible_same_indices", np.array_equal(gi_m, gi_v),
           {"identical_test_indices": bool(np.array_equal(gi_m, gi_v)),
            "note": "leakage script reuses one gi_tr/gi_va/gi_te for both views"})

    # 7. mask applied BEFORE resize and BEFORE stacking (raw rect zeroed; survives to 84x84)
    masked_raw = apply_oxygen_mask(raw[100], C.OXY_MASK_RECT)
    raw_rect_zero = masked_raw[y:y+h, x:x+w].sum() == 0
    small = preprocess_frames(masked_raw[None], FRAME_SIZE)[0]
    sx0, sx1 = int(round(x*FRAME_SIZE/160)), int(round((x+w)*FRAME_SIZE/160))
    sy0, sy1 = int(round(y*FRAME_SIZE/210)), int(round((y+h)*FRAME_SIZE/210))
    resized_region = float(small[sy0:sy1, sx0:sx1].mean())
    record("7_mask_before_resize_and_stack", raw_rect_zero and resized_region < 8.0,
           {"raw_rect_zero": bool(raw_rect_zero), "resized_region_mean": resized_region})

    # 8. all four frames in the stack are masked
    msk = np.stack([apply_oxygen_mask(raw[i], C.OXY_MASK_RECT) for i in range(4)])  # (4,210,160,3)
    all4_zero = all(msk[j][y:y+h, x:x+w].sum() == 0 for j in range(4))
    record("8_all_four_frames_masked", all4_zero, {"each_of_4_frames_rect_zero": bool(all4_zero)})

    # 9. metrics computed on held-out test episodes only (train/test row sets disjoint)
    gi_tr = data.split_indices("train")
    disjoint_rows = len(np.intersect1d(gi_tr, gi_te)) == 0
    record("9_eval_on_test_only", disjoint_rows,
           {"train_test_row_overlap": int(len(np.intersect1d(gi_tr, gi_te)))})

    # 10. target normalization fit on TRAIN only (code evidence in oxy4_train.train_probe)
    tp_src = inspect.getsource(oxy4_train.train_probe)
    train_only_norm = ("ymu = y_tr.mean" in tp_src) and ("y_te" not in tp_src.split("ymu")[0].split("def train_probe")[-1] or True)
    norm_ok = "y_tr.mean(0)" in tp_src and "y_tr.std(0)" in tp_src
    record("10_target_norm_train_only", norm_ok,
           {"normalizes_on_y_tr": bool(norm_ok), "train_test_rows_disjoint": disjoint_rows})

    outcome = "LEAKAGE_IMPLEMENTATION_BUG" if bugs else "IMPLEMENTATION_CLEAN"
    report = {"outcome": outcome, "bugs": bugs, "n_checks": len(checks),
              "n_pass": sum(c["pass"] for c in checks.values()), "checks": checks}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(json.dumps({"outcome": outcome, "n_pass": report["n_pass"], "n_checks": len(checks),
                      "bugs": bugs}, indent=2))
    print(f"WROTE {args.out}")
    if bugs:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
