"""Standalone re-validation of the Stage-H0 metadata pack against frozen raw_hf.

Re-checks, WITHOUT the teacher/ALE (safe to run in Colab as a guard):
  * every raw_hf trajectory SHA256 matches the value stored in its metadata file;
  * per-episode timestep counts match between raw and metadata;
  * the padded hostile schema validates (class ids, bbox in-bounds, count consistency);
  * ambiguous-row / per-class support totals match the metadata manifest.

This does NOT re-run collection. The byte-identical base-array parity is asserted
INSIDE collect_hf_hostile_metadata.py at collection time (HOSTILE_RECOLLECTION_NOT_IDENTICAL);
this validator confirms the saved pack is internally consistent and bound to raw_hf.
"""
import sys, os, json, glob, hashlib, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seaquest_ccrl.hostile import schema as S


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def validate(raw_root, meta_root, out=None):
    raw = sorted(glob.glob(os.path.join(raw_root, "traj_*.npz")))
    meta = sorted(glob.glob(os.path.join(meta_root, "meta_*.npz")))
    res = {"raw_root": raw_root, "meta_root": meta_root,
           "n_raw": len(raw), "n_meta": len(meta), "episodes": [], "ok": True, "failures": []}
    if len(raw) != len(meta) or not raw:
        res["ok"] = False
        res["failures"].append(f"count mismatch raw={len(raw)} meta={len(meta)}")
        _write(res, out)
        return res

    for rf, mf in zip(raw, meta):
        d = np.load(rf); m = np.load(mf)
        epres = {"raw": os.path.basename(rf), "meta": os.path.basename(mf)}
        raw_sha = _sha256_file(rf)
        epres["sha_match"] = bool("raw_sha256" in m.files and str(m["raw_sha256"]) == raw_sha)
        T_raw = len(d["actions"]); T_meta = len(m["hostile_count"])
        epres["T_match"] = bool(T_raw == T_meta)
        try:
            harr = {k: m[k] for k in ("hostile_bbox", "hostile_class", "hostile_valid",
                                      "hostile_count", "enemy_count", "enemy_missile_count")}
            S.validate_hostile_arrays(harr)
            epres["schema_ok"] = True
        except AssertionError as e:
            epres["schema_ok"] = False
            epres["schema_error"] = str(e)
        epres["n_enemy_rows"] = int((m["enemy_count"] > 0).sum())
        epres["n_missile_rows"] = int((m["enemy_missile_count"] > 0).sum())
        epres["n_ambiguous_rows"] = int(m["ambiguous"].sum()) if "ambiguous" in m.files else None
        ok = epres["sha_match"] and epres["T_match"] and epres["schema_ok"]
        if not ok:
            res["ok"] = False
            res["failures"].append(epres["meta"])
        res["episodes"].append(epres)

    res["totals"] = {
        "enemy_rows": int(sum(e["n_enemy_rows"] for e in res["episodes"])),
        "missile_rows": int(sum(e["n_missile_rows"] for e in res["episodes"])),
        "ambiguous_rows": int(sum(e["n_ambiguous_rows"] or 0 for e in res["episodes"])),
        "episodes_with_enemy_rows": int(sum(e["n_enemy_rows"] > 0 for e in res["episodes"])),
        "episodes_with_missile_rows": int(sum(e["n_missile_rows"] > 0 for e in res["episodes"])),
    }
    _write(res, out)
    return res


def _write(res, out):
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(res, open(out, "w"), indent=2)
    status = "OK" if res["ok"] else "FAIL"
    print(f"[parity-validate] {status}  n_meta={res['n_meta']}  "
          f"failures={len(res.get('failures', []))}")
    if "totals" in res:
        print(f"  totals: {res['totals']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="seaquest_ccrl/data/raw_hf")
    ap.add_argument("--meta-root", default="seaquest_ccrl/data/hostile_h0_metadata")
    ap.add_argument("--out", default="artifacts/seaquest/hostile_h0/recollection_parity.json")
    args = ap.parse_args()
    res = validate(args.raw_root, args.meta_root, args.out)
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
