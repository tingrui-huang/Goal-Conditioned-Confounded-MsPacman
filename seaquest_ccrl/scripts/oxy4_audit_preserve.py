"""Phase 2.1b step 1 — preserve the ORIGINAL Phase 2.1 leakage artifacts (never overwrite).

Copies metrics.json / predictions.npz / split_manifest.json / resolved_config.json from the
existing leakage dir into leakage/source_audit/original_phase21/ and records sha256 of every
file. Any contract-named file that is unavailable is documented explicitly (not silently
skipped). Run this BEFORE any re-run so the 0.901 result is captured verbatim.
"""
import os, json, shutil, hashlib, argparse

CONTRACT_FILES = ["metrics.json", "predictions.npz", "split_manifest.json", "resolved_config.json"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="artifacts/seaquest/oxygen_4frame/leakage")
    ap.add_argument("--dst", default="artifacts/seaquest/oxygen_4frame/leakage/source_audit/original_phase21")
    args = ap.parse_args()
    os.makedirs(args.dst, exist_ok=True)
    record = {"src": args.src, "files": {}}
    for fn in CONTRACT_FILES:
        sp = os.path.join(args.src, fn)
        if os.path.exists(sp):
            dp = os.path.join(args.dst, fn)
            shutil.copy2(sp, dp)
            record["files"][fn] = {"status": "preserved", "sha256": sha256(dp),
                                   "bytes": os.path.getsize(dp)}
        else:
            record["files"][fn] = {"status": "UNAVAILABLE",
                                   "note": "not produced by oxy4_probe_leakage / not present in this run"}
    json.dump(record, open(os.path.join(args.dst, "preservation_hashes.json"), "w"), indent=2)
    print(json.dumps({fn: record["files"][fn]["status"] for fn in CONTRACT_FILES}, indent=2))
    print(f"WROTE {args.dst}/ (+ preservation_hashes.json)")


if __name__ == "__main__":
    main()
