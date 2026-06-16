"""Bundle the Stage-H0 data pack for a self-contained Colab run.

Produces a pack directory containing:
  * a copy of the (small) hostile metadata npz files + metadata_manifest.json;
  * the frozen schema / feature-schema JSON;
  * the local Docker audit artifacts (object identity, removal) if present;
  * checksums.json listing the SHA256 of every raw_hf trajectory AND every metadata
    file, so Colab can verify the (separately uploaded) raw_hf frames byte-for-byte.

The large raw_hf frame npz files are NOT copied (multi-GB); they are referenced by
SHA256 and uploaded to Drive separately. Colab's loader (HostileH0Data) re-verifies
each raw SHA at load time.
"""
import sys, os, json, glob, hashlib, shutil, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from seaquest_ccrl.hostile import schema as S
from seaquest_ccrl.hostile import features as FEAT


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def export(raw_root, meta_root, out_pack, audit_dir=None, copy_metadata=True):
    os.makedirs(out_pack, exist_ok=True)
    meta_files = sorted(glob.glob(os.path.join(meta_root, "meta_*.npz")))
    raw_files = sorted(glob.glob(os.path.join(raw_root, "traj_*.npz")))
    assert meta_files, f"no metadata under {meta_root}"

    checksums = {"raw_hf": {}, "metadata": {}}
    for f in raw_files:
        checksums["raw_hf"][os.path.basename(f)] = _sha256_file(f)
    meta_out = os.path.join(out_pack, "hostile_h0_metadata")
    if copy_metadata:
        os.makedirs(meta_out, exist_ok=True)
    for f in meta_files:
        checksums["metadata"][os.path.basename(f)] = _sha256_file(f)
        if copy_metadata:
            shutil.copy2(f, os.path.join(meta_out, os.path.basename(f)))
    for extra in ("metadata_manifest.json",):
        src = os.path.join(meta_root, extra)
        if os.path.exists(src) and copy_metadata:
            shutil.copy2(src, os.path.join(meta_out, extra))

    json.dump(S.schema_dict(), open(os.path.join(out_pack, "schema.json"), "w"), indent=2)
    json.dump(FEAT.feature_schema(), open(os.path.join(out_pack, "feature_schema.json"), "w"), indent=2)
    json.dump(checksums, open(os.path.join(out_pack, "checksums.json"), "w"), indent=2)

    # copy local audit artifacts if they exist
    copied_audit = []
    if audit_dir and os.path.isdir(audit_dir):
        dst = os.path.join(out_pack, "audit")
        os.makedirs(dst, exist_ok=True)
        for root, _, files in os.walk(audit_dir):
            if os.path.abspath(root).startswith(os.path.abspath(out_pack)):
                continue                                   # never recurse into the pack itself
            for fn in files:
                if fn.endswith((".json", ".png", ".npz", ".csv", ".md", ".txt")):
                    rel = os.path.relpath(os.path.join(root, fn), audit_dir)
                    d = os.path.join(dst, rel)
                    os.makedirs(os.path.dirname(d), exist_ok=True)
                    shutil.copy2(os.path.join(root, fn), d)
                    copied_audit.append(rel)

    # post-write verification: re-hash every copied metadata file vs the source checksum
    verify = {"metadata_ok": True, "mismatches": []}
    if copy_metadata:
        for fn, src_sha in checksums["metadata"].items():
            dpath = os.path.join(meta_out, fn)
            if not os.path.exists(dpath) or _sha256_file(dpath) != src_sha:
                verify["metadata_ok"] = False
                verify["mismatches"].append(fn)

    pack_manifest = {
        "pack": "seaquest_hostile_h0",
        "n_raw": len(raw_files), "n_metadata": len(meta_files),
        "raw_root_reference": raw_root,
        "metadata_copied": copy_metadata,
        "audit_files_copied": copied_audit,
        "post_write_verification": verify,
        "marker": "HOSTILE_EXPORT_OK" if verify["metadata_ok"] else "HOSTILE_EXPORT_FAILED",
        "note": "raw_hf frames are referenced by SHA, uploaded separately; "
                "HostileH0Data re-verifies each raw SHA at load.",
    }
    json.dump(pack_manifest, open(os.path.join(out_pack, "pack_manifest.json"), "w"), indent=2)
    print(f"[export] {pack_manifest['marker']} pack -> {out_pack}  raw={len(raw_files)} "
          f"meta={len(meta_files)} audit_files={len(copied_audit)}")
    return out_pack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="seaquest_ccrl/data/raw_hf")
    ap.add_argument("--meta-root", default="seaquest_ccrl/data/hostile_h0_metadata")
    ap.add_argument("--out-pack", default="artifacts/seaquest/hostile_h0/colab_pack")
    ap.add_argument("--audit-dir", default="artifacts/seaquest/hostile_h0")
    ap.add_argument("--no-copy-metadata", action="store_true")
    args = ap.parse_args()
    export(args.raw_root, args.meta_root, args.out_pack, args.audit_dir,
           copy_metadata=not args.no_copy_metadata)


if __name__ == "__main__":
    main()
