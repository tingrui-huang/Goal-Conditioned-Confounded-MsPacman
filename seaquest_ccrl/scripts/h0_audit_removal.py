"""Removal audit (Section 9 / H). Local/Docker side (raw frames + metadata, no teacher).

Uses the RESOLVED palette (artifacts/.../schema/resolved_palette.json) produced by the
object-identity audit. Runs class-aware pixel removal on stratified sampled frames and
HARD-asserts the invariants via removal.verify_removal:
  * changed pixels are a subset of the union of valid hostile bboxes;
  * pixels outside hostile bboxes are byte-identical;
  * protected sprite pixels (Player/PlayerMissile/Diver) are byte-identical.
Also measures: changed/unchanged hostile-compatible px, changed protected px, changed
non-hostile px inside bbox, changed px by class / overlap category, zero-change hostile
objects, full-rectangle-fill and constant-patch occurrences. Ambiguous blue (missile/diver
overlap) rows are excluded from the accepted-primary protected-change requirement and
counted separately. Markers: HOSTILE_REMOVAL_ALIGNED / HOSTILE_REMOVAL_INVALID.
"""
import sys, os, json, glob, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from seaquest_ccrl.hostile import schema as S
from seaquest_ccrl.hostile import removal as R

DEFAULT_PALETTE = "artifacts/seaquest/hostile_h0/schema/resolved_palette.json"


def _stratified_indices(m, max_samples):
    hv, hc = m["hostile_valid"], m["hostile_class"]
    amb = m["ambiguous"] if "ambiguous" in m.files else np.zeros(len(hv), bool)
    enemy = (np.isin(hc, S.ENEMY_IDS) & hv).sum(1)
    miss = (np.isin(hc, S.MISSILE_IDS) & hv).sum(1)
    pools = {
        "enemy_only": np.where((enemy > 0) & (miss == 0) & ~amb)[0],
        "missile_present": np.where((miss > 0) & ~amb)[0],
        "multi_enemy": np.where(enemy >= 2)[0],
        "ambiguous": np.where(amb)[0],
        "empty": np.where((enemy == 0) & (miss == 0))[0],
    }
    rng = np.random.RandomState(0)
    cand = []
    for idx in pools.values():
        if len(idx):
            cand.extend(rng.choice(idx, size=min(3, len(idx)), replace=False).tolist())
    return sorted(set(int(i) for i in cand))[:max_samples]


def audit(raw_root, meta_root, out_dir, palette_path=DEFAULT_PALETTE,
          max_samples_per_ep=4, max_eps=12):
    os.makedirs(out_dir, exist_ok=True)
    palettes, tol = None, R.DEFAULT_TOL
    if os.path.exists(palette_path):
        palettes = R.load_resolved_palettes(palette_path)
        tol = json.load(open(palette_path)).get("resolved_tol", R.DEFAULT_TOL)
    raw = sorted(glob.glob(os.path.join(raw_root, "traj_*.npz")))
    meta = sorted(glob.glob(os.path.join(meta_root, "meta_*.npz")))

    agg = {"changed_outside_bbox": 0, "changed_protected_accepted": 0,
           "changed_protected_ambiguous": 0, "changed_by_class": {k: 0 for k in S.HOSTILE_ID},
           "changed_compatible_px": 0, "unchanged_compatible_px": 0,
           "changed_noncompatible_in_bbox": 0, "zero_change_objects": 0,
           "full_rectangle_fills": 0, "n_objects": 0}
    samples, violations = [], []
    n_checked = 0

    for rf, mf in list(zip(raw, meta))[:max_eps]:
        d = np.load(rf); m = np.load(mf)
        ep = int(os.path.basename(mf).split("_")[1].split(".")[0])
        amb_ep = m["ambiguous"] if "ambiguous" in m.files else np.zeros(len(m["hostile_valid"]), bool)
        for t in _stratified_indices(m, max_samples_per_ep):
            frame = d["frames"][t]
            ht = {k: m[k][t] for k in ("hostile_bbox", "hostile_class", "hostile_valid")}
            pt = {k: m[k][t] for k in ("protected_bbox", "protected_class", "protected_valid")}
            rem, changed, stats = R.remove_frame(frame, ht, pt, palettes, tol)
            try:
                R.verify_removal(frame, rem, changed, ht, pt, palettes, tol)
            except AssertionError as e:
                violations.append({"episode": ep, "t": t, "error": str(e)})
            n_checked += 1
            ambiguous = bool(amb_ep[t])
            _measure(frame, rem, changed, ht, pt, palettes, tol, agg, ambiguous)
            if len(samples) < 24:
                samples.append({"episode": ep, "t": t, "orig": frame, "removed": rem,
                                "changed": changed, "ambiguous": ambiguous,
                                "hb": ht["hostile_bbox"], "hv": ht["hostile_valid"],
                                "hc": ht["hostile_class"], "pb": pt["protected_bbox"],
                                "pv": pt["protected_valid"]})

    hard_fail = (violations or agg["changed_outside_bbox"] > 0
                 or agg["changed_protected_accepted"] > 0)
    status = "HOSTILE_REMOVAL_INVALID" if hard_fail else "HOSTILE_REMOVAL_ALIGNED"
    result = {"status": status, "pass": not hard_fail, "n_checked": n_checked,
              "palette_path": palette_path if palettes else "RAM_DEFAULT", "tol": tol,
              "aggregate": agg, "violations": violations,
              "note_ambiguous": "missile/diver blue-overlap rows excluded from accepted-primary; "
                                "protected changes there counted separately and must still be 0"}
    json.dump(result, open(os.path.join(out_dir, "removal_audit.json"), "w"), indent=2)
    if samples:
        np.savez_compressed(os.path.join(out_dir, "audit_samples.npz"),
                            orig=np.stack([s["orig"] for s in samples]),
                            removed=np.stack([s["removed"] for s in samples]),
                            changed=np.stack([s["changed"] for s in samples]),
                            episode=np.array([s["episode"] for s in samples]),
                            t=np.array([s["t"] for s in samples]),
                            ambiguous=np.array([s["ambiguous"] for s in samples]))
        _draw_grid(samples, os.path.join(out_dir, "removal_grid.png"))
    print(f"[removal-audit] {status} checked={n_checked} violations={len(violations)} "
          f"outside_bbox={agg['changed_outside_bbox']} protected_accepted={agg['changed_protected_accepted']}")
    return result


def _measure(frame, rem, changed, ht, pt, palettes, tol, agg, ambiguous):
    H, W = frame.shape[:2]
    union = np.zeros((H, W), bool)
    bbox, cls, valid = ht["hostile_bbox"], ht["hostile_class"], ht["hostile_valid"]
    for i in np.where(valid)[0]:
        x, y, w, h = (int(v) for v in bbox[i])
        if w <= 0 or h <= 0:
            continue
        agg["n_objects"] += 1
        union[y:y + h, x:x + w] = True
        name = S.HOSTILE_NAME[int(cls[i])]
        sub_change = changed[y:y + h, x:x + w]
        comp = R.class_compatible_mask(frame[y:y + h, x:x + w], name, palettes, tol)
        agg["changed_by_class"][name] += int(sub_change.sum())
        agg["changed_compatible_px"] += int((sub_change & comp).sum())
        agg["unchanged_compatible_px"] += int((~sub_change & comp).sum())
        agg["changed_noncompatible_in_bbox"] += int((sub_change & ~comp).sum())
        if sub_change.sum() == 0:
            agg["zero_change_objects"] += 1
        if sub_change.sum() == w * h and w * h >= 60:      # whole large bbox painted
            agg["full_rectangle_fills"] += 1
    # protected change accounting
    prot = R._protected_mask(frame, pt, palettes, tol)
    diff = np.any(frame != rem, axis=-1)
    agg["changed_outside_bbox"] += int((diff & ~union).sum())
    pc = int((diff & prot).sum())
    if ambiguous:
        agg["changed_protected_ambiguous"] += pc
    else:
        agg["changed_protected_accepted"] += pc


def _draw_grid(samples, out_png, n=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    sel = samples[:n]
    fig, axes = plt.subplots(len(sel), 4, figsize=(4 * 2.6, len(sel) * 2.6))
    axes = np.atleast_2d(axes)
    hcmap = {1: "lime", 2: "white", 3: "orange", 4: "red"}
    for r, s in enumerate(sel):
        diff = np.any(s["orig"] != s["removed"], axis=-1).astype(np.uint8) * 255
        for c, (img, title) in enumerate([
                (s["orig"], "original"), (s["removed"], "removed"),
                (diff, "changed mask"), (s["orig"], "bbox overlay")]):
            ax = axes[r, c]; ax.imshow(img, cmap="gray" if c == 2 else None); ax.axis("off")
            if r == 0:
                ax.set_title(title, fontsize=8)
            if c == 3:
                for i in np.where(s["hv"])[0]:
                    x, y, w, h = (int(v) for v in s["hb"][i])
                    ax.add_patch(Rectangle((x, y), w, h, fill=False,
                                           edgecolor=hcmap.get(int(s["hc"][i]), "magenta"), lw=1.0))
                for i in np.where(s["pv"])[0]:
                    x, y, w, h = (int(v) for v in s["pb"][i])
                    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="cyan",
                                           lw=0.8, linestyle=":"))
        tag = "AMBIG" if s["ambiguous"] else ""
        axes[r, 0].set_ylabel(f"ep{s['episode']} t{s['t']} {tag}", fontsize=7)
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="seaquest_ccrl/data/raw_hf")
    ap.add_argument("--meta-root", default="seaquest_ccrl/data/hostile_h0_metadata")
    ap.add_argument("--out-dir", default="artifacts/seaquest/hostile_h0/removal")
    ap.add_argument("--palette", default=DEFAULT_PALETTE)
    args = ap.parse_args()
    res = audit(args.raw_root, args.meta_root, args.out_dir, palette_path=args.palette)
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
