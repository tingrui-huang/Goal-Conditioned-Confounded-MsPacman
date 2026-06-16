"""Object-identity audit (Section 8 / D5 / D7). Local/Docker side (raw frames + metadata).

Two passes:
  PASS 1 (observe): for every VALID hostile/protected bbox, histogram the actual rendered
    RGB values, identify the dominant non-background sprite colour(s), and RESOLVE a
    per-class palette FROM OBSERVATION (never from blind RAM defaults). Writes
    observed_rgb_inventory.csv and resolved_palette.json with a small justified tolerance.
  PASS 2 (verify): using the resolved palette, count class-compatible pixels per object,
    bbox clipped fraction, overlap with protected bboxes, and flag zero-compatible objects.
    Checks resolved hostile palettes do not collide with WATER / HUD / protected colours
    (the EnemyMissile==Diver blue collision is the one EXPECTED, handled by ambiguity).

Markers: HOSTILE_OBJECT_SCHEMA_ALIGNED / HOSTILE_OBJECT_SCHEMA_INVALID.
Never broaden tolerance merely to make the audit pass.
"""
import sys, os, json, glob, csv, argparse
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from seaquest_ccrl.hostile import schema as S
from seaquest_ccrl.hostile import removal as R

# resolution parameters
RESOLVED_TOL = 6                 # tight box around each OBSERVED sprite colour
SHADE_DELTA = 12                 # only colours within this L1 of the dominant are TRUE shades
MAX_COLORS = 4                   # cap distinct colours kept per class
# background rejection when picking the sprite colour
WATER_L1 = 110                   # within this L1 of canonical water == background
DARK_MAX = 18                    # max channel value counted as near-black border
# systematic-failure criteria
ZERO_RATE_FAIL = 0.50
MIN_SUPPORT_FOR_FAIL = 50
COLLISION_TOL = 10               # hostile palette colliding with water/HUD within this == bad

WATER = np.asarray(S.WATER_COLOR, dtype=np.int16)
HUD_COLORS = {"score_lives_yellow": (210, 210, 64)}   # PlayerScore/Lives rendered yellow


def _iter_rows(raw_root, meta_root):
    raw = sorted(glob.glob(os.path.join(raw_root, "traj_*.npz")))
    meta = sorted(glob.glob(os.path.join(meta_root, "meta_*.npz")))
    for rf, mf in zip(raw, meta):
        d = np.load(rf); m = np.load(mf)
        ep = int(os.path.basename(mf).split("_")[1].split(".")[0])
        yield ep, d["frames"], m


def _is_background(rgb):
    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    if max(r, g, b) <= DARK_MAX:
        return True
    if abs(r - WATER[0]) + abs(g - WATER[1]) + abs(b - WATER[2]) <= WATER_L1:
        return True
    return False


def _bbox_pixels(frame, bbox):
    x, y, w, h = (int(v) for v in bbox)
    if w <= 0 or h <= 0:
        return np.zeros((0, 3), np.int16)
    return frame[y:y + h, x:x + w].reshape(-1, 3).astype(np.int16)


def _resolve_palette(counter):
    """counter: Counter of (r,g,b)->count over NON-background sprite pixels.

    Atari sprites are flat single colours, so the class palette = the DOMINANT colour
    plus only its true near-shades (within SHADE_DELTA L1). We deliberately do NOT pad
    with unrelated colours to hit a coverage target: a small EnemyMissile bbox contains
    ~13% adjacent-submarine gray, and a Diver bbox ~16% adjacent-shark green — those are
    OTHER classes' colours, not the sprite's, and must not enter the palette (including
    them would conflate channels and, for protected classes, shield hostile pixels).
    Returns (rgb_list, dominant_coverage, dominant, n_px)."""
    n = sum(counter.values())
    if n == 0:
        return [], 0.0, None, 0
    items = counter.most_common()
    dom = np.asarray(items[0][0], np.int16)
    chosen, covered = [list(items[0][0])], items[0][1]
    for col, c in items[1:]:
        if len(chosen) >= MAX_COLORS:
            break
        if np.abs(np.asarray(col, np.int16) - dom).sum() <= SHADE_DELTA:   # true shade only
            chosen.append(list(col)); covered += c
    return chosen, covered / n, list(items[0][0]), n


def _collisions(rgb_list, tol=COLLISION_TOL):
    out = []
    for col in rgb_list:
        c = np.asarray(col, np.int16)
        if np.abs(c - WATER).max() <= tol:
            out.append({"rgb": col, "collides_with": "water"})
        for name, hud in HUD_COLORS.items():
            if np.abs(c - np.asarray(hud, np.int16)).max() <= tol:
                out.append({"rgb": col, "collides_with": name})
    return out


def audit(raw_root, meta_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    classes = list(S.HOSTILE_ID) + list(S.PROTECTED_ID)
    counters = {c: Counter() for c in classes}
    n_obj = {c: 0 for c in classes}

    # ---- PASS 1: observe rendered RGB inside validated bboxes ----
    for ep, frames, m in _iter_rows(raw_root, meta_root):
        T = len(frames)
        for kind, bk, ck, vk, idmap in [
                ("h", "hostile_bbox", "hostile_class", "hostile_valid", S.HOSTILE_NAME),
                ("p", "protected_bbox", "protected_class", "protected_valid", S.PROTECTED_NAME)]:
            bb, cc, vv = m[bk], m[ck], m[vk]
            for t in range(T):
                for i in np.where(vv[t])[0]:
                    name = idmap[int(cc[t][i])]
                    if name == "padding":
                        continue
                    n_obj[name] += 1
                    px = _bbox_pixels(frames[t], bb[t][i])
                    for p in px:
                        if not _is_background(p):
                            counters[name][(int(p[0]), int(p[1]), int(p[2]))] += 1

    # resolve palette per class from observation
    resolved = {"classes": {}, "global_default_tol": R.DEFAULT_TOL, "resolved_tol": RESOLVED_TOL,
                "method": "dominant non-background rendered RGB inside validated OCAtari bboxes",
                "collisions": {}}
    inv_rows = []
    for c in classes:
        rgb_list, cov, dom, npx = _resolve_palette(counters[c])
        is_host = c in S.HOSTILE_ID
        coll = _collisions(rgb_list) if is_host else []
        # zero-support classes fall back to their RAM reference colour so removal stays
        # robust if such an object ever appears in a future dataset (e.g. SurfaceSubmarine).
        fallback = bool(not rgb_list)
        resolved["classes"][c] = {"rgb": rgb_list or [list(S.PALETTE_RGB.get(c, [0, 0, 0]))],
                                  "tol": RESOLVED_TOL, "observed_dominant": dom,
                                  "coverage": round(cov, 4), "n_objects": n_obj[c],
                                  "n_sprite_px": npx, "rgb_is_ram_fallback": fallback,
                                  "reference_rgb": list(S.PALETTE_RGB.get(c, []))}
        if coll:
            resolved["collisions"][c] = coll
        for col, cnt in counters[c].most_common(12):
            inv_rows.append([c, col[0], col[1], col[2], cnt,
                             round(cnt / max(1, sum(counters[c].values())), 4)])

    with open(os.path.join(out_dir, "observed_rgb_inventory.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "r", "g", "b", "count", "fraction_of_class_sprite_px"])
        w.writerows(inv_rows)
    json.dump(resolved, open(os.path.join(out_dir, "resolved_palette.json"), "w"), indent=2)
    # also surface the resolved palette at the schema location used by the export
    palettes = {c: {"rgb": v["rgb"] or [list(S.PALETTE_RGB.get(c, [0, 0, 0]))], "tol": RESOLVED_TOL}
                for c, v in resolved["classes"].items()}

    # ---- PASS 2: verify per-object compatibility with the RESOLVED palette ----
    perclass = {k: {"n": 0, "zero": 0, "px": [], "clipped": [], "overlap_protected": 0}
                for k in S.HOSTILE_ID}
    zero_examples = []
    strata = {k: None for k in [
        "one_shark", "multi_shark", "enemy_submarine", "surface_submarine", "enemy_missile",
        "missile_near_player", "missile_near_diver", "hostile_near_player",
        "hostile_near_player_missile", "edge_object", "multi_class", "object_overlap",
        "ambiguous_blue", "early_episode", "late_episode"]}
    n_rows = 0
    for ep, frames, m in _iter_rows(raw_root, meta_root):
        T = len(frames)
        hb, hc, hv = m["hostile_bbox"], m["hostile_class"], m["hostile_valid"]
        pb, pv, pc = m["protected_bbox"], m["protected_valid"], m["protected_class"]
        amb = m["ambiguous"] if "ambiguous" in m.files else np.zeros(T, bool)
        for t in range(T):
            n_rows += 1
            ht = {"hostile_bbox": hb[t], "hostile_class": hc[t], "hostile_valid": hv[t]}
            rep = R.compatible_pixel_report(frames[t], ht, palettes, RESOLVED_TOL)
            pbv = [tuple(int(v) for v in pb[t][i]) for i in np.where(pv[t])[0]]
            classes_here = set()
            for o in rep:
                name = o["class"]; perclass[name]["n"] += 1; classes_here.add(name)
                if o.get("zero_compatible"):
                    perclass[name]["zero"] += 1
                    if len(zero_examples) < 200:
                        zero_examples.append({"episode": ep, "t": t, "class": name, "bbox": o.get("bbox")})
                else:
                    perclass[name]["px"].append(o["compatible_px"])
                if "bbox" in o:
                    x, y, w, h = o["bbox"]
                    perclass[name]["clipped"].append(S.bbox_clipped_fraction(x, y, w, h))
                    if _overlap_protected(o["bbox"], pbv):
                        perclass[name]["overlap_protected"] += 1
            _assign_strata(strata, ep, t, frames[t], ht,
                           {"protected_bbox": pb[t], "protected_class": pc[t], "protected_valid": pv[t]},
                           bool(amb[t]), classes_here, T)

    summary, fail_classes = {}, []
    for k, v in perclass.items():
        n = v["n"]; zr = (v["zero"] / n) if n else 0.0
        summary[k] = {"n_valid_objects": n, "zero_compatible": v["zero"], "zero_rate": round(zr, 4),
                      "median_compatible_px": float(np.median(v["px"])) if v["px"] else 0.0,
                      "mean_clipped_fraction": float(np.mean(v["clipped"])) if v["clipped"] else 0.0,
                      "overlap_protected": v["overlap_protected"],
                      "resolved_palette": resolved["classes"][k]["rgb"],
                      "coverage": resolved["classes"][k]["coverage"]}
        if n >= MIN_SUPPORT_FOR_FAIL and zr >= ZERO_RATE_FAIL:
            fail_classes.append(k)

    bad_collisions = {c: v for c, v in resolved["collisions"].items()}   # water/HUD collisions only
    invalid = bool(fail_classes) or bool(bad_collisions)
    status = "HOSTILE_OBJECT_SCHEMA_INVALID" if invalid else "HOSTILE_OBJECT_SCHEMA_ALIGNED"
    result = {"status": status, "pass": not invalid, "n_rows": n_rows,
              "perclass": summary, "fail_classes": fail_classes,
              "palette_collisions_water_hud": bad_collisions,
              "expected_ambiguity": "EnemyMissile and Diver share blue (handled by ambiguity flag)",
              "zero_examples": zero_examples[:50],
              "resolved_palette_path": os.path.join(out_dir, "resolved_palette.json")}
    json.dump(result, open(os.path.join(out_dir, "object_identity_audit.json"), "w"), indent=2)
    _draw_overlay_grid(strata, os.path.join(out_dir, "object_overlay_grid.png"))
    print(f"[object-audit] {status} rows={n_rows} fail_classes={fail_classes} "
          f"collisions={list(bad_collisions)}")
    for k, v in summary.items():
        print(f"   {k:16s} n={v['n_valid_objects']:6d} zero={v['zero_rate']:.3f} "
              f"cov={v['coverage']:.3f} pal={v['resolved_palette']}")
    return result


def _overlap_protected(hb, pb_valid):
    x, y, w, h = hb
    for (px, py, pw, ph) in pb_valid:
        if x < px + pw and px < x + w and y < py + ph and py < y + h and pw > 0 and ph > 0:
            return True
    return False


def _center(b):
    return (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)


def _near(b1, b2, r=20):
    c1, c2 = _center(b1), _center(b2)
    return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1]) <= r


def _assign_strata(strata, ep, t, frame, ht, pt, amb, classes_here, T):
    def put(key):
        if strata.get(key) is None:
            strata[key] = (frame.copy(), ht, pt, ep, t)
    hv, hc, hb = ht["hostile_valid"], ht["hostile_class"], ht["hostile_bbox"]
    pv, pc, pb = pt["protected_valid"], pt["protected_class"], pt["protected_bbox"]
    hidx = list(np.where(hv)[0])
    hosts = [(int(hc[i]), tuple(int(v) for v in hb[i])) for i in hidx]
    prot = {S.PROTECTED_NAME[int(pc[i])]: tuple(int(v) for v in pb[i]) for i in np.where(pv)[0]}
    n_shark = sum(1 for c, _ in hosts if c == S.HOSTILE_ID["Shark"])
    if n_shark == 1: put("one_shark")
    if n_shark >= 2: put("multi_shark")
    if any(c == S.HOSTILE_ID["Submarine"] for c, _ in hosts): put("enemy_submarine")
    if any(c == S.HOSTILE_ID["SurfaceSubmarine"] for c, _ in hosts): put("surface_submarine")
    miss = [b for c, b in hosts if c == S.HOSTILE_ID["EnemyMissile"]]
    if miss: put("enemy_missile")
    if len(classes_here) >= 2: put("multi_class")
    if amb: put("ambiguous_blue")
    if t < 30: put("early_episode")
    if t > T - 30: put("late_episode")
    for c, b in hosts:
        x, y, w, h = b
        if x <= 1 or x + w >= S.FRAME_W - 1: put("edge_object")
        if "Player" in prot and _near(b, prot["Player"]): put("hostile_near_player")
        if "PlayerMissile" in prot and _near(b, prot["PlayerMissile"]): put("hostile_near_player_missile")
    for b in miss:
        if "Player" in prot and _near(b, prot["Player"]): put("missile_near_player")
        if "Diver" in prot and _near(b, prot["Diver"]): put("missile_near_diver")
    # object overlap among hostiles
    for i in range(len(hosts)):
        for j in range(i + 1, len(hosts)):
            if _near(hosts[i][1], hosts[j][1], r=8):
                put("object_overlap")


def _draw_overlay_grid(strata, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    items = [(k, v) for k, v in strata.items() if v is not None]
    if not items:
        return
    cols = 4; rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.4))
    axes = np.atleast_1d(axes).ravel()
    hcmap = {1: "lime", 2: "white", 3: "orange", 4: "red"}
    for ax, (key, (frame, ht, pt, ep, t)) in zip(axes, items):
        ax.imshow(frame); ax.set_title(f"{key}\nep{ep} t{t}", fontsize=7); ax.axis("off")
        for i in np.where(ht["hostile_valid"])[0]:
            x, y, w, h = (int(v) for v in ht["hostile_bbox"][i])
            ax.add_patch(Rectangle((x, y), w, h, fill=False,
                                   edgecolor=hcmap.get(int(ht["hostile_class"][i]), "magenta"), lw=1.2))
        for i in np.where(pt["protected_valid"])[0]:
            x, y, w, h = (int(v) for v in pt["protected_bbox"][i])
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="cyan", lw=0.8, linestyle=":"))
    for ax in axes[len(items):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(out_png, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="seaquest_ccrl/data/raw_hf")
    ap.add_argument("--meta-root", default="seaquest_ccrl/data/hostile_h0_metadata")
    ap.add_argument("--out-dir", default="artifacts/seaquest/hostile_h0/schema")
    args = ap.parse_args()
    res = audit(args.raw_root, args.meta_root, args.out_dir)
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
