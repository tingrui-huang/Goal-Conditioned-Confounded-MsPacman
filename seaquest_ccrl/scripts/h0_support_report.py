"""Support report (Section 12 / I). Reads metadata + raw player_pos ONLY (no frames,
no removal, no torch). Reports enemy / missile / joint support against the FROZEN
predeclared thresholds. Does NOT fit probes. Thresholds are never lowered after results.

Enemy   : >= 1000 valid active transitions AND >= 20 active episodes.
Missile : >=  300 valid active transitions AND >= 10 active episodes.
Insufficient missile support does NOT invalidate enemy-only continuation.
"""
import sys, os, json, glob, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from seaquest_ccrl.hostile import schema as S
from seaquest_ccrl.hostile import features as F

SPLIT_SEED = 2606
THRESH = {"enemy": {"transitions": 1000, "episodes": 20},
          "missile": {"transitions": 300, "episodes": 10}}


def _split(n_ep):
    rng = np.random.RandomState(SPLIT_SEED)
    order = rng.permutation(n_ep)
    ntr = int(round(0.70 * n_ep)); nva = int(round(0.15 * n_ep))
    return (sorted(int(x) for x in order[:ntr]),
            sorted(int(x) for x in order[ntr:ntr + nva]),
            sorted(int(x) for x in order[ntr + nva:]))


def report(raw_root, meta_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    raw = sorted(glob.glob(os.path.join(raw_root, "traj_*.npz")))
    meta = sorted(glob.glob(os.path.join(meta_root, "meta_*.npz")))
    assert raw and len(raw) == len(meta), f"raw/meta mismatch {len(raw)}/{len(meta)}"

    hc, hv, amb, ppos, ep_of, lengths = [], [], [], [], [], []
    nearest_enemy_d, nearest_missile_d = [], []
    perclass_total = {k: 0 for k in S.HOSTILE_ID}
    grid_enemy = np.zeros(9, np.int64); grid_missile = np.zeros(9, np.int64)
    per_ep_enemy, per_ep_missile = [], []

    for ei, (rf, mf) in enumerate(zip(raw, meta)):
        d = np.load(rf); m = np.load(mf)
        T = len(m["hostile_valid"])
        cls, val = m["hostile_class"], m["hostile_valid"]
        pp = np.asarray(d["player_pos"], np.float32)
        en = (np.isin(cls, S.ENEMY_IDS) & val).sum(1)
        mi = (np.isin(cls, S.MISSILE_IDS) & val).sum(1)
        a = m["ambiguous"].astype(bool) if "ambiguous" in m.files else np.zeros(T, bool)
        hc.append(cls); hv.append(val); amb.append(a); ppos.append(pp)
        ep_of.append(np.full(T, ei)); lengths.append(T)
        per_ep_enemy.append(int((en > 0).sum())); per_ep_missile.append(int((mi > 0).sum()))
        for name, cid in S.HOSTILE_ID.items():
            perclass_total[name] += int(((cls == cid) & val).sum())
        for t in range(T):
            ht = {"hostile_bbox": m["hostile_bbox"][t], "hostile_class": cls[t], "hostile_valid": val[t]}
            if en[t] > 0:
                dx, dy, ms = F.nearest_offset(ht, pp[t], "enemy")
                if not ms:
                    nearest_enemy_d.append(abs(dx) + abs(dy))
                grid_enemy += F.presence_grid(ht, pp[t], "enemy")
            if mi[t] > 0:
                dx, dy, ms = F.nearest_offset(ht, pp[t], "missile")
                if not ms:
                    nearest_missile_d.append(abs(dx) + abs(dy))
                grid_missile += F.presence_grid(ht, pp[t], "missile")

    n_ep = len(raw)
    hc = np.concatenate(hc); hv = np.concatenate(hv); amb = np.concatenate(amb)
    ep_of = np.concatenate(ep_of)
    en_mask = (np.isin(hc, S.ENEMY_IDS) & hv).sum(1) > 0
    mi_mask = (np.isin(hc, S.MISSILE_IDS) & hv).sum(1) > 0
    joint_mask = en_mask | mi_mask
    tr, va, te = _split(n_ep)
    split_of = {e: ("train" if e in tr else "val" if e in va else "test") for e in range(n_ep)}

    def comp(which, mask, exclude_ambiguous):
        accepted = mask & (~amb if exclude_ambiguous else np.ones_like(mask))
        eps = np.unique(ep_of[mask])
        d = {"active_transitions": int(mask.sum()),
             "accepted_primary_transitions": int(accepted.sum()),
             "excluded_ambiguous_rows": int((mask & amb).sum()),
             "presence_fraction": float(mask.mean()),
             "active_episodes": int(len(eps)),
             "active_per_split": {s: int(mask[np.isin(ep_of, [e for e in range(n_ep)
                                  if split_of[e] == s])].sum()) for s in ("train", "val", "test")}}
        th = THRESH.get(which)
        if th:
            ok = d["active_transitions"] >= th["transitions"] and d["active_episodes"] >= th["episodes"]
            d["threshold"] = th
            d["label"] = (f"{which.upper()}_SUPPORT_SUFFICIENT" if ok
                          else f"{which.upper()}_INSUFFICIENT_SUPPORT")
            d["pass"] = bool(ok)
        return d

    res = {
        "n_episodes": n_ep, "N": int(len(hc)), "split_seed": SPLIT_SEED,
        "split_sizes": {"train": len(tr), "val": len(va), "test": len(te)},
        "ambiguous_rows": int(amb.sum()), "ambiguous_fraction": float(amb.mean()),
        "per_class_object_count": perclass_total,
        "enemy": comp("enemy", en_mask, exclude_ambiguous=False),
        "missile": comp("missile", mi_mask, exclude_ambiguous=True),
        "joint": comp("joint", joint_mask, exclude_ambiguous=True),
        "nearest_enemy_L1_distance": _dist_summary(nearest_enemy_d),
        "nearest_missile_L1_distance": _dist_summary(nearest_missile_d),
        "enemy_3x3_grid_support": grid_enemy.tolist(),
        "missile_3x3_grid_support": grid_missile.tolist(),
        "per_episode_enemy_rows": per_ep_enemy,
        "per_episode_missile_rows": per_ep_missile,
    }
    res["joint"]["pass"] = bool(res["enemy"].get("pass") or res["missile"].get("pass"))
    json.dump(res, open(os.path.join(out_dir, "support_report.json"), "w"), indent=2)
    _write_md(res, os.path.join(out_dir, "support_report.md"))
    print(f"[support] enemy={res['enemy'].get('label')} missile={res['missile'].get('label')} "
          f"joint_pass={res['joint']['pass']}")
    print(f"   enemy: {res['enemy']['active_transitions']} tx / {res['enemy']['active_episodes']} eps")
    print(f"   missile: {res['missile']['active_transitions']} tx / {res['missile']['active_episodes']} eps "
          f"(ambiguous excluded {res['missile']['excluded_ambiguous_rows']})")
    return res


def _dist_summary(vals):
    if not vals:
        return {"n": 0}
    a = np.asarray(vals, float)
    return {"n": int(len(a)), "min": float(a.min()), "p25": float(np.percentile(a, 25)),
            "median": float(np.median(a)), "p75": float(np.percentile(a, 75)),
            "max": float(a.max()), "mean": float(a.mean())}


def _write_md(res, path):
    L = ["# Stage-H0 Support Report", "",
         f"Episodes: {res['n_episodes']}  N rows: {res['N']}  split seed {res['split_seed']}",
         f"Ambiguous rows: {res['ambiguous_rows']} ({res['ambiguous_fraction']:.4f})", "",
         "| component | active tx | active eps | accepted primary | label |",
         "|---|---|---|---|---|"]
    for c in ("enemy", "missile", "joint"):
        v = res[c]
        L.append(f"| {c} | {v['active_transitions']} | {v['active_episodes']} | "
                 f"{v.get('accepted_primary_transitions')} | {v.get('label', 'n/a')} |")
    L += ["", f"Per-class object counts: {res['per_class_object_count']}", ""]
    open(path, "w").write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="seaquest_ccrl/data/raw_hf")
    ap.add_argument("--meta-root", default="seaquest_ccrl/data/hostile_h0_metadata")
    ap.add_argument("--out-dir", default="artifacts/seaquest/hostile_h0/support")
    args = ap.parse_args()
    report(args.raw_root, args.meta_root, args.out_dir)


if __name__ == "__main__":
    main()
