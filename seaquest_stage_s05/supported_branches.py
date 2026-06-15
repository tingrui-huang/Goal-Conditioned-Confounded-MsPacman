"""P6 — supported-action forced-branch heatmaps (sections 13-14, Gate P6).
CORRECTED rerun: all action selection routes through the FROZEN Stage-S0 adapter
(teacher.greedy_action / teacher.sample_action + teacher.gumbel_from_uniform). No
inline Gumbel formula. Reuses S0 clone/restore + continuation.

Distance-metric fix (heatmap audit): the future-state distance is computed on a
CLEANED view (reliably-present components only; the missing-prone enemy_centroid is
excluded) and STANDARDIZED per component (z-units) so no single raw feature (e.g.
absolute `score`) dominates and the scale is comparable across horizons. The
divergence FRACTION (the gate metric) is scale-free: a pair diverges if the raw
cleaned integer view differs in any component beyond the same-action baseline.
Every plotted matrix's key + SHA256 + support-count is saved.
"""
import sys, os, json, argparse, hashlib
_REAL = sys.stdout
def prog(*a):
    print(*a, file=_REAL, flush=True)
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

BASE = "/work/artifacts/seaquest/stage_s05"
CFG = json.load(open(f"{BASE}/config/resolved_config.json"))
SF = CFG["state_features"]; H = [4, 8, 16]; MAXH = 16
TH = CFG["support_thresholds"]
A2CAT = C.ACTION_TO_CAT
# cleaned, reliably-present future views (enemy_centroid excluded: missing-prone)
NP_KEYS = ["n_shark", "n_submarine", "n_diver", "n_player_missile", "n_enemy_missile",
           "score", "lives", "cum_reward", "terminated"]
WO_KEYS = ["n_shark", "n_submarine", "n_diver", "n_enemy_missile",
           "score", "lives", "cum_reward", "terminated"]  # excludes own missile


def rawvec(f, keys, cum, term):
    out = []
    for k in keys:
        if k == "cum_reward":
            out.append(float(cum))
        elif k == "terminated":
            out.append(float(term))
        else:
            v = f.get(k)
            out.append(0.0 if v is None else float(v))  # counts/score/lives present; 0 = absent
    return np.array(out, dtype=np.float64)


def load_support():
    d = np.load(f"{BASE}/closed_loop/rows_O-Sampled.npz", allow_pickle=True)
    cols = list(d["columns"]); R = d["rows"]; ci = {c: i for i, c in enumerate(cols)}
    X = R[:, [ci[f] for f in SF]].astype(np.float64)
    with np.errstate(all="ignore"):
        med = np.nan_to_num(np.nanmedian(X, axis=0), nan=0.0)
    Xi = np.nan_to_num(np.where(np.isnan(X), med, X), nan=0.0)
    sc = StandardScaler().fit(Xi)
    nn = NearestNeighbors(n_neighbors=50).fit(sc.transform(Xi))
    return sc, nn, R[:, ci["sampled_action"]].astype(int), med


def supported_actions(sc, nn, actions, med, feat):
    x = np.array([np.nan if feat.get(k) is None else float(feat.get(k)) for k in SF])
    x = np.nan_to_num(np.where(np.isnan(x), med, x), nan=0.0)
    _, idx = nn.kneighbors(sc.transform(x[None]))
    h = np.bincount(actions[idx[0]], minlength=18); prop = h / h.sum()
    return [a for a in range(18) if prop[a] >= TH["local_propensity_min"] and h[a] >= TH["min_neighbor_occurrences"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", type=int, default=40)
    ap.add_argument("--seed", type=int, default=105)
    args = ap.parse_args()
    from teacher_port import SeaquestPort
    teacher = C.load_teacher("A")
    sc, nn, ds_actions, med = load_support()
    prog(f"[P6] dataset support loaded; teacher dim={teacher.action_dim} (frozen adapter)")

    port = SeaquestPort(sticky=0.0, full_action_space=True, seed=args.seed)
    rng = np.random.RandomState(args.seed + 11)
    noise_rng = np.random.RandomState(args.seed + 777)
    anchor_pool = []
    for ep in range(6):
        port.reset(seed=args.seed + ep, noop_max=30, rng=rng)
        start_lives = port.features()["lives"]; prev = None; t = 0
        while True:
            obs = port.teacher_obs()
            noise = teacher.gumbel_from_uniform(noise_rng.uniform(size=(18,)))
            a = int(teacher.sample_action(obs, noise, temperature=1.0)[0])  # FROZEN
            f = C.enrich_features(port.features(), prev)
            if t % 4 == 0:
                anchor_pool.append({"snap": port.snapshot(), "feat": f})
            rec = port.agent_step(a); prev = f; t += 1
            lv = port.features()["lives"]
            ll = (start_lives is not None and lv is not None and lv < start_lives)
            if rec["terminated"] or rec["truncated"] or ll or t >= 600:
                break
    prog(f"[P6] anchor pool {len(anchor_pool)}")
    selidx = rng.choice(len(anchor_pool), size=min(args.anchors, len(anchor_pool)), replace=False)
    anchors = [anchor_pool[i] for i in selidx]

    # shared common-random-number UNIFORM schedule -> frozen gumbel helper per step
    u_sched = noise_rng.uniform(size=(MAXH + 1, 18))
    noise_sched = teacher.gumbel_from_uniform(u_sched)

    def continue_branch(first_a):
        traj = {}; cum = 0.0; term = False
        for h in range(1, MAXH + 1):
            if h == 1:
                a = first_a
            else:
                a = int(teacher.sample_action(port.teacher_obs(), noise_sched[h], temperature=1.0)[0])  # FROZEN
            rec = port.agent_step(a); cum += rec["reward"]
            term = term or rec["terminated"] or rec["truncated"]
            f = C.enrich_features(port.features(), None)
            if h in H:
                traj[h] = {"np": rawvec(f, NP_KEYS, cum, term), "wo": rawvec(f, WO_KEYS, cum, term)}
            if term:
                for hh in H:
                    if hh >= h:
                        traj.setdefault(hh, {"np": rawvec(f, NP_KEYS, cum, term), "wo": rawvec(f, WO_KEYS, cum, term)})
                break
        return traj

    # collect raw branch vectors
    branch_data = []  # per anchor: {supp, branches, base}
    all_np = {h: [] for h in H}; all_wo = {h: [] for h in H}
    for ai, anc in enumerate(anchors):
        supp = supported_actions(sc, nn, ds_actions, med, anc["feat"])
        if len(supp) < 2:
            branch_data.append(None); continue
        br = {}
        for a in supp:
            port.restore(anc["snap"]); br[a] = continue_branch(a)
        port.restore(anc["snap"]); base = continue_branch(supp[0])
        branch_data.append({"supp": supp, "br": br, "base": base})
        for h in H:
            for a in supp:
                all_np[h].append(br[a][h]["np"]); all_wo[h].append(br[a][h]["wo"])
        if ai % 10 == 0:
            prog(f"[P6] anchor {ai}/{len(anchors)} supported={len(supp)}")

    # per-component std over ALL branch outcomes (z-standardization for distance scale)
    std_np = {h: np.maximum(np.std(np.array(all_np[h]), axis=0), 1e-6) for h in H}
    std_wo = {h: np.maximum(np.std(np.array(all_wo[h]), axis=0), 1e-6) for h in H}

    pair_np = {h: np.zeros((18, 18)) for h in H}; pair_wo = {h: np.zeros((18, 18)) for h in H}
    pair_cnt = {h: np.zeros((18, 18)) for h in H}
    eligible = 0; np_div = {h: 0 for h in H}; wo_div = {h: 0 for h in H}; base_noise = {h: 0 for h in H}
    sem_pairs = {h: set() for h in H}
    for anc in branch_data:
        if anc is None:
            continue
        eligible += 1
        supp = anc["supp"]; br = anc["br"]; base = anc["base"]
        for h in H:
            if not np.array_equal(br[supp[0]][h]["np"], base[h]["np"]):
                base_noise[h] += 1
            anp = awo = False
            for ii in range(len(supp)):
                for jj in range(ii + 1, len(supp)):
                    ai_, aj_ = supp[ii], supp[jj]
                    dnp = float(np.linalg.norm((br[ai_][h]["np"] - br[aj_][h]["np"]) / std_np[h]))
                    dwo = float(np.linalg.norm((br[ai_][h]["wo"] - br[aj_][h]["wo"]) / std_wo[h]))
                    pair_np[h][ai_, aj_] += dnp; pair_wo[h][ai_, aj_] += dwo; pair_cnt[h][ai_, aj_] += 1
                    if not np.array_equal(br[ai_][h]["np"], br[aj_][h]["np"]):  # raw scale-free divergence
                        anp = True
                        if A2CAT[ai_] != A2CAT[aj_]:
                            sem_pairs[h].add((min(A2CAT[ai_], A2CAT[aj_]), max(A2CAT[ai_], A2CAT[aj_])))
                    if not np.array_equal(br[ai_][h]["wo"], br[aj_][h]["wo"]):
                        awo = True
            if anp:
                np_div[h] += 1
            if awo:
                wo_div[h] += 1

    gate = {"eligible_anchors": eligible, "horizons": {}}; p6 = False
    for h in H:
        e = max(eligible, 1)
        npf = np_div[h] / e; wof = wo_div[h] / e
        ok = (npf >= 0.20 and wof >= 0.10 and np_div[h] > base_noise[h] and len(sem_pairs[h]) >= 2)
        gate["horizons"][h] = {"noplayer_frac": npf, "worldonly_frac": wof, "baseline_noise": base_noise[h],
                               "n_distinct_semantic_pairs": len(sem_pairs[h]), "passes": bool(ok)}
        if ok:
            p6 = True
    gate["pass"] = p6

    os.makedirs(f"{BASE}/branches", exist_ok=True)
    avg_np = {h: np.where(pair_cnt[h] > 0, pair_np[h] / np.maximum(pair_cnt[h], 1), np.nan) for h in H}
    avg_wo = {h: np.where(pair_cnt[h] > 0, pair_wo[h] / np.maximum(pair_cnt[h], 1), np.nan) for h in H}
    arrays = {}
    for h in H:
        arrays[f"np_H{h}"] = avg_np[h]; arrays[f"wo_H{h}"] = avg_wo[h]; arrays[f"count_H{h}"] = pair_cnt[h]
    np.savez_compressed(f"{BASE}/branches/raw_outputs.npz", **arrays)
    # per-matrix key + sha256 manifest (heatmap integrity)
    def sh(a):
        return hashlib.sha256(np.ascontiguousarray(np.nan_to_num(a, nan=-1.0)).tobytes()).hexdigest()
    manifest = {k: {"sha256": sh(v), "shape": list(v.shape),
                    "finite_min": float(np.nanmin(v)) if np.isfinite(v).any() else None,
                    "finite_max": float(np.nanmax(v)) if np.isfinite(v).any() else None,
                    "distance": "z-standardized L2 (per-component std over all branch outcomes)"
                                if not k.startswith("count") else "support count"}
                for k, v in arrays.items()}
    json.dump({"distance_metric": ("CLEANED view (enemy_centroid excluded as missing-prone); "
                                   "per-component z-standardization so absolute `score` no longer dominates; "
                                   "scale now comparable across horizons. Divergence FRACTION uses raw "
                                   "scale-free integer view."),
               "np_view": NP_KEYS, "wo_view": WO_KEYS,
               "views_are_distinct": NP_KEYS != WO_KEYS,
               "std_np": {h: std_np[h].tolist() for h in H},
               "std_wo": {h: std_wo[h].tolist() for h in H},
               "plotted_matrix_manifest": manifest},
              open(f"{BASE}/branches/heatmap_integrity.json", "w"), indent=2)
    json.dump({"gate_p6": gate, "np_divergence": {h: np_div[h] for h in H},
               "wo_divergence": {h: wo_div[h] for h in H}, "baseline_noise": base_noise,
               "n_distinct_semantic_pairs": {h: len(sem_pairs[h]) for h in H}},
              open(f"{BASE}/branches/action_pair_metrics.json", "w"), indent=2)
    per = [{"anchor": i, "n_supported": len(a["supp"]) if a else 0,
            "supported": a["supp"] if a else [], "eligible": a is not None}
           for i, a in enumerate(branch_data)]
    json.dump({"n_anchors": len(anchors), "eligible": eligible, "support_threshold": TH,
               "sampling": "FROZEN Stage-S0 teacher.sample_action / greedy_action", "per_anchor": per},
              open(f"{BASE}/branches/anchor_manifest.json", "w"), indent=2)
    json.dump({"per_anchor_supported": [{"anchor": p["anchor"], "supported": p["supported"]}
                                        for p in per if p["eligible"]]},
              open(f"{BASE}/branches/supported_action_pairs.json", "w"), indent=2)
    json.dump({"baseline_noise_per_horizon": base_noise,
               "note": "same-action duplicate branch; nonzero = OCAtari tracker transient (S0-documented)"},
              open(f"{BASE}/branches/same_action_baseline.json", "w"), indent=2)
    prog(f"[P6] DONE pass={p6} eligible={eligible}")
    for h in H:
        g = gate["horizons"][h]
        prog(f"  H={h}: np_frac={g['noplayer_frac']:.2f} wo_frac={g['worldonly_frac']:.2f} "
             f"sem_pairs={g['n_distinct_semantic_pairs']} base={base_noise[h]} "
             f"dist_max={manifest[f'np_H{h}']['finite_max']:.2f}")


if __name__ == "__main__":
    main()
