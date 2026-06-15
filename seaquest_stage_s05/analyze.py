"""Heatmaps A-F, P2 state-conditioning, closed-loop metrics, native-vs-ported.
Host-side (numpy/json only). Reads the collected per-mode NPZs.

Native EnvPool exposes NO object features -> object-conditioned heatmaps (A-E) and
matched-abstract-state comparison (F level 2) are computed for the O modes only;
the native side is documented as infeasible (contract s18). Native participates in
the action-marginal divergence (F) and action-only closed-loop metrics.
"""
import json, os
import numpy as np

BASE = "artifacts/seaquest/stage_s05"
CFG = json.load(open(f"{BASE}/config/resolved_config.json"))
BINS = CFG["bins"]; ROLE = {
    "UP_related": [2, 6, 7, 10, 14, 15], "DOWN_related": [5, 8, 9, 13, 16, 17],
    "LEFT_related": [4, 7, 9, 12, 15, 17], "RIGHT_related": [3, 6, 8, 11, 14, 16],
    "FIRE_related": [1, 10, 11, 12, 13, 14, 15, 16, 17]}
SEMANTIC = CFG["semantic_categories"]; CAT_NAMES = list(SEMANTIC.keys())
A2CAT = {a: i for i, (k, v) in enumerate(SEMANTIC.items()) for a in v}
MEAN = CFG["ALE_MEANINGS"]


def load(mode):
    d = np.load(f"{BASE}/closed_loop/rows_{mode}.npz", allow_pickle=True)
    cols = list(d["columns"]); R = d["rows"]
    ci = {c: i for i, c in enumerate(cols)}
    return R, ci


def col(R, ci, name):
    return R[:, ci[name]]


def probs_mat(R, ci):
    return R[:, [ci[f"prob_{i}"] for i in range(18)]]


def logits_mat(R, ci):
    return R[:, [ci[f"logit_{i}"] for i in range(18)]]


def binidx(x, edges):
    return np.digitize(x, edges[1:-1])


def role_prob(P, role):
    return P[:, ROLE[role]].sum(axis=1)


def entropy(P):
    p = np.clip(P, 1e-12, 1)
    return -(p * np.log(p)).sum(axis=1)


def heatmap_grid(rx, ry, ex, ey, vals, agg="mean"):
    nx, ny = len(ex) - 1, len(ey) - 1
    G = np.full((ny, nx), np.nan); Cnt = np.zeros((ny, nx))
    bx = binidx(rx, ex); by = binidx(ry, ey)
    for i in range(ny):
        for j in range(nx):
            m = (by == i) & (bx == j)
            Cnt[i, j] = m.sum()
            if m.sum() > 0:
                G[i, j] = np.nanmean(vals[m]) if agg == "mean" else np.nansum(vals[m])
    return G, Cnt


def heatmaps_for_mode(mode):
    R, ci = load(mode)
    P = probs_mat(R, ci); L = logits_mat(R, ci)
    sampled = col(R, ci, "sampled_action").astype(int)
    out = {}
    # need object features
    py = col(R, ci, "player_y"); oxy = col(R, ci, "oxygen")
    edx = col(R, ci, "nearest_enemy_dx"); edy = col(R, ci, "nearest_enemy_dy")
    ddx = col(R, ci, "nearest_diver_dx"); ddy = col(R, ci, "nearest_diver_dy")
    pmiss = col(R, ci, "player_missile_count"); ecnt = col(R, ci, "enemy_count")
    valid_obj = np.isfinite(py)
    out["n_rows"] = int(len(R)); out["n_rows_with_objects"] = int(valid_obj.sum())
    if valid_obj.sum() < 10:
        out["object_heatmaps"] = "INFEASIBLE: no object features (native EnvPool)"
        return out, None

    grids = {}
    # Heatmap A: player_y x oxygen
    m = valid_obj & np.isfinite(oxy)
    eyb, exb = BINS["player_y"], BINS["oxygen"]
    for val, name in [(role_prob(P, "UP_related"), "P_UP_related"),
                      (P[:, 10], "P_UPFIRE"), (P[:, 2], "P_UP"),
                      (entropy(P), "entropy")]:
        G, Cn = heatmap_grid(oxy[m], py[m], exb, eyb, val[m])
        grids[f"A__{name}"] = G; grids[f"A__{name}__count"] = Cn
    # Heatmap B: enemy_dy x enemy_dx
    m = valid_obj & np.isfinite(edx) & np.isfinite(edy)
    eyb, exb = BINS["enemy_dy"], BINS["enemy_dx"]
    for role in ["LEFT_related", "RIGHT_related", "UP_related", "DOWN_related", "FIRE_related"]:
        G, Cn = heatmap_grid(edx[m], edy[m], exb, eyb, role_prob(P, role)[m])
        grids[f"B__P_{role}"] = G; grids[f"B__P_{role}__count"] = Cn
    G, Cn = heatmap_grid(edx[m], edy[m], exb, eyb, entropy(P)[m]); grids["B__entropy"] = G; grids["B__entropy__count"] = Cn
    # Heatmap C: diver_dy x diver_dx (if enough)
    mc = valid_obj & np.isfinite(ddx) & np.isfinite(ddy)
    out["diver_support_rows"] = int(mc.sum())
    if mc.sum() >= 50:
        eyb, exb = BINS["diver_dy"], BINS["diver_dx"]
        for role in ["LEFT_related", "RIGHT_related", "UP_related", "DOWN_related", "FIRE_related"]:
            G, Cn = heatmap_grid(ddx[mc], ddy[mc], exb, eyb, role_prob(P, role)[mc])
            grids[f"C__P_{role}"] = G; grids[f"C__P_{role}__count"] = Cn
        out["diver_heatmap"] = "ok"
    else:
        out["diver_heatmap"] = f"INSUFFICIENT diver support ({int(mc.sum())} rows < 50)"
    # Heatmap D: missile active/inactive x enemy near/far
    near = np.isfinite(edx) & (np.abs(edx) + np.abs(edy) < 40)
    act = pmiss > 0.5
    Dmat = {}
    for ai, aname in [(act, "missile_active"), (~act, "missile_inactive")]:
        for ni, nname in [(near, "enemy_near"), (~near & valid_obj, "enemy_far")]:
            m = valid_obj & ai & ni
            if m.sum() > 0:
                Dmat[f"{aname}__{nname}"] = {
                    "P_FIRE_related": float(role_prob(P, "FIRE_related")[m].mean()),
                    "P_non_fire": float((1 - role_prob(P, "FIRE_related")[m]).mean()),
                    "mean_fire_logit_margin": float((L[m][:, ROLE["FIRE_related"]].max(1)
                                                     - np.delete(L[m], ROLE["FIRE_related"], 1).max(1)).mean()),
                    "count": int(m.sum())}
    out["heatmap_D_missile_fire"] = Dmat
    # Heatmap E: strata x 18 actions (mean prob, freq, mean logit)
    strata = define_strata(R, ci)
    E_prob = {}; E_freq = {}; E_logit = {}
    for sname, mask in strata.items():
        if mask.sum() == 0:
            continue
        E_prob[sname] = P[mask].mean(0).tolist()
        h = np.bincount(sampled[mask], minlength=18).astype(float); E_freq[sname] = (h / h.sum()).tolist()
        E_logit[sname] = L[mask].mean(0).tolist()
    out["heatmap_E"] = {"strata": list(E_prob.keys()), "mean_prob": E_prob,
                        "sampled_freq": E_freq, "mean_logit": E_logit}
    return out, grids


def define_strata(R, ci):
    oxy = col(R, ci, "oxygen"); edx = col(R, ci, "nearest_enemy_dx"); edy = col(R, ci, "nearest_enemy_dy")
    py = col(R, ci, "player_y"); dc = col(R, ci, "diver_count"); pm = col(R, ci, "player_missile_count")
    v = np.isfinite(py)
    near = np.isfinite(edx) & (np.abs(edx) + np.abs(edy) < 40)
    return {
        "oxygen_low": v & np.isfinite(oxy) & (oxy < 22),
        "oxygen_mid": v & np.isfinite(oxy) & (oxy >= 22) & (oxy < 44),
        "oxygen_high": v & np.isfinite(oxy) & (oxy >= 44),
        "enemy_near": v & near, "enemy_far": v & ~near,
        "diver_present": v & (dc > 0), "diver_absent": v & (dc == 0),
        "near_surface": v & (py < 73), "not_near_surface": v & (py >= 73),
        "missile_available": v & (pm < 0.5), "missile_active": v & (pm >= 0.5)}


def p2_state_conditioning(mode):
    """Effect of state vars on teacher probabilities, with episode-bootstrap CIs."""
    R, ci = load(mode)
    P = probs_mat(R, ci); ep = col(R, ci, "episode").astype(int)
    edx = col(R, ci, "nearest_enemy_dx"); oxy = col(R, ci, "oxygen")
    pm = col(R, ci, "player_missile_count"); py = col(R, ci, "player_y")
    v = np.isfinite(py)
    eps = np.unique(ep[v])

    def boot_diff(maskA, maskB, role, B=1000, seed=1):
        rng = np.random.RandomState(seed)
        diffs = []
        for _ in range(B):
            es = rng.choice(eps, size=len(eps), replace=True)
            inA = np.isin(ep, es) & maskA; inB = np.isin(ep, es) & maskB
            if inA.sum() < 5 or inB.sum() < 5:
                continue
            diffs.append(role_prob(P, role)[inA].mean() - role_prob(P, role)[inB].mean())
        if not diffs:
            return None
        return {"mean_diff": float(np.mean(diffs)),
                "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                "nA": int(maskA.sum()), "nB": int(maskB.sum())}

    eff = {}
    # enemy on left vs right -> LEFT vs RIGHT related
    L = v & np.isfinite(edx) & (edx < -15); Rr = v & np.isfinite(edx) & (edx > 15)
    eff["enemy_left_minus_right__P_LEFT_related"] = boot_diff(L, Rr, "LEFT_related")
    eff["enemy_left_minus_right__P_RIGHT_related"] = boot_diff(L, Rr, "RIGHT_related")
    # missile available vs active -> FIRE related
    ma = v & (pm < 0.5); mi = v & (pm >= 0.5)
    eff["missile_available_minus_active__P_FIRE"] = boot_diff(ma, mi, "FIRE_related")
    # low vs high oxygen -> UP related (surfacing)
    lo = v & np.isfinite(oxy) & (oxy < 22); hi = v & np.isfinite(oxy) & (oxy >= 44)
    eff["oxygen_low_minus_high__P_UP_related"] = boot_diff(lo, hi, "UP_related")
    # enemy near vs far -> FIRE related
    near = v & np.isfinite(edx) & (np.abs(edx) + np.abs(col(R, ci, "nearest_enemy_dy")) < 40)
    far = v & ~near
    eff["enemy_near_minus_far__P_FIRE"] = boot_diff(near, far, "FIRE_related")
    return eff


def closed_loop_all():
    cl = {}
    for mode in ["N-Greedy", "N-Sampled", "O-Greedy", "O-Sampled"]:
        try:
            R, ci = load(mode)
        except Exception:
            continue
        ep = col(R, ci, "episode").astype(int); sampled = col(R, ci, "sampled_action").astype(int)
        rew = col(R, ci, "reward"); ll = col(R, ci, "life_loss")
        h = np.bincount(sampled, minlength=18).astype(float); p = h / h.sum()
        ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        rets = [float(rew[ep == e].sum()) for e in np.unique(ep)]
        lens = [int((ep == e).sum()) for e in np.unique(ep)]
        d = {"return_mean": float(np.mean(rets)), "ep_len_mean": float(np.mean(lens)),
             "action_entropy": ent, "n_distinct_actions": int((h > 0).sum()),
             "score_rate_per_1000": float(1000 * sum(rets) / max(len(R), 1))}
        oxy = col(R, ci, "oxygen"); py = col(R, ci, "player_y")
        if np.isfinite(py).sum() > 10:
            v = np.isfinite(oxy)
            d["frac_time_low_oxygen(<16)"] = float((oxy[v] < 16).mean())
            d["P_UP_or_UPFIRE_in_low_oxygen"] = (
                float(role_prob(probs_mat(R, ci), "UP_related")[v & (oxy < 16)].mean())
                if (v & (oxy < 16)).sum() > 0 else None)
            # surfacing events: oxygen jumps up by >5
            surf = 0; refill = 0
            for e in np.unique(ep):
                ox = oxy[ep == e]; ox = ox[np.isfinite(ox)]
                if len(ox) > 1:
                    jumps = np.where(np.diff(ox) > 5)[0]
                    surf += len(jumps); refill += int((ox > 55).any() and (ox < 20).any())
            d["surfacing_events"] = int(surf); d["episodes_with_refill_cycle"] = int(refill)
            # oxygen at first life loss
            oxll = []
            for e in np.unique(ep):
                mm = (ep == e) & (ll > 0.5)
                if mm.sum() > 0:
                    ix = np.where(mm)[0][0]; oxll.append(float(oxy[ix]) if np.isfinite(oxy[ix]) else None)
            d["oxygen_at_life_loss"] = [x for x in oxll if x is not None]
        cl[mode] = d
    return cl


def native_ported_marginal():
    """Action-marginal divergence native vs ported (object-stratum native infeasible)."""
    def marg(mode):
        R, ci = load(mode)
        h = np.bincount(col(R, ci, "sampled_action").astype(int), minlength=18).astype(float)
        return h / h.sum()
    res = {}
    for npair in [("N-Sampled", "O-Sampled"), ("N-Greedy", "O-Greedy")]:
        try:
            pn = marg(npair[0]); pp = marg(npair[1])
        except Exception:
            continue
        tv = float(0.5 * np.abs(pn - pp).sum())
        m = 0.5 * (pn + pp)
        def kl(a, b): a = np.clip(a, 1e-12, 1); b = np.clip(b, 1e-12, 1); return float((a * np.log(a / b)).sum())
        js = 0.5 * kl(pn, m) + 0.5 * kl(pp, m)
        res[f"{npair[0]}_vs_{npair[1]}"] = {
            "total_variation": tv, "jensen_shannon": float(js),
            "argmax_native": int(np.argmax(pn)), "argmax_ported": int(np.argmax(pp)),
            "argmax_disagree": bool(np.argmax(pn) != np.argmax(pp)),
            "entropy_native": float(-(pn[pn > 0] * np.log(pn[pn > 0])).sum()),
            "entropy_ported": float(-(pp[pp > 0] * np.log(pp[pp > 0])).sum()),
            "native_dist": pn.tolist(), "ported_dist": pp.tolist()}
    res["_note"] = ("Object-stratum (matched abstract-state) native distributions are INFEASIBLE: "
                    "native EnvPool exposes no RAM/objects. Tensor parity is exact and preprocessing ops "
                    "are identical (EnvPool config: use_inter_area_resize + gray_scale), so the policy "
                    "FUNCTION is identical native vs ported; behavioral divergence is attributable to "
                    "visited-state distribution (characterized via O-mode survival/occupancy).")
    return res


def main():
    os.makedirs(f"{BASE}/policy_heatmaps", exist_ok=True)
    grids_all = {}; hm_meta = {}
    for mode in ["O-Greedy", "O-Sampled", "N-Greedy", "N-Sampled"]:
        try:
            meta, grids = heatmaps_for_mode(mode)
        except Exception as e:
            meta, grids = {"error": str(e)}, None
        hm_meta[mode] = meta
        if grids:
            for k, v in grids.items():
                grids_all[f"{mode}__{k}"] = v
    np.savez_compressed(f"{BASE}/policy_heatmaps/raw_counts.npz",
                        **{k: v for k, v in grids_all.items() if k.endswith("__count")})
    np.savez_compressed(f"{BASE}/policy_heatmaps/normalized_probabilities.npz",
                        **{k: v for k, v in grids_all.items() if not k.endswith("__count")})
    json.dump({"bins": BINS, "roles": ROLE, "semantic": SEMANTIC, "ale_meanings": MEAN,
               "heatmap_meta": hm_meta},
              open(f"{BASE}/policy_heatmaps/bin_definitions.json", "w"), indent=2)
    # P2
    p2 = {"O-Sampled": p2_state_conditioning("O-Sampled"),
          "O-Greedy": p2_state_conditioning("O-Greedy")}
    json.dump({"state_conditioning_effects": p2,
               "method": "P(role) difference between state strata, bootstrap 95% CI over EPISODES"},
              open(f"{BASE}/policy_heatmaps/native_ported_divergence.json", "w"), indent=2)
    # closed-loop + native/ported
    json.dump(closed_loop_all(), open(f"{BASE}/closed_loop/occupancy_metrics.json", "w"), indent=2)
    json.dump(native_ported_marginal(), open(f"{BASE}/closed_loop/native_ported_marginal.json", "w"), indent=2)
    # surfacing metrics extracted from closed-loop
    cl = closed_loop_all()
    surf = {m: {k: cl[m].get(k) for k in ["surfacing_events", "episodes_with_refill_cycle",
                                          "oxygen_at_life_loss", "frac_time_low_oxygen(<16)",
                                          "P_UP_or_UPFIRE_in_low_oxygen"]}
            for m in cl if "surfacing_events" in cl[m]}
    json.dump(surf, open(f"{BASE}/closed_loop/surfacing_metrics.json", "w"), indent=2)
    # episode_metrics.csv
    import csv
    with open(f"{BASE}/closed_loop/episode_metrics.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["mode", "return_mean", "ep_len_mean", "action_entropy",
                                       "n_distinct_actions", "score_rate_per_1000"])
        for m, d in cl.items():
            w.writerow([m, d["return_mean"], d["ep_len_mean"], d["action_entropy"],
                        d["n_distinct_actions"], d["score_rate_per_1000"]])
    # P2 quick verdict
    eff = p2["O-Sampled"]
    sig = sum(1 for k, v in eff.items() if v and (v["ci95"][0] > 0 or v["ci95"][1] < 0))
    print(f"heatmaps+P2 done. P2 significant effects (CI excludes 0): {sig}/{len(eff)}")
    for k, v in eff.items():
        if v:
            print(f"  {k}: {v['mean_diff']:+.3f} CI[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]")


if __name__ == "__main__":
    main()
