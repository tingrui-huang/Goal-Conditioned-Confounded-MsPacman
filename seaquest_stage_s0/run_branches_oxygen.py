"""Sections 16 + 17 + 18: tiny observational rollout, forced-first-action branch
scan (Gate C), and observational oxygen screening (Gate D). ocatari image.

NO critic, NO masking, NO causal objective. Emulator branch effects + observational
associations only. Uses the SELECTED teacher (A) in the OCAtari port env.

Key disciplines:
  * branches use a SINGLE pre-generated Gumbel-noise schedule reused across all
    forced-first-action branches at every continuation step;
  * same-action duplicate branches give the implementation-noise baseline (0 in a
    deterministic env);
  * censoring flags recorded; no branch silently crosses a reset/terminal;
  * oxygen screening uses EPISODE-LEVEL splits and never claims causation.
"""
import sys, os, json, argparse, csv
_REAL = sys.stdout
sys.stdout = open(os.devnull, "w")
def log(*a):
    print(*a, file=_REAL, flush=True)

import numpy as np
import gymnasium, ale_py  # noqa
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from teacher_port import SeaquestPort, ALE_MEANINGS
from teacher_adapter import CleanRLSeaquestTeacher

HORIZONS = [1, 2, 4, 8, 12, 16, 24, 32]
MOVE_ACTIONS = set(range(2, 18))           # everything except NOOP/FIRE involves a direction
FIRE_ACTIONS = {1, 10, 11, 12, 13, 14, 15, 16, 17}

VISIBLE_KEYS = ["player_x", "player_y", "n_shark", "n_submarine", "n_diver",
                "n_collected_diver", "n_player_missile", "n_enemy_missile",
                "score", "lives", "enemy_min_dx", "enemy_min_dy", "nearest_enemy_dist"]
NOPLAYER_KEYS = ["n_shark", "n_submarine", "n_diver", "n_collected_diver",
                 "n_player_missile", "n_enemy_missile", "score", "lives",
                 "enemy_centroid_x", "enemy_centroid_y"]
WORLDONLY_KEYS = ["n_shark", "n_submarine", "n_diver", "n_collected_diver",
                  "n_enemy_missile", "score", "lives",
                  "enemy_centroid_x", "enemy_centroid_y"]  # excludes player missile


def feat_vec(f, keys):
    out = []
    for k in keys:
        v = f.get(k)
        out.append(np.nan if v is None else float(v))
    return np.array(out, dtype=np.float64)


def enrich(f):
    """Add derived enemy-distance / centroid fields to a features dict."""
    ex, ey = f.get("enemy_xs", []), f.get("enemy_ys", [])
    ex = [v for v in ex if v is not None]; ey = [v for v in ey if v is not None]
    px, py = f.get("player_x"), f.get("player_y")
    if ex:
        f["enemy_centroid_x"] = float(np.mean(ex)); f["enemy_centroid_y"] = float(np.mean(ey))
        if px is not None and py is not None:
            d = [abs(x - px) + abs(y - py) for x, y in zip(ex, ey)]
            j = int(np.argmin(d))
            f["enemy_min_dx"] = float(ex[j] - px); f["enemy_min_dy"] = float(ey[j] - py)
            f["nearest_enemy_dist"] = float(min(d))
    f.setdefault("enemy_centroid_x", None); f.setdefault("enemy_centroid_y", None)
    f.setdefault("enemy_min_dx", None); f.setdefault("enemy_min_dy", None)
    f.setdefault("nearest_enemy_dist", None)
    return f


# ---------------------------------------------------------------- rollout
def collect_rollout(teacher, n_episodes, seed, noop_max=30, max_steps=1200, snap_stride=3):
    """Tiny observational rollout. Returns episodes list and in-memory anchor snapshots."""
    import jax, jax.numpy as jnp
    port = SeaquestPort(sticky=0.0, full_action_space=True, seed=seed)
    key = jax.random.PRNGKey(seed)
    rng = np.random.RandomState(seed + 5)
    episodes = []
    anchor_pool = []  # (epi, t, snapshot, features, oxygen)
    for ep in range(n_episodes):
        port.reset(seed=seed + ep, noop_max=noop_max, rng=rng)
        start_lives = port.features()["lives"]
        steps = []
        t = 0
        while True:
            obs = port.teacher_obs()
            logits, _ = teacher._forward(teacher.network_params, teacher.actor_params,
                                         teacher.critic_params, obs[None])
            logits = np.asarray(logits)[0]
            key, sub = jax.random.split(key)
            u = np.asarray(jax.random.uniform(sub, shape=(teacher.action_dim,)))
            gnoise = -np.log(-np.log(np.clip(u, 1e-12, 1 - 1e-12)))
            a = int(np.argmax(logits + gnoise))
            pre_f = enrich(port.features())
            if t % snap_stride == 0:
                snap = port.snapshot()
                anchor_pool.append({"ep": ep, "t": t, "snap": snap,
                                    "features": pre_f, "oxygen": pre_f.get("oxygen")})
            rec = port.agent_step(a)
            post_f = enrich(port.features())
            lives = post_f["lives"]
            life_lost = (start_lives is not None and lives is not None and lives < start_lives)
            steps.append({"t": t, "action": a, "logits": logits.tolist(),
                          "reward": rec["reward"], "oxygen_pre": pre_f.get("oxygen"),
                          "features_pre": pre_f, "features_post": post_f,
                          "terminated": rec["terminated"], "truncated": rec["truncated"],
                          "life_lost": bool(life_lost)})
            t += 1
            if rec["terminated"] or rec["truncated"] or life_lost or t >= max_steps:
                break
        episodes.append({"episode": ep, "steps": steps, "length": len(steps)})
        log(f"  [rollout] ep={ep} len={len(steps)}")
    return port, episodes, anchor_pool


# ------------------------------------------------------------ branch scan
def sample_anchors(anchor_pool, n_anchors, seed):
    rng = np.random.RandomState(seed + 7)
    valid = [a for a in anchor_pool if a["features"].get("player_x") is not None]
    # strata by oxygen
    oxy = np.array([a["oxygen"] if a["oxygen"] is not None else np.nan for a in valid])
    med = np.nanmedian(oxy)
    strata = {"low_oxygen": [a for a in valid if a["oxygen"] is not None and a["oxygen"] <= med],
              "high_oxygen": [a for a in valid if a["oxygen"] is not None and a["oxygen"] > med],
              "enemy_near": [a for a in valid if a["features"].get("nearest_enemy_dist") is not None
                             and a["features"]["nearest_enemy_dist"] < 40],
              "enemy_far": [a for a in valid if a["features"].get("nearest_enemy_dist") is not None
                            and a["features"]["nearest_enemy_dist"] >= 40]}
    chosen = {}
    per = max(1, n_anchors // 4)
    picked = []
    for name, pool in strata.items():
        if pool:
            idx = rng.choice(len(pool), size=min(per, len(pool)), replace=False)
            for i in idx:
                picked.append((name, pool[i]))
    # fill remainder uniformly
    while len(picked) < n_anchors and valid:
        picked.append(("uniform", valid[rng.randint(len(valid))]))
    return picked[:n_anchors]


def run_branches(port, teacher, anchors, seed, max_h=32):
    """For each anchor: force each first action, continue with frozen teacher using a
    SHARED pre-generated gumbel-noise schedule. Returns per-anchor branch outcomes."""
    import jax  # noqa
    rng = np.random.RandomState(seed + 13)
    # one gumbel-noise vector per continuation step, reused across ALL branches/anchors
    noise_schedule = -np.log(-np.log(np.clip(
        rng.uniform(size=(max_h + 1, teacher.action_dim)), 1e-12, 1 - 1e-12)))

    def continue_branch(first_action):
        """Roll out from the CURRENTLY restored state forcing first_action then teacher."""
        traj = []
        cum_r = 0.0
        for h in range(1, max_h + 1):
            if h == 1:
                a = first_action
            else:
                obs = port.teacher_obs()
                logits, _ = teacher._forward(teacher.network_params, teacher.actor_params,
                                             teacher.critic_params, obs[None])
                logits = np.asarray(logits)[0]
                a = int(np.argmax(logits + noise_schedule[h]))
            rec = port.agent_step(a)
            cum_r += rec["reward"]
            f = enrich(port.features())
            traj.append({"h": h, "action": a, "cum_reward": cum_r,
                         "terminated": rec["terminated"], "truncated": rec["truncated"],
                         "features": f})
            if rec["terminated"] or rec["truncated"]:
                # mark remaining horizons as censored (absorbing)
                for hh in range(h + 1, max_h + 1):
                    traj.append({"h": hh, "action": None, "cum_reward": cum_r,
                                 "terminated": True, "truncated": rec["truncated"],
                                 "features": f, "censored_absorbing": True})
                break
        return traj

    out = []
    for ai, (stratum, anc) in enumerate(anchors):
        branches = {}
        # all 18 first actions + one duplicate of action 0 as baseline-noise check
        for a in range(teacher.action_dim):
            port.restore(anc["snap"])
            branches[a] = continue_branch(a)
        port.restore(anc["snap"])
        baseline_dup = continue_branch(0)  # duplicate of action 0
        out.append({"anchor_index": ai, "stratum": stratum, "ep": anc["ep"], "t": anc["t"],
                    "anchor_features": anc["features"], "anchor_oxygen": anc["oxygen"],
                    "branches": branches, "baseline_dup_action0": baseline_dup})
        if ai % 10 == 0:
            log(f"  [branch] anchor {ai}/{len(anchors)} stratum={stratum}")
    return out, noise_schedule


def at_h(traj, h):
    for r in traj:
        if r["h"] == h:
            return r
    return None


def view_vec(f, view, include_oxygen):
    keys = {"full": VISIBLE_KEYS, "noplayer": NOPLAYER_KEYS, "worldonly": WORLDONLY_KEYS}[view]
    v = feat_vec(f, keys)
    if include_oxygen and "oxygen" in f and f["oxygen"] is not None:
        v = np.concatenate([v, [float(f["oxygen"])]])
    return v


def branch_metrics(branch_out, teacher_action_dim):
    """Compute divergence fractions per horizon and view, + component change rates."""
    results = {}
    censor_rows = []
    for view in ["full", "noplayer", "worldonly"]:
        per_h = {}
        for h in HORIZONS:
            anchors_diverge = 0
            anchors_valid = 0
            baseline_noise = 0
            comp_changes = []
            for anc in branch_out:
                vecs = []
                ok = True
                for a in range(teacher_action_dim):
                    r = at_h(anc["branches"][a], h)
                    if r is None:
                        ok = False; break
                    vecs.append(view_vec(r["features"], view, include_oxygen=False))
                if not ok:
                    continue
                anchors_valid += 1
                # baseline noise: action0 vs its duplicate at horizon h
                rb = at_h(anc["baseline_dup_action0"], h)
                r0 = at_h(anc["branches"][0], h)
                if rb is not None and r0 is not None:
                    if not _eq(view_vec(r0["features"], view, False),
                               view_vec(rb["features"], view, False)):
                        baseline_noise += 1
                # divergence between at least two first actions
                diverged = False
                base = vecs[0]
                for v in vecs[1:]:
                    if not _eq(base, v):
                        diverged = True
                        break
                if diverged:
                    anchors_diverge += 1
            per_h[h] = {"anchors_valid": anchors_valid,
                        "anchors_diverged": anchors_diverge,
                        "frac_diverged": anchors_diverge / max(anchors_valid, 1),
                        "baseline_noise_anchors": baseline_noise}
        results[view] = per_h
    return results


def _eq(a, b):
    return np.array_equal(np.nan_to_num(a, nan=-12345.0), np.nan_to_num(b, nan=-12345.0))


def component_change_rates(branch_out, teacher_action_dim):
    """Per-component fraction of anchors where the component differs across first actions."""
    comps = ["player_x", "player_y", "n_shark", "n_submarine", "n_diver",
             "n_collected_diver", "n_player_missile", "n_enemy_missile", "score",
             "lives", "enemy_centroid_x", "enemy_centroid_y", "cum_reward", "terminated"]
    out = {}
    for h in HORIZONS:
        comp_rates = {c: [0, 0] for c in comps}  # [changed, valid]
        for anc in branch_out:
            vals = {c: [] for c in comps}
            ok = True
            for a in range(teacher_action_dim):
                r = at_h(anc["branches"][a], h)
                if r is None:
                    ok = False; break
                f = enrich(dict(r["features"]))
                for c in comps:
                    if c == "cum_reward":
                        vals[c].append(r["cum_reward"])
                    elif c == "terminated":
                        vals[c].append(int(r["terminated"]))
                    else:
                        vals[c].append(f.get(c))
            if not ok:
                continue
            for c in comps:
                comp_rates[c][1] += 1
                xs = [(-99999 if v is None else v) for v in vals[c]]
                if len(set(xs)) > 1:
                    comp_rates[c][0] += 1
        out[h] = {c: {"changed": comp_rates[c][0], "valid": comp_rates[c][1],
                      "rate": comp_rates[c][0] / max(comp_rates[c][1], 1)} for c in comps}
    return out


# ------------------------------------------------------- oxygen screening
def oxygen_screen(episodes, seed):
    """Episode-level-split predictive screening. NEVER claims causation."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import log_loss, accuracy_score

    # episode-level split
    n = len(episodes)
    rng = np.random.RandomState(seed + 21)
    order = rng.permutation(n)
    n_test = max(1, n // 4); n_val = max(1, n // 4)
    test_ep = set(order[:n_test].tolist())
    val_ep = set(order[n_test:n_test + n_val].tolist())
    train_ep = set(order[n_test + n_val:].tolist())

    def build_action_xy(eps_idx):
        X, Xo, y = [], [], []
        for ei in eps_idx:
            for s in episodes[ei]["steps"]:
                f = s["features_pre"]
                vx = feat_vec(f, VISIBLE_KEYS)
                if np.any(np.isnan(vx)):
                    vx = np.nan_to_num(vx, nan=0.0)
                ox = s["oxygen_pre"]
                if ox is None:
                    continue
                X.append(vx); Xo.append(np.concatenate([vx, [float(ox)]])); y.append(int(s["action"]))
        return np.array(X), np.array(Xo), np.array(y)

    def fit_eval(Xtr, ytr, Xte, yte):
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=300, multi_class="multinomial")
        clf.fit(sc.transform(Xtr), ytr)
        proba = clf.predict_proba(sc.transform(Xte))
        labels = clf.classes_
        ll = log_loss(yte, proba, labels=labels)
        acc = accuracy_score(yte, clf.predict(sc.transform(Xte)))
        return ll, acc

    Xtr, Xotr, ytr = build_action_xy(train_ep)
    Xte, Xote, yte = build_action_xy(test_ep)
    action_assoc = {"split": {"train": sorted(train_ep), "val": sorted(val_ep), "test": sorted(test_ep)}}
    try:
        # keep only classes present in train
        mask = np.isin(yte, np.unique(ytr))
        ll0, acc0 = fit_eval(Xtr, ytr, Xte[mask], yte[mask])
        ll1, acc1 = fit_eval(Xotr, ytr, Xote[mask], yte[mask])
        action_assoc.update({
            "model1_visible_only": {"test_logloss": ll0, "test_acc": acc0},
            "model2_visible_plus_oxygen": {"test_logloss": ll1, "test_acc": acc1},
            "oxygen_logloss_improvement": ll0 - ll1,
            "oxygen_acc_uplift": acc1 - acc0,
            "oxygen_helps_action_prediction": bool((ll0 - ll1) > 0)})
    except Exception as e:
        action_assoc["error"] = str(e)

    # outcome models: future cumulative reward (>0) and life-loss within H
    def build_outcome_xy(eps_idx, H, target):
        X, Xo, y = [], [], []
        for ei in eps_idx:
            steps = episodes[ei]["steps"]
            for i, s in enumerate(steps):
                ox = s["oxygen_pre"]
                if ox is None:
                    continue
                fut = steps[i:i + H]
                if target == "reward_pos":
                    yv = int(sum(z["reward"] for z in fut) > 0)
                elif target == "life_loss":
                    yv = int(any(z["life_lost"] for z in fut))
                else:
                    continue
                vx = np.nan_to_num(feat_vec(s["features_pre"], VISIBLE_KEYS), nan=0.0)
                vx = np.concatenate([vx, [float(s["action"])]])  # visible + action
                X.append(vx); Xo.append(np.concatenate([vx, [float(ox)]])); y.append(yv)
        return np.array(X), np.array(Xo), np.array(y)

    outcome_assoc = {"horizons": {}}
    for H in [4, 8, 16, 32]:
        rowH = {}
        for target in ["reward_pos", "life_loss"]:
            Xtr, Xotr, ytr = build_outcome_xy(train_ep, H, target)
            Xte, Xote, yte = build_outcome_xy(test_ep, H, target)
            if len(np.unique(ytr)) < 2 or len(yte) == 0 or len(np.unique(yte)) < 2:
                rowH[target] = {"skipped": "degenerate labels"}
                continue
            try:
                ll0, acc0 = fit_eval(Xtr, ytr, Xte, yte)
                ll1, acc1 = fit_eval(Xotr, ytr, Xote, yte)
                rowH[target] = {"model1_visible_action": {"test_logloss": ll0, "test_acc": acc0},
                                "model2_plus_oxygen": {"test_logloss": ll1, "test_acc": acc1},
                                "oxygen_logloss_improvement": ll0 - ll1,
                                "oxygen_helps": bool((ll0 - ll1) > 0)}
            except Exception as e:
                rowH[target] = {"error": str(e)}
        outcome_assoc["horizons"][H] = rowH
    return action_assoc, outcome_assoc, {"train": sorted(train_ep),
                                         "val": sorted(val_ep), "test": sorted(test_ep)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--src", required=True)
    ap.add_argument("--tag", default="A")
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--anchors", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--art", default="/work/artifacts/seaquest/stage_s0")
    args = ap.parse_args()

    teacher = CleanRLSeaquestTeacher(args.ckpt, args.src, mod_name=f"cleanrl_src_{args.tag}")
    log(f"[branches/oxygen tag {args.tag}] action_dim={teacher.action_dim}")

    port, episodes, anchor_pool = collect_rollout(teacher, args.episodes, args.seed)
    log(f"rollout done: {len(episodes)} episodes, {len(anchor_pool)} snapshot anchors")

    anchors = sample_anchors(anchor_pool, args.anchors, args.seed)
    log(f"sampled {len(anchors)} anchors")
    branch_out, noise_schedule = run_branches(port, teacher, anchors, args.seed)

    metrics = branch_metrics(branch_out, teacher.action_dim)
    comp = component_change_rates(branch_out, teacher.action_dim)

    # Gate C: a horizon with >=20% no-player divergence, exceeding baseline noise.
    gate_c = {"pass": False, "horizons": {}}
    for h in HORIZONS:
        npd = metrics["noplayer"][h]
        wod = metrics["worldonly"][h]
        base = npd["baseline_noise_anchors"]
        ok = npd["frac_diverged"] >= 0.20 and npd["anchors_diverged"] > base
        gate_c["horizons"][h] = {"noplayer_frac": npd["frac_diverged"],
                                 "worldonly_frac": wod["frac_diverged"],
                                 "baseline_noise": base, "passes": bool(ok)}
        if ok:
            gate_c["pass"] = True
    # which components carry the effect (max over horizons)
    carrier = {}
    for h in HORIZONS:
        for c, d in comp[h].items():
            carrier.setdefault(c, 0.0)
            carrier[c] = max(carrier[c], d["rate"])
    gate_c["component_max_change_rate"] = carrier

    # oxygen screening
    action_assoc, outcome_assoc, split = oxygen_screen(episodes, args.seed)
    gate_d = {
        "oxygen_helps_action": bool(action_assoc.get("oxygen_helps_action_prediction", False)),
        "oxygen_helps_outcome": any(
            outcome_assoc["horizons"][H].get(t, {}).get("oxygen_helps", False)
            for H in outcome_assoc["horizons"] for t in ["reward_pos", "life_loss"]),
    }
    gate_d["pass"] = bool(gate_d["oxygen_helps_action"] and gate_d["oxygen_helps_outcome"])

    # ---- save artifacts
    bdir = os.path.join(args.art, "branches"); odir = os.path.join(args.art, "oxygen_screen")
    os.makedirs(bdir, exist_ok=True); os.makedirs(odir, exist_ok=True)
    anchors_meta = [{"anchor_index": i, "stratum": st, "ep": a["ep"], "t": a["t"],
                     "oxygen": a["oxygen"], "features": a["features"]}
                    for i, (st, a) in enumerate(anchors)]
    json.dump({"n_anchors": len(anchors), "horizons": HORIZONS, "anchors": anchors_meta},
              open(os.path.join(bdir, "anchors.json"), "w"), indent=2)
    json.dump({"divergence_by_view_horizon": metrics, "gate_c": gate_c},
              open(os.path.join(bdir, "horizon_metrics.json"), "w"), indent=2)
    json.dump({"component_change_rates_by_horizon": comp},
              open(os.path.join(bdir, "component_metrics.json"), "w"), indent=2)
    # censoring csv
    with open(os.path.join(bdir, "censoring.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["anchor", "action", "horizon", "terminated", "truncated", "censored_absorbing"])
        for anc in branch_out:
            for a, traj in anc["branches"].items():
                for r in traj:
                    w.writerow([anc["anchor_index"], a, r["h"], int(r.get("terminated", False)),
                                int(r.get("truncated", False)), int(r.get("censored_absorbing", False))])
    # raw branch outputs (compact npz): per (anchor, action, horizon) cum_reward + terminal + key components
    rows = []
    for anc in branch_out:
        for a, traj in anc["branches"].items():
            for r in traj:
                f = r["features"]
                rows.append([anc["anchor_index"], a, r["h"], r["cum_reward"],
                             int(r.get("terminated", False)),
                             _g(f, "player_x"), _g(f, "player_y"), _g(f, "n_shark"),
                             _g(f, "n_submarine"), _g(f, "n_diver"), _g(f, "n_player_missile"),
                             _g(f, "n_enemy_missile"), _g(f, "score"), _g(f, "lives"),
                             _g(f, "oxygen")])
    np.savez_compressed(os.path.join(bdir, "raw_branch_outputs.npz"),
                        rows=np.array(rows, dtype=np.float64),
                        columns=np.array(["anchor", "action", "horizon", "cum_reward", "terminated",
                                          "player_x", "player_y", "n_shark", "n_submarine", "n_diver",
                                          "n_player_missile", "n_enemy_missile", "score", "lives", "oxygen"]))
    json.dump(action_assoc, open(os.path.join(odir, "action_association.json"), "w"), indent=2)
    json.dump(outcome_assoc, open(os.path.join(odir, "outcome_association.json"), "w"), indent=2)
    json.dump({"episode_level_split": split, "n_episodes": len(episodes),
               "episode_lengths": [e["length"] for e in episodes]},
              open(os.path.join(odir, "split_manifest.json"), "w"), indent=2)

    summary = {"gate_c": gate_c, "gate_d": gate_d,
               "rollout_episodes": len(episodes), "n_anchors": len(anchors)}
    json.dump(summary, open(os.path.join(bdir, "_branch_oxygen_summary.json"), "w"), indent=2)
    log("GATE C pass:", gate_c["pass"])
    log("GATE D pass:", gate_d["pass"], "(action:", gate_d["oxygen_helps_action"],
        "outcome:", gate_d["oxygen_helps_outcome"], ")")
    log(json.dumps({h: gate_c["horizons"][h] for h in HORIZONS}, indent=2))


def _g(f, k):
    v = f.get(k)
    return np.nan if v is None else float(v)


if __name__ == "__main__":
    main()
