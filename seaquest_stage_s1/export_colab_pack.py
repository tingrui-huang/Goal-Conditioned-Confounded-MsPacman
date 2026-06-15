"""Stage-S1 Part A: export a portable, frozen Colab data pack (runs in the ocatari
image with the FROZEN Stage-S0 teacher). NO training here.

Builds:
  * observational dataset: learner state vector at t, one-hot action, H=16 future
    goals (no_player / world_only views reused EXACTLY from corrected S0.5), episode
    ids, timesteps. Episode-level 70/15/15 split (seed 1601); normalization fit on
    TRAIN episodes only.
  * branch pack: per-anchor, per locally-supported-action H=16 future goals from
    forced-first-action branches (frozen teacher continuation), with valid mask,
    local support counts, semantic action categories.

All action selection routes through the frozen adapter (teacher.greedy_action /
teacher.sample_action + gumbel_from_uniform). Goal views == corrected S0.5.
"""
import sys, os, json, hashlib, zipfile, io
_REAL = sys.stdout
def prog(*a):
    print(*a, file=_REAL, flush=True)
import numpy as np
sys.path.insert(0, "/work/seaquest_stage_s05")
sys.path.insert(0, "/work/seaquest_stage_s0")
import common as C
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

ART = "/work/artifacts/seaquest/stage_s1"
S05CFG = json.load(open("/work/artifacts/seaquest/stage_s05/config/resolved_config.json"))
HORIZON = 16
OSAMPLED_SEED = S05CFG["seeds"]["O-Sampled"]   # 104
BRANCH_SEED = S05CFG["seeds"]["branches"]      # 105
SPLIT_SEED = 1601
TH = S05CFG["support_thresholds"]
ALE = S05CFG["ALE_MEANINGS"]

# ---- learner state schema (fixed dim, no NaN by construction; absence is EXPLICIT
#      via has_enemy / has_diver flags). Core fields (player/oxygen/score/lives) must
#      be present or the row is excluded (never silently imputed).
STATE_SCHEMA = [
    "player_x", "player_y", "player_vx", "player_vy", "oxygen", "distance_to_surface",
    "enemy_count", "has_enemy", "nearest_enemy_dx", "nearest_enemy_dy",
    "enemy_centroid_x", "enemy_centroid_y",
    "diver_count", "has_diver", "nearest_diver_dx", "nearest_diver_dy",
    "n_shark", "n_submarine", "n_player_missile", "n_enemy_missile", "score", "lives"]
# goal views: EXACTLY the corrected S0.5 branch views
NP_KEYS = ["n_shark", "n_submarine", "n_diver", "n_player_missile", "n_enemy_missile",
           "score", "lives", "cum_reward", "terminated"]
WO_KEYS = ["n_shark", "n_submarine", "n_diver", "n_enemy_missile",
           "score", "lives", "cum_reward", "terminated"]
CORE = ["player_x", "player_y", "oxygen", "score", "lives"]


def comp(f):
    """Full per-step component dict needed for state + goal views."""
    px, py = f.get("player_x"), f.get("player_y")
    ex = [v for v in (f.get("enemy_xs") or []) if v is not None]
    ey = [v for v in (f.get("enemy_ys") or []) if v is not None]
    dx = [v for v in (f.get("diver_xs") or []) if v is not None]
    dy = [v for v in (f.get("diver_ys") or []) if v is not None]
    d = {k: f.get(k) for k in ["player_x", "player_y", "player_vx", "player_vy", "oxygen",
                               "distance_to_surface", "enemy_count", "enemy_centroid_x",
                               "enemy_centroid_y", "nearest_enemy_dx", "nearest_enemy_dy",
                               "nearest_diver_dx", "nearest_diver_dy", "diver_count",
                               "player_missile_count", "enemy_missile_count", "score", "lives"]}
    d["n_shark"] = float(f.get("n_shark", 0) or 0)
    d["n_submarine"] = float(f.get("n_submarine", 0) or 0)
    d["n_diver"] = float(f.get("n_diver", 0) or 0)
    d["n_player_missile"] = float(f.get("n_player_missile", 0) or 0)
    d["n_enemy_missile"] = float(f.get("n_enemy_missile", 0) or 0)
    d["has_enemy"] = 1.0 if ex else 0.0
    d["has_diver"] = 1.0 if dx else 0.0
    return d


def state_vec(d):
    out = []
    for k in STATE_SCHEMA:
        v = d.get(k)
        if k in ("nearest_enemy_dx", "nearest_enemy_dy", "enemy_centroid_x", "enemy_centroid_y") and not d.get("has_enemy"):
            v = 0.0
        if k in ("nearest_diver_dx", "nearest_diver_dy") and not d.get("has_diver"):
            v = 0.0
        if k in ("player_vx", "player_vy") and v is None:
            v = 0.0  # velocity undefined at episode start -> 0 (explicit, documented)
        out.append(np.nan if v is None else float(v))
    return np.array(out, dtype=np.float64)


def goal_vec(d, keys, cum, term):
    out = []
    for k in keys:
        if k == "cum_reward":
            out.append(float(cum))
        elif k == "terminated":
            out.append(float(term))
        else:
            out.append(float(d.get(k, 0.0) or 0.0))
    return np.array(out, dtype=np.float64)


def core_ok(d):
    return all(d.get(k) is not None for k in CORE)


# --------------------------------------------------------- observational rollout
def regen_observational(teacher):
    from teacher_port import SeaquestPort
    port = SeaquestPort(sticky=0.0, full_action_space=True, seed=OSAMPLED_SEED)
    rng = np.random.RandomState(OSAMPLED_SEED + 11)
    noise_rng = np.random.RandomState(OSAMPLED_SEED + 777)
    episodes = []
    ep = 0; n = 0
    while ep < 20 and n < 12000:
        port.reset(seed=OSAMPLED_SEED + ep, noop_max=30, rng=rng)
        start_lives = port.features()["lives"]; prev = None; steps = []
        while True:
            obs = port.teacher_obs()
            noise = teacher.gumbel_from_uniform(noise_rng.uniform(size=(18,)))
            a = int(teacher.sample_action(obs, noise, temperature=1.0)[0])  # FROZEN
            f = C.enrich_features(port.features(), prev)
            d = comp(f)
            rec = port.agent_step(a)
            lv = port.features()["lives"]
            term = bool(rec["terminated"] or rec["truncated"])
            ll = (start_lives is not None and lv is not None and lv < start_lives)
            steps.append({"action": a, "comp": d, "reward": float(rec["reward"]),
                          "terminated": term, "life_lost": bool(ll)})
            prev = f; n += 1
            if term or ll or len(steps) >= 4000 or n >= 12000:
                break
        episodes.append(steps); ep += 1
        prog(f"  [obs] ep={ep-1} len={len(steps)}")
    return episodes


def build_obs_dataset(episodes):
    """Transitions with a valid H=16 future, no boundary cross, no missing core."""
    S, A, GNP, GWO, EID, TS = [], [], [], [], [], []
    for ei, steps in enumerate(episodes):
        L = len(steps)
        for t in range(L):
            if t + HORIZON >= L:
                continue  # no valid H=16 future inside this episode
            dt = steps[t]["comp"]; dfut = steps[t + HORIZON]["comp"]
            # episode-internal terminal within (t, t+H] would have ended the episode;
            # since episodes end at terminal/life-loss, t+H<L already guarantees no
            # crossing. Require core present at t and t+H.
            if not core_ok(dt) or not core_ok(dfut):
                continue
            sv = state_vec(dt)
            if np.any(np.isnan(sv)):
                continue
            cum = sum(steps[k]["reward"] for k in range(t, t + HORIZON))
            term = float(any(steps[k]["terminated"] for k in range(t, t + HORIZON)))
            S.append(sv); A.append(steps[t]["action"])
            GNP.append(goal_vec(dfut, NP_KEYS, cum, term))
            GWO.append(goal_vec(dfut, WO_KEYS, cum, term))
            EID.append(ei); TS.append(t)
    return (np.array(S), np.array(A, dtype=np.int64), np.array(GNP), np.array(GWO),
            np.array(EID, dtype=np.int64), np.array(TS, dtype=np.int64))


# --------------------------------------------------------------- branch pack
def build_branch_pack(teacher):
    from teacher_port import SeaquestPort
    # support reference from corrected O-Sampled rows
    d = np.load("/work/artifacts/seaquest/stage_s05/closed_loop/rows_O-Sampled.npz", allow_pickle=True)
    cols = list(d["columns"]); R = d["rows"]; ci = {c: i for i, c in enumerate(cols)}
    SF = C.STATE_FEATURES
    X = R[:, [ci[f] for f in SF]].astype(np.float64)
    with np.errstate(all="ignore"):
        med = np.nan_to_num(np.nanmedian(X, axis=0), nan=0.0)
    Xi = np.nan_to_num(np.where(np.isnan(X), med, X), nan=0.0)
    scaler = StandardScaler().fit(Xi)
    nn = NearestNeighbors(n_neighbors=50).fit(scaler.transform(Xi))
    ds_actions = R[:, ci["sampled_action"]].astype(int)

    def supported(feat):
        x = np.array([np.nan if feat.get(k) is None else float(feat.get(k)) for k in SF])
        x = np.nan_to_num(np.where(np.isnan(x), med, x), nan=0.0)
        _, idx = nn.kneighbors(scaler.transform(x[None]))
        h = np.bincount(ds_actions[idx[0]], minlength=18); prop = h / h.sum()
        supp = [a for a in range(18) if prop[a] >= TH["local_propensity_min"]
                and h[a] >= TH["min_neighbor_occurrences"]]
        return supp, h

    port = SeaquestPort(sticky=0.0, full_action_space=True, seed=BRANCH_SEED)
    rng = np.random.RandomState(BRANCH_SEED + 11)
    noise_rng = np.random.RandomState(BRANCH_SEED + 777)
    pool = []
    for ep in range(6):
        port.reset(seed=BRANCH_SEED + ep, noop_max=30, rng=rng)
        start_lives = port.features()["lives"]; prev = None; t = 0
        while True:
            obs = port.teacher_obs()
            noise = teacher.gumbel_from_uniform(noise_rng.uniform(size=(18,)))
            a = int(teacher.sample_action(obs, noise, temperature=1.0)[0])
            f = C.enrich_features(port.features(), prev)
            if t % 4 == 0:
                pool.append({"snap": port.snapshot(), "feat": f})
            rec = port.agent_step(a); prev = f; t += 1
            lv = port.features()["lives"]
            if rec["terminated"] or rec["truncated"] or (start_lives and lv and lv < start_lives) or t >= 600:
                break
    selidx = rng.choice(len(pool), size=min(40, len(pool)), replace=False)
    anchors = [pool[i] for i in selidx]
    u_sched = noise_rng.uniform(size=(HORIZON + 1, 18))
    noise_sched = teacher.gumbel_from_uniform(u_sched)

    def continue_branch(first_a):
        cum = 0.0; term = False; d_fut = None
        for h in range(1, HORIZON + 1):
            a = first_a if h == 1 else int(teacher.sample_action(port.teacher_obs(), noise_sched[h], 1.0)[0])
            rec = port.agent_step(a); cum += rec["reward"]
            term = term or rec["terminated"] or rec["truncated"]
            d_fut = comp(C.enrich_features(port.features(), None))
            if term:
                break
        gnp = goal_vec(d_fut, NP_KEYS, cum, term); gwo = goal_vec(d_fut, WO_KEYS, cum, term)
        return gnp, gwo, term, (h >= HORIZON)

    n_anchor = len(anchors)
    anchor_states = np.zeros((n_anchor, len(STATE_SCHEMA)))
    fut_np = np.full((n_anchor, 18, len(NP_KEYS)), np.nan)
    fut_wo = np.full((n_anchor, 18, len(WO_KEYS)), np.nan)
    valid = np.zeros((n_anchor, 18), dtype=np.int64)
    supp_cnt = np.zeros((n_anchor, 18), dtype=np.int64)
    for ai, anc in enumerate(anchors):
        anchor_states[ai] = state_vec(comp(anc["feat"]))
        supp, hist = supported(anc["feat"])
        supp_cnt[ai] = hist
        for a in supp:
            port.restore(anc["snap"])
            gnp, gwo, term, reached = continue_branch(a)
            if reached:  # only keep branches that reached H=16 uncensored
                fut_np[ai, a] = gnp; fut_wo[ai, a] = gwo; valid[ai, a] = 1
        if ai % 10 == 0:
            prog(f"  [branch] anchor {ai}/{n_anchor} supported={len(supp)}")
    sem_cat = np.array([C.ACTION_TO_CAT[a] for a in range(18)], dtype=np.int64)
    return anchor_states, fut_np, fut_wo, valid, supp_cnt, sem_cat


# ------------------------------------------------------------------ pack
def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    teacher = C.load_teacher("A")
    prog("[export] frozen teacher loaded (adapter)")
    eps = regen_observational(teacher)
    S, A, GNP, GWO, EID, TS = build_obs_dataset(eps)
    prog(f"[export] obs dataset: {S.shape[0]} transitions, state_dim={S.shape[1]}, "
         f"np_dim={GNP.shape[1]}, wo_dim={GWO.shape[1]}, episodes={len(set(EID.tolist()))}")
    anchor_states, fut_np, fut_wo, valid, supp_cnt, sem_cat = build_branch_pack(teacher)
    prog(f"[export] branch pack: {anchor_states.shape[0]} anchors, valid branches={int(valid.sum())}")

    # episode-level split (seed 1601) + train-only normalization
    ep_ids = np.unique(EID)
    rng = np.random.RandomState(SPLIT_SEED)
    order = rng.permutation(ep_ids)
    ntr = int(round(0.70 * len(ep_ids))); nva = int(round(0.15 * len(ep_ids)))
    tr_e = sorted(int(x) for x in order[:ntr])
    va_e = sorted(int(x) for x in order[ntr:ntr + nva])
    te_e = sorted(int(x) for x in order[ntr + nva:])
    tr_mask = np.isin(EID, tr_e)
    # std floor: near-constant features map to ~0 instead of exploding (avoid /1e-6).
    def floored_std(x):
        s = x.std(0)
        return np.where(s < 1e-2, 1.0, s)
    smean = S[tr_mask].mean(0); sstd = floored_std(S[tr_mask])
    gnp_mean = GNP[tr_mask].mean(0); gnp_std = floored_std(GNP[tr_mask])
    gwo_mean = GWO[tr_mask].mean(0); gwo_std = floored_std(GWO[tr_mask])

    os.makedirs(ART, exist_ok=True)
    files = {}

    def add(name, arr):
        buf = io.BytesIO(); np.save(buf, arr); files[name] = buf.getvalue()

    add("observational/states.npy", S.astype(np.float32))
    add("observational/actions.npy", A.astype(np.int64))
    add("observational/goals_no_player_H16.npy", GNP.astype(np.float32))
    add("observational/goals_world_only_H16.npy", GWO.astype(np.float32))
    add("observational/episode_ids.npy", EID.astype(np.int64))
    add("observational/timesteps.npy", TS.astype(np.int64))
    add("branches/anchor_states.npy", anchor_states.astype(np.float32))
    add("branches/candidate_actions.npy", np.arange(18, dtype=np.int64))
    add("branches/future_no_player_H16.npy", fut_np.astype(np.float32))
    add("branches/future_world_only_H16.npy", fut_wo.astype(np.float32))
    add("branches/valid_mask.npy", valid.astype(np.int64))
    add("branches/local_support_counts.npy", supp_cnt.astype(np.int64))
    add("branches/semantic_action_categories.npy", sem_cat.astype(np.int64))

    schema = {"state_schema": STATE_SCHEMA, "state_dim": len(STATE_SCHEMA),
              "no_player_keys": NP_KEYS, "world_only_keys": WO_KEYS,
              "np_dim": len(NP_KEYS), "wo_dim": len(WO_KEYS),
              "action_dim": 18, "action_meanings": ALE,
              "absence_encoding": ("has_enemy/has_diver flags; absent relative coords set to 0 (explicit, "
                                   "not imputed). player_vx/vy = 0 at episode start (velocity undefined -> 0). "
                                   "Rows with missing CORE fields are EXCLUDED, never imputed."),
              "core_required_present": CORE}
    files["feature_schema.json"] = json.dumps(schema, indent=2).encode()
    norm = {"state_mean": smean.tolist(), "state_std": sstd.tolist(),
            "no_player_mean": gnp_mean.tolist(), "no_player_std": gnp_std.tolist(),
            "world_only_mean": gwo_mean.tolist(), "world_only_std": gwo_std.tolist(),
            "fit_on": "train_episodes_only", "split_seed": SPLIT_SEED}
    files["normalization.json"] = json.dumps(norm, indent=2).encode()
    split = {"split_seed": SPLIT_SEED, "fractions": {"train": 0.70, "val": 0.15, "test": 0.15},
             "train_episode_ids": tr_e, "val_episode_ids": va_e, "test_episode_ids": te_e,
             "n_train_ep": len(tr_e), "n_val_ep": len(va_e), "n_test_ep": len(te_e)}
    files["split_manifest.json"] = json.dumps(split, indent=2).encode()

    def git(*a):
        import subprocess
        try:
            return subprocess.check_output(["git", "-C", "/work", *a], text=True).strip()
        except Exception:
            return None
    s05hashes = {}
    for rel in ["config/resolved_config.json", "closed_loop/rows_O-Sampled.npz",
                "branches/heatmap_integrity.json", "audit_report.json"]:
        p = f"/work/artifacts/seaquest/stage_s05/{rel}"
        if os.path.exists(p):
            s05hashes[rel] = sha(open(p, "rb").read())
    import datetime
    manifest = {
        "stage": "seaquest_stage_s1", "source_git_commit": git("rev-parse", "HEAD"),
        "s05_artifact_sha256": s05hashes, "horizon": HORIZON,
        "collection_mode": "corrected O-Sampled (frozen adapter)", "primary_seed": 0,
        "feature_names": STATE_SCHEMA, "state_dim": len(STATE_SCHEMA),
        "no_player_keys": NP_KEYS, "world_only_keys": WO_KEYS,
        "action_mapping": {str(i): ALE[i] for i in range(18)},
        "goal_views": {"no_player": NP_KEYS, "world_only": WO_KEYS},
        "censoring_rules": ["no episode-boundary cross", "valid H=16 future required",
                            "branch must reach H=16 uncensored", "core fields present (no silent imputation)"],
        "split_seed": SPLIT_SEED, "n_episodes": int(len(ep_ids)),
        "n_transitions": int(S.shape[0]), "n_branch_anchors": int(anchor_states.shape[0]),
        "n_valid_branches": int(valid.sum()),
        "dimensions": {"state": list(S.shape), "goal_no_player": list(GNP.shape),
                       "goal_world_only": list(GWO.shape), "branch_future_np": list(fut_np.shape)},
        "file_sha256": {k: sha(v) for k, v in files.items()},
        "export_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    files["manifest.json"] = json.dumps(manifest, indent=2).encode()

    zpath = f"{ART}/seaquest_s1_colab_pack.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    prog(f"WROTE {zpath} ({os.path.getsize(zpath)} bytes, {len(files)} members)")
    json.dump(manifest, open(f"{ART}/pack_manifest.json", "w"), indent=2)
    prog(f"[export] transitions={manifest['n_transitions']} anchors={manifest['n_branch_anchors']} "
         f"valid_branches={manifest['n_valid_branches']} episodes={manifest['n_episodes']}")
    prog(f"[export] split: train={len(tr_e)} val={len(va_e)} test={len(te_e)} episodes")


if __name__ == "__main__":
    main()
