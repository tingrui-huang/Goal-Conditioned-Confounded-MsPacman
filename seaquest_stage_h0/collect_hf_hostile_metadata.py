"""Stage-H0 Part A: EXACT HF recollection enriched with hostile/protected metadata.

Runs inside the FROZEN Stage-S0 `seaquest-s0:ocatari` Docker image (jax teacher +
OCAtari coexist), exactly like seaquest_stage_s05/collect_hf_raw.py. This script
re-runs the byte-identical frozen collection loop (same teacher, adapter, port, RNGs,
seeds, noop-reset, sticky/frameskip, action mapping, pre-action timing) and ADDS, at
every pre-action decision point t, the OCAtari object metadata. The action source and
behaviour distribution are NOT changed — this is metadata ENRICHMENT.

CONFIG SOURCE OF TRUTH (D1): the ORIGINAL `raw_hf/manifest.json` is authoritative.
`base_seed`, `max_steps_per_ep` and `n_episodes` are READ from it; the frozen teacher
checkpoint / adapter / port SHA256 are asserted to match the original manifest. CLI
flags are CONSISTENCY ASSERTIONS only — if a CLI value is given and disagrees with the
manifest, the run aborts. We never invent a seed schedule.

For every episode the newly collected base arrays (frames, actions, player_pos, oxygen,
done, target, theta) are compared with EXACT equality to the original raw_hf trajectory
(first mismatch reports episode/field/index/dtype/length). Any difference aborts with
HOSTILE_RECOLLECTION_NOT_IDENTICAL. Metadata is saved to a SEPARATE directory (never
written into raw_hf, which would silently flip the legacy SeaquestOfflineDataset into
'enemy' mode); each metadata file references the SHA256 of its raw_hf trajectory.

Local (Docker) only. Do NOT run in Colab.
"""
import sys, os, json, hashlib, argparse, subprocess
_REAL_STDOUT = sys.stdout                              # captured before teacher_port redirects stdout


def prog(*a, **kw):
    kw.pop("flush", None)
    print(*a, file=_REAL_STDOUT, flush=True, **kw)


import numpy as np

sys.path.insert(0, "/work")
sys.path.insert(0, "/work/seaquest_stage_s05")
sys.path.insert(0, "/work/seaquest_stage_s0")

import common as CM                                    # frozen S0 teacher loader + paths
from seaquest_ccrl import config as C
from seaquest_ccrl.envs.seaquest_gc import _player_pos, _oxygen
from seaquest_ccrl.hostile import schema as S
from seaquest_ccrl.hostile import extraction as EX
from seaquest_ccrl.hostile import removal as RM

RAW_DEFAULT = "/work/seaquest_ccrl/data/raw_hf"
META_DEFAULT = "/work/seaquest_ccrl/data/hostile_h0_metadata"
PARITY_DIR = "/work/artifacts/seaquest/hostile_h0/parity"

BASE_FIELDS = ("frames", "actions", "player_pos", "oxygen", "done", "target", "theta")


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _sample_target(rng):
    x = rng.randint(C.TARGET_X_RANGE[0], C.TARGET_X_RANGE[1] + 1)
    y = rng.randint(C.TARGET_Y_RANGE[0], C.TARGET_Y_RANGE[1] + 1)
    return (float(x), float(y))


def _lives_of(objs):
    for o in objs:
        if getattr(o, "category", "") == "Lives":
            return float(getattr(o, "value", np.nan))
    return np.nan


def _compare_field(new, orig, key):
    """Exact comparison. Returns (ok, detail_dict). equal_nan only for float arrays."""
    a = np.asarray(new); b = np.asarray(orig)
    det = {"field": key, "new_shape": list(a.shape), "orig_shape": list(b.shape),
           "new_dtype": str(a.dtype), "orig_dtype": str(b.dtype)}
    if a.shape != b.shape:
        det["mismatch"] = "shape"
        return False, det
    if str(a.dtype) != str(b.dtype):
        det["mismatch"] = "dtype"
        return False, det
    if np.issubdtype(a.dtype, np.floating):
        ok = np.array_equal(a, b, equal_nan=True)
        diff = ~(np.isclose(a, b, rtol=0, atol=0, equal_nan=True))
    else:
        ok = np.array_equal(a, b)
        diff = a != b
    if not ok:
        flat = np.argwhere(diff)
        det["mismatch"] = "value"
        det["n_diff"] = int(diff.sum())
        det["first_index"] = [int(x) for x in flat[0]] if len(flat) else None
        if len(flat):
            idx = tuple(int(x) for x in flat[0])
            det["new_value"] = float(np.asarray(a[idx]))
            det["orig_value"] = float(np.asarray(b[idx]))
    return ok, det


def _resolve_config(raw_root, cli_episodes, cli_max_steps, cli_seed):
    """raw_hf/manifest.json is authoritative; CLI values are assertion-only."""
    mpath = os.path.join(raw_root, "manifest.json")
    if not os.path.exists(mpath):
        raise FileNotFoundError(f"original raw_hf manifest missing: {mpath}")
    om = json.load(open(mpath))
    base_seed = int(om["base_seed"])
    max_steps = int(om["max_steps_per_ep"])
    n_episodes = int(om.get("n_episodes", om.get("n_episodes_requested")))
    # CLI assertions
    for name, cli, resolved in [("--seed", cli_seed, base_seed),
                                ("--max-steps", cli_max_steps, max_steps),
                                ("--episodes", cli_episodes, n_episodes)]:
        if cli is not None and int(cli) != int(resolved):
            raise AssertionError(
                f"CLI {name}={cli} disagrees with raw_hf manifest ({resolved}); "
                f"manifest is the source of truth. Omit the flag or pass the manifest value.")
    # teacher identity assertions (recover the EXACT frozen teacher)
    teacher_checks = {}
    for key, path in [("teacher_ckpt_sha256", CM.CKPT),
                      ("teacher_adapter_sha256", "/work/seaquest_stage_s0/teacher_adapter.py"),
                      ("teacher_port_sha256", "/work/seaquest_stage_s0/teacher_port.py")]:
        if key in om:
            got = _sha256_file(path)
            teacher_checks[key] = {"manifest": om[key], "current": got, "match": got == om[key]}
            if got != om[key]:
                raise AssertionError(
                    f"HOSTILE_RECOLLECTION_NOT_IDENTICAL: {key} mismatch — frozen teacher "
                    f"path differs from the one that produced raw_hf ({path})")
    return {"base_seed": base_seed, "max_steps_per_ep": max_steps, "n_episodes": n_episodes,
            "config_source": mpath, "teacher_checks": teacher_checks,
            "original_manifest_action_mapping": om.get("action_mapping"),
            "original_env": om.get("env")}


def collect(raw_root, meta_root, cli_episodes=None, cli_max_steps=None, cli_seed=None,
            verbose=True, limit_episodes=None):
    os.makedirs(meta_root, exist_ok=True)
    os.makedirs(PARITY_DIR, exist_ok=True)
    cfg = _resolve_config(raw_root, cli_episodes, cli_max_steps, cli_seed)
    base_seed, max_steps, n_episodes = cfg["base_seed"], cfg["max_steps_per_ep"], cfg["n_episodes"]
    # ENGINEERING DRY-RUN cap only: process the first K episodes (seeds are per-episode
    # base_seed+ep, so the schedule for those episodes is unchanged). NOT for the real gate.
    if limit_episodes is not None:
        n_episodes = min(n_episodes, int(limit_episodes))
        prog(f"[dry-run] limiting to first {n_episodes} episodes (NOT the full parity gate)")

    from teacher_port import SeaquestPort                # S0 frozen port
    teacher = CM.load_teacher("A")                       # S0 frozen adapter
    CM.prog = prog
    # IDENTICAL RNG construction order to seaquest_stage_s05/collect_hf_raw.py
    port = SeaquestPort(sticky=0.0, full_action_space=True, seed=base_seed)
    tgt_rng = np.random.RandomState(base_seed)
    noop_rng = np.random.RandomState(base_seed + 11)
    noise_rng = np.random.RandomState(base_seed + 777)

    try:
        oc_commit = subprocess.check_output(
            ["git", "-C", "/work/OC_Atari", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        oc_commit = None
    try:
        src_commit = subprocess.check_output(
            ["git", "-C", "/work", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        src_commit = None
    try:
        import ocatari as _oc
        oc_version = getattr(_oc, "__version__", None)
    except Exception:
        oc_version = None

    manifest = {
        "experiment": "seaquest_hostile_h0_metadata_enrichment",
        "purpose": "EXACT HF recollection enriched with hostile/protected OCAtari metadata",
        "config_source": cfg["config_source"], "resolved_config": {
            "base_seed": base_seed, "max_steps_per_ep": max_steps, "n_episodes": n_episodes},
        "teacher_identity_checks": cfg["teacher_checks"],
        "base_arrays_compared": list(BASE_FIELDS),
        "parity_rule": "exact equality (equal_nan for float); else HOSTILE_RECOLLECTION_NOT_IDENTICAL",
        "schema": S.schema_dict(),
        "teacher_ckpt": CM.CKPT, "teacher_ckpt_sha256": _sha256_file(CM.CKPT),
        "teacher_adapter_sha256": _sha256_file("/work/seaquest_stage_s0/teacher_adapter.py"),
        "teacher_port_sha256": _sha256_file("/work/seaquest_stage_s0/teacher_port.py"),
        "collect_hf_raw_sha256": _sha256_file("/work/seaquest_stage_s05/collect_hf_raw.py"),
        "ocatari_git_commit": oc_commit, "ocatari_version": oc_version,
        "source_git_commit": src_commit,
        "env": {"env_id": "ALE/Seaquest-v5 (OCAtari port)", "frameskip": SeaquestPort.FRAME_SKIP,
                "repeat_action_probability": 0.0, "full_action_space": True, "noop_max": 30},
        "raw_root": raw_root, "meta_root": meta_root, "episodes": [],
    }
    parity = {"config_source": cfg["config_source"], "resolved_config": manifest["resolved_config"],
              "teacher_identity_checks": cfg["teacher_checks"], "episodes": [],
              "n_episodes": n_episodes}

    for ep in range(n_episodes):
        raw_path = os.path.join(raw_root, f"traj_{ep:04d}.npz")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"original raw_hf trajectory missing: {raw_path}")
        orig = np.load(raw_path)

        port.reset(seed=base_seed + ep, noop_max=30, rng=noop_rng)
        target = _sample_target(tgt_rng)
        frames, actions, positions, oxygens, dones, targets = [], [], [], [], [], []
        extracted_by_t = []     # plain-value snapshots (OCAtari MUTATES object instances in place)
        rewards, lives_before, lives_after, life_lost = [], [], [], []
        last_pos = last_oxy = None                                     # EXACT collect_hf_raw forward-fill

        for t in range(max_steps):
            objs = [o for o in port.env.objects if getattr(o, "category", "") != "NoObject"]
            pos = _player_pos(objs); oxy = _oxygen(objs)
            if pos is None:
                pos = last_pos
            else:
                last_pos = pos
            if oxy is None:
                oxy = last_oxy
            else:
                last_oxy = oxy
            frame = np.asarray(port.env.render(), dtype=np.uint8)      # (210,160,3) PRE-action
            obs = port.teacher_obs()                                   # (4,84,84) teacher view
            noise = teacher.gumbel_from_uniform(noise_rng.uniform(size=(18,)))
            a = int(teacher.sample_action(obs, noise, temperature=1.0)[0])  # FROZEN O-Sampled

            frames.append(frame); actions.append(a)
            positions.append(pos if pos is not None else (np.nan, np.nan))
            oxygens.append(oxy if oxy is not None else -1)
            targets.append(target)
            extracted_by_t.append(EX.extract_objects(objs))            # snapshot values NOW
            lives_before.append(_lives_of(objs))

            rec = port.agent_step(a)
            done = bool(rec["terminated"] or rec["truncated"])
            dones.append(done)
            rewards.append(float(rec["reward"]))                       # reward AFTER action_t
            la = _lives_of(port.env.objects)
            lives_after.append(la)
            lb = lives_before[-1]
            life_lost.append(bool(np.isfinite(lb) and np.isfinite(la) and la < lb))
            if done:
                break

        # --- assemble base arrays in the ORIGINAL schema ---
        base = {"frames": np.asarray(frames, dtype=np.uint8),
                "actions": np.asarray(actions, dtype=np.int64),
                "player_pos": np.asarray(positions, dtype=np.float32),
                "oxygen": np.asarray(oxygens, dtype=np.int32),
                "done": np.asarray(dones, dtype=np.bool_),
                "target": np.asarray(targets, dtype=np.float32),
                "theta": np.int32(C.THETA)}

        # --- EXACT parity vs the REAL original trajectory (all base arrays) ---
        ep_fields, ep_ok = {}, True
        ep_fields["episode_length"] = {"new": int(len(actions)), "orig": int(len(orig["actions"])),
                                       "match": int(len(actions)) == int(len(orig["actions"]))}
        ep_ok &= ep_fields["episode_length"]["match"]
        for k in BASE_FIELDS:
            okk, det = _compare_field(base[k], orig[k], k)
            ep_fields[k] = {"match": bool(okk), **det}
            ep_ok &= okk
        if not ep_ok:
            first_bad = next((v for v in ep_fields.values() if not v["match"]), None)
            parity["episodes"].append({"episode_id": ep, "ok": False, "fields": ep_fields})
            _write_parity(parity, marker="HOSTILE_RECOLLECTION_NOT_IDENTICAL", ok=False)
            raise AssertionError(
                f"HOSTILE_RECOLLECTION_NOT_IDENTICAL: episode {ep} field mismatch: {first_bad}")

        # --- build hostile/protected metadata from the PRE-REMOVAL value snapshots ---
        T = len(frames)
        harr = S.empty_hostile_arrays(T)
        parr = S.empty_protected_arrays(T)
        perclass = {k: np.zeros(T, dtype=np.int32) for k in S.HOSTILE_ID}
        for t, ex in enumerate(extracted_by_t):
            EX.fill_step(harr, parr, t, ex)
            for r in ex["hostile"]:
                perclass[S.HOSTILE_NAME[r["class_id"]]][t] += 1
        S.validate_hostile_arrays(harr)
        amb = np.array([RM.ambiguous_row({kk: harr[kk][t] for kk in harr if harr[kk].shape[0] == T},
                                         {kk: parr[kk][t] for kk in parr})
                        for t in range(T)], dtype=bool)
        raw_sha = _sha256_file(raw_path)
        meta = dict(harr); meta.update(parr)
        meta.update({
            "ambiguous": amb,
            "reward": np.asarray(rewards, dtype=np.float32),
            "lives_before": np.asarray(lives_before, dtype=np.float32),
            "lives_after": np.asarray(lives_after, dtype=np.float32),
            "life_lost_after_action": np.asarray(life_lost, dtype=np.bool_),
            "raw_sha256": raw_sha, "episode_id": np.int32(ep),
            "ocatari_git_commit": str(oc_commit), "ocatari_version": str(oc_version),
        })
        meta_path = os.path.join(meta_root, f"meta_{ep:04d}.npz")
        np.savez_compressed(meta_path, **meta)

        parity["episodes"].append({"episode_id": ep, "ok": True,
                                   "fields": {k: v["match"] for k, v in ep_fields.items()},
                                   "raw_path": raw_path, "raw_sha256": raw_sha,
                                   "meta_path": meta_path, "meta_sha256": _sha256_file(meta_path),
                                   "length": T})
        manifest["episodes"].append({
            "episode_id": ep, "steps": T,
            "raw_file": os.path.basename(raw_path), "raw_sha256": raw_sha,
            "meta_file": os.path.basename(meta_path), "meta_sha256": _sha256_file(meta_path),
            "parity": "IDENTICAL",
            "n_enemy_rows": int((harr["enemy_count"] > 0).sum()),
            "n_missile_rows": int((harr["enemy_missile_count"] > 0).sum()),
            "n_ambiguous_rows": int(amb.sum()),
            "perclass_total": {k: int(v.sum()) for k, v in perclass.items()},
        })
        if verbose:
            e = manifest["episodes"][-1]
            prog(f"[ep {ep:03d}] steps={T:4d} PARITY=IDENTICAL enemy_rows={e['n_enemy_rows']} "
                 f"missile_rows={e['n_missile_rows']} ambig={e['n_ambiguous_rows']} -> {meta_path}")

    manifest["n_episodes"] = len(manifest["episodes"])
    manifest["all_parity_identical"] = True
    with open(os.path.join(meta_root, "metadata_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    _write_parity(parity, marker="HOSTILE_RECOLLECTION_ALIGNED", ok=True)
    prog(f"\nDONE: {manifest['n_episodes']} episodes enriched, all base arrays IDENTICAL -> {meta_root}")
    prog("HOSTILE_RECOLLECTION_ALIGNED")
    return meta_root


def _write_parity(parity, marker, ok):
    parity["marker"] = marker
    parity["all_identical"] = ok
    parity["n_episodes_compared"] = len([e for e in parity["episodes"]])
    os.makedirs(PARITY_DIR, exist_ok=True)
    json.dump(parity, open(os.path.join(PARITY_DIR, "recollection_parity.json"), "w"), indent=2)
    lines = ["# Stage-H0 — Recollection Parity", "",
             f"**Marker:** `{marker}`  ",
             f"**Config source:** {parity['config_source']}  ",
             f"**Resolved:** {parity['resolved_config']}  ",
             f"**Episodes compared:** {parity['n_episodes_compared']}", "",
             "| ep | ok | fields |", "|---|---|---|"]
    for e in parity["episodes"]:
        f = e["fields"]
        if isinstance(f, dict) and all(isinstance(v, bool) for v in f.values()):
            fld = ",".join(k for k, v in f.items() if v)
            lines.append(f"| {e['episode_id']} | {e['ok']} | all({len(f)}) ok |")
        else:
            lines.append(f"| {e['episode_id']} | {e['ok']} | MISMATCH (see json) |")
    open(os.path.join(PARITY_DIR, "recollection_parity.md"), "w").write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=None, help="assertion-only; manifest is source of truth")
    ap.add_argument("--max-steps", type=int, default=None, help="assertion-only")
    ap.add_argument("--seed", type=int, default=None, help="assertion-only (raw_hf base_seed)")
    ap.add_argument("--raw-root", type=str, default=RAW_DEFAULT)
    ap.add_argument("--meta-root", type=str, default=META_DEFAULT)
    ap.add_argument("--limit-episodes", type=int, default=None,
                    help="ENGINEERING DRY-RUN cap (first K episodes); NOT the full parity gate")
    args = ap.parse_args()
    collect(args.raw_root, args.meta_root, cli_episodes=args.episodes,
            cli_max_steps=args.max_steps, cli_seed=args.seed, limit_episodes=args.limit_episodes)


if __name__ == "__main__":
    main()
