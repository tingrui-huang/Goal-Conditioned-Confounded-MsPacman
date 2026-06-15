"""Validate the Stage-S1 Colab pack. Fails on hash mismatch, wrong dims, NaN/Inf,
action out of [0,17], episode leakage across split, invalid H=16 future, or
schema/column mismatch. Used locally AND mirrored in the Colab notebook.
Stdlib + numpy only (no torch / no env deps).
"""
import sys, json, hashlib, zipfile, io
import numpy as np


def load_pack(zip_path):
    z = zipfile.ZipFile(zip_path)
    raw = {n: z.read(n) for n in z.namelist()}
    def arr(name):
        return np.load(io.BytesIO(raw[name]), allow_pickle=False)
    def js(name):
        return json.loads(raw[name].decode())
    return raw, arr, js


def validate(zip_path, strict=True):
    raw, arr, js = load_pack(zip_path)
    errors = []; checks = {}

    manifest = js("manifest.json"); schema = js("feature_schema.json")
    split = js("split_manifest.json"); norm = js("normalization.json")

    # 1) file hashes
    for name, expected in manifest["file_sha256"].items():
        if name == "manifest.json":
            continue
        got = hashlib.sha256(raw[name]).hexdigest()
        if got != expected:
            errors.append(f"hash mismatch: {name}")
    checks["hashes_ok"] = not any("hash mismatch" in e for e in errors)

    S = arr("observational/states.npy"); A = arr("observational/actions.npy")
    GNP = arr("observational/goals_no_player_H16.npy")
    GWO = arr("observational/goals_world_only_H16.npy")
    EID = arr("observational/episode_ids.npy"); TS = arr("observational/timesteps.npy")

    # 2) dims & schema
    if S.shape[1] != schema["state_dim"] or schema["state_dim"] != len(schema["state_schema"]):
        errors.append("state_dim mismatch")
    if GNP.shape[1] != schema["np_dim"] or GWO.shape[1] != schema["wo_dim"]:
        errors.append("goal_dim mismatch")
    if not (len(S) == len(A) == len(GNP) == len(GWO) == len(EID) == len(TS)):
        errors.append("observational length mismatch")
    checks["dims_ok"] = not any(("dim" in e or "length" in e) for e in errors)

    # 3) NaN/Inf
    for nm, a in [("states", S), ("goals_np", GNP), ("goals_wo", GWO)]:
        if not np.isfinite(a).all():
            errors.append(f"non-finite in {nm}")
    checks["finite_ok"] = not any("non-finite" in e for e in errors)

    # 4) action range
    if A.min() < 0 or A.max() > 17:
        errors.append("action outside [0,17]")
    checks["action_range_ok"] = not (A.min() < 0 or A.max() > 17)

    # 5) episode leakage across split
    tr = set(split["train_episode_ids"]); va = set(split["val_episode_ids"]); te = set(split["test_episode_ids"])
    if (tr & va) or (tr & te) or (va & te):
        errors.append("episode leakage across split")
    obs_eps = set(int(x) for x in np.unique(EID))
    if not obs_eps <= (tr | va | te):
        errors.append("observational episode not assigned to a split")
    checks["no_episode_leakage"] = not any("leakage" in e or "not assigned" in e for e in errors)

    # 6) valid H=16 future: goals finite (already) + transitions exist
    checks["has_transitions"] = bool(len(S) > 0)
    if len(S) == 0:
        errors.append("no transitions")

    # 7) branch pack
    AS = arr("branches/anchor_states.npy"); FNP = arr("branches/future_no_player_H16.npy")
    FWO = arr("branches/future_world_only_H16.npy"); VM = arr("branches/valid_mask.npy")
    SC = arr("branches/local_support_counts.npy"); SEM = arr("branches/semantic_action_categories.npy")
    if AS.shape[1] != schema["state_dim"]:
        errors.append("anchor state_dim mismatch")
    if FNP.shape[1:] != (18, schema["np_dim"]) or FWO.shape[1:] != (18, schema["wo_dim"]):
        errors.append("branch future dims mismatch")
    if VM.shape != (AS.shape[0], 18):
        errors.append("valid_mask shape mismatch")
    # valid branch futures must be finite where valid
    vm = VM.astype(bool)
    if vm.any() and not np.isfinite(FNP[vm]).all():
        errors.append("non-finite valid branch np future")
    checks["branch_ok"] = not any("branch" in e or "anchor" in e or "valid_mask" in e for e in errors)
    checks["n_valid_branches"] = int(VM.sum())

    summary = {"zip": zip_path, "n_transitions": int(len(S)), "state_dim": int(S.shape[1]),
               "np_dim": int(GNP.shape[1]), "wo_dim": int(GWO.shape[1]),
               "n_episodes": len(obs_eps), "n_anchors": int(AS.shape[0]),
               "n_valid_branches": int(VM.sum()),
               "split": {"train": len(tr), "val": len(va), "test": len(te)},
               "checks": checks, "errors": errors, "PASS": len(errors) == 0}
    if strict and errors:
        print(json.dumps(summary, indent=2))
        raise AssertionError(f"pack validation FAILED: {errors}")
    return summary


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "artifacts/seaquest/stage_s1/seaquest_s1_colab_pack.zip"
    s = validate(p, strict=False)
    print(json.dumps(s, indent=2))
    print("PASS" if s["PASS"] else "FAIL")
