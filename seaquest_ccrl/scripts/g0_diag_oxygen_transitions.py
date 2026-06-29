"""VERIFY every oxygen reset in the teacher reference episode (seed 1000) before any claim of
"surfacing". An oxygen jump 0->64 is ambiguous: Seaquest also refills oxygen on death/respawn.
Re-runs the deterministic episode locally, captures per step {oxygen, player_y, lives, reward,
terminated, truncated, player_present}, finds each low->high oxygen reset, and classifies it as:
  (1) successful surfacing + refill,  (2) death/life reset,  (3) unresolved
from the MEASURED signals (lives change, player-absent/respawn, surface region, terminated).
"""
import json
import numpy as np
import ocatari.ram.seaquest as _sq
_orig = _sq._detect_objects_ram
_sq._detect_objects_ram = lambda o, r, h: _orig(o, np.asarray(r, np.int64), h)
from seaquest_ccrl.scripts.g0_closed_loop_eval import SeaquestPort, pcenter   # noqa: E402

OUT = "artifacts/seaquest/behavioral_diagnosis_compact"
FV = "artifacts/seaquest/goal_control/full_view/evaluation"
SEED = 1000
LOW, HIGH = 15.0, 55.0       # oxygen "low" trough / "high" refilled thresholds
WIN = 60                     # refill is gradual (~2/step over ~30 steps), so look back up to 60


def capture():
    acts = np.load(f"{FV}/episode_actions.npz")[f"actions_{SEED}"]
    port = SeaquestPort(sticky=0.0, full_action_space=True, seed=SEED)
    port.reset(seed=SEED, noop_max=0)
    O, PY, LV, RW, TM, TR, PR = [], [], [], [], [], [], []
    for s in range(len(acts)):
        f = port.features()
        O.append(f.get("oxygen") if f.get("oxygen") is not None else -1.0)
        PY.append(f.get("player_y") if f.get("player_y") is not None else np.nan)
        LV.append(f.get("lives") if f.get("lives") is not None else np.nan)
        PR.append(f.get("player_y") is not None)
        rec = port.agent_step(int(acts[s]))
        RW.append(float(rec["reward"])); TM.append(bool(rec["terminated"])); TR.append(bool(rec["truncated"]))
    return (np.array(O), np.array(PY), np.array(LV), np.array(RW),
            np.array(TM), np.array(TR), np.array(PR), len(acts))


def find_resets(O):
    """Each upward crossing into HIGH preceded by a low trough within WIN steps.
    Returns [(t_low, t_high), ...] where t_low = the oxygen trough (min valid O in the window)."""
    resets = []; n = len(O); t = 1
    while t < n:
        if O[t] >= HIGH and O[t - 1] < HIGH:                       # crossing up into "refilled"
            w0 = max(0, t - WIN); seg = O[w0:t]
            valid = np.where(seg >= 0)[0]
            if len(valid):
                tlo = w0 + int(valid[np.argmin(seg[valid])])
                if O[tlo] <= LOW:                                  # a genuine low trough preceded it
                    resets.append((tlo, t))
        t += 1
    return resets


def main():
    O, PY, LV, RW, TM, TR, PR, n = capture()
    surf_thresh = float(np.nanmin(PY)) + 10.0                      # "surface region" = within 10px of the highest the sub ever goes
    resets = find_resets(O)
    rows = []
    for (a, b) in resets:
        w0, w1 = max(0, a - 3), min(n - 1, b + 3)
        lv_before = LV[a]; lv_after = LV[b]
        lv_min_win = np.nanmin(LV[w0:w1 + 1]) if np.isfinite(LV[w0:w1 + 1]).any() else np.nan
        life_lost = bool(np.isfinite(lv_before) and np.isfinite(lv_min_win) and lv_min_win < lv_before)
        player_absent = bool((~PR[w0:w1 + 1]).any())
        py_min_win = float(np.nanmin(PY[w0:w1 + 1])) if np.isfinite(PY[w0:w1 + 1]).any() else np.nan
        reached_surface = bool(np.isfinite(py_min_win) and py_min_win <= surf_thresh)
        term = bool(TM[w0:w1 + 1].any()); trunc = bool(TR[w0:w1 + 1].any())
        rew = float(RW[a:b + 1].sum())
        if a <= 5:
            cls = "initial_spawn (episode start)"
        elif life_lost or term or player_absent:
            cls = "death/life reset"
        elif reached_surface and not life_lost and not player_absent and not term:
            cls = "successful surfacing+refill"
        else:
            cls = "unresolved"
        rows.append({
            "t_before": int(a), "t_after": int(b),
            "oxygen_before": float(O[a]), "oxygen_after": float(O[b]),
            "player_y_before": (None if not np.isfinite(PY[a]) else float(PY[a])),
            "player_y_after": (None if not np.isfinite(PY[b]) else float(PY[b])),
            "player_y_min_in_window": (None if not np.isfinite(py_min_win) else py_min_win),
            "surface_region_thresh_y": round(surf_thresh, 1), "reached_surface_region": reached_surface,
            "lives_before": (None if not np.isfinite(lv_before) else float(lv_before)),
            "lives_after": (None if not np.isfinite(lv_after) else float(lv_after)),
            "lives_min_in_window": (None if not np.isfinite(lv_min_win) else float(lv_min_win)),
            "reward_in_window": rew, "life_lost": life_lost,
            "terminated": term, "truncated": trunc, "player_absent_nearby": player_absent,
            "classification": cls,
        })
    counts = {}
    for r in rows:
        counts[r["classification"]] = counts.get(r["classification"], 0) + 1
    # all life-loss events (lives decrease), independent of oxygen detection
    deaths = [{"step": int(s), "lives_before": float(LV[s - 1]), "lives_after": float(LV[s]),
               "oxygen_at": float(O[s]), "player_y_at": (None if not np.isfinite(PY[s]) else float(PY[s]))}
              for s in range(1, n) if np.isfinite(LV[s]) and np.isfinite(LV[s - 1]) and LV[s] < LV[s - 1]]
    out = {"seed": SEED, "n_steps": n, "n_oxygen_resets": len(rows),
           "thresholds": {"low": LOW, "high": HIGH, "window": WIN, "surface_y": round(surf_thresh, 1)},
           "class_counts": counts, "n_deaths_lives_drop": len(deaths), "death_events": deaths, "resets": rows}
    json.dump(out, open(f"{OUT}/oxygen_transition_verification.json", "w"), indent=2)
    print(f"oxygen resets found: {len(rows)}  | class counts: {counts}\n")
    hdr = ["t_before", "t_after", "O2_b", "O2_a", "py_b", "py_a", "py_min", "surf?",
           "lives_b", "lives_a", "reward", "life_lost", "term", "absent", "CLASS"]
    print("{:>8}{:>8}{:>6}{:>6}{:>7}{:>7}{:>7}{:>6}{:>8}{:>8}{:>8}{:>10}{:>6}{:>7}  {}".format(*hdr))
    for r in rows:
        print("{:>8}{:>8}{:>6.0f}{:>6.0f}{:>7}{:>7}{:>7}{:>6}{:>8}{:>8}{:>8.0f}{:>10}{:>6}{:>7}  {}".format(
            r["t_before"], r["t_after"], r["oxygen_before"], r["oxygen_after"],
            str(r["player_y_before"]), str(r["player_y_after"]),
            ("-" if r["player_y_min_in_window"] is None else round(r["player_y_min_in_window"], 0)),
            str(r["reached_surface_region"]), str(r["lives_before"]), str(r["lives_after"]),
            r["reward_in_window"], str(r["life_lost"]), str(r["terminated"]),
            str(r["player_absent_nearby"]), r["classification"]))
    print(f"\nWROTE {OUT}/oxygen_transition_verification.json")


if __name__ == "__main__":
    main()
