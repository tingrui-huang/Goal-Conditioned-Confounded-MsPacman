"""Verify first-life-loss terminal semantics end-to-end and recompute the lightweight pilot metrics
on first-life-only data. Collects a small first-life pilot (reusing the now-terminal collect_episode),
then runs the required assertions and the corrected metrics + masked oxygen probe.

NO full recollection, NO model training. Outputs PASS/FAIL per check + corrected metrics.
"""
import glob
import json
import os
import sys
from unittest.mock import MagicMock
for _m in ("envpool", "gym"):
    sys.modules.setdefault(_m, MagicMock())
import numpy as np
import ocatari.ram.seaquest as _sq
_orig = _sq._detect_objects_ram
_sq._detect_objects_ram = lambda o, r, h: _orig(o, np.asarray(r, np.int64), h)
from seaquest_ccrl.scripts.g0_closed_loop_eval import (SeaquestPort, TEACHER_CKPT, TEACHER_SRC,
                                                       run_policy, _prep)
from seaquest_ccrl.scripts.g0_pilot_oxyaware import collect_episode, _small, ridge_probe, AT_SURF_C
from seaquest_ccrl.data.dataset import SeaquestOfflineDataset
from seaquest_ccrl.data.masking import apply_oxygen_mask
from seaquest_ccrl.training.dataset_sampler import HindsightSampler
from seaquest_ccrl.training.config import TrainConfig
from seaquest_ccrl.games import get_game
from seaquest_ccrl import config as C
from seaquest_stage_s0.teacher_adapter import CleanRLSeaquestTeacher

OUT = "artifacts/seaquest/pilot_oxyaware_fl"
DATA = f"{OUT}/data"
SEEDS = list(range(7000, 7014))
TRIGGER, REFILLED = 20, 58
UPFAM = {2, 6, 7, 10, 14, 15}
HORIZONS = (16, 32, 64)


def main():
    os.makedirs(DATA, exist_ok=True)
    teacher = CleanRLSeaquestTeacher(TEACHER_CKPT, TEACHER_SRC, mod_name="cleanrl_src_A")
    checks = {}

    # ---- collect first-life pilot ----
    eps = []
    for i, seed in enumerate(SEEDS):
        arrays, log, tcause = collect_episode(teacher, seed, store_frames=True, first_life=True)
        np.savez_compressed(f"{DATA}/traj_{i:04d}.npz", **{k: v for k, v in arrays.items() if v is not None})
        eps.append({"arrays": arrays, "terminal": tcause})
        print(f"  ep{i} seed{seed}: {len(arrays['actions'])} steps, terminal={tcause}")
    json.dump({"confounder": "oxygen", "schema": "raw_hf + first_life", "policy": "P1 trigger=20",
               "episodes": [{"file": f"traj_{i:04d}.npz", "terminal": e["terminal"],
                             "steps": int(len(e["arrays"]["actions"]))} for i, e in enumerate(eps)]},
              open(f"{DATA}/manifest.json", "w"), indent=2)

    # ===== VERIFICATIONS =====
    # 1. every saved episode contains at most one life (lives constant, <=1 terminal, no death frames)
    ok = True
    for e in eps:
        a = e["arrays"]; lv = a["lives"][a["lives"] >= 0]
        ok &= (len(np.unique(lv)) <= 1) and int(a["done"].sum()) <= 1 \
            and bool(a["player_present"].all()) and int((a["death_respawn"] != 0).sum()) == 0
    checks["at_most_one_life"] = ok

    # 2. no saved frame after the first life loss (no absent-player / death frames stored)
    checks["no_post_life_loss_frames"] = all(bool(e["arrays"]["player_present"].all())
                                             and int((e["arrays"]["death_respawn"] != 0).sum()) == 0 for e in eps)

    # 3 + 4. sampler: future goals never cross a life boundary, and no goal has an invalid position
    game = get_game("seaquest"); cfg = TrainConfig(frame_stack=4)
    smp = HindsightSampler(game, oracle=False, cfg=cfg, device="cpu", rng=np.random.default_rng(0), root=DATA)
    off, L, se = smp.offsets, smp.lengths, smp.seg_end
    cross = False
    for e_ in range(smp.n_ep):
        s0 = int(off[e_]); s1 = s0 + int(L[e_]); idx = np.arange(s0, s1)
        cross |= bool((se[s0:s1] < idx).any() or (se[s0:s1] >= s1).any())   # future-only & never leaves episode
    # Monte-Carlo the actual sampling formula
    rng = np.random.default_rng(1)
    for _ in range(20000):
        e_ = rng.integers(0, smp.n_ep); t = rng.integers(0, L[e_]); k = rng.geometric(1 - cfg.gamma)
        gt = int(off[e_] + t); gf = int(min(gt + k, se[gt]))
        if not (off[e_] <= gf < off[e_] + L[e_] and gf >= gt):
            cross = True; break
    goals_finite = bool(np.isfinite(smp.goals.cpu().numpy()).all())
    checks["no_goal_crosses_life"] = (not cross)
    checks["no_invalid_goal_position"] = goals_finite

    # 5. evaluation stops at first life loss (teacher drowns within a long horizon -> rollout truncates)
    work = SeaquestPort(sticky=0.0, full_action_space=True, seed=SEEDS[0])
    traj = run_policy(SEEDS[0], np.zeros(700, np.int64), 0, 700, "B3b", port=work, teacher=teacher, aid=0)
    checks["eval_stops_at_life_loss"] = bool(0 <= traj["life_step"] < 700
                                             and len(traj["positions"]) == traj["life_step"]
                                             and traj["term_step"] == -1)

    # 6. full vs masked exact sample parity (same transitions; obs differ ONLY in the oxygen rect)
    df = SeaquestOfflineDataset(DATA, oracle=True); dm = SeaquestOfflineDataset(DATA, oracle=False)
    tf, tm = df.trajectory(0), dm.trajectory(0)
    parity = (np.array_equal(tf["action"], tm["action"]) and np.array_equal(tf["achieved_goal"], tm["achieved_goal"])
              and np.array_equal(tf["done"], tm["done"]) and np.array_equal(tf["target"], tm["target"])
              and tf["episode_id"] == tm["episode_id"])
    x, y, w, h = C.OXY_MASK_RECT
    diff = (tf["obs"].astype(int) != tm["obs"].astype(int)).any(3)            # (T,210,160) where views differ
    band = np.zeros((210, 160), bool); band[y:y + h, x:x + w] = True
    diff_outside_rect = int(diff[:, ~band].sum())
    checks["full_masked_exact_parity"] = bool(parity and diff_outside_rect == 0 and diff[:, band].sum() > 0)

    # ===== CORRECTED PILOT METRICS (first-life only) =====
    up_lo = up_hi = nlo = nhi = trig = anchors = clean = surf_steps = 0
    tcause = {}; dyl = {H: [] for H in HORIZONS}; dyh = {H: [] for H in HORIZONS}
    pm, pf, po = [], [], []
    for e in eps:
        a = e["arrays"]; act = a["actions"]; oxy = a["oxygen"].astype(float)
        surf = a["surfacing"]; py = a["player_pos"][:, 1].astype(float); div = a["divers"]; T1 = len(act)
        anchors += T1; surf_steps += int(surf.sum()); tcause[e["terminal"]] = tcause.get(e["terminal"], 0) + 1
        for i in range(T1):
            if not (0 <= oxy[i] <= 63):
                continue
            isup = int(act[i]) in UPFAM
            if oxy[i] < TRIGGER:
                nlo += 1; up_lo += isup
                if i > 0 and oxy[i - 1] >= TRIGGER:
                    trig += 1
            else:
                nhi += 1; up_hi += isup
        t = 0
        while t < T1:
            if not surf[t]:
                t += 1; continue
            s0 = t
            while t < T1 and surf[t]:
                t += 1
            blk = py[s0:t]
            if np.isfinite(blk).any() and np.nanmin(blk) <= AT_SURF_C and oxy[min(t, T1 - 1)] >= REFILLED:
                clean += 1
        for H in HORIZONS:
            for tt in range(T1 - H):
                (dyl if oxy[tt] < TRIGGER else dyh)[H].append(py[tt + H] - py[tt])
        fr = a["frames"]
        for tt in range(0, T1, 6):
            pm.append(_small(_prep(fr[tt], 84, True))); pf.append(_small(_prep(fr[tt], 84, False)))
            po.append(int(oxy[tt]))
    plo = up_lo / nlo if nlo else float("nan"); phi = up_hi / nhi if nhi else float("nan")
    fut = {H: {"n_low": len(dyl[H]), "n_high": len(dyh[H]),
               "diff": round(float(np.mean(dyl[H]) - np.mean(dyh[H])), 2) if dyl[H] and dyh[H] else None}
           for H in HORIZONS}
    r2m, aucm, nte, lowf = ridge_probe(pm, po); r2f, aucf, _, _ = ridge_probe(pf, po)
    override_term = sum(tcause.get(k, 0) for k in ("nodiver_surface", "enemy_ascent", "ascent_other"))
    metrics = {"n_episodes": len(eps), "total_anchors": anchors,
               "mean_life_len": round(anchors / len(eps), 1),
               "oxygen_to_action": {"P_up_low": round(plo, 4), "P_up_high": round(phi, 4),
                                    "gradient": round(plo - phi, 4), "low_oxygen_events": trig,
                                    "override_fraction": round(surf_steps / anchors, 4)},
               "oxygen_to_future": fut, "clean_refills": clean, "terminal_causes": tcause,
               "frac_terminated_by_override": round(override_term / len(eps), 3),
               "masked_probe_first_life": {"masked_R2": round(r2m, 3), "masked_AUC": round(aucm, 3),
                                           "fullview_R2": round(r2f, 3), "fullview_AUC": round(aucf, 3),
                                           "n_test": nte, "low_fraction": round(lowf, 3)}}

    summary = {"checks": checks, "corrected_pilot_metrics": metrics, "data": DATA}
    json.dump(summary, open(f"{OUT}/first_life_verification.json", "w"), indent=2)

    print("\n=== FIRST-LIFE VERIFICATION ===")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    m = metrics
    print("\n=== CORRECTED PILOT METRICS (first-life only) ===")
    print(f"  episodes={m['n_episodes']} mean_life_len={m['mean_life_len']} total_anchors={m['total_anchors']}")
    print(f"  oxygen->action: grad={m['oxygen_to_action']['gradient']:+} "
          f"(P_up_low={m['oxygen_to_action']['P_up_low']} high={m['oxygen_to_action']['P_up_high']}) "
          f"low_oxy_events={m['oxygen_to_action']['low_oxygen_events']}")
    print(f"  oxygen->future: " + " ".join(f"H{H} d={fut[H]['diff']}" for H in HORIZONS))
    print(f"  clean_refills={m['clean_refills']} terminal_causes={m['terminal_causes']} "
          f"frac_terminated_by_override={m['frac_terminated_by_override']}")
    print(f"  masked probe (first-life): R2={m['masked_probe_first_life']['masked_R2']} "
          f"AUC={m['masked_probe_first_life']['masked_AUC']} (full R2={m['masked_probe_first_life']['fullview_R2']})")
    print(f"\nchanged files: collect/collect_dataset.py, scripts/g0_pilot_oxyaware.py, training/dataset_sampler.py, "
          f"scripts/g0_closed_loop_eval.py, scripts/g0_actor_closed_loop_eval.py")
    print(f"WROTE {OUT}/first_life_verification.json , dataset -> {DATA}/")


if __name__ == "__main__":
    main()
