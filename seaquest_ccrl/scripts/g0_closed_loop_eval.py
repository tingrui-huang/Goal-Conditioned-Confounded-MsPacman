"""Stage-G0 steps 12-19 — closed-loop goal-reaching evaluation (full-view critic vs baselines).

Deterministic, episode-balanced, difficulty/direction-stratified anchor subset (default 20/episode
-> 480 per horizon). Common-random-number: every policy restores the IDENTICAL cloned state per
anchor (hash-verified). Policies: full-view critic (argmax over 18 actions), B0 reference-action
replay, B1 random, B2 NOOP, B3a greedy teacher (matches the reference-goal generator), B3b native
stochastic Gumbel teacher (secondary). B4 (masked critic) = N/A (no authoritative checkpoint local).

Runs in seaquest-s0:ocatari-torch. Recomputes anchors deterministically from episode_actions.npz
(reset(seed) -> replay actions[0:t] -> snapshot). Does NOT retrain or modify the critic.
"""
import sys, os, json, hashlib, argparse, csv
from collections import deque, Counter
import numpy as np

from seaquest_stage_s0 import teacher_port as TP
sys.stdout = TP._REAL_STDOUT
from seaquest_stage_s0.teacher_port import SeaquestPort
from seaquest_stage_s0.teacher_adapter import CleanRLSeaquestTeacher

import torch
from seaquest_ccrl.training.train_critic import load_critic
from seaquest_ccrl.models.sa_encoder import preprocess_frames

EVAL = "artifacts/seaquest/goal_control/full_view/evaluation"
HORIZONS = [16, 32, 64]
RADIUS = 8.0
CKPT = "artifacts/seaquest/seaquest_stage_g0_fullview_train/critic_full_view.pt"
TEACHER_CKPT = ("artifacts/seaquest/stage_s0/teacher/_downloads/cand_A/"
                "sebulba_ppo_envpool_impala_atari_wrapper.cleanrl_model")
TEACHER_SRC = ("artifacts/seaquest/stage_s0/teacher/_downloads/cand_A/"
               "sebulba_ppo_envpool_impala_atari_wrapper.py")
# gate thresholds (predeclared, contract section 17)
G_SUCCESS, G_OVER_BASELINE, G_PER_H, G_MEDHARD = 0.60, 0.30, 0.40, 0.25


def difficulty(d):
    return "easy" if d <= 16 else ("medium" if d <= 32 else "hard")
def direction(player_y, goal_y):
    return "up" if goal_y < player_y else "down"
def dist(p, g):
    return float(np.hypot(p[0] - g[0], p[1] - g[1])) if p is not None else np.inf
def pxy(f):
    return None if f.get("player_x") is None or f.get("player_y") is None else (float(f["player_x"]), float(f["player_y"]))
def pcenter(port):
    """Player CENTER (x+w/2, y+h/2) — matches the raw_hf training-goal convention
    (seaquest_gc._player_pos), unlike teacher_port.features which returns top-left (x,y)."""
    for o in port.env.objects:
        if getattr(o, "category", "") == "Player":
            return (float(o.x) + float(o.w) / 2.0, float(o.y) + float(o.h) / 2.0)
    return None


def select_subset(manifest, per_ep):
    """Episode-balanced, 6-cell (difficulty x direction) round-robin, deterministic by t."""
    anchors = manifest["anchors"]
    chosen = {}                                   # H -> list of anchor dicts
    for H in HORIZONS:
        chosen[H] = []
        for seed in manifest["seeds"]:
            cand = [a for a in anchors if a["H"] == H and a["seed"] == seed]
            cells = {}
            for a in cand:
                key = (difficulty(a["displacement"]), direction(a["player_t"][1], a["goal"][1]))
                cells.setdefault(key, []).append(a)
            order = [(d, u) for d in ("easy", "medium", "hard") for u in ("up", "down")]
            for k in order:
                cells.setdefault(k, [])
                cells[k].sort(key=lambda a: a["t"])
            ptr = {k: 0 for k in order}; picked = []
            while len(picked) < per_ep and any(ptr[k] < len(cells[k]) for k in order):
                for k in order:
                    if ptr[k] < len(cells[k]):
                        picked.append(cells[k][ptr[k]]); ptr[k] += 1
                        if len(picked) >= per_ep:
                            break
            for a in picked:
                a2 = dict(a); a2["difficulty"] = difficulty(a["displacement"])
                a2["direction"] = direction(a["player_t"][1], a["goal"][1])
                chosen[H].append(a2)
    return chosen


def run_policy(seed, full_actions, t, H, kind, *, port, critic=None, cfg=None, teacher=None,
               device="cpu", aid=0, goal=None):
    """Reach s_t by reset(seed)+replay (deterministic; avoids ale_py cloneState aliasing),
    then roll one policy for H steps. Every policy reproduces the identical s_t -> genuine CRN."""
    port.reset(seed=seed, noop_max=0)
    rgbq = deque(maxlen=4)
    r0 = np.asarray(port.ale.getScreenRGB(), np.uint8)
    for _ in range(4):
        rgbq.append(r0.copy())
    for s in range(t):
        port.agent_step(int(full_actions[s]))
        rgbq.append(np.asarray(port.ale.getScreenRGB(), np.uint8).copy())
    # CRN identity = emulator RAM at the deterministically-reached s_t
    crn = hashlib.sha256(np.asarray(port.ale.getRAM(), np.uint8).tobytes()).hexdigest()[:16]
    buf = deque(maxlen=4)
    if kind == "critic":
        for f in rgbq:
            buf.append(preprocess_frames(f[None], cfg.frame_size)[0])
    ref = full_actions[t:t + H]
    rng = np.random.default_rng(10_000 + aid)         # B1 / B3b per-anchor stream
    gnorm = cfg.normalize_goal(goal) if (cfg is not None) else None
    positions, acts = [], []
    term_step, life_step = -1, -1
    start_lives = port.features().get("lives")
    for k in range(H):
        if kind == "critic":
            obs = np.concatenate(list(buf), axis=2)
            a = int(np.argmax(critic.score_all_actions(obs, gnorm, device)))
        elif kind == "B0":
            a = int(ref[k])                            # exact reference-action replay
        elif kind == "B1":
            a = int(rng.integers(0, 18))
        elif kind == "B2":
            a = 0
        elif kind == "B3a":
            a = int(teacher.greedy_action(port.teacher_obs()[None])[0])
        elif kind == "B3b":
            noise = teacher.gumbel_from_uniform(rng.uniform(size=18))
            a = int(teacher.sample_action(port.teacher_obs(), noise)[0])
        rec = port.agent_step(a)
        if kind == "critic":
            buf.append(preprocess_frames(np.asarray(port.ale.getScreenRGB(), np.uint8)[None], cfg.frame_size)[0])
        lives = port.features().get("lives")
        positions.append(pcenter(port)); acts.append(a)        # CENTER coords (match training)
        if life_step < 0 and start_lives is not None and lives is not None and lives < start_lives:
            life_step = k
        if rec["terminated"] or rec["truncated"]:
            term_step = k; break
    return {"crn": crn, "positions": positions, "actions": acts,
            "term_step": term_step, "life_step": life_step}


def metrics_for(traj, goal, H):
    """Success metrics for horizon H from a (possibly longer) trajectory vs a fixed goal."""
    pos = traj["positions"][:H]
    ds = np.array([dist(p, goal) for p in pos]) if pos else np.array([np.inf])
    reached = ds <= RADIUS
    tstep, lstep = traj["term_step"], traj["life_step"]
    return {"success_by_H": bool(reached.any()),
            "success_at_H": bool(len(pos) == H and ds[-1] <= RADIUS),
            "min_dist": float(ds.min()), "final_dist": float(ds[-1]),
            "time_to_first_hit": (int(np.argmax(reached) + 1) if reached.any() else -1),
            "life_lost": bool(0 <= lstep < H), "terminated": bool(0 <= tstep < H),
            "n_steps": len(pos), "actions": [int(x) for x in traj["actions"][:H]], "dists": ds.tolist()}


def boot_ep(vals, eps, n=2000, seed=0):
    vals = np.asarray(vals, float); eps = np.asarray(eps)
    uniq = np.unique(eps); by = {e: vals[eps == e] for e in uniq}
    rng = np.random.RandomState(seed); ms = []
    for _ in range(n):
        pk = rng.choice(uniq, len(uniq), replace=True)
        ms.append(np.concatenate([by[e] for e in pk]).mean())
    return {"mean": float(vals.mean()), "ci95": [float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))]}


def boot_paired(a, b, eps, n=2000, seed=0):
    d = np.asarray(a, float) - np.asarray(b, float)
    r = boot_ep(d, eps, n, seed)
    r["excludes_0"] = bool(r["ci95"][0] > 0 or r["ci95"][1] < 0)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-episode", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-dir", default=EVAL)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.makedirs(f"{args.out_dir}/figures", exist_ok=True)
    os.makedirs(f"{args.out_dir}/trajectory_examples", exist_ok=True)
    manifest = json.load(open(f"{args.out_dir}/anchor_manifest.json"))
    ep_npz = np.load(f"{args.out_dir}/episode_actions.npz")
    critic, cfg, oracle = load_critic(CKPT, args.device); assert oracle is True
    teacher = CleanRLSeaquestTeacher(TEACHER_CKPT, TEACHER_SRC, mod_name="cleanrl_src_A")

    per_ep = 3 if args.smoke else args.per_episode
    seeds = manifest["seeds"][:2] if args.smoke else manifest["seeds"]
    chosen = select_subset(manifest, per_ep)
    if args.smoke:
        chosen = {H: [a for a in chosen[H] if a["seed"] in seeds][:3] for H in HORIZONS}
    subset = {"per_episode": per_ep, "horizons": HORIZONS, "n_per_horizon": {H: len(chosen[H]) for H in HORIZONS},
              "radius": RADIUS, "policies": ["critic", "B0", "B1", "B2", "B3a", "B3b"], "B4": "N/A",
              "strata_counts": {H: dict(Counter((a["difficulty"], a["direction"]) for a in chosen[H]))
                                and {f"{d}|{u}": sum(1 for a in chosen[H] if a["difficulty"] == d and a["direction"] == u)
                                     for d in ("easy", "medium", "hard") for u in ("up", "down")} for H in HORIZONS},
              "anchors": {H: [{k: a[k] for k in ("anchor_id", "seed", "t", "H", "player_t", "goal",
                                                 "displacement", "difficulty", "direction")} for a in chosen[H]]
                          for H in HORIZONS}}
    json.dump(subset, open(f"{args.out_dir}/selected_subset_manifest.json", "w"), indent=2)
    print(f"[subset] per_horizon={subset['n_per_horizon']} strata(H16)={subset['strata_counts'][16]}")

    POL = ["critic", "B0", "B1", "B2", "B3a", "B3b"]
    rows = []                                   # per (anchor,H,policy)
    by_seed = {}
    for H in HORIZONS:
        for a in chosen[H]:
            by_seed.setdefault(a["seed"], {}).setdefault(a["t"], []).append((H, a))
    def _row(a, seed, t, H, kind, crn, crn_ref, m):
        return {"anchor_id": a["anchor_id"], "seed": seed, "t": t, "H": H, "policy": kind,
                "difficulty": a["difficulty"], "direction": a["direction"], "displacement": a["displacement"],
                "crn": crn, "crn_ok": crn == crn_ref,
                **{k: m[k] for k in ("success_by_H", "success_at_H", "min_dist", "final_dist",
                   "time_to_first_hit", "life_lost", "terminated", "n_steps")},
                "actions": m["actions"], "dists": m["dists"]}
    BASE = ["B0", "B1", "B2", "B3a", "B3b"]
    work = SeaquestPort(sticky=0.0, full_action_space=True, seed=seeds[0])
    for seed in seeds:
        if seed not in by_seed:
            continue
        acts_full = ep_npz[f"actions_{seed}"]
        ts = sorted(by_seed[seed].keys())
        maxneed = max(t + H for t in ts for (H, _) in by_seed[seed][t])
        # CENTER reference trajectory (one replay) -> center goals matching the training convention
        work.reset(seed=seed, noop_max=0)
        cpos = [pcenter(work)]
        for s in range(maxneed):
            work.agent_step(int(acts_full[s])); cpos.append(pcenter(work))
        for t in ts:
            hlist = by_seed[seed][t]
            maxH = max(H for H, _ in hlist); aid0 = hlist[0][1]["anchor_id"]; crn_ref = None
            cinfo = {}                                    # H -> (center-relabeled anchor, center goal)
            for (H, a) in hlist:
                g, p0 = cpos[t + H], cpos[t]
                d = dist(p0, g) if (g and p0) else 0.0
                a2 = dict(a); a2["goal"] = list(g) if g else a["goal"]
                a2["player_t"] = list(p0) if p0 else a["player_t"]; a2["displacement"] = d
                a2["difficulty"] = difficulty(d)
                a2["direction"] = direction(p0[1], g[1]) if (g and p0) else a["direction"]
                cinfo[H] = (a2, g)
            for kind in BASE:                             # goal-agnostic -> ONE rollout / horizon-set
                traj = run_policy(seed, acts_full, t, maxH, kind, port=work, teacher=teacher,
                                  device=args.device, aid=aid0)
                if crn_ref is None:
                    crn_ref = traj["crn"]
                for (H, a) in hlist:
                    a2, g = cinfo[H]
                    rows.append(_row(a2, seed, t, H, kind, traj["crn"], crn_ref, metrics_for(traj, tuple(g), H)))
            for (H, a) in hlist:                          # critic: goal-conditioned, per horizon
                a2, g = cinfo[H]
                traj = run_policy(seed, acts_full, t, H, "critic", port=work, critic=critic, cfg=cfg,
                                  device=args.device, aid=a2["anchor_id"], goal=tuple(g))
                rows.append(_row(a2, seed, t, H, "critic", traj["crn"], crn_ref, metrics_for(traj, tuple(g), H)))
        print(f"  seed {seed}: {sum(1 for r in rows if r['seed']==seed)} rows")

    _aggregate_and_gate(rows, chosen, args.out_dir, manifest)


def _aggregate_and_gate(rows, chosen, out, manifest):
    crn_ok = all(r["crn_ok"] for r in rows)
    def sel(policy, H=None, pred=None):
        return [r for r in rows if r["policy"] == policy and (H is None or r["H"] == H)
                and (pred is None or pred(r))]
    POL = ["critic", "B0", "B1", "B2", "B3a", "B3b"]
    agg = {"crn_verified": crn_ok, "n_rows": len(rows), "radius": RADIUS, "policies": POL, "per_horizon": {}, "aggregate": {}}
    for policy in POL:
        rs = sel(policy)
        if not rs:
            continue
        agg["aggregate"][policy] = boot_ep([r["success_by_H"] for r in rs], [r["seed"] for r in rs])
        agg["aggregate"][policy]["success_at_H"] = float(np.mean([r["success_at_H"] for r in rs]))
        agg["per_horizon"][policy] = {}
        for H in HORIZONS:
            rh = sel(policy, H)
            agg["per_horizon"][policy][H] = {
                "success_by_H": boot_ep([r["success_by_H"] for r in rh], [r["seed"] for r in rh]),
                "success_at_H": float(np.mean([r["success_at_H"] for r in rh])),
                "n": len(rh), "n_episodes": len(set(r["seed"] for r in rh)),
                "min_dist_mean": float(np.mean([r["min_dist"] for r in rh]))}
    # difficulty/direction stratification (critic success_by_H)
    strat = {}
    for d in ("easy", "medium", "hard"):
        for u in ("up", "down"):
            rs = sel("critic", pred=lambda r: r["difficulty"] == d and r["direction"] == u)
            strat[f"{d}|{u}"] = {"n": len(rs), "success_by_H": (float(np.mean([r["success_by_H"] for r in rs])) if rs else None)}
    medhard = sel("critic", pred=lambda r: r["difficulty"] in ("medium", "hard"))
    medhard_succ = float(np.mean([r["success_by_H"] for r in medhard])) if medhard else 0.0
    agg["critic_stratified_success_by_H"] = strat
    agg["critic_medium_hard_success_by_H"] = {"n": len(medhard), "success_by_H": medhard_succ}
    json.dump(agg, open(f"{out}/aggregate_metrics.json", "w"), indent=2)

    # paired comparisons (anchor-aligned within episode bootstrap)
    def paired(p):
        cr = {(r["anchor_id"], r["H"]): r for r in rows if r["policy"] == "critic"}
        ot = {(r["anchor_id"], r["H"]): r for r in rows if r["policy"] == p}
        keys = [k for k in cr if k in ot]
        return boot_paired([cr[k]["success_by_H"] for k in keys], [ot[k]["success_by_H"] for k in keys],
                           [cr[k]["seed"] for k in keys])
    paired_cmp = {f"critic_minus_{p}": paired(p) for p in ("B1", "B2", "B3a", "B3b", "B0")}
    json.dump(paired_cmp, open(f"{out}/paired_comparisons.json", "w"), indent=2)

    # competence gate (section 17)
    parity = json.load(open(f"{out}/clone_restore_parity.json"))
    ref_ok = all(parity["per_horizon"][str(H)]["within_goal_radius_rate"] >= 0.98 for H in HORIZONS)
    crit_agg = agg["aggregate"]["critic"]["mean"]
    per_h_ok = all(agg["per_horizon"]["critic"][H]["success_by_H"]["mean"] > G_PER_H for H in HORIZONS)
    over_rand = crit_agg - agg["aggregate"]["B1"]["mean"]
    over_noop = crit_agg - agg["aggregate"]["B2"]["mean"]
    ci_rand_excl0 = paired_cmp["critic_minus_B1"]["excludes_0"] and paired_cmp["critic_minus_B1"]["mean"] > 0
    cond = {"1_reference_replay_ge_0.98": bool(ref_ok),
            "2_critic_success_ge_0.60": bool(crit_agg >= G_SUCCESS),
            "3_critic_over_random_ge_0.30": bool(over_rand >= G_OVER_BASELINE),
            "4_critic_over_noop_ge_0.30": bool(over_noop >= G_OVER_BASELINE),
            "5_critic_each_H_gt_0.40": bool(per_h_ok),
            "6_ci_critic_minus_random_excludes_0": bool(ci_rand_excl0),
            "7_medhard_ge_0.25": bool(medhard_succ >= G_MEDHARD)}
    easy_only = (strat.get("easy|up", {}).get("success_by_H") or 0) and medhard_succ < 0.10
    if all(cond.values()):
        outcome = "FULL_VIEW_GOAL_CONTROL_PASS"
    elif crit_agg < 0.30 or over_rand < 0.05:
        outcome = "FULL_VIEW_GOAL_CONTROL_FAIL"
    elif 0.30 <= crit_agg < 0.60 or easy_only:
        outcome = "FULL_VIEW_GOAL_CONTROL_WEAK"
    else:
        outcome = "FULL_VIEW_GOAL_CONTROL_WEAK"
    # borderline flag (escalation trigger)
    def near(x, thr): return abs(x - thr) < 0.05
    borderline = (outcome == "FULL_VIEW_GOAL_CONTROL_WEAK" or near(crit_agg, G_SUCCESS)
                  or near(over_rand, G_OVER_BASELINE) or near(over_noop, G_OVER_BASELINE)
                  or near(medhard_succ, G_MEDHARD)
                  or (crit_agg - 0.0 and agg["aggregate"]["critic"]["ci95"][0] <= G_SUCCESS <= agg["aggregate"]["critic"]["ci95"][1]))
    gate = {"outcome": outcome, "conditions": cond, "all_pass": all(cond.values()),
            "critic_aggregate_success_by_H": crit_agg, "critic_minus_random": over_rand,
            "critic_minus_noop": over_noop, "medhard_success_by_H": medhard_succ,
            "borderline_escalate": bool(borderline),
            "thresholds": {"success": G_SUCCESS, "over_baseline": G_OVER_BASELINE,
                           "per_horizon": G_PER_H, "medhard": G_MEDHARD}}
    json.dump(gate, open(f"{out}/competence_gate.json", "w"), indent=2)

    _diagnostics(rows, out)
    _raw_and_csv(rows, out)
    _summary(agg, paired_cmp, gate, subset_path=f"{out}/selected_subset_manifest.json", out=out)
    print(json.dumps({"outcome": outcome, "crn_verified": crn_ok,
                      "critic_success_by_H": round(crit_agg, 3),
                      "B0": round(agg["aggregate"]["B0"]["mean"], 3),
                      "random": round(agg["aggregate"]["B1"]["mean"], 3),
                      "noop": round(agg["aggregate"]["B2"]["mean"], 3),
                      "B3a_teacher": round(agg["aggregate"]["B3a"]["mean"], 3),
                      "medhard": round(medhard_succ, 3), "borderline": bool(borderline)}, indent=2))
    print(f"WROTE {out}/competence_gate.json + aggregate_metrics.json + paired_comparisons.json + SUMMARY.md")


def _diagnostics(rows, out):
    cr = [r for r in rows if r["policy"] == "critic"]
    alla = [a for r in cr for a in r["actions"]]
    hist = np.bincount(alla, minlength=18).tolist()
    h = np.array(hist, float); h = h / max(h.sum(), 1)
    ent = float(-(h[h > 0] * np.log(h[h > 0])).sum())
    runs = []
    for r in cr:
        a = r["actions"]; c = 1
        for i in range(1, len(a)):
            if a[i] == a[i - 1]:
                c += 1
            else:
                runs.append(c); c = 1
        runs.append(c)
    toward = float(np.mean([(r["dists"][0] - r["dists"][-1]) > 0 for r in cr if r["dists"]]))
    json.dump({"action_frequency": hist, "action_entropy_nats": ent,
               "mean_repeated_run_len": float(np.mean(runs)) if runs else 0,
               "frac_noop": hist[0] / max(sum(hist), 1), "frac_fire": hist[1] / max(sum(hist), 1),
               "frac_rollouts_moving_toward_goal": toward,
               "n_critic_rollouts": len(cr)},
              open(f"{out}/action_frequency.json", "w"), indent=2)
    fails = sorted([r for r in cr if not r["success_by_H"]], key=lambda r: -r["min_dist"])[:20]
    json.dump({"n_failures_examined": len(fails),
               "examples": [{"anchor_id": r["anchor_id"], "seed": r["seed"], "t": r["t"], "H": r["H"],
                             "difficulty": r["difficulty"], "direction": r["direction"],
                             "min_dist": r["min_dist"], "final_dist": r["final_dist"],
                             "actions": r["actions"], "dists": r["dists"]} for r in fails]},
              open(f"{out}/failure_diagnostics.json", "w"), indent=2)
    for i, r in enumerate(fails):
        np.savez(f"{out}/trajectory_examples/fail_{i:02d}_a{r['anchor_id']}_H{r['H']}.npz",
                 actions=np.array(r["actions"]), dists=np.array(r["dists"]),
                 goal=np.array([0]), H=r["H"])


def _raw_and_csv(rows, out):
    keys = ["anchor_id", "seed", "t", "H", "policy", "difficulty", "direction", "displacement",
            "success_by_H", "success_at_H", "min_dist", "final_dist", "time_to_first_hit",
            "life_lost", "terminated", "n_steps", "crn", "crn_ok"]
    with open(f"{out}/per_anchor_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})
    np.savez_compressed(f"{out}/raw_rollouts.npz",
                        anchor_id=np.array([r["anchor_id"] for r in rows]),
                        seed=np.array([r["seed"] for r in rows]), H=np.array([r["H"] for r in rows]),
                        policy=np.array([r["policy"] for r in rows]),
                        success_by_H=np.array([r["success_by_H"] for r in rows]),
                        success_at_H=np.array([r["success_at_H"] for r in rows]),
                        min_dist=np.array([r["min_dist"] for r in rows]),
                        final_dist=np.array([r["final_dist"] for r in rows]),
                        time_to_first_hit=np.array([r["time_to_first_hit"] for r in rows]),
                        life_lost=np.array([r["life_lost"] for r in rows]),
                        terminated=np.array([r["terminated"] for r in rows]))


def _summary(agg, paired, gate, subset_path, out):
    a = agg["aggregate"]
    L = ["# Stage-G0 Closed-Loop Goal-Control Evaluation — SUMMARY", "",
         f"**Outcome: `{gate['outcome']}`**  (CRN verified: {agg['crn_verified']}; borderline-escalate: {gate['borderline_escalate']})", "",
         "## Aggregate success_by_H (episode-bootstrap 95% CI)",
         "| policy | success_by_H | CI | success_at_H |", "|---|---|---|---|"]
    names = {"critic": "full-view critic", "B0": "B0 ref replay", "B1": "B1 random",
             "B2": "B2 NOOP", "B3a": "B3a greedy teacher", "B3b": "B3b native teacher"}
    for p in ["critic", "B0", "B1", "B2", "B3a", "B3b"]:
        if p in a:
            L.append(f"| {names[p]} | {a[p]['mean']:.3f} | [{a[p]['ci95'][0]:.3f},{a[p]['ci95'][1]:.3f}] | {a[p]['success_at_H']:.3f} |")
    L += ["", "## Per-horizon critic success_by_H"]
    for H in HORIZONS:
        ph = agg["per_horizon"]["critic"][H]
        L.append(f"- H={H}: {ph['success_by_H']['mean']:.3f} CI[{ph['success_by_H']['ci95'][0]:.3f},{ph['success_by_H']['ci95'][1]:.3f}] (n={ph['n']})")
    L += ["", "## Gate conditions"]
    for k, v in gate["conditions"].items():
        L.append(f"- {k}: {v}")
    L += ["", "## Paired (critic - baseline) success_by_H"]
    for k, v in paired.items():
        L.append(f"- {k}: {v['mean']:+.3f} CI[{v['ci95'][0]:.3f},{v['ci95'][1]:.3f}] excl0={v['excludes_0']}")
    L += ["", "## Difficulty x direction (critic success_by_H)"]
    for k, v in agg["critic_stratified_success_by_H"].items():
        L.append(f"- {k}: {v['success_by_H']} (n={v['n']})")
    L += ["", f"medium+hard success_by_H = {agg['critic_medium_hard_success_by_H']['success_by_H']:.3f} "
          f"(n={agg['critic_medium_hard_success_by_H']['n']})",
          "", "B4 (masked critic): N/A — no authoritative local checkpoint (smoke not used)."]
    open(f"{out}/SUMMARY.md", "w", encoding="utf-8").write("\n".join(L))


if __name__ == "__main__":
    main()
