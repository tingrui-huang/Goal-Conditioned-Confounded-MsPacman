"""Stage-G0 steps 8-11 — evaluation anchor collection + clone/restore PARITY gate.

Runs in the validated local OCAtari/ALE env (seaquest-s0:ocatari). Collects FRESH HF-teacher
episodes (greedy, seeds >= 1000, disjoint from train 0..39), selects pre-action anchors, builds
reference continuations for H in {16,32,64} with goal = player position at t+H, filters
eligibility, then verifies clone/restore by restoring each anchor and REPLAYING the reference
action sequence and comparing the final player position + episode status.

Gate: reference-replay success within goal radius >= 0.98, else CLONE_RESTORE_OR_TIMING_INVALID.
Does NOT load/modify the critic. Deterministic (greedy teacher + fixed seed), so step 12 can
recollect the identical anchors from the recorded seeds + action sequences.
"""
import sys, os, json, argparse
from collections import deque
import numpy as np

# teacher_port redirects stdout to devnull at import; restore it.
from seaquest_stage_s0 import teacher_port as TP
sys.stdout = TP._REAL_STDOUT
from seaquest_stage_s0.teacher_port import SeaquestPort
from seaquest_stage_s0.teacher_adapter import CleanRLSeaquestTeacher

GOAL_RADIUS = 8.0
HORIZONS = [16, 32, 64]
SEED_OFFSET = 1000
OUT = "artifacts/seaquest/goal_control/full_view/evaluation"


def pxy(f):
    if f.get("player_x") is None or f.get("player_y") is None:
        return None
    return (float(f["player_x"]), float(f["player_y"]))


def roll_episode(teacher, seed, max_steps, warmup, stride):
    """Greedy-teacher rollout to game-over. Returns positions[t] (s_t, len L+1),
    actions[t], life_lost[t] (lives dropped on t->t+1), and anchor snapshots at strided t."""
    port = SeaquestPort(sticky=0.0, full_action_space=True, seed=seed)
    port.reset(seed=seed, noop_max=0)
    rgbq = deque(maxlen=4)
    r0 = np.asarray(port.ale.getScreenRGB(), np.uint8)
    for _ in range(4):
        rgbq.append(r0.copy())
    positions, actions, life_lost = [], [], []
    anchors = []                      # {t, snap, rgb_stack, player_t}
    start_lives = port.features().get("lives")
    t = 0
    while t < max_steps:
        feat = port.features()
        p_t = pxy(feat)
        positions.append(p_t)
        if t >= warmup and (t - warmup) % stride == 0 and p_t is not None:
            anchors.append({"t": t, "snap": port.snapshot(),
                            "rgb_stack": np.stack(list(rgbq), 0), "player_t": p_t})
        a = int(teacher.greedy_action(port.teacher_obs()[None])[0])
        rec = port.agent_step(a)
        rgbq.append(np.asarray(port.ale.getScreenRGB(), np.uint8).copy())
        lives = port.features().get("lives")
        life_lost.append(bool(start_lives is not None and lives is not None and lives < start_lives))
        if lives is not None and start_lives is not None and lives < start_lives:
            start_lives = lives
        actions.append(a)
        t += 1
        if rec["terminated"] or rec["truncated"]:
            positions.append(pxy(port.features()))
            return dict(seed=seed, positions=positions, actions=actions, life_lost=life_lost,
                        anchors=anchors, length=len(actions), terminated=True), port
    positions.append(pxy(port.features()))
    return dict(seed=seed, positions=positions, actions=actions, life_lost=life_lost,
                anchors=anchors, length=len(actions), terminated=False), port


def eligible(ep, t, H):
    """(eligible_raw, eligible_nontrivial, goal, dist) for anchor t at horizon H."""
    L = ep["length"]
    if t + H > L:                                   # t+H must be a valid recorded state
        return False, False, None, None
    p_t, p_f = ep["positions"][t], ep["positions"][t + H]
    if p_t is None or p_f is None:
        return False, False, None, None
    if any(ep["life_lost"][t:t + H]):               # no reset/life-loss in (t, t+H]
        return False, False, None, None
    d = float(np.hypot(p_f[0] - p_t[0], p_f[1] - p_t[1]))
    return True, (d > GOAL_RADIUS), p_f, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/seaquest/stage_s0/teacher/_downloads/cand_A/"
                                      "sebulba_ppo_envpool_impala_atari_wrapper.cleanrl_model")
    ap.add_argument("--src", default="artifacts/seaquest/stage_s0/teacher/_downloads/cand_A/"
                                     "sebulba_ppo_envpool_impala_atari_wrapper.py")
    ap.add_argument("--n-episodes", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=2500)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out-dir", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    teacher = CleanRLSeaquestTeacher(args.ckpt, args.src, mod_name="cleanrl_src_A")
    print(f"[anchors] teacher action_dim={teacher.action_dim}; seeds "
          f"{SEED_OFFSET}..{SEED_OFFSET + args.n_episodes - 1}; H={HORIZONS}")

    anchors_meta, ep_meta = [], []
    ep_actions, ep_positions = {}, {}
    rgb_store, refact_store = {}, {}
    # parity accumulators per horizon
    par = {H: {"n": 0, "exact": 0, "within1": 0, "within_r": 0, "first_mismatch": []} for H in HORIZONS}
    eligible_eps = {H: set() for H in HORIZONS}
    raw_count = {H: 0 for H in HORIZONS}; raw_eps = {H: set() for H in HORIZONS}
    aid = 0
    for i in range(args.n_episodes):
        seed = SEED_OFFSET + i
        ep, port = roll_episode(teacher, seed, args.max_steps, args.warmup, args.stride)
        ep_meta.append({"seed": seed, "length": ep["length"], "terminated": ep["terminated"],
                        "n_life_losses": int(sum(ep["life_lost"])), "n_anchor_steps": len(ep["anchors"])})
        ep_actions[f"actions_{seed}"] = np.asarray(ep["actions"], np.int16)
        ep_positions[f"pos_{seed}"] = np.asarray([(p if p else (np.nan, np.nan)) for p in ep["positions"]], np.float32)
        print(f"  ep seed={seed} len={ep['length']} life_losses={sum(ep['life_lost'])} "
              f"anchors={len(ep['anchors'])} terminated={ep['terminated']}")

        for anc in ep["anchors"]:
            t = anc["t"]
            elig_H = {}
            for H in HORIZONS:
                er, ent, goal, d = eligible(ep, t, H)
                if er:                               # raw eligibility (secondary metric)
                    raw_count[H] += 1; raw_eps[H].add(seed)
                if er and ent:                       # eligible + nontrivial -> parity + count
                    elig_H[H] = (goal, d)
            if not elig_H:
                continue
            # ---- step 11: clone/restore parity (restore snapshot, replay reference actions) ----
            for H, (goal, d) in elig_H.items():
                port.restore(anc["snap"])
                ref = ep["actions"][t:t + H]
                first_mis = -1
                for k, a in enumerate(ref):
                    port.agent_step(a)
                    pk = pxy(port.features())
                    truth = ep["positions"][t + k + 1]
                    if first_mis < 0 and (pk is None or truth is None or pk != truth):
                        first_mis = k
                final = pxy(port.features())
                truth_final = ep["positions"][t + H]
                ex = bool(final is not None and final == truth_final)
                w1 = bool(final is not None and truth_final is not None and
                          abs(final[0] - truth_final[0]) <= 1 and abs(final[1] - truth_final[1]) <= 1)
                wr = bool(final is not None and truth_final is not None and
                          np.hypot(final[0] - truth_final[0], final[1] - truth_final[1]) <= GOAL_RADIUS)
                P = par[H]; P["n"] += 1; P["exact"] += ex; P["within1"] += w1; P["within_r"] += wr
                P["first_mismatch"].append(first_mis)
                eligible_eps[H].add(seed)
                rgb_store[f"rgb_{aid}"] = anc["rgb_stack"].astype(np.uint8)
                refact_store[f"ref_{aid}"] = np.asarray(ref, np.int16)
                anchors_meta.append({"anchor_id": aid, "seed": seed, "t": t, "H": H,
                                     "player_t": list(anc["player_t"]), "goal": list(goal),
                                     "displacement": d, "eligible_nontrivial": True,
                                     "parity_exact": ex, "parity_within1": w1, "parity_within_radius": wr,
                                     "first_mismatch_step": first_mis})
                aid += 1
        del port

    # ---- eligibility report: nontrivial (primary) + raw (secondary, no nontrivial filter) ----
    counts = {H: {"eligible_nontrivial": par[H]["n"], "n_episodes_nontrivial": len(eligible_eps[H]),
                  "eligible_raw": raw_count[H], "n_episodes_raw": len(raw_eps[H])} for H in HORIZONS}

    parity_summary, gate_ok = {}, True
    for H in HORIZONS:
        P = par[H]; n = max(P["n"], 1)
        fm = [x for x in P["first_mismatch"] if x >= 0]
        parity_summary[H] = {"n_replays": P["n"], "exact_match_rate": P["exact"] / n,
                             "within_1px_rate": P["within1"] / n, "within_goal_radius_rate": P["within_r"] / n,
                             "n_first_mismatch": len(fm), "min_first_mismatch": (min(fm) if fm else None)}
        if P["within_r"] / n < 0.98:
            gate_ok = False
    enough = all(counts[H]["eligible_nontrivial"] >= 300 and counts[H]["n_episodes_nontrivial"] >= 20 for H in HORIZONS)

    if not enough:
        outcome = "INSUFFICIENT_ANCHORS_STOP_FOR_REVIEW"
    elif gate_ok:
        outcome = "CLONE_RESTORE_PARITY_PASS"
    else:
        outcome = "CLONE_RESTORE_OR_TIMING_INVALID"

    manifest = {"seeds": [SEED_OFFSET + i for i in range(args.n_episodes)],
                "seed_offset": SEED_OFFSET, "horizons": HORIZONS, "goal_radius": GOAL_RADIUS,
                "warmup": args.warmup, "stride": args.stride, "greedy_teacher": True,
                "train_seed_range": "0..39 (disjoint)", "eligible_counts": counts,
                "target_per_horizon": 300, "target_episodes": 20,
                "n_episodes": args.n_episodes, "n_eligible_anchors": len(anchors_meta),
                "episodes": ep_meta, "anchors": anchors_meta}
    json.dump(manifest, open(f"{args.out_dir}/anchor_manifest.json", "w"), indent=2)
    json.dump({"outcome": outcome, "gate_threshold_within_radius": 0.98, "gate_pass": gate_ok,
               "enough_anchors": enough, "per_horizon": parity_summary,
               "eligible_counts": counts}, open(f"{args.out_dir}/clone_restore_parity.json", "w"), indent=2)
    np.savez_compressed(f"{args.out_dir}/episode_actions.npz", **ep_actions, **ep_positions)
    np.savez_compressed(f"{args.out_dir}/anchor_refactions.npz", **refact_store)
    np.savez_compressed(f"{args.out_dir}/anchor_rgb_stacks.npz", **rgb_store)
    print(json.dumps({"outcome": outcome, "eligible_counts": counts,
                      "parity": {H: parity_summary[H]["within_goal_radius_rate"] for H in HORIZONS}}, indent=2))
    print(f"WROTE {args.out_dir}/anchor_manifest.json + clone_restore_parity.json (+ npz)")
    if outcome != "CLONE_RESTORE_PARITY_PASS":
        raise SystemExit(11)


if __name__ == "__main__":
    main()
