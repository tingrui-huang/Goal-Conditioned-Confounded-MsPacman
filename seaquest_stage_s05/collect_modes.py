"""Collect fixed diagnostic rollouts for the 4 modes (Stage-S0.5 sections 5-6).

N-Greedy / N-Sampled: native EnvPool (jax325 image) — action/logits/probs only
(EnvPool exposes no objects; object fields are null, documented).
O-Greedy / O-Sampled: OCAtari port (ocatari image) — full canonical row.

Writes per-mode NPZ (numeric matrix + columns + obs hashes) and a JSON summary.
"""
import sys, os, json, argparse
_REAL_STDOUT = sys.stdout  # captured before teacher_port redirects stdout to devnull
def prog(*a):
    print(*a, file=_REAL_STDOUT, flush=True)
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ART = "/work/artifacts/seaquest/stage_s05"
# Action selection routes through the FROZEN Stage-S0 adapter only
# (teacher.greedy_action / teacher.sample_action). Gumbel-Max is an internal
# property of the original CleanRL policy; no per-step Gumbel noise is recorded.
NUM_COLS = (["episode", "timestep", "sampled_action", "argmax_action", "reward",
             "lives", "life_loss", "terminated", "truncated"]
            + [f"logit_{i}" for i in range(18)] + [f"prob_{i}" for i in range(18)]
            + C.STATE_FEATURES)


def native_collect(teacher, mode, seed, episodes, cap):
    make_env = teacher.src.make_env
    envs = make_env("Seaquest-v5", seed, num_envs=1)()
    noise_rng = np.random.RandomState(seed + 777)  # uniform draws for frozen gumbel helper
    rows = []; hashes = []; obs_samples = []
    sampled = (mode == "N-Sampled")
    ep = 0; n = 0
    while ep < episodes and n < cap:
        next_obs = envs.reset(); start_lives = None; t = 0
        terminated = False
        while not terminated:
            o = np.asarray(next_obs)  # (1,4,84,84)
            logits = teacher.logits(o[0])[0]  # (18,)
            probs = C.softmax(logits, 1.0)
            argmax_a = int(teacher.greedy_action(o[0])[0])  # FROZEN adapter
            if sampled:
                noise = teacher.gumbel_from_uniform(noise_rng.uniform(size=(18,)))
                a = int(teacher.sample_action(o[0], noise, temperature=1.0)[0])  # FROZEN
            else:
                a = argmax_a
            if len(obs_samples) < 6:
                obs_samples.append(o[0].copy()); hashes.append(C.obs_hash(o[0]))
            next_obs, r, d, info = envs.step(np.array([a]))
            lives = int(info["lives"][0]); rew = float(info["reward"][0])
            if start_lives is None:
                start_lives = lives
            life_lost = int(lives < start_lives)
            terminated = int(sum(info["terminated"])) == 1
            row = ([ep, t, a, argmax_a, rew, lives, life_lost, int(terminated), 0]
                   + logits.tolist() + probs.tolist() + [np.nan] * len(C.STATE_FEATURES))
            rows.append(row); t += 1; n += 1
            if n >= cap:
                break
        ep += 1
        prog(f"  [{mode}] ep={ep-1} len={t}")
    envs.close()
    return rows, obs_samples, hashes


def ported_collect(teacher, mode, seed, episodes, cap):
    from teacher_port import SeaquestPort
    port = SeaquestPort(sticky=0.0, full_action_space=True, seed=seed)
    rng = np.random.RandomState(seed + 11)          # noop reset RNG
    noise_rng = np.random.RandomState(seed + 777)   # uniform draws for frozen gumbel helper
    sampled = (mode == "O-Sampled")
    rows = []; obs_samples = []; hashes = []
    ep = 0; n = 0
    while ep < episodes and n < cap:
        port.reset(seed=seed + ep, noop_max=30, rng=rng)
        start_lives = port.features()["lives"]; prev = None; t = 0
        while True:
            obs = port.teacher_obs()
            logits = teacher.logits(obs)[0]
            probs = C.softmax(logits, 1.0)
            argmax_a = int(teacher.greedy_action(obs)[0])  # FROZEN adapter
            if sampled:
                noise = teacher.gumbel_from_uniform(noise_rng.uniform(size=(18,)))
                a = int(teacher.sample_action(obs, noise, temperature=1.0)[0])  # FROZEN
            else:
                a = argmax_a
            f = C.enrich_features(port.features(), prev)
            if len(obs_samples) < 6:
                obs_samples.append(obs.copy()); hashes.append(C.obs_hash(obs))
            rec = port.agent_step(a)
            post = C.enrich_features(port.features(), f)
            lives = post["lives"]
            life_lost = int(start_lives is not None and lives is not None and lives < start_lives)
            term = int(rec["terminated"]); trunc = int(rec["truncated"])
            row = ([ep, t, a, argmax_a, rec["reward"],
                    (np.nan if lives is None else lives), life_lost, term, trunc]
                   + logits.tolist() + probs.tolist() + C.feat_row(f))
            rows.append(row); prev = f; t += 1; n += 1
            if term or trunc or life_lost or n >= cap or t >= 4000:
                break
        ep += 1
        prog(f"  [{mode}] ep={ep-1} len={t}")
    return rows, obs_samples, hashes


def summarize(rows, mode):
    R = np.array(rows, dtype=np.float64)
    ci = {c: i for i, c in enumerate(NUM_COLS)}
    acts = R[:, ci["sampled_action"]].astype(int)
    h = np.bincount(acts, minlength=18).astype(float)
    p = h / max(h.sum(), 1)
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    # per-episode returns
    eps = R[:, ci["episode"]].astype(int)
    rets = [float(R[eps == e, ci["reward"]].sum()) for e in np.unique(eps)]
    lens = [int((eps == e).sum()) for e in np.unique(eps)]
    return {"mode": mode, "n_transitions": len(rows), "n_episodes": int(len(np.unique(eps))),
            "return_mean": float(np.mean(rets)), "return_std": float(np.std(rets)),
            "ep_len_mean": float(np.mean(lens)), "action_histogram": h.astype(int).tolist(),
            "action_entropy_nats": ent, "n_distinct_actions": int((h > 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--cap", type=int, default=12000)
    args = ap.parse_args()
    teacher = C.load_teacher("A")
    prog(f"[collect {args.mode}] action_dim={teacher.action_dim}")
    if args.mode.startswith("N-"):
        rows, obs, hashes = native_collect(teacher, args.mode, args.seed, args.episodes, args.cap)
    else:
        rows, obs, hashes = ported_collect(teacher, args.mode, args.seed, args.episodes, args.cap)
    R = np.array(rows, dtype=np.float64)
    out = f"{ART}/closed_loop/rows_{args.mode}.npz"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, rows=R, columns=np.array(NUM_COLS),
                        obs_samples=np.array(obs, dtype=np.uint8), obs_hashes=np.array(hashes))
    summ = summarize(rows, args.mode)
    json.dump(summ, open(f"{ART}/closed_loop/summary_{args.mode}.json", "w"), indent=2)
    prog(f"WROTE {out}  n={len(rows)} ret_mean={summ['return_mean']:.0f} entropy={summ['action_entropy_nats']:.3f} distinct={summ['n_distinct_actions']}")


if __name__ == "__main__":
    main()
