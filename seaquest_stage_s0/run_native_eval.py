"""Native EnvPool teacher evaluation (runs INSIDE the legacy JAX container).

Reproduces each candidate in its EXACT native EnvPool wrapper (the make_env from
its own downloaded training script) and the native Gumbel-Max sampling used by the
vendored cleanrl eval helper (cleanrl_utils/evals/ppo_envpool_jax_eval.py):

    action = argmax(logits - log(-log(u))),  u ~ uniform   (T = 1.0)
    episode ends when sum(infos["terminated"]) == 1
    return   = sum(infos["reward"])   (raw, unclipped game score)

Also runs a uniform-random-policy baseline under the identical wrapper.

Usage (in container):
    python run_native_eval.py --tag A --episodes 10 --smoke 2 --seed 1 \
        --ckpt /work/.../X.cleanrl_model --src /work/.../X.py --out /work/.../native_eval.json
"""
import argparse, json, os, time, sys
import numpy as np


def build(tag, ckpt, src):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from teacher_adapter import CleanRLSeaquestTeacher
    teacher = CleanRLSeaquestTeacher(ckpt, src, mod_name=f"cleanrl_src_{tag}")
    return teacher


def run_eval(teacher, env_id, episodes, seed, policy, max_steps=27000):
    """policy in {'teacher','random'}. Returns per-episode dicts."""
    import jax, jax.numpy as jnp
    make_env = teacher.src.make_env
    envs = make_env(env_id, seed, num_envs=1)()
    n_act = int(envs.single_action_space.n)
    obs_space = envs.single_observation_space

    key = jax.random.PRNGKey(seed)

    def teacher_action(next_obs, key):
        logits, _ = teacher._forward(teacher.network_params, teacher.actor_params,
                                     teacher.critic_params, np.asarray(next_obs))
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        return np.asarray(action), key

    rng = np.random.RandomState(seed + 777)
    results = []
    act_hist = np.zeros(n_act, dtype=np.int64)
    invalid = 0
    for ep in range(episodes):
        next_obs = envs.reset()
        if ep == 0:
            obs_shape = tuple(np.asarray(next_obs).shape)
            obs_dtype = str(np.asarray(next_obs).dtype)
            obs_min = int(np.asarray(next_obs).min()); obs_max = int(np.asarray(next_obs).max())
        ep_ret = 0.0; ep_len = 0; term_count = 0; trunc_count = 0
        terminated = False
        while not terminated:
            if policy == "teacher":
                actions, key = teacher_action(next_obs, key)
            else:
                actions = rng.randint(0, n_act, size=(1,))
            a0 = int(actions[0])
            if a0 < 0 or a0 >= n_act:
                invalid += 1
            else:
                act_hist[a0] += 1
            next_obs, _, _, infos = envs.step(np.array(actions))
            ep_ret += float(infos["reward"][0])
            ep_len += 1
            term = int(sum(infos["terminated"]))
            terminated = term == 1
            if ep_len >= max_steps:
                trunc_count = 1
                break
        results.append({"episode": ep, "return": ep_ret, "length": ep_len,
                        "terminated": int(terminated), "truncated": int(trunc_count)})
        print(f"  [{policy}] ep={ep} return={ep_ret:.1f} len={ep_len}", flush=True)
    envs.close()
    return {
        "policy": policy, "env_id": env_id, "seed": seed, "n_actions": n_act,
        "episodes": results,
        "obs_shape": obs_shape, "obs_dtype": obs_dtype,
        "obs_min": obs_min, "obs_max": obs_max,
        "action_histogram": act_hist.tolist(),
        "invalid_action_outputs": int(invalid),
    }


def summarize(block):
    rets = np.array([e["return"] for e in block["episodes"]], dtype=np.float64)
    lens = np.array([e["length"] for e in block["episodes"]], dtype=np.float64)
    h = np.array(block["action_histogram"], dtype=np.float64)
    p = h / max(h.sum(), 1)
    ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    block["return_mean"] = float(rets.mean()); block["return_std"] = float(rets.std())
    block["return_median"] = float(np.median(rets))
    block["length_mean"] = float(lens.mean())
    block["action_entropy_nats"] = ent
    block["n_distinct_actions_used"] = int((h > 0).sum())
    block["terminated_count"] = int(sum(e["terminated"] for e in block["episodes"]))
    block["truncated_count"] = int(sum(e["truncated"] for e in block["episodes"]))
    return block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--env-id", default="Seaquest-v5")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--smoke", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model-card-ref", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    teacher = build(args.tag, args.ckpt, args.src)
    print(f"[tag {args.tag}] action_dim={teacher.action_dim} "
          f"recurrent={teacher.recurrent_state}", flush=True)

    out = {"tag": args.tag, "env_id": args.env_id, "seed": args.seed,
           "action_dim": teacher.action_dim,
           "recurrent_state": teacher.recurrent_state,
           "model_card_ref_return": args.model_card_ref}

    print("== smoke (teacher) ==", flush=True)
    smoke = summarize(run_eval(teacher, args.env_id, args.smoke, args.seed, "teacher"))
    out["smoke_teacher"] = smoke

    print("== eval (teacher) ==", flush=True)
    out["eval_teacher"] = summarize(run_eval(teacher, args.env_id, args.episodes, args.seed, "teacher"))

    print("== baseline (random) ==", flush=True)
    out["eval_random"] = summarize(run_eval(teacher, args.env_id, args.episodes, args.seed, "random"))

    tm = out["eval_teacher"]["return_mean"]; rm = out["eval_random"]["return_mean"]
    out["teacher_above_random"] = bool(tm > rm)
    if args.model_card_ref:
        out["pct_of_model_card"] = float(tm / args.model_card_ref)
        out["meets_60pct_gate"] = bool(tm >= 0.60 * args.model_card_ref)
    out["wall_seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"WROTE {args.out}  teacher_mean={tm:.1f} random_mean={rm:.1f} "
          f"wall={out['wall_seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
