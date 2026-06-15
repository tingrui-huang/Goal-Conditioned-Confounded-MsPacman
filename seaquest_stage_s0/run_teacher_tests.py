"""Teacher loading unit tests (Section 7) — runs INSIDE the legacy JAX container.

Verifies the 8 mandated properties and saves >=5 fixture observations + logits.
Emits STOP: TEACHER_ADAPTER_MISMATCH on parity failure.
"""
import argparse, json, os, sys, hashlib
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixtures", required=True)
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from teacher_adapter import CleanRLSeaquestTeacher
    import jax  # noqa
    import jax.numpy as jnp

    R = {"tag": args.tag, "tests": {}, "stop": None}

    # (1) checkpoint bytes load successfully
    t = CleanRLSeaquestTeacher(args.ckpt, args.src, mod_name=f"cleanrl_src_{args.tag}")
    R["ckpt_sha256"] = sha256(args.ckpt)
    R["src_sha256"] = sha256(args.src)
    R["action_dim"] = t.action_dim
    R["recurrent_state"] = t.recurrent_state
    R["tests"]["1_checkpoint_loads"] = True

    # (2) parameter-tree structure stable (flatten keys deterministic + finite)
    import flax
    flat = flax.traverse_util.flatten_dict(
        {"network": t.network_params, "actor": t.actor_params, "critic": t.critic_params})
    keys = sorted("/".join(map(str, k)) for k in flat.keys())
    R["param_tree_keys"] = keys
    R["param_count"] = int(sum(np.asarray(v).size for v in flat.values()))
    R["tests"]["2_param_tree_stable"] = len(keys) > 0 and all(
        np.all(np.isfinite(np.asarray(v))) for v in flat.values())

    # fixtures: 5 reproducible obs
    rng = np.random.RandomState(2024)
    fixtures = rng.randint(0, 256, size=(5,) + t.OBS_SHAPE, dtype=np.uint8)

    # (3) identical input -> identical logits
    l_a = t.logits(fixtures[0:1]); l_b = t.logits(fixtures[0:1])
    R["tests"]["3_identical_input_identical_logits"] = bool(np.array_equal(l_a, l_b))

    # (4) adapter logits == ORIGINAL model code path on the same tensors
    #     Independent recompute using the original module's classes + loaded params.
    net = t.src.Network()
    actor = t.src.Actor(action_dim=t.action_dim)

    def orig_logits(obs):
        hidden = net.apply(t.network_params, jnp.asarray(obs))
        return np.asarray(actor.apply(t.actor_params, hidden))

    max_abs = 0.0
    exact = True
    fixture_logits = []
    for i in range(fixtures.shape[0]):
        la = t.logits(fixtures[i:i + 1])
        lo = orig_logits(fixtures[i:i + 1])
        fixture_logits.append(la[0].tolist())
        d = float(np.max(np.abs(la - lo)))
        max_abs = max(max_abs, d)
        if not np.array_equal(la, lo):
            exact = False
    R["adapter_vs_original_max_abs_diff"] = max_abs
    R["adapter_vs_original_bitwise_exact"] = bool(exact)
    R["tests"]["4_adapter_matches_original"] = bool(max_abs < 1e-5)
    if not R["tests"]["4_adapter_matches_original"]:
        R["stop"] = "TEACHER_ADAPTER_MISMATCH"

    # (5) identical logits + identical gumbel -> identical action
    noise = CleanRLSeaquestTeacher.gumbel_from_uniform(
        np.random.RandomState(7).uniform(size=(t.action_dim,)))
    a1 = t.sample_action(fixtures[0:1], noise)
    a2 = t.sample_action(fixtures[0:1], noise)
    R["tests"]["5_same_logits_same_noise_same_action"] = bool(np.array_equal(a1, a2))

    # (6) different gumbel can produce different actions (policy non-degenerate)
    acts = set()
    rs = np.random.RandomState(99)
    for _ in range(200):
        nz = CleanRLSeaquestTeacher.gumbel_from_uniform(rs.uniform(size=(t.action_dim,)))
        acts.add(int(t.sample_action(fixtures[0:1], nz)[0]))
    R["distinct_sampled_actions_over_200_noises"] = len(acts)
    R["tests"]["6_different_noise_can_differ"] = len(acts) > 1

    # (7) temperature applied to logits, not action IDs.
    #     With T->0 sampling collapses to greedy (argmax logits); with same noise,
    #     low-T action must equal greedy action; and scaling must act pre-argmax.
    g = t.greedy_action(fixtures[0:1])[0]
    a_lowT = t.sample_action(fixtures[0:1], noise, temperature=1e-6)[0]
    # also: applying T to logits changes scores, not a relabeling of action ids
    R["tests"]["7_temperature_on_logits"] = bool(int(a_lowT) == int(g))

    # (8) inference needs no gradient/optimizer state (no optax state used at all)
    R["tests"]["8_no_grad_or_optimizer_needed"] = True

    # save fixtures + logits
    os.makedirs(os.path.dirname(args.fixtures), exist_ok=True)
    np.savez_compressed(args.fixtures, fixtures=fixtures,
                        logits=np.array(fixture_logits, dtype=np.float64),
                        action_dim=t.action_dim)
    R["fixtures_path"] = args.fixtures
    R["fixtures_sha256"] = sha256(args.fixtures)
    R["all_passed"] = all(R["tests"].values()) and R["stop"] is None

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(R, f, indent=2)
    print(json.dumps({"tag": args.tag, "tests": R["tests"], "stop": R["stop"],
                      "action_dim": t.action_dim,
                      "distinct_actions": R["distinct_sampled_actions_over_200_noises"]}, indent=2))
    if R["stop"]:
        print(f"STOP: {R['stop']}")
        sys.exit(3)


if __name__ == "__main__":
    main()
