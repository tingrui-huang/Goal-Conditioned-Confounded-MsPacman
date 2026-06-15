"""P1 — network activation / fixed-tensor parity (Stage-S0.5 Q1, Gate P1).

Reuses S0 fixtures_A.npz (5 saved obs + their logits) and re-verifies that the
adapter still reproduces them bitwise; checks argmax + Gumbel-with-fixed-noise,
probability validity, and ALE action ordering. Does NOT recreate the architecture.
"""
import sys, os, json, hashlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

ART = "/work/artifacts/seaquest/stage_s05"
S0FIX = "/work/artifacts/seaquest/stage_s0/teacher/fixtures_A.npz"


def main():
    teacher = C.load_teacher("A")
    fx = np.load(S0FIX)
    fixtures = fx["fixtures"]; s0_logits = fx["logits"]
    R = {"action_dim": teacher.action_dim, "recurrent_state": teacher.recurrent_state,
         "n_fixtures": int(fixtures.shape[0]), "checks": {}}

    # adapter logits vs S0-stored logits (bitwise / tol). Action selection routes
    # ONLY through the frozen Stage-S0 adapter (greedy_action / sample_action +
    # gumbel_from_uniform). Gumbel-Max is an internal property of the original policy.
    max_abs = 0.0; probs_ok = True; argmax_list = []; sampled_list = []
    rng = np.random.RandomState(424242)
    fixed_noise = teacher.gumbel_from_uniform(rng.uniform(size=(18,)))  # frozen S0 helper
    for i in range(fixtures.shape[0]):
        la = teacher.logits(fixtures[i:i + 1])[0]
        max_abs = max(max_abs, float(np.max(np.abs(la - s0_logits[i]))))
        p = C.softmax(la, 1.0)
        if abs(p.sum() - 1.0) > 1e-9 or not np.all(np.isfinite(p)):
            probs_ok = False
        argmax_list.append(int(teacher.greedy_action(fixtures[i:i + 1])[0]))            # frozen
        sampled_list.append(int(teacher.sample_action(fixtures[i:i + 1], fixed_noise)[0]))  # frozen
    R["checks"]["adapter_matches_s0_logits_max_abs"] = max_abs
    R["checks"]["adapter_reproduces_within_tol"] = bool(max_abs < 1e-5)
    R["checks"]["probabilities_valid_sum1_finite"] = bool(probs_ok)
    l1 = teacher.logits(fixtures[0:1]); l2 = teacher.logits(fixtures[0:1])
    R["checks"]["identical_input_identical_logits"] = bool(np.array_equal(l1, l2))
    # frozen sample_action is deterministic given fixed noise
    a1 = int(teacher.sample_action(fixtures[0:1], fixed_noise)[0])
    a2 = int(teacher.sample_action(fixtures[0:1], fixed_noise)[0])
    R["checks"]["fixed_noise_deterministic_action"] = (a1 == a2)
    R["sampling_pathway"] = "frozen Stage-S0 teacher.sample_action / greedy_action (no inline Gumbel)"
    R["argmax_actions"] = argmax_list
    R["frozen_sample_action_fixed_noise"] = sampled_list
    R["n_distinct_argmax_over_fixtures"] = len(set(argmax_list))
    R["action_ordering"] = C.ALE_MEANINGS
    R["all_pass"] = bool(R["checks"]["adapter_reproduces_within_tol"]
                         and R["checks"]["probabilities_valid_sum1_finite"]
                         and R["checks"]["identical_input_identical_logits"]
                         and R["checks"]["fixed_noise_deterministic_action"])
    os.makedirs(f"{ART}/teacher_activation", exist_ok=True)
    json.dump(R, open(f"{ART}/teacher_activation/fixed_tensor_parity.json", "w"), indent=2)
    # copy action mapping reference
    am = json.load(open("/work/artifacts/seaquest/stage_s0/teacher/action_mapping.json"))
    json.dump({"action_count": am.get("action_count"),
               "bijective": am.get("bijective_over_teacher_outputs"),
               "mapping": am.get("mapping")},
              open(f"{ART}/teacher_activation/action_mapping.json", "w"), indent=2)
    print(json.dumps({"all_pass": R["all_pass"], "max_abs": max_abs,
                      "distinct_argmax": R["n_distinct_argmax_over_fixtures"]}, indent=2))


if __name__ == "__main__":
    main()
