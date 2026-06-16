"""Stage-H0 qualification logic: component-level gates -> final outcome.

`decide(results)` is a PURE function over a structured metrics dict (so the smoke
test can feed synthetic numbers). It never collapses everything to one number: it
emits per-component (enemy / missile / joint) gate outcomes AND a single final code,
and the written report separates scientific failures, implementation failures,
insufficient support, and inconclusive outcomes.

Predeclared thresholds (Sections 13-16) are frozen here and must NOT be tuned after
reading results.
"""
import os, json, argparse

# -- frozen thresholds -------------------------------------------------------
HIDDEN_UPPER_CI_MAX = 0.50      # recovery upper-95 <= 0.50 -> HIDDEN_ENOUGH
NOT_HIDDEN_LOWER_CI = 0.80      # recovery lower-95 >= 0.80 -> NOT_HIDDEN
NOT_HIDDEN_NEAREST_R2 = 0.60    # masked nearest-position R2 >= 0.60 -> NOT_HIDDEN
ACTION_MIN_IMPROVEMENT = 0.005  # mean exact-action log-loss improvement

ALLOWED = {
    "HOSTILE_JOINT_QUALIFIED", "ENEMY_ONLY_QUALIFIED", "MISSILE_ONLY_QUALIFIED",
    "HOSTILE_HIDDENNESS_INCONCLUSIVE", "HOSTILE_NOT_HIDDEN",
    "HOSTILE_ACTION_CHANNEL_FAILED", "HOSTILE_FUTURE_CHANNEL_FAILED",
    "HOSTILE_SUPPORT_FAILED", "HOSTILE_REMOVAL_INVALID",
    "HOSTILE_RECOLLECTION_NOT_IDENTICAL", "HOSTILE_OBJECT_SCHEMA_INVALID",
}


def _hiddenness_outcome(h):
    """h: {recovery_ci:[lo,hi], visible_better_than_prior:bool,
           masked_nearest_r2:float|None, adequate_support:bool}."""
    if h is None:
        return "INCONCLUSIVE_HIDDENNESS", "no result"
    if not h.get("visible_better_than_prior", False):
        return "INCONCLUSIVE_HIDDENNESS", "visible probe not better than prior (target/probe unvalidated)"
    r2 = h.get("masked_nearest_r2")
    if h.get("adequate_support") and r2 is not None and r2 >= NOT_HIDDEN_NEAREST_R2:
        return "NOT_HIDDEN", f"masked nearest-position R2={r2:.3f} >= {NOT_HIDDEN_NEAREST_R2}"
    lo, hi = h["recovery_ci"]
    if hi <= HIDDEN_UPPER_CI_MAX:
        return "HIDDEN_ENOUGH", f"recovery upper-95={hi:.3f} <= {HIDDEN_UPPER_CI_MAX}"
    if lo >= NOT_HIDDEN_LOWER_CI:
        return "NOT_HIDDEN", f"recovery lower-95={lo:.3f} >= {NOT_HIDDEN_LOWER_CI}"
    return "INCONCLUSIVE_HIDDENNESS", f"recovery CI=[{lo:.3f},{hi:.3f}]"


def _action_pass(a):
    """a: {improvement_mean, improvement_ci:[lo,hi], shuffled_mean, shuffled_ci:[lo,hi],
           one_episode:bool}."""
    if a is None:
        return False, "no result"
    lo, hi = a["improvement_ci"]
    if a["improvement_mean"] < ACTION_MIN_IMPROVEMENT:
        return False, f"mean improvement {a['improvement_mean']:.4f} < {ACTION_MIN_IMPROVEMENT}"
    if not (lo > 0):
        return False, f"improvement lower-95={lo:.4f} not > 0"
    s_lo, s_hi = a.get("shuffled_ci", [0.0, 0.0])
    if s_lo > 0 and a.get("shuffled_mean", 0) >= 0.5 * a["improvement_mean"]:
        return False, "shuffled-U shows comparable improvement"
    if a.get("one_episode", False):
        return False, "improvement explained by a single episode"
    return True, f"improvement {a['improvement_mean']:.4f}, CI=[{lo:.4f},{hi:.4f}]"


def _future_pass(f):
    """f: {per_horizon: {H: {target, mse_red_mean, mse_red_ci:[lo,hi]}}, shuffled_reproduces:bool}."""
    if f is None:
        return False, "no result"
    if f.get("shuffled_reproduces", False):
        return False, "shuffled-U reproduces the improvement"
    ph = {h: v for h, v in f.get("per_horizon", {}).items() if "mse_red_ci" in v}
    short = [(int(h), v) for h, v in ph.items() if int(h) <= 16 and v["mse_red_ci"][0] > 0
             and v["mse_red_mean"] > 0]
    other = [(int(h), v) for h, v in ph.items() if v["mse_red_ci"][0] > 0 and v["mse_red_mean"] > 0]
    if not short:
        return False, "no primary player target with positive MSE reduction (lower-95>0) at H<=16"
    other_h = [h for h, _ in other if h != short[0][0]]
    if not other_h:
        return False, "improvement not confirmed at a second horizon"
    return True, f"H<=16 ok ({short[0][0]}) + second horizon {other_h[0]}"


def decide(results):
    """Return a full report dict including per-component gates and a single final code."""
    rep = {"components": {}, "gates": {}, "thresholds": {
        "hidden_upper_ci_max": HIDDEN_UPPER_CI_MAX, "not_hidden_lower_ci": NOT_HIDDEN_LOWER_CI,
        "not_hidden_nearest_r2": NOT_HIDDEN_NEAREST_R2, "action_min_improvement": ACTION_MIN_IMPROVEMENT}}

    # -- hard infrastructure gates (precedence) --
    if not results.get("recollection", {}).get("identical", False):
        return _final(rep, "HOSTILE_RECOLLECTION_NOT_IDENTICAL", "implementation",
                      "base-array parity failed")
    if not results.get("object_schema", {}).get("pass", False):
        return _final(rep, "HOSTILE_OBJECT_SCHEMA_INVALID", "implementation",
                      "object identity audit failed")
    if not results.get("removal", {}).get("pass", False):
        return _final(rep, "HOSTILE_REMOVAL_INVALID", "implementation",
                      "removal audit failed")

    support = results.get("support", {})
    comps = ["enemy", "missile", "joint"]
    comp_pass = {}
    for c in comps:
        sup = support.get(c, {})
        sup_ok = bool(sup.get("pass", False))
        h_out, h_why = _hiddenness_outcome(results.get("hiddenness", {}).get(c))
        a_ok, a_why = _action_pass(results.get("action", {}).get(c))
        f_ok, f_why = _future_pass(results.get("future", {}).get(c))
        hidden_ok = (h_out == "HIDDEN_ENOUGH")
        qualifies = sup_ok and hidden_ok and a_ok and f_ok
        comp_pass[c] = qualifies
        rep["components"][c] = {
            "support_pass": sup_ok, "support_detail": sup,
            "hiddenness": h_out, "hiddenness_why": h_why,
            "action_pass": a_ok, "action_why": a_why,
            "future_pass": f_ok, "future_why": f_why,
            "qualifies": qualifies,
        }

    # -- final outcome precedence --
    enemy_sup = support.get("enemy", {}).get("pass", False)
    missile_sup = support.get("missile", {}).get("pass", False)
    if not enemy_sup and not missile_sup:
        return _final(rep, "HOSTILE_SUPPORT_FAILED", "support",
                      "both enemy and missile components lack support")

    if comp_pass.get("joint"):
        return _final(rep, "HOSTILE_JOINT_QUALIFIED", "qualified", "joint hostile field qualified")
    if comp_pass.get("enemy") and not comp_pass.get("missile"):
        return _final(rep, "ENEMY_ONLY_QUALIFIED", "qualified", "enemy component qualified")
    if comp_pass.get("missile") and not comp_pass.get("enemy"):
        return _final(rep, "MISSILE_ONLY_QUALIFIED", "qualified", "missile component qualified")

    # neither/both-partial: report the dominant scientific blocker on supported components
    sup_comps = [c for c in ["enemy", "missile", "joint"]
                 if support.get(c, {}).get("pass", False)]
    hidden_states = [rep["components"][c]["hiddenness"] for c in sup_comps]
    if "NOT_HIDDEN" in hidden_states:
        return _final(rep, "HOSTILE_NOT_HIDDEN", "scientific",
                      "a supported component is recoverable from the removed state")
    if "INCONCLUSIVE_HIDDENNESS" in hidden_states:
        return _final(rep, "HOSTILE_HIDDENNESS_INCONCLUSIVE", "inconclusive",
                      "hiddenness CI inconclusive for a supported component")
    if not all(rep["components"][c]["action_pass"] for c in sup_comps):
        return _final(rep, "HOSTILE_ACTION_CHANNEL_FAILED", "scientific",
                      "U does not incrementally predict the HF action")
    return _final(rep, "HOSTILE_FUTURE_CHANNEL_FAILED", "scientific",
                  "U does not incrementally predict future player outcomes")


def _final(rep, code, kind, reason):
    assert code in ALLOWED, f"illegal outcome {code}"
    rep["final_outcome"] = code
    rep["failure_kind"] = kind          # qualified | scientific | implementation | support | inconclusive
    rep["reason"] = reason
    return rep


def write_report(rep, out_json, out_md):
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump(rep, open(out_json, "w"), indent=2)
    lines = ["# Seaquest Stage-H0 Hostile-Field Qualification", "",
             f"**Final outcome:** `{rep['final_outcome']}`  ",
             f"**Kind:** {rep['failure_kind']}  ",
             f"**Reason:** {rep['reason']}", "",
             "## Per-component gates", "",
             "| component | support | hiddenness | U→action | U→future | qualifies |",
             "|---|---|---|---|---|---|"]
    for c, v in rep["components"].items():
        lines.append(f"| {c} | {v['support_pass']} | {v['hiddenness']} | "
                     f"{v['action_pass']} | {v['future_pass']} | **{v['qualifies']}** |")
    lines += ["", "## Gate detail", ""]
    for c, v in rep["components"].items():
        lines.append(f"### {c}")
        lines.append(f"- hiddenness: {v['hiddenness']} — {v['hiddenness_why']}")
        lines.append(f"- U→action: {v['action_pass']} — {v['action_why']}")
        lines.append(f"- U→future: {v['future_pass']} — {v['future_why']}")
        lines.append("")
    lines += ["## Interpretation key", "",
              "- **scientific**: the hostile field genuinely fails a channel (not a bug).",
              "- **implementation**: a hard assertion failed (recollection/schema/removal).",
              "- **support**: insufficient active transitions/episodes.",
              "- **inconclusive**: CI does not separate hidden vs not-hidden.", ""]
    open(out_md, "w").write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="path to a results JSON for decide()")
    ap.add_argument("--out-json", default="artifacts/seaquest/hostile_h0/hostile_qualification.json")
    ap.add_argument("--out-md", default="artifacts/seaquest/hostile_h0/hostile_qualification.md")
    args = ap.parse_args()
    results = json.load(open(args.results))
    rep = decide(results)
    write_report(rep, args.out_json, args.out_md)
    print(f"[qualify] final_outcome={rep['final_outcome']} ({rep['failure_kind']})")


if __name__ == "__main__":
    main()
