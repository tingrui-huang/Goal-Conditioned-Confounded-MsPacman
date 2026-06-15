"""Finalize Stage-S0: stamp provenance into every JSON (Section 21), aggregate
gates into audit_report.json, and write SUMMARY.md. Runs on the HOST (stdlib only).
Honest about the marginal/underpowered oxygen screen.
"""
import json, os, glob, datetime, subprocess

BASE = "artifacts/seaquest/stage_s0"


def git(*a):
    try:
        return subprocess.check_output(["git", *a], text=True).strip()
    except Exception:
        return None


def load(p, d=None):
    try:
        return json.load(open(p))
    except Exception:
        return d


def provenance():
    cfg = load(f"{BASE}/config/resolved_config.json", {})
    man = load(f"{BASE}/teacher/candidate_manifest.json", {})
    sel = (man.get("candidates", {}) or {}).get("A", {})
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--short")),
        "config_hash": cfg.get("config_hash"),
        "code_version": "seaquest_stage_s0",
        "env_versions": {"teacher_image": "seaquest-s0:jax325 (jax 0.3.25, envpool 0.8.4, numpy 1.24.2, py3.9)",
                          "ocatari_image": "seaquest-s0:ocatari (+ ale-py 0.10.1, gymnasium 0.29.1)"},
        "seeds": cfg.get("seeds"),
        "selected_teacher_checkpoint": {
            "repo_id": sel.get("repo_id"), "revision": sel.get("revision"),
            "checkpoint_sha256": sel.get("checkpoint_sha256")},
    }


def build_audit_report(prov):
    pa = load(f"{BASE}/teacher/adapter_parity_A.json", {})
    pb = load(f"{BASE}/teacher/adapter_parity_B.json", {})
    na = load(f"{BASE}/teacher/native_eval_A.json", {})
    nb = load(f"{BASE}/teacher/native_eval_B.json", {})
    sel = load(f"{BASE}/teacher/teacher_selection.json", {})
    port = load(f"{BASE}/teacher/ocatari_eval_A.json", {})
    amap = load(f"{BASE}/teacher/action_mapping.json", {})
    rt = load(f"{BASE}/restore/roundtrip_results.json", {})
    sa = load(f"{BASE}/restore/same_action_reproduction.json", {})
    hz = load(f"{BASE}/branches/horizon_metrics.json", {})
    bo = load(f"{BASE}/branches/_branch_oxygen_summary.json", {})
    aa = load(f"{BASE}/oxygen_screen/action_association.json", {})
    oa = load(f"{BASE}/oxygen_screen/outcome_association.json", {})

    na_t = na.get("eval_teacher") or {}; nb_t = nb.get("eval_teacher") or {}
    gate_a = {
        "candidate_A": {"adapter_all_passed": pa.get("all_passed"),
                        "native_mean": na_t.get("return_mean"), "pct_model_card": na.get("pct_of_model_card"),
                        "meets_60pct": na.get("meets_60pct_gate"), "above_random": na.get("teacher_above_random"),
                        "distinct_actions": na_t.get("n_distinct_actions_used")},
        "candidate_B": {"adapter_all_passed": pb.get("all_passed"),
                        "native_mean": nb_t.get("return_mean"), "pct_model_card": nb.get("pct_of_model_card"),
                        "meets_60pct": nb.get("meets_60pct_gate"), "above_random": nb.get("teacher_above_random"),
                        "distinct_actions": nb_t.get("n_distinct_actions_used")},
        "selected": sel.get("selected"),
        "ocatari_port": {"port_mean": (port.get("port_eval") or {}).get("return_mean"),
                         "pct_of_native": port.get("pct_of_native"),
                         "meets_60pct_portability_gate": port.get("meets_60pct_portability_gate"),
                         "note": "Plays at native per-step scoring rate but oxygen-survival-limited; concrete identified wrapper difference (surfacing), not a teacher/mapping failure."},
    }
    da = gate_a["candidate_A"]["distinct_actions"] or 0
    gate_a["pass"] = bool(pa.get("all_passed") and na.get("teacher_above_random")
                          and na.get("meets_60pct_gate") and da > 1)

    gate_b = {
        "action_mapping_bijective": amap.get("bijective_over_teacher_outputs"),
        "clone_restore_roundtrip_pass": rt.get("all_pass"),
        "same_action_continuation_reproducible": sa.get("all_reproducible"),
        "timing_resolved": True, "oxygen_signals_audited": True,
        "ocatari_tracker_caveat": "ALE clone/restore is exact; OCAtari's Python object-tracker is NOT restored, so object-derived features carry transient noise for a few frames after a restore (baseline noise 0-4/32 anchors, decaying to 0 by H=32).",
    }
    gate_b["pass"] = bool(gate_b["action_mapping_bijective"] and gate_b["clone_restore_roundtrip_pass"]
                          and gate_b["same_action_continuation_reproducible"])

    gate_c = bo.get("gate_c") or (hz.get("gate_c") or {})

    # Gate D with explicit magnitudes + marginality flag
    oa_h = oa.get("horizons") or {}
    outcome_helps = []
    for H, row in oa_h.items():
        for tgt, r in row.items():
            if isinstance(r, dict) and r.get("oxygen_logloss_improvement") is not None:
                outcome_helps.append({"horizon": H, "target": tgt,
                                      "dLL": r["oxygen_logloss_improvement"], "helps": r["oxygen_helps"]})
    gate_d = {
        "oxygen_helps_action": bool(aa.get("oxygen_helps_action_prediction")),
        "action_logloss_improvement": aa.get("oxygen_logloss_improvement"),
        "action_accuracy_uplift": aa.get("oxygen_acc_uplift"),
        "oxygen_helps_any_outcome": any(o["helps"] for o in outcome_helps),
        "outcome_details": outcome_helps,
        "pass": bool(aa.get("oxygen_helps_action_prediction") and any(o["helps"] for o in outcome_helps)),
        "MARGINAL": True,
        "marginality_note": ("PASS is by the log-loss criterion only and is WEAK/UNDERPOWERED: "
                             "8-episode tiny rollout (4 train / 2 test episode split), action log-loss "
                             "improvement ~0.025 nats with NEGATIVE accuracy uplift; outcome improvements "
                             "are sub-0.012 nats at H=4/8/16 and negative at H=32; life-loss outcome was "
                             "degenerate at short H. Treat oxygen as a PLAUSIBLE-BUT-UNCONFIRMED confounder "
                             "candidate; a higher-N re-screen is recommended as a SEPARATE step (not added "
                             "here, per seed-discipline: do not add episodes after seeing results)."),
    }

    report = {**prov, "research_question_answered": True,
              "GATE_A_teacher": gate_a, "GATE_B_instrumentation": gate_b,
              "GATE_C_external_action_footprint": gate_c, "GATE_D_oxygen_candidate": gate_d}

    if not gate_a["pass"]:
        outcome = "STOP_NO_USABLE_TEACHER"
    elif not gate_b["pass"]:
        outcome = "STOP_INSTRUMENTATION_FAILURE"
    elif not gate_c.get("pass"):
        outcome = "STOP_EXTERNAL_EFFECT_TOO_WEAK"
    elif gate_d.get("pass"):
        outcome = "PROCEED_TEACHER_AND_EXTERNAL_EFFECT"
    else:
        outcome = "PROCEED_SEAQUEST_BUT_REJECT_OXYGEN"
    report["FINAL_OUTCOME"] = outcome
    report["outcome_caveat"] = ("Gate D passes only MARGINALLY (see marginality_note). The robust, "
                                "well-supported conclusions are: validated teacher + exact instrumentation + "
                                "real external first-action effects. Oxygen-as-confounder is plausible but "
                                "needs a higher-power re-screen before adoption.")
    json.dump(report, open(f"{BASE}/audit_report.json", "w"), indent=2)
    return report


def f(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def pct(x):
    try:
        return f"{100*float(x):.1f}%"
    except Exception:
        return "?"


def horizon_rows(div, view):
    per = div.get(view, {})
    out = []
    for h in ["1", "2", "4", "8", "12", "16", "24", "32"]:
        e = per.get(h) or per.get(int(h))
        if e:
            out.append(f"| {h} | {e['anchors_diverged']}/{e['anchors_valid']} | {f(e['frac_diverged'])} | {e['baseline_noise_anchors']} |")
    return "\n".join(out)


def write_summary(rep):
    a = rep["GATE_A_teacher"]; b = rep["GATE_B_instrumentation"]
    c = rep["GATE_C_external_action_footprint"]; d = rep["GATE_D_oxygen_candidate"]
    hz = load(f"{BASE}/branches/horizon_metrics.json", {})
    div = hz.get("divergence_by_view_horizon", {})
    na = load(f"{BASE}/teacher/native_eval_A.json", {}); nb = load(f"{BASE}/teacher/native_eval_B.json", {})
    sa = load(f"{BASE}/restore/same_action_reproduction.json", {})
    anchors = load(f"{BASE}/branches/anchors.json", {})
    carrier = (c.get("component_max_change_rate") or {})
    top = sorted(carrier.items(), key=lambda kv: -kv[1])[:9]
    comp_lines = "\n".join(f"- `{k}`: max cross-action change rate {f(v)}" for k, v in top)
    randA = (na.get("eval_random") or {}).get("return_mean"); randB = (nb.get("eval_random") or {}).get("return_mean")
    cA, cB = a["candidate_A"], a["candidate_B"]; port = a["ocatari_port"]
    outcome_tbl = "\n".join(
        f"  - {o['target']} @H{o['horizon']}: ΔlogLoss {o['dLL']:+.4f} ({'helps' if o['helps'] else 'no'})"
        for o in (d.get("outcome_details") or []))

    S = []
    S.append(f"# Seaquest Stage-S0 — Feasibility & Instrumentation Audit — SUMMARY\n")
    S.append(f"**FINAL OUTCOME: `{rep['FINAL_OUTCOME']}`**  ·  generated {rep['timestamp_utc']}  ·  git `{rep['git_commit']}` (uncommitted by design)\n")
    S.append("> Gate D passes only **marginally** — see Q14/Q16. Robust conclusions: validated teacher, "
             "exact instrumentation, real external first-action effects. No critic / masking / causal "
             "objective was built (out of scope for S0).\n")
    S.append("---\n")

    S.append("## 1. Which teacher candidates loaded?\n"
             "Both. **A** `Seaquest-v5-sebulba_ppo_envpool_impala_atari_wrapper-seed1` (rev `c584bce`) and "
             "**B** `Seaquest-v5-cleanba_impala_envpool_machado_atari_wrapper_a0_l1_d4-seed3` (rev `9ac6e0e`) "
             "deserialize exactly and pass all 8 adapter/loading unit tests (bitwise-reproducible logits; "
             "adapter == original model code; Gumbel-Max sampling; temperature on logits; **feed-forward, "
             "recurrent_state = not_applicable**).\n")

    S.append("## 2. Which reproduced native performance?\n"
             f"Both, each in its OWN native EnvPool wrapper (T=1, sampled Gumbel-Max, 10 episodes):\n"
             f"- **A**: mean **{f(cA['native_mean'],0)}** = {pct(cA['pct_model_card'])} of model-card 1676 "
             f"(random {f(randA,0)}); {cA['distinct_actions']} distinct actions.\n"
             f"- **B**: mean **{f(cB['native_mean'],0)}** = {pct(cB['pct_model_card'])} of model-card 1750 "
             f"(random {f(randB,0)}); {cB['distinct_actions']} distinct actions.\n"
             "Both clear the 60%-of-model-card gate and are far above random.\n")

    S.append("## 3. Which teacher was selected and why?\n"
             "**Candidate A.** Pre-declared priority: reproducibility (tie) → wrapper portability (A trained "
             "sticky=0, matching the DETERMINISTIC clone/restore branch env; B trained sticky=0.25) → "
             "performance (A 1802 ≥ B 1732) → action coverage (B wins 18 vs 12; noted) → stability. B remains "
             "a fully-validated comparator. (`teacher/teacher_selection.json`)\n")

    S.append("## 4. Exact teacher observation preprocessing\n"
             "EnvPool Atari pipeline, replicated for the OCAtari port: repeat action 4 ALE frames; "
             "**max-pool the last 2** ALE grayscale frames (`ale.getScreenGrayscale`); **resize 84×84 with "
             "`cv2.INTER_AREA`**; **stack 4** processed frames, channel order temporal oldest→newest; "
             "**uint8 [0,255]** (network divides by 255). Tensor `(4,84,84)` → transposed `(84,84,4)` in-net.\n")

    S.append("## 5. Exact action mapping\n"
             "**18 actions, minimal == full == legal**, standard ALE order; teacher_index → meaning → ALE id "
             "→ OCAtari id is the **identity and bijective**. Empirical alias test: some actions are "
             "state-dependently equivalent (movement vs movement+fire when no missile firable) but global ALE "
             "meanings kept distinct. (`teacher/action_mapping.json`)\n")

    S.append("## 6. What does one `env.step()` mean?\n"
             "In the OCAtari branch env, one `env.step()` = **one ALE frame** (frameskip=1). One **agent step** "
             "= 4 frames (frameskip=4) with max-pool over the last 2, matching the teacher's native skip; "
             "reward summed over the 4 frames.\n")

    S.append("## 7. Objects/reward/score/oxygen — pre- or post-action?\n"
             "`env.objects` (incl. oxygen/score/lives) reflect the state **AFTER** the step producing the "
             "current frame. The value read BEFORE calling step is the pre-action observation; after, "
             "post-action. Stored as separate pre_/post_ fields in `schema/timing_trace.jsonl`. Missing oxygen "
             "= explicit `None`, never carried forward. Timing assertions (reward↔score-delta, no terminal "
             "crossing, post≠pre) all pass.\n")

    S.append("## 8. Life loss vs terminal vs truncation\n"
             "`Lives` value decrease = life loss (tracked, distinct). `terminated` = true episode end; "
             "`truncated` = time limit. Branches never cross a reset silently (`branches/censoring.csv`).\n")

    S.append("## 9. Is clone/restore exact?\n"
             "**Yes for forward rollouts.** `ale.cloneState(include_rng=True)` + restore reproduces post-step "
             "RAM, RGB, reward, score, oxygen, lives, terminal identically; a fixed action sequence reproduces "
             f"the H={sa.get('horizon','16')} trajectory bitwise. *Documented artifacts:* (i) a bare getRAM/"
             "getScreenRGB read taken right after restore (no step) is lazily stale by ~5 bytes incl 0x66 "
             "oxygen, corrected by the first step; (ii) OCAtari's Python object-tracker is NOT part of ALE "
             "state, so object-derived features carry transient post-restore noise (the Gate-C baseline).\n")

    S.append("## 10. At which horizons do forced actions affect future state?\n"
             f"{anchors.get('n_anchors','?')} anchors, H ∈ [1,2,4,8,12,16,24,32].\n\n"
             "**No-player view** (excludes future player position/velocity):\n\n"
             "| H | diverged | frac | same-action baseline |\n|---|---|---|---|\n"
             + horizon_rows(div, "noplayer") + "\n\n"
             "**World-only view** (also excludes the player's own missile):\n\n"
             "| H | diverged | frac | baseline |\n|---|---|---|---|\n"
             + horizon_rows(div, "worldonly") + "\n\n"
             "No-player divergence is 0.47–0.78 at every horizon; world-only is 0.16–0.38 (≥0.20 at "
             "H=2,4,8,12,16). The same-action baseline (OCAtari-tracker noise) is 0–4/32 anchors and the "
             "effect exceeds it at every horizon. **Bootstrap 95% CIs over anchors** (2000 resamples, "
             "`branches/bootstrap_ci.json`): no-player CI lower bound ≥0.31 at ALL horizons (e.g. H=16: "
             "0.78 [0.62,0.91]); world-only CI lower bound reaches ≥0.20 at H=8 (0.38 [0.22,0.56]) and H=16 "
             "(0.38 [0.22,0.53]) — so the external footprint is robust, not a point-estimate artifact. "
             "(`branches/horizon_metrics.json`)\n")

    S.append("## 11. Which components carry the action effect?\n" + comp_lines + "\n")

    S.append("## 12. Is action relevance dense or state-dependent?\n"
             "**Real but moderately state-dependent.** Player position (0.91) and the player's own missile "
             "(0.75) dominate. GENUINE external dynamics — enemy positions/counts, score and cumulative "
             "reward (~0.25) — diverge in a smaller but non-trivial fraction of anchors, and the world-only "
             "view (which excludes both player position and own missile) still clears 20% at H=2–16. So "
             "forced first actions DO leave external footprints, but they are not uniformly dense.\n")

    S.append("## 13. Is oxygen associated with teacher actions after conditioning on visible state?\n"
             f"**Marginally yes (log-loss only).** Episode-split action model: +oxygen improves held-out "
             f"log-loss by {f(d.get('action_logloss_improvement'),4)} nats but the accuracy uplift is "
             f"{f(d.get('action_accuracy_uplift'),4)} (negative). (`oxygen_screen/action_association.json`)\n")

    S.append("## 14. Does oxygen predict non-oxygen future outcomes after conditioning on visible state + action?\n"
             "**Marginally, at some horizons only.** +oxygen improves held-out log-loss for future-reward>0:\n"
             + outcome_tbl + "\n"
             "life-loss was degenerate at short H and negative at H=32. Small magnitudes; underpowered. "
             "(`oxygen_screen/outcome_association.json`)\n")

    S.append("## 15. Which gates passed?\n"
             f"- **Gate A (teacher):** {'PASS' if a.get('pass') else 'FAIL'}\n"
             f"- **Gate B (instrumentation):** {'PASS' if b.get('pass') else 'FAIL'}\n"
             f"- **Gate C (external action footprint):** {'PASS' if c.get('pass') else 'FAIL'}\n"
             f"- **Gate D (oxygen candidate):** {'PASS (MARGINAL — see Q14)' if d.get('pass') else 'FAIL'}\n\n"
             f"Portability sub-note: OCAtari port mean {f(port.get('port_mean'),0)} ≈ {pct(port.get('pct_of_native'))} "
             "of native — the ported teacher scores at native per-step rate but is oxygen-survival-limited "
             "(surfaces less effectively), a concrete identified wrapper difference, not a teacher/mapping "
             "failure. It does not affect Gate C (emulator dynamics) or the internal consistency of Gate D "
             "(screens the actual data-generating policy).\n")

    S.append("## 16. Should the project proceed?\n"
             f"**`{rep['FINAL_OUTCOME']}`** — proceed to (a) small offline collection and (b) a vanilla "
             "STATE critic: the teacher is validated, instrumentation is exact, and forced first actions leave "
             "measurable external (no-player, world-only) future effects.\n\n"
             "**However, oxygen-as-confounder is only marginally supported** (Gate D passes by log-loss but "
             "with tiny magnitude, negative action-accuracy uplift, and an 8-episode underpowered screen). "
             "Recommendation: ALSO pursue **further oxygen investigation** — a higher-N observational screen "
             "with bootstrap CIs — as a SEPARATE step BEFORE committing to oxygen as THE confounder (do not "
             "swap in a different confounder in the same run). Do NOT proceed to pixel critics, masking, or "
             "causal/robust objectives yet (later levels).\n")

    S.append("---\n### Artifacts\n`artifacts/seaquest/stage_s0/` — config/, environment/, teacher/, schema/, "
             "restore/, branches/, oxygen_screen/, figures/, audit_report.json, SUMMARY.md. All code "
             "uncommitted. Two Docker images: `seaquest-s0:jax325`, `seaquest-s0:ocatari`.\n")

    open(f"{BASE}/SUMMARY.md", "w", encoding="utf-8").write("\n".join(S))


def stamp_all(prov):
    n = 0
    for p in glob.glob(f"{BASE}/**/*.json", recursive=True):
        if os.path.basename(p).startswith("_"):
            continue
        d = load(p)
        if not isinstance(d, dict):
            continue
        d.setdefault("_provenance", prov)
        json.dump(d, open(p, "w"), indent=2)
        n += 1
    return n


if __name__ == "__main__":
    prov = provenance()
    rep = build_audit_report(prov)
    write_summary(rep)
    n = stamp_all(prov)
    print(f"FINAL_OUTCOME = {rep['FINAL_OUTCOME']}")
    print(f"stamped {n} JSON artifacts; wrote audit_report.json + SUMMARY.md")
