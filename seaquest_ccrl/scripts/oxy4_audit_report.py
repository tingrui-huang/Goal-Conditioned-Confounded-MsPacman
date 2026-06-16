"""Phase 2.1b steps 8-9 — one primary leakage-source outcome + the 9 decision answers.

Thresholds are PREDECLARED (do not change after viewing results). Aggregates the
implementation audit, the V1-V4 visual probes, the P1-P3 proxies and the sanity controls.
If the visual probes are absent (proxy-only run), the outcome is PENDING_VISUAL_PROBES.
"""
import os, json, argparse

# --- predeclared thresholds ---
STRONG_R2 = 0.50            # R2 >= this  => "strongly predictive"
SUBSTANTIAL_DROP_R2 = 0.15  # absolute R2 reduction => "substantially worse"
PROXY_EXPLAINS_FRAC = 0.50  # best-proxy R2 / V4 R2 >= this => proxies explain most of V4
PERMUTE_MAX_R2 = 0.05       # label-permuted probe must stay below this (else pipeline bug)

B = "artifacts/seaquest/oxygen_4frame/leakage/source_audit"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=B)
    args = ap.parse_args()
    impl = json.load(open(f"{args.base}/implementation_audit.json"))
    M = json.load(open(f"{args.base}/audit_probe_metrics.json"))
    vp = M.get("visual_probes", {})

    def r2(name): return vp.get(name, {}).get("regression", {}).get("r2")
    def acc(name): return vp.get(name, {}).get("classification", {}).get("accuracy")
    V1, V2, V3, V4 = (r2(n) for n in ["V1_newest_oxybar_masked", "V2_four_oxybar_masked",
                                      "V3_four_bottomhud_masked", "V4_four_gameplay_crop"])
    VIS = r2("visible_four")
    proxies = M.get("proxy_baselines", {})
    best_proxy = max([p["regression"]["r2"] for p in proxies.values()], default=0.0)
    sanity = M.get("sanity", {})

    have_visual = None not in (V1, V2, V3, V4)
    notes, secondary = [], []

    if impl["outcome"] == "LEAKAGE_IMPLEMENTATION_BUG":
        primary = "LEAKAGE_IMPLEMENTATION_BUG"
    elif sanity.get("label_permutation_r2", 0.0) > PERMUTE_MAX_R2:
        primary = "SANITY_FAILED_INSPECT_PIPELINE"
        notes.append(f"label-permuted R2={sanity['label_permutation_r2']:.3f} > {PERMUTE_MAX_R2}")
    elif not have_visual:
        primary = "PENDING_VISUAL_PROBES"
        notes.append("V1-V4 not present; run oxy4_audit_probes on GPU to finalize.")
    else:
        dV2_V3 = V2 - V3   # leakage from rest of bottom HUD
        dV3_V4 = V3 - V4   # leakage from top HUD
        dV2_V1 = V2 - V1   # temporal-history contribution
        cands = []
        if dV2_V3 >= SUBSTANTIAL_DROP_R2: cands.append(("LEAKAGE_FROM_INCOMPLETE_HUD_MASK", dV2_V3))
        if dV3_V4 >= SUBSTANTIAL_DROP_R2: cands.append(("LEAKAGE_FROM_TOP_HUD", dV3_V4))
        if cands:
            cands.sort(key=lambda x: -x[1])
            primary = cands[0][0]
            secondary += [c[0] for c in cands[1:]]
        elif dV2_V1 >= SUBSTANTIAL_DROP_R2 and V4 >= STRONG_R2:
            primary = "LEAKAGE_FROM_TEMPORAL_DYNAMICS"
        elif V1 >= STRONG_R2 and V4 >= STRONG_R2:
            primary = "LEAKAGE_FROM_STATIC_GAMEPLAY_PROXIES"
        elif V4 and best_proxy / max(V4, 1e-6) >= PROXY_EXPLAINS_FRAC:
            primary = "LEAKAGE_FROM_EPISODE_TIME_OR_PLAYER_POSITION"
        else:
            primary = "LEAKAGE_SOURCE_INCONCLUSIVE"
        if dV2_V1 >= SUBSTANTIAL_DROP_R2 and "TEMPORAL" not in primary:
            secondary.append("temporal_history_contributes")
        if best_proxy / max(V4 or 1e-6, 1e-6) >= PROXY_EXPLAINS_FRAC and "EPISODE_TIME" not in primary:
            secondary.append("episode_time_or_player_position_contributes")

    answers = {
        "1_masked_0901_reproducible": (None if V2 is None else {"V2_masked_r2": V2,
            "reproduces_original_~0.901": bool(V2 is not None and abs(V2 - 0.901) <= 0.10)}),
        "2_single_masked_frame_predicts": (None if V1 is None else {"V1_r2": V1, "strong": V1 >= STRONG_R2}),
        "3_four_frame_history_adds": (None if (V1 is None or V2 is None) else {"V2_minus_V1_r2": V2 - V1}),
        "4_bottom_hud_mask_reduces": (None if (V2 is None or V3 is None) else {"V2_minus_V3_r2": V2 - V3}),
        "5_removing_all_hud_reduces": (None if (V3 is None or V4 is None) else {"V3_minus_V4_r2": V3 - V4,
            "visible_minus_V4_r2": (None if VIS is None else VIS - V4)}),
        "6_timestep_alone_predicts": {"P1_timestep_r2": proxies.get("P1_timestep", {}).get("regression", {}).get("r2")},
        "7_player_position_predicts": {"P2_player_y_r2": proxies.get("P2_player_y", {}).get("regression", {}).get("r2")},
        "8_hidden_partial_or_observed": _hidden_verdict(V2, V4, VIS),
        "9_recommendation": _recommendation(primary),
    }
    report = {
        "primary_outcome": primary, "secondary_causes": secondary, "notes": notes,
        "predeclared_thresholds": {"strong_r2": STRONG_R2, "substantial_drop_r2": SUBSTANTIAL_DROP_R2,
                                   "proxy_explains_frac": PROXY_EXPLAINS_FRAC, "permute_max_r2": PERMUTE_MAX_R2},
        "r2": {"V1_newest_masked": V1, "V2_four_masked": V2, "V3_bottomhud_masked": V3,
               "V4_gameplay_crop": V4, "visible": VIS, "best_proxy": best_proxy},
        "accuracy": {n: acc(n) for n in vp},
        "implementation_outcome": impl["outcome"],
        "sanity": sanity,
        "decision_answers": answers,
    }
    json.dump(report, open(f"{args.base}/leakage_source_report.json", "w"), indent=2)
    print(json.dumps({"primary_outcome": primary, "secondary_causes": secondary,
                      "r2": report["r2"], "notes": notes}, indent=2))
    print(f"WROTE {args.base}/leakage_source_report.json")


def _hidden_verdict(V2, V4, VIS):
    if V2 is None or V4 is None:
        return None
    if V4 >= STRONG_R2:
        return "PARTIALLY_OBSERVED_VIA_GAMEPLAY (oxygen recoverable from gameplay content/motion)"
    if V2 >= STRONG_R2 and V4 < STRONG_R2:
        return "OBSERVED_VIA_HUD_RESIDUAL (recoverability collapses once HUD pixels are removed -> fixable with a stronger mask)"
    return "GENUINELY_HIDDEN (low recoverability from masked pixels)"


def _recommendation(primary):
    if primary in ("LEAKAGE_FROM_INCOMPLETE_HUD_MASK", "LEAKAGE_FROM_TOP_HUD"):
        return "CONTINUE_WITH_STRONGER_OBSERVATION_MASK (extend the mask to the leaking HUD region, re-run leakage)"
    if primary in ("LEAKAGE_FROM_STATIC_GAMEPLAY_PROXIES", "LEAKAGE_FROM_EPISODE_TIME_OR_PLAYER_POSITION"):
        return "RECONSIDER_CONFOUNDER (oxygen is observable from gameplay proxies; masking the bar cannot hide it)"
    if primary == "LEAKAGE_FROM_TEMPORAL_DYNAMICS":
        return "CONTINUE_WITH_CARE (temporal dynamics carry oxygen; document as partial observability)"
    if primary == "LEAKAGE_IMPLEMENTATION_BUG":
        return "STOP_FIX_IMPLEMENTATION"
    if primary == "SANITY_FAILED_INSPECT_PIPELINE":
        return "STOP_INSPECT_PIPELINE"
    return "PENDING (finalize visual probes)"


if __name__ == "__main__":
    main()
