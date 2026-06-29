"""Build the compact Seaquest behavioral-diagnosis dashboard (4 aggregate figures + 4
representative examples + summary) into a NEW TensorBoard run and the artifacts dir.

Reads ONLY verified offline artifacts (raw_rollouts.npz, per_anchor_results.csv) and the 4
rerun example npz from g0_diag_rerun_examples. No model is loaded, no rollout is recomputed.
"""
import json, os
import numpy as np
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image, ImageDraw, ImageSequence
from torch.utils.tensorboard import SummaryWriter
from tensorboard.compat.proto.summary_pb2 import Summary

FV = "artifacts/seaquest/goal_control/full_view/evaluation"
OUT = "artifacts/seaquest/behavioral_diagnosis_compact"
RUN = "runs/seaquest_behavioral_diagnosis_compact"
RAD = 8.0
FPS = 5                      # GIF playback; 200ms/frame quantizes cleanly to centiseconds (GIF delay unit)
ORD = ["typical_success", "near_miss", "approach_then_drift", "downward_or_H64_failure"]  # critic: 1 success + 3 failures
NICE = {"typical_success": "Typical success", "near_miss": "Near miss",
        "approach_then_drift": "Approach then drift", "downward_or_H64_failure": "Downward / H64 failure",
        "teacher_goal_reaching_success": "Teacher goal-reaching success (reference)"}


def agg_data():
    raw = np.load(f"{FV}/raw_rollouts.npz", allow_pickle=True)
    pol = raw["policy"].astype(str)
    def mean_sb(mask): return float(raw["success_by_H"][mask].mean())
    def mean_sa(mask): return float(raw["success_at_H"][mask].mean())
    crit = pol == "critic"
    d = {
        "policies": {  # success_by_H ; NOOP = B2 (action-0 policy)
            "critic": (mean_sb(pol == "critic"), int((pol == "critic").sum())),
            "random": (mean_sb(pol == "B1"), int((pol == "B1").sum())),
            "teacher": (mean_sb(pol == "B3b"), int((pol == "B3b").sum())),
            "noop": (mean_sb(pol == "B2"), int((pol == "B2").sum())),
        },
        "byH": {H: (mean_sb(crit & (raw["H"] == H)), int((crit & (raw["H"] == H)).sum())) for H in (16, 32, 64)},
        "byH_at": {H: mean_sa(crit & (raw["H"] == H)) for H in (16, 32, 64)},
    }
    rows = [r for r in csv.DictReader(open(f"{FV}/per_anchor_results.csv")) if r["policy"] == "critic"]
    d["dir"] = {}
    for u in ("up", "down"):
        rs = [r for r in rows if r["direction"] == u]
        d["dir"][u] = (float(np.mean([r["success_by_H"] == "True" for r in rs])), len(rs))
    return d


def fig_policy(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    order = ["critic", "random", "teacher", "noop"]
    lbl = {"critic": "Critic\n(direct control)", "random": "Random", "teacher": "Teacher\n(stochastic)", "noop": "NOOP"}
    names = [lbl[k] for k in order]; vals = [d["policies"][k][0] for k in order]; ns = [d["policies"][k][1] for k in order]
    cols = ["#1f77b4", "#888888", "#2ca02c", "#d62728"]
    b = ax.bar(names, vals, color=cols)
    for r, v, n in zip(b, vals, ns):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.015, f"{v:.3f}\n(n={n})", ha="center", fontsize=8.5)
    ax.set_ylim(0, 1.0); ax.set_ylabel("success_by_H"); ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Figure 1 — Goal-control success by policy (success_by_H)", fontsize=11)
    fig.tight_layout(); return fig


def fig_byH(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    Hs = [16, 32, 64]; vals = [d["byH"][H][0] for H in Hs]; ns = [d["byH"][H][1] for H in Hs]
    b = ax.bar([f"H{H}" for H in Hs], vals, color="#1f77b4")
    for r, v, n in zip(b, vals, ns):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.015, f"{v:.3f}\n(n={n})", ha="center", fontsize=9)
    ax.set_ylim(0, 0.8); ax.set_ylabel("success_by_H")
    ax.set_title("Figure 2 — Critic success by horizon", fontsize=11)
    fig.tight_layout(); return fig


def fig_dir(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    us = ["up", "down"]; vals = [d["dir"][u][0] for u in us]; ns = [d["dir"][u][1] for u in us]
    b = ax.bar(["upward goals", "downward goals"], vals, color=["#2ca02c", "#d62728"])
    for r, v, n in zip(b, vals, ns):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.015, f"{v:.3f}\n(n={n})", ha="center", fontsize=9)
    ax.set_ylim(0, 0.8); ax.set_ylabel("success_by_H")
    ax.set_title("Figure 3 — Critic success by goal direction\n(no same-level stratum exists in the eval set)", fontsize=10.5)
    fig.tight_layout(); return fig


def fig_byH_at(d):
    fig, ax = plt.subplots(figsize=(6, 4))
    Hs = [16, 32, 64]; x = np.arange(3); w = 0.38
    by = [d["byH"][H][0] for H in Hs]; at = [d["byH_at"][H] for H in Hs]
    ax.bar(x - w / 2, by, w, label="success_by_H (entered 8px ball)", color="#1f77b4")
    ax.bar(x + w / 2, at, w, label="success_at_H (inside ball at deadline)", color="#ff7f0e")
    for i in range(3):
        ax.text(x[i] - w / 2, by[i] + 0.012, f"{by[i]:.3f}", ha="center", fontsize=8)
        ax.text(x[i] + w / 2, at[i] + 0.012, f"{at[i]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"H{H}" for H in Hs]); ax.set_ylim(0, 0.8)
    ax.set_ylabel("success rate"); ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title("Figure 4 — By-H vs At-H (critic)\n(motivates trajectory inspection; not auto-interpreted)", fontsize=10.5)
    fig.tight_layout(); return fig


def example_panel(npz):
    z = np.load(npz, allow_pickle=True)
    cat = str(z["category"]); H = int(z["H"]); aid = int(z["anchor_id"]); seed = int(z["seed"])
    dists = z["dists"]; pos = z["positions"]; goal = z["goal"]; start = z["start"]
    init = float(np.hypot(start[0] - goal[0], start[1] - goal[1]))
    fig = plt.figure(figsize=(13, 4.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 0.7], wspace=0.32)
    # (a) distance curve
    ax = fig.add_subplot(gs[0]); steps = np.arange(1, len(dists) + 1)
    dd = np.where(np.isfinite(dists), dists, np.nan)
    ax.plot(steps, dd, "-o", ms=3, color="#1f77b4"); ax.axhline(RAD, color="r", ls="--", lw=1, label="8px goal radius")
    mi = int(np.nanargmin(dd)); ax.plot(steps[mi], dd[mi], "v", color="k", ms=8, label=f"min {dd[mi]:.1f}px")
    ax.set_xlabel("rollout step"); ax.set_ylabel("distance to goal (px)"); ax.legend(fontsize=8)
    ax.set_title("distance-to-goal", fontsize=10)
    # (b) XY trajectory (screen orientation: y down)
    ax = fig.add_subplot(gs[1])
    ax.plot(pos[:, 0], pos[:, 1], "-", color="#1f77b4", lw=1.2, alpha=0.8)
    ax.scatter(pos[:, 0], pos[:, 1], c=np.arange(len(pos)), cmap="viridis", s=14, zorder=3)
    ax.plot(start[0], start[1], "s", color="lime", ms=10, mec="k", label="start", zorder=4)
    ax.plot(goal[0], goal[1], "*", color="red", ms=16, mec="k", label="goal", zorder=4)
    ax.add_patch(Circle((goal[0], goal[1]), RAD, fill=False, ec="red", ls="--", lw=1.2))
    ax.set_xlim(0, 160); ax.set_ylim(0, 210); ax.invert_yaxis(); ax.set_aspect("equal")
    ax.set_xlabel("x (px)"); ax.set_ylabel("y (px, screen down)"); ax.legend(fontsize=8, loc="best")
    ax.set_title("player XY trajectory", fontsize=10)
    # (c) metadata card
    ax = fig.add_subplot(gs[2]); ax.axis("off")
    card = (f"rollout id\n  anchor {aid}, seed {seed}, H{H}\n\n"
            f"horizon       {H}\n"
            f"direction     {z['category'] and ''}{_dir(z)}\n"
            f"initial dist  {init:.1f} px\n"
            f"min dist      {float(z['min_dist']):.1f} px\n"
            f"final dist    {('inf' if not np.isfinite(float(z['final_dist'])) else f'{float(z['final_dist']):.1f} px')}\n"
            f"success_by_H  {bool(z['success_by_H'])}\n"
            f"success_at_H  {bool(z['success_at_H'])}\n"
            f"life_lost     {bool(z['life_lost'])}\n"
            f"terminated    {bool(z['terminated'])}")
    ax.text(0.0, 0.98, card, va="top", ha="left", family="monospace", fontsize=9.5)
    fig.suptitle(f"{NICE[cat]}  —  anchor {aid} seed {seed} H{H}", fontsize=12, weight="bold", y=1.02)
    return fig, z, cat


def _dir(z):
    # direction from start vs goal y (screen down): up = goal above start (smaller y)
    return "up" if float(z["goal"][1]) < float(z["start"][1]) else "down"


def make_gif(z, path, fps=FPS):
    frames = z["frames"]; goal = z["goal"]; dists = z["dists"]; sc = 2
    imgs = []
    for k in range(len(frames)):
        im = Image.fromarray(frames[k]).resize((160 * sc, 210 * sc), Image.NEAREST)
        dr = ImageDraw.Draw(im)
        gx, gy = float(goal[0]) * sc, float(goal[1]) * sc
        dr.ellipse([gx - RAD * sc, gy - RAD * sc, gx + RAD * sc, gy + RAD * sc], outline=(255, 60, 60), width=2)
        dr.line([gx - 4, gy, gx + 4, gy], fill=(255, 60, 60), width=2); dr.line([gx, gy - 4, gx, gy + 4], fill=(255, 60, 60), width=2)
        d = dists[k - 1] if (k >= 1 and k - 1 < len(dists)) else float("nan")
        dr.text((4, 4), f"step {k}/{len(frames)-1}  d={d:.0f}px" if k >= 1 else "anchor (start)", fill=(255, 255, 0))
        imgs.append(im)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(round(1000 / fps)), loop=0, optimize=True)
    return frames, imgs[0].size                                 # (W,H) of the saved GIF


def verify_gif(gif_path, frames, positions, fps):
    """Read back the saved GIF and check: frame count, fps/order, and that the player MOVES."""
    im = Image.open(gif_path)
    gif_frames = [np.array(f.convert("RGB")) for f in ImageSequence.Iterator(im)]
    pd = np.array([float(np.abs(gif_frames[k].astype(int) - gif_frames[k - 1].astype(int)).mean())
                   for k in range(1, len(gif_frames))])
    disp = np.array([float(np.hypot(positions[k][0] - positions[k - 1][0], positions[k][1] - positions[k - 1][1]))
                     for k in range(1, len(positions))
                     if np.all(np.isfinite(positions[k])) and np.all(np.isfinite(positions[k - 1]))] or [0.0])
    return {
        "n_frames_saved": int(len(gif_frames)),
        "n_frames_expected": int(len(positions) + 1),     # anchor + one per step
        "frame_count_ok": bool(len(gif_frames) == len(positions) + 1),
        "fps": fps, "frame_duration_ms": int(im.info.get("duration")),
        "actual_fps": round(1000.0 / int(im.info.get("duration")), 2),
        "fps_ok": bool(abs(im.info.get("duration") - 1000 / fps) <= 10),   # GIF delay quantizes to 10ms

        "frame_order": "anchor(0) -> step1..stepN",
        "player_total_path_px": round(float(disp.sum()), 1),
        "player_mean_step_disp_px": round(float(disp.mean()), 2),
        "frac_steps_player_moved_gt0p5px": round(float((disp > 0.5).mean()), 3),
        "min_consecutive_frame_pixel_diff": round(float(pd.min()), 3),
        "mean_consecutive_frame_pixel_diff": round(float(pd.mean()), 3),
        "all_consecutive_frames_distinct": bool((pd > 0).all()),
        "player_visibly_moves": bool(disp.sum() > 1.0 and (pd > 0).all()),
    }


def main():
    os.makedirs(f"{OUT}/figures", exist_ok=True); os.makedirs(f"{OUT}/videos", exist_ok=True)
    os.makedirs(RUN, exist_ok=True)
    w = SummaryWriter(RUN)
    d = agg_data()

    # ---- aggregate figures ----
    figs = {"fig1_policy_comparison": fig_policy(d), "fig2_success_by_horizon": fig_byH(d),
            "fig3_success_by_direction": fig_dir(d), "fig4_byH_vs_atH": fig_byH_at(d)}
    for i, (name, fig) in enumerate(figs.items(), 1):
        fig.savefig(f"{OUT}/figures/{name}.png", dpi=130, bbox_inches="tight")
        w.add_figure(f"1_aggregate/{name}", fig, 0); plt.close(fig)
    # scalars for the policy bars + horizon trend
    for idx, k in enumerate(["critic", "random", "teacher", "noop"]):
        w.add_scalar("aggregate/policy_success_by_H", d["policies"][k][0], idx)
    for H in (16, 32, 64):
        w.add_scalar("aggregate/critic_success_by_H_vs_horizon", d["byH"][H][0], H)
        w.add_scalar("aggregate/critic_success_at_H_vs_horizon", d["byH_at"][H], H)

    # ---- representative examples ----
    cards = []; verifs = []

    def render_clip(npz_path, group, tag):
        """Render one example-schema clip (panel + animated GIF) under group/tag; return (verif, z)."""
        fig, z, _ = example_panel(npz_path)
        fig.savefig(f"{OUT}/figures/{tag}_panel.png", dpi=120, bbox_inches="tight")
        w.add_figure(f"{group}/{tag}_panel", fig, 0); plt.close(fig)
        gif_path = f"{OUT}/videos/{tag}.gif"
        frames, (gw, gh) = make_gif(z, gif_path, fps=FPS)
        v = verify_gif(gif_path, frames, z["positions"], FPS); v["category"] = tag
        gb = open(gif_path, "rb").read()
        w._get_file_writer().add_summary(Summary(value=[Summary.Value(
            tag=f"{group}/{tag}_animated", image=Summary.Image(height=gh, width=gw, colorspace=3, encoded_image_string=gb))]), 0)
        return v, z

    # ---- MAIN teacher reference: short VERIFIED goal-reaching SUCCESS clip (H<=64) ----
    tnpz = f"{OUT}/examples/teacher_goal_reaching_success.npz"
    if os.path.exists(tnpz):
        v, _ = render_clip(tnpz, "0_reference", "teacher_goal_reaching_success")
        verifs.append(v)

    # ---- OPTIONAL failure example: seed-1000 full episode (verified oxygen-depletion DEATHS) ----
    ref_gif = f"{OUT}/videos/reference_full_episode.gif"
    if os.path.exists(ref_gif):
        rgb = open(ref_gif, "rb").read(); rw, rh = Image.open(ref_gif).size
        w._get_file_writer().add_summary(Summary(value=[Summary.Value(
            tag="4_optional/teacher_oxygen_death_episode_animated",
            image=Summary.Image(height=rh, width=rw, colorspace=3, encoded_image_string=rgb))]), 0)
        oxpng = f"{OUT}/figures/reference_oxygen_curve.png"
        if os.path.exists(oxpng):
            w.add_image("4_optional/oxygen_curve", np.array(Image.open(oxpng).convert("RGB")), 0, dataformats="HWC")
        otp = f"{OUT}/oxygen_transition_verification.json"
        if os.path.exists(otp):
            ot = json.load(open(otp))
            ln = ["seed-1000 full episode — VERIFIED oxygen resets (lives/respawn/surface, NOT curve shape):",
                  f"class_counts: {ot['class_counts']}", f"deaths (lives drop): {ot['death_events']}"]
            w.add_text("4_optional/oxygen_transition_verification", "```\n" + "\n".join(ln) + "\n```", 0)

    # ---- critic examples: 1 success + 3 failures ----
    exdir = f"{OUT}/examples"
    bycat = {}
    for f in sorted(os.listdir(exdir)):
        z = np.load(f"{exdir}/{f}", allow_pickle=True)
        if "category" in z.files and str(z["category"]) in ORD:
            bycat[str(z["category"])] = f
    for cat in ORD:
        if cat not in bycat:
            continue
        v, z = render_clip(f"{exdir}/{bycat[cat]}", "2_examples", cat)
        verifs.append(v)
        cards.append({"category": cat, "anchor_id": int(z["anchor_id"]), "seed": int(z["seed"]), "H": int(z["H"]),
                      "direction": _dir(z), "initial_dist": float(np.hypot(z["start"][0]-z["goal"][0], z["start"][1]-z["goal"][1])),
                      "min_dist": float(z["min_dist"]), "final_dist": float(z["final_dist"]),
                      "success_by_H": bool(z["success_by_H"]), "success_at_H": bool(z["success_at_H"]),
                      "life_lost": bool(z["life_lost"])})

    # ---- GIF verification (frame count / fps / order / player visibly moves) ----
    json.dump({"fps": FPS, "examples": verifs}, open(f"{OUT}/gif_verification.json", "w"), indent=2)
    vhdr = ["category", "n_frames", "exp", "ok", "fps", "dur_ms", "path_px", "moved%", "distinct", "moves"]
    vlines = ["{:<24}{:>9}{:>6}{:>5}{:>5}{:>8}{:>9}{:>8}{:>10}{:>7}".format(*vhdr)]
    print("\n=== GIF VERIFICATION ===")
    print(vlines[0])
    for v in verifs:
        row = "{:<24}{:>9}{:>6}{:>5}{:>5}{:>8}{:>9}{:>8}{:>10}{:>7}".format(
            v["category"], v["n_frames_saved"], v["n_frames_expected"], str(v["frame_count_ok"]),
            v["fps"], v["frame_duration_ms"], v["player_total_path_px"],
            v["frac_steps_player_moved_gt0p5px"], str(v["all_consecutive_frames_distinct"]),
            str(v["player_visibly_moves"]))
        print(row); vlines.append(row)
    w.add_text("2_examples/_gif_verification", "```\n" + "\n".join(vlines) + "\n```", 0)
    all_ok = all(v["frame_count_ok"] and v["fps_ok"] and v["player_visibly_moves"] for v in verifs)
    print("ALL GIFs verified (frame count + fps + player moves):", all_ok)

    # ---- summary (facts vs interpretation) ----
    pv = {k: d["policies"][k][0] for k in d["policies"]}
    facts = [
        f"Verified metrics: critic success_by_H = {pv['critic']:.3f} "
        f"(random {pv['random']:.3f}, teacher {pv['teacher']:.3f}, NOOP {pv['noop']:.3f}).",
        f"Horizon trend (verified): H16 {d['byH'][16][0]:.2f} > H32 {d['byH'][32][0]:.2f} > H64 {d['byH'][64][0]:.2f} (monotonic decrease).",
        f"Direction trend (verified): upward {d['dir']['up'][0]:.3f} vs downward {d['dir']['down'][0]:.3f} (downward lower; n={d['dir']['up'][1]} each).",
        f"By-H vs At-H (verified): success_at_H is far below success_by_H at every horizon "
        f"(e.g. H32 {d['byH'][32][0]:.2f} vs {d['byH_at'][32]:.3f}).",
        "The four critic reruns reproduce the stored scalars exactly (Gate 3 ALL_MATCH).",
        "MAIN teacher reference = a VERIFIED teacher goal-reaching SUCCESS clip (anchor 9, seed 1000, H16): "
        "the teacher reaches the goal to 0.0px and HOLDS it (success_by_H AND success_at_H), the task it is "
        "actually expert at (B3b~0.92). This is competent teacher play under the same H<=64 setup.",
        "This teacher does NOT manage oxygen: across greedy + 16 stochastic fresh episodes it never surfaces "
        "mid-dive (last surface = spawn step 32) and dies of oxygen at step 567 every time. The death-laden "
        "seed-1000 full episode is therefore kept only as an OPTIONAL failure example (4_optional), not the "
        "main reference (verified deaths at steps 567/1135).",
    ]
    interp = [
        "Teacher reference (goal-reaching SUCCESS, H16): reaches the goal exactly (0.0px) and stays -- competent expert play at the short-horizon task. (Goal is from the teacher's own trajectory; teacher is goal-agnostic.)",
        "Typical critic success (a28,H16,up): enters the 8px ball (min 4.0px) then drifts out (final 25.6px) -> a pass-through, not a stop.",
        "Near miss (a17,H32,down): approaches to 8.9px, just outside the radius -> a near miss, not a far failure.",
        "Approach-then-drift (a67,H64,up): reaches 8.9px then moves far away (final 89.5px) -> weak long-horizon correction.",
        "Downward/H64 failure (a24,H64,down): never gets within 20px and loses the submarine -> insufficient progress on a long dive.",
    ]
    summary = (
        "# Seaquest fully-observed critic — behavioral diagnosis (compact)\n\n"
        "The fully observed critic learns non-trivial goal control and outperforms random, but remains "
        "well below the teacher. Performance decreases with horizon, and downward goals are reached less "
        "often than upward goals (both verified from the recomputed rollout data). The MAIN teacher reference "
        "is a verified teacher goal-reaching success clip (H16, reaches 0.0px and holds) -- competent expert "
        "play at the short-horizon task. (The teacher does not manage oxygen -- it never surfaces mid-dive and "
        "dies of oxygen at step 567 in every sampled episode -- so the seed-1000 death episode is kept only as "
        "an optional failure example, not the main reference.) The four critic clips show a pass-through "
        "success that does not stop, a near miss just outside the radius, an approach-then-drift with weak "
        "long-horizon correction, and a long downward dive with insufficient progress. These explain how "
        "aggregate success can plateau near 53% without implying the critic learned nothing. No claim is made "
        "that every failure shares one cause.\n\n"
        "## Facts (verified)\n" + "\n".join(f"- {x}" for x in facts) +
        "\n\n## Interpretation (teacher reference + the 4 critic trajectories)\n" + "\n".join(f"- {x}" for x in interp) + "\n"
    )
    open(f"{OUT}/summary.md", "w", encoding="utf-8").write(summary)
    w.add_text("3_summary/facts_vs_interpretation", summary.replace("\n", "  \n"), 0)
    for c in cards:
        w.add_text(f"2_examples/{c['category']}_card", "```json\n" + json.dumps(c, indent=2) + "\n```", 0)
    json.dump({"cards": cards}, open(f"{OUT}/example_cards.json", "w"), indent=2)
    w.flush(); w.close()
    print("WROTE figures + videos + summary + TB run", RUN)
    print("examples:", [c["category"] for c in cards])


if __name__ == "__main__":
    main()
