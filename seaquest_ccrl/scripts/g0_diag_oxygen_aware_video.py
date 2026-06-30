"""Annotated gameplay video for the OxygenAwareTeacher wrapper (surface_trigger=25, refilled=58),
for VISUAL verification before any data re-collection.

Produces, in artifacts/seaquest/oxygen_aware_teacher_video/:
  * wrapped_annotated.gif    - the oxygen-aware wrapped teacher (>= 2 complete oxygen cycles)
  * original_annotated.gif   - the ORIGINAL HF teacher on the SAME seed (reference; never surfaces)
  * side_by_side.gif         - wrapped (left) vs original (right), aligned by timestep
  * wrapped_log.csv / original_log.csv  - raw per-step logs
  * summary.json             - cycle boundaries, control-switch steps, deaths, verification stats

Each frame is annotated with: timestep, oxygen, lives, player_y, surfacing-mode flag, and BOTH the
teacher's would-be action and the actually-emitted action. Control switches (HF teacher <-> surfacing
override) are flashed as a banner. Both runs use GREEDY (deterministic) actions so that, until the
first oxygen override, the two trajectories are pixel-identical -- the strongest visual proof that the
wrapper changes ONLY the oxygen response.

Does NOT collect or modify any dataset.
"""
import csv
import json
import os
import sys
from unittest.mock import MagicMock

for _m in ("envpool", "gym"):
    sys.modules.setdefault(_m, MagicMock())            # teacher source imports these (unused here)
import numpy as np
import ocatari.ram.seaquest as _sq
_orig = _sq._detect_objects_ram
_sq._detect_objects_ram = lambda o, r, h: _orig(o, np.asarray(r, np.int64), h)  # uint8->int64 (numpy ver)
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

from seaquest_ccrl.scripts.g0_closed_loop_eval import SeaquestPort, TEACHER_CKPT, TEACHER_SRC
from seaquest_stage_s0.teacher_adapter import CleanRLSeaquestTeacher
from seaquest_ccrl.policies.oxygen_aware_teacher import OxygenAwareTeacher

import argparse

OUT = "artifacts/seaquest/oxygen_aware_teacher_video"
SEED = 3000
SURFACE_TRIGGER = 20
REFILLED = 58
SURFACE_ACTION = 10            # UPFIRE
MAX_STEPS = 1100               # hard cap
TARGET_CYCLES = 2              # stop wrapped run a bit after this many completed surfacing cycles
SCALE = 3                      # game-frame upscale for legibility
GIF_STEP = 2                   # subsample frames into the GIF (log keeps every step)
GIF_FPS = 12

ACTION_NAMES = {0: "NOOP", 1: "FIRE", 2: "UP", 3: "RIGHT", 4: "LEFT", 5: "DOWN",
                6: "UPRIGHT", 7: "UPLEFT", 8: "DOWNRIGHT", 9: "DOWNLEFT", 10: "UPFIRE",
                11: "RIGHTFIRE", 12: "LEFTFIRE", 13: "DOWNFIRE", 14: "UPRIGHTFIRE",
                15: "UPLEFTFIRE", 16: "DOWNRIGHTFIRE", 17: "DOWNLEFTFIRE"}


def _font(sz):
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), sz)


F_HDR = None
F_BIG = None


def aname(a):
    return f"{ACTION_NAMES.get(int(a), '?')}({int(a)})"


def annotate(frame_rgb, info, switch=None):
    """Upscale the game frame and draw a header bar with the per-step annotations.
    `info` keys: t, oxy, lives, py, surfacing, teacher_a, emitted_a, title.
    `switch` (or None): banner string drawn over the frame on control-switch steps.
    """
    h, w, _ = frame_rgb.shape
    game = Image.fromarray(frame_rgb).resize((w * SCALE, h * SCALE), Image.NEAREST)
    gw, gh = game.size
    hdr_h = 84
    canvas = Image.new("RGB", (gw, gh + hdr_h), (12, 12, 18))
    canvas.paste(game, (0, hdr_h))
    d = ImageDraw.Draw(canvas)

    surf = info["surfacing"]
    mode_col = (255, 110, 70) if surf else (90, 200, 255)
    mode_txt = "SURFACING override" if surf else "HF teacher"
    oxy = info["oxy"]
    oxy_col = (255, 80, 80) if (oxy is not None and 0 <= oxy < SURFACE_TRIGGER) else (200, 230, 200)
    py = info["py"]
    py_s = "n/a" if py is None or (isinstance(py, float) and np.isnan(py)) else f"{py:.0f}"
    oxy_s = "n/a" if oxy is None or oxy < 0 else f"{oxy:.1f}"

    d.text((6, 3), info["title"], font=F_HDR, fill=(230, 230, 120))
    d.text((6, 22), f"t={info['t']:<4d}", font=F_HDR, fill=(220, 220, 220))
    d.text((86, 22), f"O2={oxy_s}", font=F_HDR, fill=oxy_col)
    d.text((196, 22), f"lives={info['lives']}", font=F_HDR, fill=(220, 220, 220))
    d.text((300, 22), f"player_y={py_s}", font=F_HDR, fill=(220, 220, 220))
    d.text((6, 41), "MODE:", font=F_HDR, fill=(180, 180, 180))
    d.text((58, 41), mode_txt, font=F_HDR, fill=mode_col)
    # teacher vs emitted action
    ta, ea = info["teacher_a"], info["emitted_a"]
    d.text((6, 60), f"teacher={aname(ta)}", font=F_HDR, fill=(160, 200, 160))
    if surf:
        d.text((210, 60), f"-> OVERRIDE emitted={aname(ea)}", font=F_HDR, fill=(255, 150, 90))
    else:
        d.text((210, 60), f"emitted={aname(ea)}  (= teacher)", font=F_HDR, fill=(160, 200, 160))

    if switch:
        bw = gw
        d.rectangle([0, hdr_h, bw, hdr_h + 26], fill=(0, 0, 0))
        col = (255, 140, 60) if "-> SURFACING" in switch else (90, 210, 255)
        d.text((10, hdr_h + 4), switch, font=F_BIG, fill=col)
    return canvas


def rollout(teacher, wrap, seed, max_steps, stop_after_cycles=None, prime_steps=0):
    """Run a greedy rollout. wrap=None -> original teacher. Returns dict with frames + per-step log.
    `prime_steps`: number of initial teacher-greedy steps to run WITHOUT capture (skips the
    reset transient where the oxygen RAM reads spuriously low before the bar fills)."""
    p = SeaquestPort(sticky=0.0, full_action_space=True, seed=seed)
    p.reset(seed=seed, noop_max=0)
    if wrap is not None:
        wrap.reset()
    for _ in range(prime_steps):                       # un-captured warmup so oxygen reaches full
        p.agent_step(int(teacher.greedy_action(p.teacher_obs())[0]))
    frames, log = [], []
    prev_surf = False
    prev_lives = None
    cycles = 0
    switches = []
    deaths = []
    for s in range(max_steps):
        f = p.features()
        oxy = f.get("oxygen")
        oxy = -1.0 if oxy is None else float(oxy)
        lives = f.get("lives")
        py = f.get("player_y")
        obs = p.teacher_obs()
        teacher_a = int(teacher.greedy_action(obs)[0])
        if wrap is None:
            emitted_a, surfacing = teacher_a, False
        else:
            emitted_a, surfacing = wrap.act(obs, oxy, py, mode="greedy")
        # control switch?
        switch = None
        if wrap is not None and surfacing != prev_surf:
            if surfacing:
                switch = ">>> SWITCH: HF teacher -> SURFACING override <<<"
            else:
                switch = "<<< SWITCH: SURFACING override -> HF teacher >>>"
                cycles += 1
            switches.append({"t": s, "to": "surfacing" if surfacing else "teacher"})
        # death?
        if prev_lives is not None and lives is not None and lives < prev_lives:
            deaths.append({"t": s, "lives_after": int(lives), "surfacing": bool(prev_surf)})
        prev_lives = lives if lives is not None else prev_lives

        frames.append(np.asarray(p.ale.getScreenRGB(), np.uint8).copy())
        log.append({"t": s, "oxygen": round(oxy, 2) if oxy >= 0 else None, "lives": lives,
                    "player_y": None if py is None else float(py), "surfacing": bool(surfacing),
                    "teacher_action": teacher_a, "teacher_action_name": ACTION_NAMES.get(teacher_a),
                    "emitted_action": emitted_a, "emitted_action_name": ACTION_NAMES.get(emitted_a),
                    "overridden": bool(surfacing and emitted_a != teacher_a), "switch": switch})
        prev_surf = surfacing
        p.agent_step(emitted_a)
        if stop_after_cycles is not None and cycles >= stop_after_cycles and s > 30:
            # play a little past the last refill, then stop
            if s - switches[-1]["t"] > 35:
                break
    return {"frames": frames, "log": log, "switches": switches, "deaths": deaths,
            "cycles": cycles, "n": len(frames)}


def build_gif(roll, title, path):
    sw_by_t = {s["t"]: (">>> SWITCH: HF teacher -> SURFACING override <<<" if s["to"] == "surfacing"
                        else "<<< SWITCH: SURFACING override -> HF teacher >>>") for s in roll["switches"]}
    imgs = []
    for i in range(0, roll["n"], GIF_STEP):
        r = roll["log"][i]
        info = {"t": r["t"], "oxy": r["oxygen"], "lives": r["lives"], "py": r["player_y"],
                "surfacing": r["surfacing"], "teacher_a": r["teacher_action"],
                "emitted_a": r["emitted_action"], "title": title}
        # show a switch banner on the switch frame and a couple frames around it
        sw = None
        for dt in (0, -1, 1, -2, 2):
            if (r["t"] + dt) in sw_by_t:
                sw = sw_by_t[r["t"] + dt]; break
        imgs.append(annotate(roll["frames"][i], info, switch=sw))
    dur = int(round(1000.0 / GIF_FPS / 10.0)) * 10
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=dur, loop=0, optimize=True)
    return imgs, dur


def build_side_by_side(roll_w, imgs_w, roll_o, imgs_o, path, dur):
    n = min(len(imgs_w), len(imgs_o))
    gap = 14
    out = []
    for i in range(n):
        a, b = imgs_w[i], imgs_o[i]
        h = max(a.height, b.height)
        canvas = Image.new("RGB", (a.width + gap + b.width, h + 22), (8, 8, 12))
        d = ImageDraw.Draw(canvas)
        d.text((6, 3), "WRAPPED (oxygen-aware)", font=F_HDR, fill=(255, 150, 90))
        d.text((a.width + gap + 6, 3), "ORIGINAL HF teacher", font=F_HDR, fill=(120, 200, 255))
        canvas.paste(a, (0, 22)); canvas.paste(b, (a.width + gap, 22))
        d.rectangle([a.width + gap // 2 - 1, 22, a.width + gap // 2 + 1, h + 22], fill=(60, 60, 70))
        out.append(canvas)
    out[0].save(path, save_all=True, append_images=out[1:], duration=dur, loop=0, optimize=True)


def write_log(roll, path):
    cols = ["t", "oxygen", "lives", "player_y", "surfacing", "teacher_action", "teacher_action_name",
            "emitted_action", "emitted_action_name", "overridden", "switch"]
    with open(path, "w", newline="") as fh:
        wcsv = csv.DictWriter(fh, fieldnames=cols); wcsv.writeheader()
        for r in roll["log"]:
            wcsv.writerow(r)


def render_from_traj(npz_path, out_path, title, terminal_cause="", step=2, fps=12):
    """Render an annotated GIF directly from a COLLECTED first-life trajectory (collect_corpus output):
    shows the oxygen-aware surfacing within the single life and marks the terminal (first life loss)."""
    z = np.load(npz_path)
    fr = z["frames"]; oxy = z["oxygen"]; lv = z["lives"]; py = z["player_pos"][:, 1]
    surf = z["surfacing"]; act = z["actions"]; tact = z["teacher_actions"]; n = len(act)
    sw = {}; prev = False
    for t in range(n):
        if bool(surf[t]) != prev:
            sw[t] = (">>> SWITCH: HF teacher -> SURFACING override <<<" if surf[t]
                     else "<<< SWITCH: SURFACING override -> HF teacher >>>")
            prev = bool(surf[t])
    term_banner = f">>> TERMINAL: first life loss (cause: {terminal_cause}) <<<"
    imgs = []
    idxs = list(range(0, n, step))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)                       # always include the exact terminal frame
    for i in idxs:
        info = {"t": int(i), "oxy": float(oxy[i]), "lives": int(lv[i]), "py": float(py[i]),
                "surfacing": bool(surf[i]), "teacher_a": int(tact[i]), "emitted_a": int(act[i]), "title": title}
        s = sw.get(i) or sw.get(i - 1) or sw.get(i + 1)
        if i >= n - 1 - step:
            s = term_banner
        imgs.append(annotate(fr[i], info, switch=s))
    dur = int(round(1000.0 / fps / 10.0)) * 10
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=dur, loop=0, optimize=True)
    print(f"  rendered {len(imgs)} frames (life={n} steps, terminal={terminal_cause}) -> {out_path}")
    return len(imgs)


def main():
    global F_HDR, F_BIG, OUT, SEED, SURFACE_TRIGGER, REFILLED, SURFACE_ACTION
    ap = argparse.ArgumentParser()
    ap.add_argument("--trigger", type=int, default=SURFACE_TRIGGER)
    ap.add_argument("--refilled", type=int, default=REFILLED)
    ap.add_argument("--surface-action", type=int, default=SURFACE_ACTION)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=None, help="output dir (default: ..._video_t<trigger>)")
    ap.add_argument("--from-traj", default=None, help="render an annotated GIF from a collected first-life traj npz")
    ap.add_argument("--terminal", default="", help="terminal cause label for --from-traj")
    ap.add_argument("--out-gif", default=None, help="output gif path for --from-traj")
    ap.add_argument("--title", default="OXYGEN-AWARE HF teacher (first-life)")
    a = ap.parse_args()
    if a.from_traj:
        F_HDR = _font(14); F_BIG = _font(15)
        os.makedirs(os.path.dirname(a.out_gif), exist_ok=True)
        render_from_traj(a.from_traj, a.out_gif, a.title, terminal_cause=a.terminal)
        return
    SURFACE_TRIGGER, REFILLED, SURFACE_ACTION, SEED = a.trigger, a.refilled, a.surface_action, a.seed
    OUT = a.out or f"{OUT}_t{SURFACE_TRIGGER}"
    F_HDR = _font(14); F_BIG = _font(15)
    os.makedirs(OUT, exist_ok=True)
    teacher = CleanRLSeaquestTeacher(TEACHER_CKPT, TEACHER_SRC, mod_name="cleanrl_src_A")
    wrap = OxygenAwareTeacher(teacher, surface_trigger=SURFACE_TRIGGER, refilled=REFILLED,
                              surface_action=SURFACE_ACTION)

    # determine how many startup steps until the oxygen bar first reads full (skip reset transient)
    pp = SeaquestPort(sticky=0.0, full_action_space=True, seed=SEED); pp.reset(seed=SEED, noop_max=0)
    prime = 0
    for _ in range(20):
        ox = pp.features().get("oxygen")
        if ox is not None and ox >= REFILLED:
            break
        pp.agent_step(int(teacher.greedy_action(pp.teacher_obs())[0])); prime += 1
    print(f"[prime] skipping {prime} startup steps until oxygen first reads >= {REFILLED}")

    print(f"[wrapped] rolling out (seed={SEED}, trigger={SURFACE_TRIGGER}, refilled={REFILLED}, "
          f"surface_action={ACTION_NAMES[SURFACE_ACTION]}) ...")
    roll_w = rollout(teacher, wrap, SEED, MAX_STEPS, stop_after_cycles=TARGET_CYCLES, prime_steps=prime)
    n = roll_w["n"]
    print(f"  wrapped: {n} steps, completed surfacing cycles={roll_w['cycles']}, "
          f"switches={len(roll_w['switches'])}, deaths={len(roll_w['deaths'])}")

    print(f"[original] rolling out same seed for {n} steps ...")
    roll_o = rollout(teacher, None, SEED, n, stop_after_cycles=None, prime_steps=prime)

    print("[render] wrapped_annotated.gif ...")
    imgs_w, dur = build_gif(roll_w, "OXYGEN-AWARE WRAPPED HF teacher", f"{OUT}/wrapped_annotated.gif")
    print("[render] original_annotated.gif ...")
    imgs_o, _ = build_gif(roll_o, "ORIGINAL HF teacher (reference)", f"{OUT}/original_annotated.gif")
    print("[render] side_by_side.gif ...")
    build_side_by_side(roll_w, imgs_w, roll_o, imgs_o, f"{OUT}/side_by_side.gif", dur)

    write_log(roll_w, f"{OUT}/wrapped_log.csv")
    write_log(roll_o, f"{OUT}/original_log.csv")

    # divergence: first step where wrapped vs original emitted action differs (== first override)
    first_div = next((r_w["t"] for r_w, r_o in zip(roll_w["log"], roll_o["log"])
                      if r_w["emitted_action"] != r_o["emitted_action"]), None)
    ox_w = np.array([r["oxygen"] if r["oxygen"] is not None else np.nan for r in roll_w["log"]])
    ox_o = np.array([r["oxygen"] if r["oxygen"] is not None else np.nan for r in roll_o["log"]])
    summary = {
        "seed": SEED, "surface_trigger": SURFACE_TRIGGER, "refilled": REFILLED,
        "surface_action": ACTION_NAMES[SURFACE_ACTION], "n_steps_wrapped": n, "gif_fps": GIF_FPS,
        "gif_frame_step": GIF_STEP, "scale": SCALE,
        "wrapped_completed_oxygen_cycles": roll_w["cycles"],
        "control_switches": roll_w["switches"], "deaths_wrapped": roll_w["deaths"],
        "deaths_original": roll_o["deaths"],
        "first_divergence_step (== first oxygen override)": first_div,
        "identical_until_first_override": bool(first_div is not None
            and all(roll_w["log"][i]["emitted_action"] == roll_o["log"][i]["emitted_action"]
                    for i in range(first_div))),
        "wrapped_oxygen_min_max": [float(np.nanmin(ox_w)), float(np.nanmax(ox_w))],
        "original_oxygen_min_max": [float(np.nanmin(ox_o)), float(np.nanmax(ox_o))],
        "wrapped_refill_events": int(roll_w["cycles"]),
        "files": ["wrapped_annotated.gif", "original_annotated.gif", "side_by_side.gif",
                  "wrapped_log.csv", "original_log.csv"],
    }
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2)

    print("\n=== oxygen-aware teacher video ===")
    print(f"  wrapped: {n} steps, {roll_w['cycles']} oxygen cycle(s), {len(roll_w['switches'])} control switches")
    print(f"  switches (t -> mode): " + ", ".join(f"{s['t']}->{s['to']}" for s in roll_w["switches"]))
    print(f"  wrapped O2 range [{summary['wrapped_oxygen_min_max'][0]:.1f},{summary['wrapped_oxygen_min_max'][1]:.1f}]"
          f"  original O2 range [{summary['original_oxygen_min_max'][0]:.1f},{summary['original_oxygen_min_max'][1]:.1f}]")
    print(f"  identical until first override (step {first_div}): {summary['identical_until_first_override']}")
    print(f"  deaths wrapped={len(roll_w['deaths'])} original={len(roll_o['deaths'])}")
    print(f"WROTE -> {OUT}/  (wrapped_annotated.gif, original_annotated.gif, side_by_side.gif, *_log.csv, summary.json)")


if __name__ == "__main__":
    main()
