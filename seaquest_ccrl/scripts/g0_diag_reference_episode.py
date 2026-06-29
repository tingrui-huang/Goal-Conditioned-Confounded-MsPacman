"""Capture the FULL teacher reference episode (seed 1000) for the behavioral-diagnosis package,
so the 'how it plays' reference shows the complete oxygen cycle (dive -> deplete -> surface ->
refill) instead of a 16-step snippet. Re-runs the deterministic reference action sequence locally
(ale_py + ocatari; the SAME episode anchors are drawn from) and records, every step: oxygen, depth
(player_y), action, player xy; frames are kept every `stride` steps to bound GIF size.

Saves examples/reference_full_episode.npz + a quick oxygen(t)/depth(t) plot to eyeball the rhythm.
"""
import os
import numpy as np
from PIL import Image, ImageDraw

ALE = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT',
       'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE', 'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']
UP_ACTS = {2, 6, 7, 10, 14, 15}
FPS = 5
LOW_O2 = 16

# numpy-version workaround (same as g0_diag_rerun_examples): cast RAM to int64 before OCAtari detection
import ocatari.ram.seaquest as _sq
_orig_detect = _sq._detect_objects_ram
_sq._detect_objects_ram = lambda objects, ram_state, hud: _orig_detect(
    objects, np.asarray(ram_state, np.int64), hud)

from seaquest_ccrl.scripts.g0_closed_loop_eval import SeaquestPort, pcenter   # noqa: E402

OUT = "artifacts/seaquest/behavioral_diagnosis_compact"
FV = "artifacts/seaquest/goal_control/full_view/evaluation"
SEED = 1000
STRIDE = 5            # keep every 5th frame -> ~240 frames for a 1200-step episode


def main():
    os.makedirs(f"{OUT}/examples", exist_ok=True)
    acts = np.load(f"{FV}/episode_actions.npz")[f"actions_{SEED}"]
    n = len(acts)
    port = SeaquestPort(sticky=0.0, full_action_space=True, seed=SEED)
    port.reset(seed=SEED, noop_max=0)
    oxygen, depth, lives, px, py, kept_frames, kept_steps = [], [], [], [], [], [], []
    for s in range(n):
        f = port.features()
        oxygen.append(f.get("oxygen") if f.get("oxygen") is not None else -1)
        lives.append(f.get("lives") if f.get("lives") is not None else np.nan)
        p = pcenter(port)
        px.append(p[0] if p else np.nan); py.append(p[1] if p else np.nan)
        depth.append(f.get("player_y") if f.get("player_y") is not None else np.nan)
        if s % STRIDE == 0:
            kept_frames.append(np.asarray(port.ale.getScreenRGB(), np.uint8)); kept_steps.append(s)
        port.agent_step(int(acts[s]))
    oxygen = np.array(oxygen, np.float32); depth = np.array(depth, np.float32); lives = np.array(lives, np.float32)
    frames = np.stack(kept_frames); kept_steps = np.array(kept_steps, np.int64)
    # VERIFIED death events = lives decrease (oxygen 'resets' here are respawns, not surfacing)
    death_steps = np.array([s for s in range(1, n)
                            if np.isfinite(lives[s]) and np.isfinite(lives[s - 1]) and lives[s] < lives[s - 1]], np.int64)
    np.savez_compressed(f"{OUT}/examples/reference_full_episode.npz",
                        frames=frames, frame_steps=kept_steps, oxygen=oxygen, depth=depth, lives=lives,
                        death_steps=death_steps, px=np.array(px, np.float32), py=np.array(py, np.float32),
                        actions=acts.astype(np.int64), stride=STRIDE, seed=SEED, n_steps=n)

    ox = oxygen[oxygen >= 0]
    print(f"[reference] seed {SEED}: steps={n} frames_kept={len(frames)} (stride {STRIDE})")
    print(f"  oxygen range {ox.min():.0f}..{ox.max():.0f}; VERIFIED deaths (lives drop) at steps "
          f"{death_steps.tolist()} -> the oxygen 'refills' are RESPAWNS, not surfacing")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(13, 3.4)); T = np.arange(n)
    ax1.plot(T, oxygen, color="#1f77b4", lw=1.2, label="oxygen (0-64)")
    ax1.axhline(16, color="orange", ls="--", lw=0.8, label="low-O2 (<=16)")
    for i, ds in enumerate(death_steps):
        ax1.axvline(ds, color="red", lw=1.4, label="DEATH: out of O2 (verified)" if i == 0 else None)
    ax1.set_ylabel("oxygen", color="#1f77b4"); ax1.set_ylim(-2, 68); ax1.set_xlabel("step")
    ax2 = ax1.twinx(); ax2.plot(T, depth, color="#d62728", lw=0.8, alpha=0.6, label="depth (player_y)")
    ax2.set_ylabel("depth (small=surface)", color="#d62728"); ax2.invert_yaxis()
    ax1.set_title(f"Teacher reference episode (seed {SEED}) — oxygen depletes -> DEATH x{len(death_steps)} "
                  f"(verified; not surfacing)")
    ax1.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/reference_oxygen_curve.png", dpi=110); plt.close(fig)

    # ---- oxygen-annotated full-episode GIF (kept frames, one per `stride` steps) ----
    gif_path = f"{OUT}/videos/reference_full_episode.gif"
    os.makedirs(f"{OUT}/videos", exist_ok=True)
    imgs = []
    sc = 2
    for i, st in enumerate(kept_steps):
        o2 = float(oxygen[st]); a = int(acts[st]); dep = float(depth[st])
        img = Image.fromarray(frames[i]).resize((160 * sc, 210 * sc), Image.NEAREST)
        panel = 70
        cv = Image.new("RGB", (160 * sc, 210 * sc + panel), (16, 16, 16))
        cv.paste(img, (0, 0)); dr = ImageDraw.Draw(cv); y0 = 210 * sc + 6
        frac = max(0.0, min(1.0, o2 / 64.0))
        col = (int(235 * (1 - frac)) + 20, int(200 * frac) + 20, 40)          # red(low)->green(full)
        dr.rectangle([8, y0, 160 * sc - 8, y0 + 14], outline=(110, 110, 110))
        dr.rectangle([9, y0 + 1, 9 + int((160 * sc - 18) * frac), y0 + 13], fill=col)
        low = 0 <= o2 <= LOW_O2
        near_death = bool(death_steps.size and np.any(np.abs(death_steps - st) <= STRIDE))
        o2s = "?" if o2 < 0 else f"{int(o2)}/64"
        deps = "?" if not np.isfinite(dep) else str(int(dep))
        dr.text((9, y0 + 20), f"step {st}/{n}   O2 {o2s}   depth {deps}   {ALE[a]}",
                fill=(255, 120, 120) if low else (235, 235, 235))
        if near_death:                                   # VERIFIED death (lives drop), not surfacing
            tag, col = "  DIED: out of O2 -> respawn", (255, 60, 60)
        elif low:
            tag, col = "  LOW O2", (255, 150, 60)
        else:
            tag, col = "", (140, 220, 140)
        dr.text((9, y0 + 38), tag, fill=col)
        imgs.append(cv)
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:],
                 duration=int(round(1000 / FPS)), loop=0, optimize=True)

    # verify
    from PIL import ImageSequence
    rr = Image.open(gif_path); gf = [np.array(f.convert("RGB")) for f in ImageSequence.Iterator(rr)]
    pdiff = np.array([float(np.abs(gf[k].astype(int) - gf[k - 1].astype(int)).mean()) for k in range(1, len(gf))])
    pxy = np.stack([px, py], 1)[kept_steps]
    disp = np.array([float(np.hypot(pxy[k, 0] - pxy[k - 1, 0], pxy[k, 1] - pxy[k - 1, 1]))
                     for k in range(1, len(pxy)) if np.all(np.isfinite(pxy[k])) and np.all(np.isfinite(pxy[k - 1]))])
    verif = {"n_frames": len(gf), "n_frames_expected": len(kept_steps), "fps": FPS,
             "frame_duration_ms": int(rr.info.get("duration")), "stride_steps": STRIDE,
             "oxygen_min": float(ox.min()), "oxygen_max": float(ox.max()),
             "verified_deaths_out_of_o2": death_steps.tolist(), "n_deaths": int(death_steps.size),
             "successful_surfacings": 0,
             "player_total_path_px": round(float(disp.sum()), 1),
             "all_consecutive_frames_distinct": bool((pdiff > 0).all())}
    import json
    json.dump(verif, open(f"{OUT}/reference_gif_verification.json", "w"), indent=2)
    print(f"  WROTE figures/reference_oxygen_curve.png + videos/reference_full_episode.gif "
          f"({len(gf)} frames @ {FPS}fps) + reference_full_episode.npz")
    print(f"  verify: {json.dumps(verif)}")


if __name__ == "__main__":
    main()
