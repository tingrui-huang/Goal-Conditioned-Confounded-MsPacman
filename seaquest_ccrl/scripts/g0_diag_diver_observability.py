"""Is carried_divers (the collected-diver count) reliably present in the critic/actor observation?

The obs the critic/actor see = _prep: full 210x160 RGB, oxygen rect masked, area-resized to 84x84x3
(NO crop). The collected-diver icons render in the bottom HUD, so they are PHYSICALLY in the obs but
heavily downscaled. This probe quantifies whether the COUNT survives:
  1. locate the CollectedDiver sprites in raw coords; confirm they are OUTSIDE the oxygen mask rect;
  2. map that band into the 84x84 obs; measure a simple readout (foreground-pixel mass in the band)
     vs the true count -> is 0 vs >=1 (and the exact count) linearly/monotonically recoverable?
  3. save a few 84x84 obs at different counts for visual confirmation.
"""
import os
import sys
from unittest.mock import MagicMock
for _m in ("envpool", "gym"):
    sys.modules.setdefault(_m, MagicMock())
import numpy as np
import ocatari.ram.seaquest as _sq
_orig = _sq._detect_objects_ram
_sq._detect_objects_ram = lambda o, r, h: _orig(o, np.asarray(r, np.int64), h)
from PIL import Image
from seaquest_ccrl.scripts.g0_closed_loop_eval import SeaquestPort, TEACHER_CKPT, TEACHER_SRC, _prep
from seaquest_ccrl.scripts.g0_diag_oxygen_trigger_sweep import prime_len
from seaquest_ccrl import config as C
from seaquest_stage_s0.teacher_adapter import CleanRLSeaquestTeacher
from seaquest_ccrl.policies.oxygen_aware_teacher import OxygenAwareTeacher

OUT = "artifacts/seaquest/diver_observability"
FS = 84
SEEDS = [5000, 5001, 5002, 5003]
STEPS = 1200


def collected_diver_boxes(port):
    boxes = []
    for o in port.env.objects:
        if getattr(o, "category", "") == "CollectedDiver":
            boxes.append((float(o.x), float(o.y), float(getattr(o, "w", 0) or 0), float(getattr(o, "h", 0) or 0)))
    return boxes


def main():
    os.makedirs(OUT, exist_ok=True)
    teacher = CleanRLSeaquestTeacher(TEACHER_CKPT, TEACHER_SRC, mod_name="cleanrl_src_A")
    ox = C.OXY_MASK_RECT
    print(f"OXY_MASK_RECT (x,y,w,h) = {ox}  -> masks raw rows y[{ox[1]},{ox[1]+ox[3]}) cols x[{ox[0]},{ox[0]+ox[2]})\n")

    rows = []          # (count, raw_frame, diver_box_union_y)
    diver_y_lo, diver_y_hi, diver_x_lo, diver_x_hi = 1e9, -1e9, 1e9, -1e9
    seen_counts = {}
    for sd in SEEDS:
        pr = prime_len(teacher, sd)
        wrap = OxygenAwareTeacher(teacher, surface_trigger=20, refilled=58, surface_action=10)
        p = SeaquestPort(sticky=0.0, full_action_space=True, seed=sd); p.reset(seed=sd, noop_max=0)
        wrap.reset(); rng = np.random.default_rng(sd)
        for _ in range(pr):
            p.agent_step(int(teacher.sample_action(p.teacher_obs(), teacher.gumbel_from_uniform(rng.uniform(size=18)))[0]))
        for s in range(STEPS):
            f = p.features(); oxy = f.get("oxygen"); oxy = -1.0 if oxy is None else float(oxy)
            cnt = f.get("n_collected_diver", 0)
            boxes = collected_diver_boxes(p)
            for (x, y, w, h) in boxes:
                diver_y_lo = min(diver_y_lo, y); diver_y_hi = max(diver_y_hi, y + h)
                diver_x_lo = min(diver_x_lo, x); diver_x_hi = max(diver_x_hi, x + w)
            frame = np.asarray(p.ale.getScreenRGB(), np.uint8)
            rows.append((cnt, frame.copy()))
            seen_counts[cnt] = seen_counts.get(cnt, 0) + 1
            a, _ = wrap.act(p.teacher_obs(), oxy, f.get("player_y"), mode="stochastic", rng=rng)
            p.agent_step(a)

    print(f"counts observed (count: #frames): {dict(sorted(seen_counts.items()))}")
    if diver_y_hi < 0:
        print("NO CollectedDiver sprites ever rendered -> count is NEVER shown as an icon.");
    else:
        print(f"CollectedDiver sprites raw region: y[{diver_y_lo:.0f},{diver_y_hi:.0f}] x[{diver_x_lo:.0f},{diver_x_hi:.0f}]")
        inside = (diver_y_lo >= ox[1] and diver_y_hi <= ox[1] + ox[3])
        print(f"  inside oxygen mask rect? {inside}  (mask y[{ox[1]},{ox[1]+ox[3]}))")
        # map diver band into 84x84 rows
        r0 = int(np.floor(diver_y_lo * FS / 210.0)); r1 = int(np.ceil(diver_y_hi * FS / 210.0))
        print(f"  diver band maps to obs rows [{r0},{r1}) of {FS}  ({r1-r0} row(s) at 84x84)")

    # ---- decodability: TIGHT diver-icon box, RAW vs 84x84 obs, vs true count ----
    if diver_y_hi > 0:
        # raw box (generous in x so 1 vs 2 icons differ), and its image of it in the 84x84 obs
        ry0, ry1, rx0, rx1 = int(diver_y_lo), int(diver_y_hi) + 1, 48, 96
        oy0, oy1 = max(0, int(ry0 * FS / 210)), min(FS, int(np.ceil(ry1 * FS / 210)) + 1)
        ox0, ox1 = int(rx0 * FS / 160), int(np.ceil(rx1 * FS / 160))
        print(f"\n  diver icon box: RAW y[{ry0},{ry1}) x[{rx0},{rx1}) -> OBS y[{oy0},{oy1}) x[{ox0},{ox1})")

        def colored(px):  # non-black, non-grey-seabed coloured pixels (divers are saturated colour)
            px = px.astype(np.int32); mx = px.max(2); mn = px.min(2)
            return ((mx > 50) & ((mx - mn) > 40)).sum()                # saturated & bright

        raw_by, obs_by = {}, {}
        for cnt, frame in rows:
            raw_by.setdefault(cnt, []).append(colored(frame[ry0:ry1, rx0:rx1]))
            obs = _prep(frame, FS, mask_oxygen=False)
            obs_by.setdefault(cnt, []).append(colored(obs[oy0:oy1, ox0:ox1]))

        def sep01(by):
            m0 = by.get(0, []); mp = [m for c in by if c >= 1 for m in by[c]]
            if not m0 or not mp:
                return None
            thr = (np.mean(m0) + np.mean(mp)) / 2
            return (sum(x <= thr for x in m0) + sum(x > thr for x in mp)) / (len(m0) + len(mp)), np.mean(m0), np.mean(mp)

        print("\n  RAW frame (210x160) coloured-pixel count in diver box, by true count:")
        for cnt in sorted(raw_by):
            v = raw_by[cnt]; print(f"    divers={cnt}: mean={np.mean(v):6.2f} std={np.std(v):5.2f}")
        s = sep01(raw_by);  print(f"    -> 0-vs->=1 acc={s[0]:.3f} (mean0={s[1]:.2f} vs meanPos={s[2]:.2f})" if s else "")
        print("\n  84x84 OBS (what the critic/actor sees) coloured-pixel count in diver box, by true count:")
        for cnt in sorted(obs_by):
            v = obs_by[cnt]; print(f"    divers={cnt}: mean={np.mean(v):6.2f} std={np.std(v):5.2f}")
        s = sep01(obs_by);  print(f"    -> 0-vs->=1 acc={s[0]:.3f} (mean0={s[1]:.2f} vs meanPos={s[2]:.2f})" if s else "")

        # ---- save sample 84x84 obs (upscaled) at a few counts ----
        saved = set()
        for cnt, frame in rows:
            if cnt in saved or cnt > 4:
                continue
            saved.add(cnt)
            obs = _prep(frame, FS, mask_oxygen=False)
            Image.fromarray(obs).resize((FS * 5, FS * 5), Image.NEAREST).save(f"{OUT}/obs84_divers{cnt}.png")
            Image.fromarray(frame).save(f"{OUT}/raw_divers{cnt}.png")
        print(f"\n  saved sample obs/raw frames for counts {sorted(saved)} -> {OUT}/")


if __name__ == "__main__":
    main()
