"""STRICT per-case forensic diagnosis of the 5 ENEMY_DEATH_SURFACE cases at surface_trigger=20.
DO NOT assume enemy deaths. For each life loss we extract object-level evidence and classify into:
  1 ENEMY_OR_PROJECTILE  - an enemy/EnemyMissile bounding box actually contacts the player at the
                           contact frame (last valid player pos before the death animation);
  2 SURFACE_RULE_NODIVER - player touches the surface carrying 0 collected divers, no object contact
                           (Atari Seaquest: surfacing with no divers costs a life);
  3 OXYGEN_DEPLETION      - oxygen reached 0 at/just before the life loss;
  4 RESET_OR_ARTIFACT     - life drop with no death animation / terminated / score anomaly;
  5 UNRESOLVED            - none of the above established.

Reproduces the IDENTICAL trigger=20 stochastic rollouts (wrap.act + rng=default_rng(seed), greedy
prime) so the flagged deaths recur. Saves, per case, an annotated clip [death-30 .. death+15] and a
full per-step CSV. Also profiles several UNWRAPPED-teacher deaths on the same fields for contrast.
"""
import csv
import os
import sys
from collections import deque
from unittest.mock import MagicMock

for _m in ("envpool", "gym"):
    sys.modules.setdefault(_m, MagicMock())
import numpy as np
import ocatari.ram.seaquest as _sq
_orig = _sq._detect_objects_ram
_sq._detect_objects_ram = lambda o, r, h: _orig(o, np.asarray(r, np.int64), h)
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

from seaquest_ccrl.scripts.g0_closed_loop_eval import SeaquestPort, TEACHER_CKPT, TEACHER_SRC
from seaquest_ccrl.scripts.g0_diag_oxygen_trigger_sweep import prime_len
from seaquest_stage_s0.teacher_adapter import CleanRLSeaquestTeacher
from seaquest_ccrl.policies.oxygen_aware_teacher import OxygenAwareTeacher

OUT = "artifacts/seaquest/surface_death_forensics"
TRIGGER, REFILLED, SURFACE_ACTION = 20, 58, 10
SURFACE_Y = 50.0          # player_y <= this == at the surface
CONTACT_PAD = 2.0         # bbox inflation (px) for a "contact"; also report center distance
STEPS = 1300
CASES = [(4000, 372), (4002, 868), (4003, 860), (4004, 852), (4005, 852)]  # (seed, block_start_t)
ACTNAME = {0: "NOOP", 1: "FIRE", 2: "UP", 3: "RIGHT", 4: "LEFT", 5: "DOWN", 6: "UPRIGHT", 7: "UPLEFT",
           8: "DOWNRIGHT", 9: "DOWNLEFT", 10: "UPFIRE", 11: "RIGHTFIRE", 12: "LEFTFIRE", 13: "DOWNFIRE",
           14: "UPRIGHTFIRE", 15: "UPLEFTFIRE", 16: "DOWNRIGHTFIRE", 17: "DOWNLEFTFIRE"}
_F = None


def objstate(port):
    """Pull object-level state straight from OCAtari objects (positions + sizes)."""
    P = None; sharks = []; subs = []; emiss = []; pmiss = []; divers = 0
    oxy = lives = score = None
    for o in port.env.objects:
        c = getattr(o, "category", "")
        if c == "NoObject":
            continue
        box = (float(o.x), float(o.y), float(getattr(o, "w", 0) or 0), float(getattr(o, "h", 0) or 0))
        if c == "Player":
            P = box
        elif c == "Shark":
            sharks.append(box)
        elif c == "Submarine":
            subs.append(box)
        elif c == "EnemyMissile":
            emiss.append(box)
        elif c == "PlayerMissile":
            pmiss.append(box)
        elif c == "CollectedDiver":
            divers += 1
        elif c == "OxygenBar":
            oxy = getattr(o, "value", None)
        elif c == "Lives":
            lives = getattr(o, "value", None)
        elif c == "PlayerScore":
            score = getattr(o, "value", None)
    return {"P": P, "sharks": sharks, "subs": subs, "emiss": emiss, "pmiss": pmiss,
            "divers": divers, "oxy": None if oxy is None else float(oxy),
            "lives": None if lives is None else float(lives), "score": None if score is None else float(score)}


def _center(b):
    return (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)


def _overlap(a, b, pad=CONTACT_PAD):
    return not (a[0] + a[2] + pad < b[0] or b[0] + b[2] + pad < a[0]
                or a[1] + a[3] + pad < b[1] or b[1] + b[3] + pad < a[1])


def nearest(P, boxes):
    """Return (min_center_dist, contact_bool, best_box) of P vs a list of boxes."""
    if P is None or not boxes:
        return (float("inf"), False, None)
    pc = _center(P); best = (float("inf"), False, None)
    for b in boxes:
        bc = _center(b); d = float(np.hypot(pc[0] - bc[0], pc[1] - bc[1]))
        if d < best[0]:
            best = (d, _overlap(P, b), b)
    return best


def rollout(teacher, wrap, seed, steps, prime, keep_frames=False):
    """Exact reproduction of the sweep's stochastic rollout; logs full per-step object state."""
    p = SeaquestPort(sticky=0.0, full_action_space=True, seed=seed); p.reset(seed=seed, noop_max=0)
    if wrap is not None:
        wrap.reset()
    rng = np.random.default_rng(seed)
    def stoch(obs):
        return int(teacher.sample_action(obs, teacher.gumbel_from_uniform(rng.uniform(size=18)))[0])
    for _ in range(prime):
        p.agent_step(stoch(p.teacher_obs()))
    log, frames = [], []
    cum = 0.0
    for s in range(steps):
        st = objstate(p)
        obs = p.teacher_obs()
        if wrap is None:
            executed = stoch(obs); surf = False; tsel = executed
        else:
            executed, surf = wrap.act(obs, -1.0 if st["oxy"] is None else st["oxy"],
                                      None if st["P"] is None else st["P"][1], mode="stochastic", rng=rng)
            tsel = executed if not surf else int(teacher.greedy_action(obs)[0])  # greedy proxy (no rng draw)
        if keep_frames:
            frames.append(np.asarray(p.ale.getScreenRGB(), np.uint8).copy())
        rec = p.agent_step(executed); cum += rec["reward"]
        ne = nearest(st["P"], st["sharks"] + st["subs"]); nm = nearest(st["P"], st["emiss"])
        py = None if st["P"] is None else st["P"][1]
        log.append({"t": s, "lives": st["lives"], "reward": rec["reward"], "cum_score": cum,
                    "hud_score": st["score"], "oxygen": st["oxy"],
                    "player_x": None if st["P"] is None else st["P"][0], "player_y": py,
                    "divers_carried": st["divers"], "surfacing": surf,
                    "teacher_sel": tsel, "teacher_sel_name": ACTNAME.get(tsel),
                    "executed": executed, "executed_name": ACTNAME.get(executed),
                    "n_shark": len(st["sharks"]), "n_sub": len(st["subs"]), "n_enemy_missile": len(st["emiss"]),
                    "nearest_enemy_dist": round(ne[0], 1) if np.isfinite(ne[0]) else None,
                    "enemy_contact": bool(ne[1]), "nearest_missile_dist": round(nm[0], 1) if np.isfinite(nm[0]) else None,
                    "missile_contact": bool(nm[1]),
                    "at_surface": bool(py is not None and py <= SURFACE_Y),
                    "terminated": bool(rec["terminated"]), "truncated": bool(rec["truncated"])})
    return log, frames


def deaths(log):
    out = []
    for i in range(1, len(log)):
        a, b = log[i - 1]["lives"], log[i]["lives"]
        if a is not None and b is not None and b < a:
            out.append(i)
    return out


def contact_frame(log, dstep):
    """Walk back over the death animation (player_y None) to the last valid player position."""
    k = dstep - 1
    while k >= 0 and log[k]["player_y"] is None:
        k -= 1
    anim_len = (dstep - 1) - k
    return max(k, 0), anim_len


def classify(log, dstep):
    cstep, anim = contact_frame(log, dstep)
    c = log[cstep]
    # search a small pre-contact window for the closest object approach + min oxygen
    w = [log[j] for j in range(max(0, cstep - 4), cstep + 1)]
    enemy_contact = any(x["enemy_contact"] for x in w)
    miss_contact = any(x["missile_contact"] for x in w)
    min_enemy = min([x["nearest_enemy_dist"] for x in w if x["nearest_enemy_dist"] is not None], default=None)
    min_miss = min([x["nearest_missile_dist"] for x in w if x["nearest_missile_dist"] is not None], default=None)
    min_oxy = min([x["oxygen"] for x in w if x["oxygen"] is not None], default=None)
    at_surf = bool(c["at_surface"])
    divers = c["divers_carried"]
    reward_at_death = sum(log[j]["reward"] for j in range(cstep, min(len(log), dstep + 2)))
    if c["terminated"]:
        cat = "4 RESET_OR_ARTIFACT"
    elif anim == 0 and not enemy_contact and not miss_contact:
        cat = "4 RESET_OR_ARTIFACT"             # life drop with no death animation and no contact
    elif min_oxy is not None and min_oxy <= 0:
        cat = "3 OXYGEN_DEPLETION"
    elif enemy_contact or miss_contact:
        cat = "1 ENEMY_OR_PROJECTILE"
    elif at_surf and divers == 0:
        cat = "2 SURFACE_RULE_NODIVER"
    else:
        cat = "5 UNRESOLVED"
    return {"death_step": dstep, "contact_step": cstep, "anim_len": anim, "at_surface": at_surf,
            "divers_carried": divers, "min_oxy_5": min_oxy, "min_enemy_dist_5": min_enemy,
            "enemy_contact": enemy_contact, "min_missile_dist_5": min_miss, "missile_contact": miss_contact,
            "reward_around_death": reward_at_death, "category": cat,
            "player_y_at_contact": c["player_y"], "player_x_at_contact": c["player_x"]}


def annotate(frame, r, is_death):
    h, w, _ = frame.shape
    g = Image.fromarray(frame).resize((w * 3, h * 3), Image.NEAREST); gw, gh = g.size
    cv = Image.new("RGB", (gw, gh + 92), (12, 12, 18)); cv.paste(g, (0, 92)); d = ImageDraw.Draw(cv)
    py = r["player_y"]; pys = "None" if py is None else f"{py:.0f}"
    oxs = "n/a" if r["oxygen"] is None else f"{r['oxygen']:.0f}"
    ned = "inf" if r["nearest_enemy_dist"] is None else f"{r['nearest_enemy_dist']:.0f}"
    nmd = "inf" if r["nearest_missile_dist"] is None else f"{r['nearest_missile_dist']:.0f}"
    d.text((6, 2), f"t={r['t']}  lives={r['lives']}  score={r['hud_score']}  O2={oxs}", font=_F, fill=(230, 230, 120))
    d.text((6, 20), f"player_y={pys}  AT_SURFACE={r['at_surface']}  divers_carried={r['divers_carried']}",
           font=_F, fill=(120, 220, 255) if r["at_surface"] else (210, 210, 210))
    ecol = (255, 80, 80) if r["enemy_contact"] else (180, 220, 180)
    d.text((6, 38), f"nearest_enemy={ned}px contact={r['enemy_contact']}  "
                    f"missile={nmd}px contact={r['missile_contact']}", font=_F, fill=ecol)
    mcol = (255, 140, 60) if r["surfacing"] else (150, 200, 150)
    d.text((6, 56), f"mode={'SURFACING' if r['surfacing'] else 'HF teacher'}  "
                    f"teacher={r['teacher_sel_name']}  exec={r['executed_name']}", font=_F, fill=mcol)
    if is_death:
        d.rectangle([0, 74, gw, 92], fill=(120, 0, 0))
        d.text((6, 75), ">>> LIFE LOST THIS FRAME (respawn) <<<", font=_F, fill=(255, 230, 230))
    return cv


def save_clip(log, frames, dstep, path):
    lo, hi = max(0, dstep - 30), min(len(log), dstep + 16)
    imgs = [annotate(frames[i], log[i], i == dstep) for i in range(lo, hi)]
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=120, loop=0, optimize=True)
    return lo, hi


def write_csv(log, lo, hi, path):
    cols = ["t", "lives", "reward", "cum_score", "hud_score", "oxygen", "player_x", "player_y",
            "divers_carried", "surfacing", "teacher_sel_name", "executed_name", "n_shark", "n_sub",
            "n_enemy_missile", "nearest_enemy_dist", "enemy_contact", "nearest_missile_dist",
            "missile_contact", "at_surface", "terminated"]
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore"); wr.writeheader()
        for i in range(lo, hi):
            wr.writerow(log[i])


def main():
    global _F
    _F = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 13)
    os.makedirs(OUT, exist_ok=True)
    teacher = CleanRLSeaquestTeacher(TEACHER_CKPT, TEACHER_SRC, mod_name="cleanrl_src_A")

    print("=== 5 SURFACE-DEATH CASES (trigger=20) ===\n")
    summary = []
    for seed, blk in CASES:
        pr = prime_len(teacher, seed)
        wrap = OxygenAwareTeacher(teacher, surface_trigger=TRIGGER, refilled=REFILLED, surface_action=SURFACE_ACTION)
        log, frames = rollout(teacher, wrap, seed, STEPS, pr, keep_frames=True)
        ds = deaths(log)
        # the flagged death = first life loss at/after the block start
        cand = [d for d in ds if d >= blk - 2]
        dstep = cand[0] if cand else (ds[0] if ds else None)
        if dstep is None:
            print(f"seed {seed}: NO death found"); continue
        info = classify(log, dstep)
        clip = f"{OUT}/case_s{seed}_t{dstep}.gif"; trace = f"{OUT}/case_s{seed}_t{dstep}.csv"
        lo, hi = save_clip(log, frames, dstep, clip); write_csv(log, lo, hi, trace)
        info.update({"seed": seed, "block_start": blk})
        summary.append(info)
        c = log[info["contact_step"]]
        print(f"seed {seed}: life lost @t={dstep} (contact frame t={info['contact_step']}, anim={info['anim_len']})")
        print(f"   player_y@contact={info['player_y_at_contact']}  AT_SURFACE={info['at_surface']}  "
              f"divers_carried={info['divers_carried']}  min_O2(5)={info['min_oxy_5']}")
        print(f"   nearest_enemy={info['min_enemy_dist_5']}px contact={info['enemy_contact']}  |  "
              f"nearest_missile={info['min_missile_dist_5']}px contact={info['missile_contact']}")
        print(f"   reward_around_death={info['reward_around_death']}  terminated={c['terminated']}")
        print(f"   ==> CLASS: {info['category']}")
        print(f"   clip={clip}\n")

    # ---- contrast: unwrapped-teacher normal deaths ----
    print("=== UNWRAPPED TEACHER normal deaths (contrast) ===\n")
    norm = []
    for seed in (4000, 4001, 4002):
        pr = prime_len(teacher, seed)
        log, _ = rollout(teacher, None, seed, STEPS, pr, keep_frames=False)
        for dstep in deaths(log):
            info = classify(log, dstep); cstep = info["contact_step"]
            norm.append((seed, dstep, info))
            if len(norm) <= 8:
                print(f"seed {seed} @t={dstep}: player_y@contact={info['player_y_at_contact']} "
                      f"AT_SURFACE={info['at_surface']} divers={info['divers_carried']} "
                      f"nearest_enemy={info['min_enemy_dist_5']}px contact={info['enemy_contact']} "
                      f"missile={info['min_missile_dist_5']}px contact={info['missile_contact']} "
                      f"anim={info['anim_len']} -> {info['category']}")

    # ---- compact comparison table ----
    print("\n=== COMPARISON SUMMARY ===")
    def agg(rows):
        n = len(rows)
        if not n:
            return "none"
        ec = sum(r["enemy_contact"] or r["missile_contact"] for r in rows)
        surf = sum(r["at_surface"] for r in rows)
        d0 = sum(r["divers_carried"] == 0 for r in rows)
        anim = np.mean([r["anim_len"] for r in rows])
        return f"n={n} | obj_contact={ec}/{n} | at_surface={surf}/{n} | divers==0={d0}/{n} | mean_anim={anim:.1f}"
    print(f"  5 surface cases : {agg(summary)}")
    print(f"  normal teacher  : {agg([i for _, _, i in norm])}")
    from collections import Counter
    print("\n  surface-case categories: " + str(dict(Counter(r["category"] for r in summary))))
    print(f"\nWROTE clips + traces -> {OUT}/")


if __name__ == "__main__":
    main()
