"""Acceptance checks A-F for the Level-1 confounded Seaquest build.

A. Env sanity            : Player+OxygenBar present; oxygen depletes underwater &
                           refills on surfacing; frame is (210,160,3).
B. Mask completeness     : every oxygen-rendering pixel (OxygenBar union
                           OxygenBarDepleted) lies inside OXY_MASK_RECT, and the masked
                           region is constant across oxygen levels (unidentifiability).
C. Confounding present   : U->A (surfacing actions correlate with low oxygen);
                           U->S' (oxygen depletion -> drown -> state/life change).
D. Oracle-for-free       : unmasked frame recoverable; oracle dataset == data, no mask.
E. Strength knob         : two THETA values yield different oxygen->action dependence.
F. Determinism           : same seed + same action sequence => identical trajectory.

Run: python -m seaquest_ccrl.scripts.validate
"""
import os
import shutil
import tempfile

import numpy as np

from seaquest_ccrl import config as C
from seaquest_ccrl.envs.seaquest_gc import SeaquestGCEnv
from seaquest_ccrl.policies.scripted_behavior import ScriptedBehaviorPolicy
from seaquest_ccrl.data.masking import apply_oxygen_mask, oracle
from seaquest_ccrl.data.dataset import SeaquestOfflineDataset

UP_ACTIONS = {2, 6, 7, 10, 14, 15}     # any action with an upward (surfacing) component
DOWN = 5

_results = []
def record(name, ok, detail=""):
    _results.append((name, ok, detail))
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}: {detail}")
    return ok


def _objects(env):
    return [o for o in env.env.objects if o.category != "NoObject"]


def _bar_boxes(env):
    boxes = []
    for o in _objects(env):
        if o.category in ("OxygenBar", "OxygenBarDepleted"):
            boxes.append((o.x, o.y, o.w, o.h))
    return boxes


# --------------------------------------------------------------------------
def check_A_and_B():
    print("\n== A. Env sanity  +  B. Mask completeness ==")
    cfg = C.DEFAULT
    env = SeaquestGCEnv(cfg)
    policy = ScriptedBehaviorPolicy(cfg)

    frame, state = env.reset(seed=0)
    shape_ok = (tuple(frame.shape) == C.FRAME_SHAPE and frame.dtype == np.uint8)

    have_player = have_oxy = True
    oxy_series = []
    bar_union = None
    frames_by_oxy = {}      # oxygen-level -> a sample frame (for constancy test)
    target = (80.0, 150.0)

    for t in range(800):
        a = policy.act(state, target)
        # alternate: force diving for a stretch to deplete, then let it surface
        if t < 400:
            a = DOWN
        frame, _, term, trunc, state = env.step(a)
        objs = _objects(env)
        cats = {o.category for o in objs}
        if "Player" not in cats and state["player_pos"] is None:
            have_player = False
        if state["oxygen"] is None:
            have_oxy = False
        if state["oxygen"] is not None:
            oxy_series.append(state["oxygen"])
            frames_by_oxy.setdefault(state["oxygen"], frame.copy())
        for (x, y, w, h) in _bar_boxes(env):
            b = (x, y, x + w, y + h)
            bar_union = b if bar_union is None else (
                min(bar_union[0], b[0]), min(bar_union[1], b[1]),
                max(bar_union[2], b[2]), max(bar_union[3], b[3]))
        if term or trunc:
            frame, state = env.reset(seed=0)

    # A: oxygen depletes underwater (first 400 forced-dive steps trend down)
    oxy = np.asarray(oxy_series)
    depletes = oxy.min() < oxy.max() and oxy.min() <= cfg.theta
    refills = oxy.max() >= cfg.oxy_full_width - 5  # came back up near full at some point
    record("A.frame_shape", bool(shape_ok), f"{frame.shape} {frame.dtype}")
    record("A.objects_present", have_player and have_oxy, "Player & OxygenBar each step")
    record("A.oxygen_dynamics", bool(depletes and refills),
           f"oxy range [{oxy.min()},{oxy.max()}], theta={cfg.theta}")

    # B1: every oxygen-rendering pixel is inside OXY_MASK_RECT
    mx, my, mw, mh = cfg.oxy_mask_rect
    rect = (mx, my, mx + mw, my + mh)
    if bar_union is not None:
        inside = (bar_union[0] >= rect[0] and bar_union[1] >= rect[1] and
                  bar_union[2] <= rect[2] and bar_union[3] <= rect[3])
    else:
        inside = False
    record("B.coverage", bool(inside),
           f"bar union {bar_union} subset-of mask rect {rect}")

    # B2: masked region constant across oxygen levels (min vs max oxygen frame)
    if len(frames_by_oxy) >= 2:
        lo = frames_by_oxy[min(frames_by_oxy)]
        hi = frames_by_oxy[max(frames_by_oxy)]
        # raw frames DIFFER inside rect (proof oxygen leaks there if unmasked)...
        raw_diff = int(np.abs(lo[my:my+mh, mx:mx+mw].astype(int) -
                              hi[my:my+mh, mx:mx+mw].astype(int)).sum())
        # ...but AFTER masking they are identical inside rect (constant => no leak)
        ml = apply_oxygen_mask(lo); mh_ = apply_oxygen_mask(hi)
        masked_equal = np.array_equal(ml[my:my+mh, mx:mx+mw], mh_[my:my+mh, mx:mx+mw])
        record("B.unmasked_leaks", raw_diff > 0,
               f"unmasked rect differs by {raw_diff} across oxygen (leak exists)")
        record("B.masked_constant", bool(masked_equal),
               "masked rect identical across oxygen (leak removed)")
    else:
        record("B.masked_constant", False, "insufficient oxygen variation sampled")
    env.close()


def check_C():
    print("\n== C. Confounding present (U->A, U->S') ==")
    cfg = C.DEFAULT
    # ---- C1: U -> A. Surfacing actions correlate with low oxygen. ----
    env = SeaquestGCEnv(cfg)
    policy = ScriptedBehaviorPolicy(cfg)
    frame, state = env.reset(seed=1)
    rng = np.random.RandomState(0)
    target = (rng.randint(30, 130), rng.randint(60, 160))
    up_low = tot_low = up_high = tot_high = 0
    for t in range(1500):
        a = policy.act(state, target)
        oxy = state["oxygen"]
        if oxy is not None:
            if oxy < cfg.theta:
                tot_low += 1; up_low += int(a in UP_ACTIONS)
            else:
                tot_high += 1; up_high += int(a in UP_ACTIONS)
        frame, _, term, trunc, state = env.step(a)
        if policy.reached(state, target):
            target = (rng.randint(30, 130), rng.randint(60, 160))
        if term or trunc:
            frame, state = env.reset(seed=1)
    p_low = up_low / max(tot_low, 1)
    p_high = up_high / max(tot_high, 1)
    record("C.U->A", p_low > 0.9 and p_low > p_high + 0.3,
           f"P(surface|oxy<theta)={p_low:.2f} vs P(surface|oxy>=theta)={p_high:.2f}")
    env.close()

    # ---- C2: U -> S'. Never surface -> oxygen depletes to 0 -> drown (life lost). ----
    env = SeaquestGCEnv(cfg)
    frame, state = env.reset(seed=2)
    def lives(e):
        for o in _objects(e):
            if o.category == "Lives":
                return o.value if hasattr(o, "value") else None
        return None
    start_lives = lives(env)
    drowned = False
    oxy_at_drown = None
    prev_oxy = None
    for t in range(4000):
        frame, _, term, trunc, state = env.step(DOWN)   # dive & stay -> no surfacing
        oxy = state["oxygen"]
        cur_lives = lives(env)
        # drown event: oxygen hit ~0 and then a life is lost (S' transition from U)
        if prev_oxy is not None and prev_oxy <= 1 and cur_lives is not None and \
           start_lives is not None and cur_lives < start_lives:
            drowned = True; oxy_at_drown = prev_oxy; break
        prev_oxy = oxy if oxy is not None else prev_oxy
        if term or trunc:
            break
    record("C.U->S'", bool(drowned),
           f"oxygen->{oxy_at_drown} preceded life loss "
           f"({start_lives}->{lives(env)}): depletion ends the episode")
    env.close()


def check_D():
    print("\n== D. Oracle-for-free ==")
    cfg = C.Config(n_episodes=1, max_steps_per_ep=120,
                   data_root=tempfile.mkdtemp(prefix="sq_val_"))
    from seaquest_ccrl.collect.collect_dataset import collect
    collect(cfg, out_root=cfg.data_root, verbose=False)

    ds_masked = SeaquestOfflineDataset(cfg.data_root, oracle=False)
    ds_oracle = SeaquestOfflineDataset(cfg.data_root, oracle=True)
    tm = ds_masked.trajectory(0)
    to = ds_oracle.trajectory(0)

    # oracle obs == raw unmasked frames
    oracle_eq = np.array_equal(to["obs"], to["frames_unmasked"])
    # masked obs differs ONLY within the rect; identical elsewhere
    mx, my, mw, mh = cfg.oxy_mask_rect
    out_eq = True
    rect_zero = True
    for fm, fu in zip(tm["obs"], tm["frames_unmasked"]):
        a = fm.copy(); b = fu.copy()
        a[my:my+mh, mx:mx+mw] = 0; b[my:my+mh, mx:mx+mw] = 0
        if not np.array_equal(a, b):
            out_eq = False
        if fm[my:my+mh, mx:mx+mw].any():
            rect_zero = False
    record("D.oracle_is_unmasked", bool(oracle_eq), "oracle obs == stored frames")
    record("D.mask_local", bool(out_eq and rect_zero),
           "masked obs == unmasked outside rect; zero inside rect")
    shutil.rmtree(cfg.data_root, ignore_errors=True)


def check_E():
    print("\n== E. Strength knob (THETA moves the oxygen->action threshold) ==")
    cfg = C.DEFAULT
    UP = 2

    def override_threshold(theta):
        """Highest oxygen at which the policy issues a SURFACE override.

        An override is isolated as: action==UP while the target is BELOW the sub
        (navigation alone would go DOWN) -> the UP can only come from the oxygen
        rule. Max oxygen over such steps approximates THETA, free of navigation-up
        pollution. Returns (max, mean) oxygen over override steps.
        """
        env = SeaquestGCEnv(cfg)
        pol = ScriptedBehaviorPolicy(cfg, theta=theta)
        frame, state = env.reset(seed=3)
        rng = np.random.RandomState(1)
        target = (rng.randint(30, 130), rng.randint(60, 160))
        oxys = []
        for t in range(1500):
            pos = state["player_pos"]; oxy = state["oxygen"]
            a = pol.act(state, target)
            if a == UP and pos is not None and oxy is not None and \
               (target[1] - pos[1]) > cfg.move_tol:        # nav wanted DOWN -> override
                oxys.append(oxy)
            frame, _, term, trunc, state = env.step(a)
            if pol.reached(state, target):
                target = (rng.randint(30, 130), rng.randint(60, 160))
            if term or trunc:
                frame, state = env.reset(seed=3)
        env.close()
        return (max(oxys) if oxys else float("nan"),
                float(np.mean(oxys)) if oxys else float("nan"))

    lo_max, lo_mean = override_threshold(10)
    hi_max, hi_mean = override_threshold(35)
    # The override fires only when oxy<THETA, so the threshold scales with THETA.
    record("E.knob_moves", (hi_max > lo_max + 5) and (lo_max < 10) and (hi_max < 35),
           f"override oxygen max(mean): theta=10 -> {lo_max:.0f}({lo_mean:.1f}), "
           f"theta=35 -> {hi_max:.0f}({hi_mean:.1f})")


def check_F():
    print("\n== F. Determinism ==")
    cfg = C.DEFAULT
    seq = [5, 5, 3, 3, 2, 2, 4, 0, 5, 3] * 8
    def rollout():
        env = SeaquestGCEnv(cfg)
        frame, state = env.reset(seed=7)
        frames = [frame.copy()]; poss = [state["player_pos"]]
        for a in seq:
            frame, _, term, trunc, state = env.step(a)
            frames.append(frame.copy()); poss.append(state["player_pos"])
            if term or trunc:
                break
        env.close()
        return np.stack(frames), poss
    f1, p1 = rollout()
    f2, p2 = rollout()
    record("F.deterministic", np.array_equal(f1, f2) and p1 == p2,
           f"{len(f1)} frames identical across two seeded runs")


def main():
    print("=" * 70)
    print("Level-1 Confounded Seaquest - acceptance checks A-F")
    print("=" * 70)
    check_A_and_B()
    check_C()
    check_D()
    check_E()
    check_F()
    print("\n" + "=" * 70)
    n_pass = sum(1 for _, ok, _ in _results if ok)
    for name, ok, _ in _results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\n  {n_pass}/{len(_results)} checks passed")
    print("=" * 70)
    return 0 if n_pass == len(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
