"""Upstream-style perturbation saliency for the Seaquest HF CleanRL teacher (actor + critic),
faithful to the teacher repo's saliency/saliency.py (Greydanus-2017, the 84x84 `score_frame_sebulba`
path) + gen_saliency.py overlay. Runs in the Stage-S0 ocatari Docker image (jax teacher + OCAtari).

Method: occlude(I,m)=I*(1-m)+gaussian_filter(I,sigma=3)*m over a density-d grid on the 84x84 teacher
obs; score = 0.5*||L-l||^2 on the 18 logits (ACTOR) or on the value scalar (CRITIC); per-frame
normalize; overlay (resized to 160x210) on the RGB render frame. No training, no model change.

Visualization-only diagnostic to see what the Seaquest expert attends to (player sub / enemies /
divers / oxygen), drawn the SAME way as the upstream teacher figures.
"""
import sys, os, argparse
_REAL = sys.stdout
def prog(*a, **k): k.pop("flush", None); print(*a, file=_REAL, flush=True, **k)
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
sys.path.insert(0, "/work"); sys.path.insert(0, "/work/seaquest_stage_s05")
import common as CM

OUT = "/work/artifacts/seaquest/upstream_saliency"
RADIUS, DENSITY, BLUR, SIZE = 4, 4, 3, 84


def get_mask(center, r=RADIUS, size=(SIZE, SIZE)):
    y, x = np.ogrid[-center[0]:size[0] - center[0], -center[1]:size[1] - center[1]]
    m = np.zeros(size); m[x * x + y * y <= 1] = 1
    m = gaussian_filter(m, sigma=r)
    return (m / m.max()).astype(np.float32)


def occlude(I, mask):
    return (I * (1 - mask) + gaussian_filter(I, sigma=BLUR) * mask).astype(np.float32)


def score_frame(teacher, obs, mode):
    """obs (4,84,84) uint8. Returns saliency resized to (210,160)."""
    base = teacher.logits(obs)[0] if mode == "actor" else np.atleast_1d(teacher.value(obs))
    g = SIZE // DENSITY + 1
    scores = np.zeros((g, g), np.float32)
    of = obs.astype(np.float32)
    for i in range(0, SIZE, DENSITY):
        for j in range(0, SIZE, DENSITY):
            pert = occlude(of, get_mask((i, j))).astype(np.uint8)
            out = teacher.logits(pert)[0] if mode == "actor" else np.atleast_1d(teacher.value(pert))
            scores[i // DENSITY, j // DENSITY] = 0.5 * float(((base - out) ** 2).sum())
    pmax = scores.max()
    scores = cv2.resize(scores, (160, 210), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return pmax * scores / (scores.max() + 1e-8)


def overlay(saliency, frame_hwc, fudge=100, ch=0):
    S = saliency.astype(np.float32); pmax = S.max(); S = S - S.min()
    S = fudge * pmax * S / (S.max() + 1e-8)
    I = frame_hwc.astype("uint16").copy(); I[:, :, ch] += S.astype("uint16")
    return I.clip(1, 255).astype("uint8")


def collect_states(teacher, seed, want=4, min_enemies=1, max_steps=12000, stride=5):
    """Collect states with >=min_enemies hostile objects (sharks/enemy subs) visible. Enemies appear
    deeper, so play across episode resets up to max_steps."""
    from teacher_port import SeaquestPort
    port = SeaquestPort(sticky=0.0, full_action_space=True, seed=seed)
    rng = np.random.RandomState(seed + 11); noise = np.random.RandomState(seed + 777)
    port.reset(seed=seed, noop_max=30, rng=rng); ep = 0
    picks = []
    for t in range(max_steps):
        obs = port.teacher_obs(); f = port.features()
        n_enemy = f["n_shark"] + f["n_submarine"]
        if f["player_x"] is not None and n_enemy >= min_enemies and t % stride == 0:
            picks.append({"obs": obs.copy(), "render": np.asarray(port.env.render(), np.uint8),
                          "t": t, "feat": {k: f[k] for k in ["player_x", "player_y", "oxygen",
                                          "n_shark", "n_submarine", "n_diver"]},
                          "enemy_xy": list(zip(f.get("enemy_xs", []), f.get("enemy_ys", [])))})
            if len(picks) >= want:
                break
        a = int(teacher.sample_action(obs, teacher.gumbel_from_uniform(noise.uniform(size=18)), 1.0)[0])
        rec = port.agent_step(a)
        if rec["terminated"] or rec["truncated"]:
            ep += 1; port.reset(seed=seed + 1000 + ep, noop_max=30, rng=rng)
    return picks


def draw(teacher, st, tag):
    actor = score_frame(teacher, st["obs"], "actor")
    critic = score_frame(teacher, st["obs"], "critic")
    frame = st["render"]                                      # (210,160,3)
    a_ov = overlay(actor, frame, ch=0)                        # actor -> red
    c_ov = overlay(critic, frame, ch=0)                       # critic -> red
    # side-by-side: frame | actor | critic
    panel = np.concatenate([frame, a_ov, c_ov], axis=1)
    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/seaquest_upstream_{tag}.png"
    cv2.imwrite(p, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    return p, float(actor.max()), float(critic.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=12345); ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--min-enemies", type=int, default=1)
    args = ap.parse_args()
    teacher = CM.load_teacher("A"); CM.prog = prog
    prog(f"[seaquest upstream saliency] action_dim={teacher.action_dim} min_enemies={args.min_enemies}")
    picks = collect_states(teacher, args.seed, want=args.n, min_enemies=args.min_enemies)
    prog(f"collected {len(picks)} states (frame | ACTOR | CRITIC, all upstream-style)")
    for k, st in enumerate(picks):
        p, am, cm = draw(teacher, st, f"s{k}")
        prog(f"  {p}  t={st['t']} feat={st['feat']}  actor_max={am:.2f} critic_max={cm:.4f}")


if __name__ == "__main__":
    main()
