"""Reproduce the UPSTREAM teacher-repo saliency drawing (Greydanus-2017, saliency/saliency.py +
gen_saliency.py) on our DIAMOND Pong teacher, to explain why the upstream figures appear to "look at
the opponent" while our Stage-1K (actor/policy, semantic-JS) figures do not.

Two upstream choices we did NOT use in Stage-1K:
  1. it draws BOTH actor (policy logits) AND critic (value) saliency  -> score = 0.5*||L-l||^2 on
     logits (actor) OR on the value scalar (critic);
  2. it normalizes EACH frame to its own max -> the relatively-strongest region is always bright.

Hypothesis: the opponent lights up in the CRITIC (value) saliency, not the actor — i.e. the value
of the state depends on the opponent, but the action does not (consistent with Stage-1J/1K).

This is a visualization-only diagnostic. Faithful reimplementation of the upstream method/params
(radius=4, density=4, occlude blur sigma=3); no training, no model change.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from PIL import Image

from .. import config as C
from ..objects import extract_pong_objects
from ..teacher.load_teacher import TeacherPolicy, load_teacher, make_env

OUT = Path("pong_action_gate/results/stage1k_teacher_policy_saliency/upstream_style")
RADIUS, DENSITY, BLUR = 4, 4, 3


# ---- upstream saliency primitives (faithful to saliency/saliency.py) -------- #
def get_mask(center, size=(64, 64), r=RADIUS):
    y, x = np.ogrid[-center[0]:size[0] - center[0], -center[1]:size[1] - center[1]]
    m = np.zeros(size); m[x * x + y * y <= 1] = 1
    m = gaussian_filter(m, sigma=r)
    return m / m.max()


def occlude(I, mask):
    return I * (1 - mask) + gaussian_filter(I, sigma=BLUR) * mask     # blur the masked region


def imresize(img, size_wh):
    return np.array(Image.fromarray(img).resize(size_wh, Image.BILINEAR)).astype(np.float32)


@torch.no_grad()
def _out(model, obs4, h, mode):
    t = torch.as_tensor(obs4, dtype=torch.float32)
    lg, val, _ = model.predict_act_value(t, h)
    return lg if mode == "actor" else val.reshape(1, -1)


def score_frame(model, obs4, h, mode, r=RADIUS, d=DENSITY, res_wh=(160, 210)):
    """obs4: (1,3,64,64) float in [-1,1]; h: (hx,cx). Returns saliency resized to (210,160)."""
    L = _out(model, obs4, h, mode)
    g = 64 // d + 1
    scores = np.zeros((g, g), np.float32)
    for i in range(0, 64, d):
        for j in range(0, 64, d):
            l = _out(model, occlude(obs4, get_mask((i, j))), h, mode)
            scores[i // d, j // d] = float((L - l).pow(2).sum().mul(0.5))
    pmax = scores.max()
    scores = imresize(scores, res_wh)                                 # -> (210,160)
    return pmax * scores / (scores.max() + 1e-8)


def saliency_on_frame(saliency, atari_chw, fudge=100, channel=0):
    pmax = saliency.max(); S = saliency.astype(np.float32); S -= S.min()
    S = fudge * pmax * S / (S.max() + 1e-8)
    I = atari_chw.astype("uint16"); I[channel] += S.astype("uint16")
    return I.clip(1, 255).astype("uint8")


# --------------------------------------------------------------------------- #
def collect_decision_state(seed=30000, max_steps=2000):
    model, _ = load_teacher(device="cpu"); teacher = TeacherPolicy(model, "cpu")
    env = make_env(replace(C.M1Config(), device="cpu"), num_envs=1); ale = env.env.ale
    torch.manual_seed(seed); obs, _ = env.reset(seed=[seed]); hx, cx = teacher.initial_state(1)
    prev = extract_pong_objects(ale.getRAM()); picks = []
    for t in range(max_steps):
        o = extract_pong_objects(ale.getRAM())
        dx = (o.ball_x - prev.ball_x) if (o.ball_present and prev.ball_present and prev.ball_x is not None) else None
        is_dec = bool(o.ball_present and dx is not None and dx > 0 and o.ball_x is not None and o.ball_x >= 120
                      and o.opp_y is not None)
        with torch.no_grad():
            lg, val, (hx2, cx2) = model.predict_act_value(obs[:, C.TEACHER_OBS_SLICE], (hx, cx))
        if is_dec:
            picks.append({"obs4": obs[:, C.TEACHER_OBS_SLICE].numpy().copy(),
                          "hx": hx.clone(), "cx": cx.clone(),
                          "render": np.asarray(env.env.render(), np.uint8),
                          "logits": lg.numpy()[0], "value": float(val.reshape(-1)[0]),
                          "ball": (o.ball_x, o.ball_y), "opp_y": o.opp_y, "player_y": o.player_y, "t": t})
            if len(picks) >= 8:
                break
        a = int(torch.distributions.Categorical(logits=lg / C.BEHAVIOR_TEMPERATURE).sample().item())
        prev = o; obs, *_ = env.step(a); hx, cx = hx2, cx2
    env.close()
    return model, picks


def collect_matched_state(seed=30000, want_ag=0, want_op=2, max_steps=6000):
    """Find a state matching a target score (agent vs opponent points) with the opponent paddle as
    HIGH (near the top, small opp_y) as possible, ball visible — to match a specific game frame."""
    model, _ = load_teacher(device="cpu"); teacher = TeacherPolicy(model, "cpu")
    env = make_env(replace(C.M1Config(), device="cpu"), num_envs=1); ale = env.env.ale
    torch.manual_seed(seed); obs, _ = env.reset(seed=[seed]); hx, cx = teacher.initial_state(1)
    ag = op = 0; best = None
    for t in range(max_steps):
        o = extract_pong_objects(ale.getRAM())
        with torch.no_grad():
            lg, val, (hx2, cx2) = model.predict_act_value(obs[:, C.TEACHER_OBS_SLICE], (hx, cx))
        score_ok = (want_ag is None or ag == want_ag) and (want_op is None or op == want_op)
        # opponent in the TOP region (small opp_y) and ball approaching the agent (decision-like)
        if score_ok and o.ball_present and o.opp_y is not None and o.opp_y <= 110:
            cand = {"obs4": obs[:, C.TEACHER_OBS_SLICE].numpy().copy(), "hx": hx.clone(), "cx": cx.clone(),
                    "render": np.asarray(env.env.render(), np.uint8), "logits": lg.numpy()[0],
                    "value": float(val.reshape(-1)[0]), "ball": (o.ball_x, o.ball_y), "opp_y": o.opp_y,
                    "player_y": o.player_y, "t": t, "score": (ag, op)}
            if best is None or o.opp_y < best["opp_y"]:     # opp near TOP = smallest y
                best = cand
        a = int(torch.distributions.Categorical(logits=lg / C.BEHAVIOR_TEMPERATURE).sample().item())
        obs, rew, end, trunc, _ = env.step(a); hx, cx = hx2, cx2
        r = float(rew.item()); ag += int(r > 0); op += int(r < 0)
        if want_op is not None and (op > want_op or ag > want_ag) and best is not None:
            break
        if bool((end | trunc).item()):
            obs, _ = env.reset(seed=[seed + 7000 + t]); hx, cx = teacher.initial_state(1); ag = op = 0
    env.close()
    return model, best


def draw(model, st, tag):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = (st["hx"], st["cx"]); obs4 = st["obs4"]
    actor = score_frame(model, obs4, h, "actor")
    critic = score_frame(model, obs4, h, "critic")
    frame_chw = st["render"].transpose(2, 0, 1).astype(np.uint8)      # (3,210,160)

    # (a) faithful upstream combined: actor + critic both on the RED channel
    combined = saliency_on_frame(critic, saliency_on_frame(actor, frame_chw.copy(), channel=0), channel=0)
    # (b) separated so we can SEE which one attends to the opponent: actor->red, critic->green
    sep = saliency_on_frame(critic, saliency_on_frame(actor, frame_chw.copy(), channel=0), channel=1)

    fig, ax = plt.subplots(1, 4, figsize=(15, 4.2))
    ax[0].imshow(st["render"]); ax[0].set_title(f"frame (t={st['t']})  ball={st['ball']} opp_y={st['opp_y']}"); ax[0].axis("off")
    ax[1].imshow(np.kron(actor, np.ones((1, 1))), cmap="hot"); ax[1].set_title("ACTOR (policy) saliency\nupstream L2-logit, per-frame norm"); ax[1].axis("off")
    ax[2].imshow(critic, cmap="hot"); ax[2].set_title("CRITIC (value) saliency\nupstream L2-value, per-frame norm"); ax[2].axis("off")
    ax[3].imshow(sep.transpose(1, 2, 0)); ax[3].set_title("overlay: actor=RED, critic=GREEN"); ax[3].axis("off")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"upstream_style_{tag}.png"
    fig.suptitle("Upstream-style saliency on our DIAMOND Pong teacher (actor vs critic)", fontsize=11)
    fig.tight_layout(); fig.savefig(p, dpi=90); plt.close(fig)
    # quantify opponent-region saliency for actor vs critic (opponent paddle native x=16 -> mid-left)
    return p, {"actor_max": float(actor.max()), "critic_max": float(critic.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=30000)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--match-score", type=int, nargs=2, default=None,
                    help="AGENT OPP target score; finds that state with opponent paddle near the top")
    args = ap.parse_args()
    if args.match_score is not None:
        ag, op = args.match_score
        for sd in range(args.seed, args.seed + 40):              # search seeds until the score is reached
            model, st = collect_matched_state(sd, want_ag=ag, want_op=op)
            if st is not None:
                p, info = draw(model, st, f"score{ag}_{op}")
                print(f"  saved {p}  seed={sd} t={st['t']} score(agent,opp)={st['score']} "
                      f"opp_y={st['opp_y']} ball={st['ball']}  actor_max={info['actor_max']:.2f} "
                      f"critic_max={info['critic_max']:.3f}")
                return
        print(f"could not reach score agent={ag} opp={op} in the searched seeds")
        return
    model, picks = collect_decision_state(args.seed)
    print(f"collected {len(picks)} decision states")
    for k, st in enumerate(picks[:args.n]):
        p, info = draw(model, st, f"s{k}")
        print(f"  saved {p}  actor_max={info['actor_max']:.3f} critic_max={info['critic_max']:.3f}")


if __name__ == "__main__":
    main()
