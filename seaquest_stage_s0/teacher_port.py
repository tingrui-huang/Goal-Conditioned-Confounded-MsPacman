"""OCAtari teacher PORT + branch environment (Sections 14 & 17) — ocatari image.

Replicates EnvPool's Atari preprocessing so the frozen JAX teacher can act on the
OCAtari/ALE clone-restore-able env:
  * frame_skip = 4 (action repeated 4 ALE frames);
  * per agent step: max-pool over the LAST 2 of the 4 raw ALE grayscale frames
    (ALE getScreenGrayscale, the emulator's own luminance — what EnvPool uses);
  * resize to 84x84 via cv2.INTER_AREA;
  * stack the last 4 processed frames, channel order = temporal [oldest..newest]
    (matches the cleanrl eval helper which renders next_obs[0][-1] as the newest);
  * obs dtype uint8 in [0,255]; the teacher Network divides by 255 internally.

Snapshot = (ALE state include_rng, frame-stack deque, max-pool raw buffer, score/
lives bookkeeping). Restore is exact for forward rollouts (verified in Section 15).

Nothing here masks oxygen or alters dynamics.
"""
import sys, os
_REAL_STDOUT = sys.stdout
sys.stdout = open(os.devnull, "w")  # silence vendored OCAtari debug prints
def log(*a):
    print(*a, file=_REAL_STDOUT, flush=True)

from collections import deque
import numpy as np
import cv2
import gymnasium, ale_py  # noqa

ENV_ID = "ALE/Seaquest-v5"
ALE_MEANINGS = ['NOOP', 'FIRE', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT',
                'DOWNRIGHT', 'DOWNLEFT', 'UPFIRE', 'RIGHTFIRE', 'LEFTFIRE', 'DOWNFIRE',
                'UPRIGHTFIRE', 'UPLEFTFIRE', 'DOWNRIGHTFIRE', 'DOWNLEFTFIRE']


def _num(v):
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


class SeaquestPort:
    FRAME_SKIP = 4
    STACK = 4
    SIZE = 84

    def __init__(self, sticky=0.0, full_action_space=True, seed=0,
                 gray_mode="ale", interp="area", maxpool=True):
        from ocatari.core import OCAtari
        self.env = OCAtari(ENV_ID, mode="ram", hud=True, render_mode="rgb_array",
                           frameskip=1, repeat_action_probability=sticky,
                           full_action_space=full_action_space)
        self.env.reset(seed=seed)
        self.ale = self.env._env.unwrapped.ale
        self.n_actions = int(self.env.action_space.n)
        self._stack = deque(maxlen=self.STACK)
        self._prev_gray = None
        self.gray_mode = gray_mode
        self.maxpool = maxpool
        self._interp = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR,
                        "nearest": cv2.INTER_NEAREST}[interp]

    # -- preprocessing ------------------------------------------------------
    def _gray(self):
        if self.gray_mode == "ale":
            g = np.asarray(self.ale.getScreenGrayscale(), dtype=np.uint8)
            if g.ndim == 3:
                g = g[..., 0]
            return g
        # cv2 luminosity from RGB
        rgb = np.asarray(self.ale.getScreenRGB(), dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    def _process(self, two_frames):
        if self.maxpool and two_frames[0] is not None:
            m = np.maximum(two_frames[0], two_frames[1])
        else:
            m = two_frames[1]
        return cv2.resize(m, (self.SIZE, self.SIZE), interpolation=self._interp).astype(np.uint8)

    def reset(self, seed=0, noop_max=0, rng=None):
        self.env.reset(seed=seed)
        self.ale = self.env._env.unwrapped.ale
        n = 0
        if noop_max and rng is not None:
            n = int(rng.randint(1, noop_max + 1))
        for _ in range(n):
            self.env.step(0)
        g = self._gray()
        proc = cv2.resize(g, (self.SIZE, self.SIZE), interpolation=self._interp).astype(np.uint8)
        self._stack.clear()
        for _ in range(self.STACK):
            self._stack.append(proc)
        self._prev_gray = g
        return self.teacher_obs()

    def agent_step(self, action):
        """Repeat `action` for FRAME_SKIP ALE frames; return post-step record."""
        reward = 0.0
        frames = []
        terminated = trunc = False
        for i in range(self.FRAME_SKIP):
            _, r, term, tr, info = self.env.step(int(action))
            reward += float(r)
            frames.append(self._gray())
            terminated = terminated or bool(term)
            trunc = trunc or bool(tr)
            if term or tr:
                break
        # max-pool last 2 frames available
        if len(frames) >= 2:
            two = (frames[-2], frames[-1])
        else:
            two = (self._prev_gray, frames[-1])
        proc = self._process(two)
        self._stack.append(proc)
        self._prev_gray = frames[-1]
        return {"reward": reward, "terminated": terminated, "truncated": trunc,
                "teacher_obs": self.teacher_obs()}

    def teacher_obs(self):
        return np.stack(list(self._stack), axis=0).astype(np.uint8)  # (4,84,84)

    # -- snapshot / restore -------------------------------------------------
    def snapshot(self):
        return {"ale": self.ale.cloneState(include_rng=True),
                "stack": [f.copy() for f in self._stack],
                "prev_gray": None if self._prev_gray is None else self._prev_gray.copy()}

    def restore(self, snap):
        self.ale.restoreState(snap["ale"])
        self._stack = deque(snap["stack"], maxlen=self.STACK)
        self._prev_gray = None if snap["prev_gray"] is None else snap["prev_gray"].copy()

    # -- audited feature extraction (post agent step) -----------------------
    def features(self, include_oxygen=True):
        objs = self.env.objects
        f = {"player_x": None, "player_y": None, "oxygen": None, "score": None,
             "lives": None, "n_shark": 0, "n_submarine": 0, "n_diver": 0,
             "n_collected_diver": 0, "n_player_missile": 0, "n_enemy_missile": 0,
             "enemy_xs": [], "enemy_ys": [], "diver_xs": [], "diver_ys": []}
        for o in objs:
            c = getattr(o, "category", "")
            if c == "Player":
                f["player_x"] = _num(o.x); f["player_y"] = _num(o.y)
            elif c == "OxygenBar":
                f["oxygen"] = _num(getattr(o, "value", None))
            elif c == "PlayerScore":
                f["score"] = _num(getattr(o, "value", None))
            elif c == "Lives":
                f["lives"] = _num(getattr(o, "value", None))
            elif c == "Shark":
                f["n_shark"] += 1; f["enemy_xs"].append(_num(o.x)); f["enemy_ys"].append(_num(o.y))
            elif c == "Submarine":
                f["n_submarine"] += 1; f["enemy_xs"].append(_num(o.x)); f["enemy_ys"].append(_num(o.y))
            elif c == "Diver":
                f["n_diver"] += 1; f["diver_xs"].append(_num(o.x)); f["diver_ys"].append(_num(o.y))
            elif c == "CollectedDiver":
                f["n_collected_diver"] += 1
            elif c == "PlayerMissile":
                f["n_player_missile"] += 1
            elif c == "EnemyMissile":
                f["n_enemy_missile"] += 1
        if not include_oxygen:
            f.pop("oxygen")
        return f


# --------------------------------------------------------------- port eval
def run_port_eval(teacher, episodes=10, seed=1, noop_max=30, episodic_life=True,
                  max_agent_steps=6000):
    import jax, jax.numpy as jnp
    port = SeaquestPort(sticky=0.0, full_action_space=True, seed=seed)
    key = jax.random.PRNGKey(seed)
    rng = np.random.RandomState(seed + 11)
    act_hist = np.zeros(port.n_actions, dtype=np.int64)
    results = []
    for ep in range(episodes):
        port.reset(seed=seed, noop_max=noop_max, rng=rng)
        start_lives = port.features()["lives"]
        ret = 0.0; steps = 0
        while True:
            obs = port.teacher_obs()
            logits, _ = teacher._forward(teacher.network_params, teacher.actor_params,
                                         teacher.critic_params, obs[None])
            key, sub = jax.random.split(key)
            u = jax.random.uniform(sub, shape=logits.shape)
            a = int(np.asarray(jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1))[0])
            act_hist[a] += 1
            rec = port.agent_step(a)
            ret += rec["reward"]; steps += 1
            lives = port.features()["lives"]
            life_lost = (start_lives is not None and lives is not None and lives < start_lives)
            if rec["terminated"] or rec["truncated"] or (episodic_life and life_lost) or steps >= max_agent_steps:
                break
        results.append({"episode": ep, "return": ret, "agent_steps": steps})
        log(f"  [port] ep={ep} return={ret:.1f} steps={steps}")
    rets = np.array([r["return"] for r in results], dtype=np.float64)
    h = act_hist / max(act_hist.sum(), 1)
    ent = float(-(h[h > 0] * np.log(h[h > 0])).sum())
    return {"episodes": results, "return_mean": float(rets.mean()),
            "return_std": float(rets.std()), "return_median": float(np.median(rets)),
            "action_histogram": act_hist.tolist(), "action_entropy_nats": ent,
            "n_distinct_actions": int((act_hist > 0).sum())}


def main():
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--src", required=True)
    ap.add_argument("--tag", default="A")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--native-mean", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from teacher_adapter import CleanRLSeaquestTeacher
    teacher = CleanRLSeaquestTeacher(args.ckpt, args.src, mod_name=f"cleanrl_src_{args.tag}")
    log(f"[port-eval tag {args.tag}] action_dim={teacher.action_dim}")
    res = run_port_eval(teacher, episodes=args.episodes, seed=args.seed)
    out = {"tag": args.tag, "preprocessing": {
            "frame_skip": 4, "maxpool_last2": True, "grayscale": "ale.getScreenGrayscale",
            "resize": "cv2.INTER_AREA 84x84", "stack": 4, "channel_order": "temporal oldest..newest",
            "obs_shape": [4, 84, 84], "dtype": "uint8"},
           "port_eval": res, "native_mean": args.native_mean}
    if args.native_mean:
        out["pct_of_native"] = float(res["return_mean"] / args.native_mean)
        out["meets_60pct_portability_gate"] = bool(res["return_mean"] >= 0.60 * args.native_mean)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    log(f"WROTE {args.out} port_mean={res['return_mean']:.1f} "
        f"pct_native={out.get('pct_of_native')}")


if __name__ == "__main__":
    main()
