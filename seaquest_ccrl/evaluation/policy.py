"""Goal-conditioned policies (argmax over critic, or a trained actor).

Both maintain a k-frame stack buffer (cfg.frame_stack). The raw frame passed in is
already the run's view (masked for naive, unmasked for oracle), so MASK-then-STACK
holds: a naive policy stacks k ghost-inpainted frames (only Pac-Man motion visible,
no ghost-motion leak). reset() MUST be called at each episode start.
"""
import numpy as np

from seaquest_ccrl.models.sa_encoder import preprocess_frames


class _StackBuffer:
    """Resize each raw frame to (fs,fs,3) and keep the last k; stack on channel axis."""
    def __init__(self, frame_size, frame_stack):
        self.fs = frame_size
        self.k = max(1, int(frame_stack))
        self.buf = []

    def reset(self):
        self.buf = []

    def push_stack(self, frame_uint8) -> np.ndarray:
        small = preprocess_frames(frame_uint8[None], self.fs)[0]   # (fs,fs,3) uint8
        self.buf.append(small)
        if len(self.buf) > self.k:
            self.buf = self.buf[-self.k:]
        frames = self.buf[:]                                       # pad front with oldest
        while len(frames) < self.k:
            frames = [frames[0]] + frames
        return np.concatenate(frames, axis=2)                      # (fs,fs,3k) oldest->newest


class ContrastiveGCPolicy:
    def __init__(self, critic, cfg, device="cpu", temperature: float = 0.0):
        self.critic = critic
        self.cfg = cfg
        self.device = device
        self.temperature = temperature   # 0 => greedy argmax
        self._rng = np.random.default_rng(cfg.seed)
        self._stack = _StackBuffer(critic.frame_size, getattr(cfg, "frame_stack", 1))

    def reset(self):
        self._stack.reset()

    def act(self, frame_uint8: np.ndarray, goal_xy) -> int:
        goal_norm = self.cfg.normalize_goal(goal_xy)
        obs_small = self._stack.push_stack(frame_uint8)
        scores = self.critic.score_all_actions(obs_small, goal_norm, self.device)
        if self.temperature and self.temperature > 0:
            logits = scores / self.temperature
            p = np.exp(logits - logits.max()); p /= p.sum()
            return int(self._rng.choice(len(scores), p=p))
        return int(np.argmax(scores))


class ActorGCPolicy:
    """Policy from a trained goal-conditioned actor pi(a|s,g): argmax (or sample)
    over the actor's action logits. Stays on demonstrated actions (BC-regularized)."""

    def __init__(self, actor, cfg, device="cpu", temperature: float = 0.0):
        self.actor = actor
        self.cfg = cfg
        self.device = device
        self.temperature = temperature
        self._rng = np.random.default_rng(cfg.seed)
        self._stack = _StackBuffer(actor.frame_size, getattr(actor, "frame_stack", 1))

    def reset(self):
        self._stack.reset()

    def act(self, frame_uint8: np.ndarray, goal_xy) -> int:
        import torch
        goal_norm = self.cfg.normalize_goal(goal_xy)
        obs_small = self._stack.push_stack(frame_uint8)
        with torch.no_grad():
            frames = torch.from_numpy(obs_small[None]).to(self.device)
            goals = torch.as_tensor(goal_norm, dtype=torch.float32, device=self.device)[None]
            logits = self.actor(frames, goals).squeeze(0).cpu().numpy()
        if self.temperature and self.temperature > 0:
            z = logits / self.temperature
            p = np.exp(z - z.max()); p /= p.sum()
            return int(self._rng.choice(len(logits), p=p))
        return int(np.argmax(logits))
