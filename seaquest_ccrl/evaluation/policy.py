"""Goal-conditioned policy = argmax over actions of the contrastive critic.

For discrete actions no separate actor is needed (Eysenbach et al. 2022, sec on
discrete control): at state s with goal g, score f(s, a, g) for all 18 actions
and pick the best (greedy) or sample softmax(scores / temperature).
"""
import numpy as np


class ContrastiveGCPolicy:
    def __init__(self, critic, cfg, device="cpu", temperature: float = 0.0):
        self.critic = critic
        self.cfg = cfg
        self.device = device
        self.temperature = temperature   # 0 => greedy argmax
        self._rng = np.random.default_rng(cfg.seed)

    def act(self, frame_uint8: np.ndarray, goal_xy) -> int:
        """frame: raw (210,160,3) uint8 (already masked/oracle per the run);
        goal_xy: (2,) pixel position target."""
        goal_norm = self.cfg.normalize_goal(goal_xy)
        scores = self.critic.score_all_actions(frame_uint8, goal_norm, self.device)
        if self.temperature and self.temperature > 0:
            logits = scores / self.temperature
            p = np.exp(logits - logits.max())
            p /= p.sum()
            return int(self._rng.choice(len(scores), p=p))
        return int(np.argmax(scores))


class ActorGCPolicy:
    """Policy from a trained goal-conditioned actor pi(a|s,g): argmax (or sample)
    over the actor's action logits. This is the offline BC-regularized policy that
    stays on demonstrated actions, unlike argmax-over-critic."""

    def __init__(self, actor, cfg, device="cpu", temperature: float = 0.0):
        self.actor = actor
        self.cfg = cfg
        self.device = device
        self.temperature = temperature
        self._rng = np.random.default_rng(cfg.seed)

    def act(self, frame_uint8: np.ndarray, goal_xy) -> int:
        import torch
        from seaquest_ccrl.models.sa_encoder import preprocess_frames
        goal_norm = self.cfg.normalize_goal(goal_xy)
        small = preprocess_frames(frame_uint8[None], self.actor.frame_size)
        with torch.no_grad():
            frames = torch.from_numpy(small).to(self.device)
            goals = torch.as_tensor(goal_norm, dtype=torch.float32, device=self.device)[None]
            logits = self.actor(frames, goals).squeeze(0).cpu().numpy()
        if self.temperature and self.temperature > 0:
            z = logits / self.temperature
            p = np.exp(z - z.max()); p /= p.sum()
            return int(self._rng.choice(len(logits), p=p))
        return int(np.argmax(logits))
