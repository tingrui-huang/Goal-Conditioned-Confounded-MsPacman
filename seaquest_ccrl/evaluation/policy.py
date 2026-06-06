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
