"""Goal-conditioned actor pi(a | s, g) for the OFFLINE contrastive-RL variant.

Eysenbach et al. 2022, offline objective:
    max_pi  E[ (1-lambda) * f(s, a, g)  +  lambda * log pi(a_orig | s, g) ]
The second term is goal-conditioned behavioral cloning (BC): it keeps the
extracted policy ON the demonstrated actions, so argmax/sampling does NOT exploit
out-of-distribution actions where the critic is unreliable (which is what made the
plain argmax-over-critic policy ram into walls). lambda=1 => pure GCBC.

Discrete actions: the actor outputs logits over the action set; the critic term is
the exact expectation E_{a~pi}[f] = sum_a pi(a) f(s,a,g) (differentiable, no reparam).
"""
import torch
import torch.nn as nn


class GoalConditionedActor(nn.Module):
    def __init__(self, frame_size: int = 84, nb_actions: int = 9, frame_stack: int = 1):
        super().__init__()
        self.frame_size = frame_size
        self.nb_actions = nb_actions
        self.frame_stack = frame_stack
        in_ch = 3 * frame_stack
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, in_ch, frame_size, frame_size)).flatten(1).shape[1]
        self.head = nn.Sequential(
            nn.Linear(flat + 2, 512), nn.ReLU(inplace=True),
            nn.Linear(512, nb_actions),
        )

    def forward(self, frames, goals) -> torch.Tensor:
        """frames (B,H,W,3) uint8/float; goals (B,2) normalized -> action logits (B,A)."""
        if frames.dtype == torch.uint8:
            frames = frames.float()
        x = frames.permute(0, 3, 1, 2) / 255.0
        h = self.conv(x).flatten(1)
        return self.head(torch.cat([h, goals.float()], dim=1))
