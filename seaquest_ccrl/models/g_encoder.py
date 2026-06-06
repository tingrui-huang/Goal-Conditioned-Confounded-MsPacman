"""ψ(s_g): goal encoder.

Goals are 2D submarine POSITIONS (x, y), NOT pixel frames (acceptance check F).
A small MLP maps the normalized position to the same d-dim space as ϕ, so the
critic f(s,a,s_g) = ϕ(s,a) · ψ(s_g) is a plain dot product.
"""
import torch
import torch.nn as nn


class GEncoder(nn.Module):
    def __init__(self, repr_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 256), nn.ReLU(inplace=True),
            nn.Linear(256, repr_dim),
        )

    def forward(self, goal_pos: torch.Tensor) -> torch.Tensor:
        """goal_pos: (B,2) float (normalized) -> (B,d)."""
        return self.net(goal_pos.float())
