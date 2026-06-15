"""Small fixed CNN probe over the four-frame state, with optional scalar extras
(action one-hot, oxygen). Same backbone for every matched pair; the ONLY difference
between a pair is whether the oxygen scalar is appended to `extras`.
"""
import torch
import torch.nn as nn


class ProbeNet(nn.Module):
    def __init__(self, extra_dim=0, out_dim=1, hidden=256, frame_size=84):
        super().__init__()
        self.extra_dim = extra_dim
        self.conv = nn.Sequential(
            nn.Conv2d(12, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, 12, frame_size, frame_size)).flatten(1).shape[1]
        self.head = nn.Sequential(
            nn.Linear(flat + extra_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim))

    def forward(self, frames_uint8, extras=None):
        x = frames_uint8.float().permute(0, 3, 1, 2) / 255.0   # (B,12,84,84)
        h = self.conv(x).flatten(1)
        if self.extra_dim:
            h = torch.cat([h, extras], dim=1)
        return self.head(h)
