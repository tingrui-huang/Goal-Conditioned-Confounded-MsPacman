"""ϕ(s, a): state-action encoder.

CNN over a SINGLE masked pixel frame (Markov constraint, acceptance check B --
no frame stacking, no recurrence) concatenated with the one-hot action, mapped
to a d-dim embedding. Frame preprocessing (resize + normalize to [0,1]) lives
here so train and eval share the exact same pipeline.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def preprocess_frames(frames_uint8: np.ndarray, size: int, chunk: int = 256) -> np.ndarray:
    """(N,210,160,3) uint8 -> (N,size,size,3) uint8 via area-average resize.

    Resize is done ONCE at load (sampler) and ONCE per eval step, never inside
    the forward pass. Normalisation to float[0,1] happens in `forward`. Uses
    torch's area interpolation (no cv2 dependency); chunked to bound memory.
    """
    out = np.empty((frames_uint8.shape[0], size, size, 3), dtype=np.uint8)
    for s in range(0, len(frames_uint8), chunk):
        e = min(s + chunk, len(frames_uint8))
        t = torch.from_numpy(frames_uint8[s:e]).permute(0, 3, 1, 2).float()
        t = F.interpolate(t, size=(size, size), mode="area")
        out[s:e] = t.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8).numpy()
    return out


class SAEncoder(nn.Module):
    def __init__(self, repr_dim: int = 256, frame_size: int = 84, nb_actions: int = 18):
        super().__init__()
        self.frame_size = frame_size
        self.nb_actions = nb_actions
        # 3 conv layers, channels [32,64,128], 3x3, stride 2, padding 1, ReLU
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, frame_size, frame_size)
            flat = self.conv(dummy).flatten(1).shape[1]
        self.flat_dim = flat
        self.head = nn.Sequential(
            nn.Linear(flat + nb_actions, 512), nn.ReLU(inplace=True),
            nn.Linear(512, repr_dim),
        )

    def forward(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """frames: (B,size,size,3) uint8 OR float; actions: (B,) int64 -> (B,d)."""
        if frames.dtype == torch.uint8:
            frames = frames.float()
        x = frames.permute(0, 3, 1, 2) / 255.0          # (B,3,H,W) in [0,1]
        h = self.conv(x).flatten(1)                       # (B, flat)
        a = F.one_hot(actions.long(), self.nb_actions).float()  # (B, nb_actions)
        return self.head(torch.cat([h, a], dim=1))        # (B, d)
