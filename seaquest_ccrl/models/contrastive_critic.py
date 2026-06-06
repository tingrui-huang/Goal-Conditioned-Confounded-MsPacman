"""Contrastive critic f(s, a, s_g) = ϕ(s, a) · ψ(s_g)  (Eysenbach et al. 2022).

NO reconstruction, NO decoder, NO world model (acceptance check A). The critic
is purely the dot product of the two encoders' embeddings.
"""
import numpy as np
import torch
import torch.nn as nn

from seaquest_ccrl.models.sa_encoder import SAEncoder, preprocess_frames
from seaquest_ccrl.models.g_encoder import GEncoder


class ContrastiveCritic(nn.Module):
    def __init__(self, repr_dim: int = 256, frame_size: int = 84, nb_actions: int = 18):
        super().__init__()
        self.frame_size = frame_size
        self.nb_actions = nb_actions
        self.sa_encoder = SAEncoder(repr_dim, frame_size, nb_actions)
        self.g_encoder = GEncoder(repr_dim)

    # -- training: full B x B logit matrix ----------------------------------
    def forward(self, frames, actions, goals) -> torch.Tensor:
        """frames (B,H,W,3); actions (B,); goals (B,2) -> logits (B,B).

        logits[i,j] = f(s_i, a_i, s_g_j). Diagonal = positives (acceptance C).
        """
        sa = self.sa_encoder(frames, actions)            # (B,d)
        g = self.g_encoder(goals)                        # (B,d)
        return torch.einsum("ik,jk->ij", sa, g)          # (B,B)

    # -- eval: score one state against all actions for a fixed goal ---------
    @torch.no_grad()
    def score_all_actions(self, frame_uint8: np.ndarray, goal_norm: np.ndarray,
                          device="cpu") -> np.ndarray:
        """frame: (210,160,3) uint8 raw env frame; goal_norm: (2,) normalized.

        Returns (nb_actions,) critic scores f(s, a, g) for every discrete action.
        Resize/normalize handled here so eval matches training preprocessing.
        """
        self.eval()
        small = preprocess_frames(frame_uint8[None], self.frame_size)  # (1,size,size,3)
        frames = torch.from_numpy(small).to(device).repeat(self.nb_actions, 1, 1, 1)
        actions = torch.arange(self.nb_actions, device=device)
        sa = self.sa_encoder(frames, actions)                          # (A,d)
        g = self.g_encoder(torch.as_tensor(goal_norm, dtype=torch.float32,
                                           device=device)[None])       # (1,d)
        return (sa @ g.T).squeeze(1).cpu().numpy()                     # (A,)
