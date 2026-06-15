"""Eysenbach-style in-batch NCE (sigmoid BCE over the full BxB logit matrix).
NO TD/Bellman/target-net/actor/reward/aux losses.
"""
import torch
import torch.nn.functional as F


def nce_logits(sa_repr, g_repr):
    """logits[i,j] = phi(s_i,a_i) . psi(g_j) / sqrt(D)."""
    d = sa_repr.shape[-1]
    return sa_repr @ g_repr.t() / (d ** 0.5)


def nce_loss(sa_repr, g_repr):
    """Symmetric-free in-batch contrastive: sigmoid BCE, labels = identity matrix.
    Returns (loss, diagnostics dict). Hard-asserts finiteness."""
    B = sa_repr.shape[0]
    logits = nce_logits(sa_repr, g_repr)
    labels = torch.eye(B, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    assert torch.isfinite(loss).all(), "non-finite NCE loss"
    with torch.no_grad():
        diag = torch.diagonal(logits)
        off = logits[~torch.eye(B, dtype=torch.bool, device=logits.device)]
        diags = {
            "loss": float(loss.item()),
            "pos_logit_mean": float(diag.mean().item()),
            "neg_logit_mean": float(off.mean().item()),
            "pos_neg_margin": float(diag.mean().item() - off.mean().item()),
            "repr_norm": float(sa_repr.norm(dim=-1).mean().item()),
        }
    return loss, diags
