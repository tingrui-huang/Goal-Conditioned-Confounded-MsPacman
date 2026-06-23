"""NCE training loop for the contrastive critic (Algorithm 1).

Loss = sigmoid binary cross-entropy on the B x B logit matrix, labels = identity
(diagonal positives, off-diagonal in-batch negatives -> B-1 negatives/positive
for free). No actor network: the discrete policy is argmax over actions at eval
(evaluation/policy.py).

Train TWICE via the `oracle` flag (naive = masked, oracle = unmasked); identical
architecture + hyperparameters otherwise.
"""
import os
import json
import time

import numpy as np
import torch
import torch.nn as nn

from seaquest_ccrl import config as C
from seaquest_ccrl.training.config import TrainConfig, DEFAULT
from seaquest_ccrl.training.dataset_sampler import HindsightSampler
from seaquest_ccrl.models.contrastive_critic import ContrastiveCritic


@torch.no_grad()
def _action_shuffle_delta(critic, sampler, cfg, device, n=512):
    """Lightweight in-training action-use signal: mean f(s,a_true,g) - f(s,a_shuffled,g).
    Positive => the critic scores the demonstrated action above a random one."""
    fr, ac, go = sampler.sample(n)
    sa = critic.sa_encoder(fr, ac); g = critic.g_encoder(go)
    s_true = (sa * g).sum(1)
    perm = torch.randperm(len(ac), device=device)
    s_shuf = (critic.sa_encoder(fr, ac[perm]) * g).sum(1)
    return float((s_true - s_shuf).mean().item())


def train(oracle: bool, cfg: TrainConfig = DEFAULT, game=None, root: str = None,
          device: str = "cpu", verbose: bool = True,
          eval_every: int = 0, eval_episodes: int = 30,
          eval_max_steps: int = 600, tb_logdir: str = None, tb_shuffle_every: int = 2000) -> str:
    """Train one critic. If eval_every > 0, periodically run goal-reaching eval to
    build a success-rate-vs-step learning curve. Loss/diag-acc and eval curves are
    saved next to the checkpoint as history_{tag}.json."""
    if game is None:
        from seaquest_ccrl.games import get_game
        game = get_game("seaquest")
    tag = "oracle" if oracle else "naive"
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    sampler = HindsightSampler(game, oracle=oracle, cfg=cfg, device=device, rng=rng, root=root)
    critic = ContrastiveCritic(cfg.repr_dim, cfg.frame_size, cfg.nb_actions,
                               getattr(cfg, "frame_stack", 1)).to(device)
    opt = torch.optim.Adam(critic.parameters(), lr=cfg.lr)
    # NCE: per anchor i, sigmoid-BCE over the B candidate goals (1 positive on the
    # diagonal + B-1 in-batch negatives), summed over candidates then meaned over
    # anchors. A plain mean over all B*B entries lets the 1:(B-1) imbalance collapse
    # the critic to "predict all-negative" (diag-acc stuck at chance); summing per
    # anchor keeps each positive at full weight against its negatives.
    bce = nn.BCEWithLogitsLoss(reduction="none")
    labels_eye = torch.eye(cfg.batch_size, device=device)
    # Goal-collision fix: when goal positions cluster (Pac-Man), many in-batch
    # negatives sit within the eval radius of the positive -> false negatives ->
    # irreducible BCE floor + capped diag-acc. If cfg.goal_radius > 0, treat ANY
    # in-batch goal within that radius (px) as a positive (label 1), aligning the
    # objective with "reach an eps-cell" instead of "hit the exact pixel".
    goal_radius = float(getattr(cfg, "goal_radius", 0.0) or 0.0)
    if goal_radius > 0:
        _glo = torch.tensor([cfg.goal_x_lo, cfg.goal_y_lo], device=device)
        _gspan = torch.tensor([cfg.goal_x_hi - cfg.goal_x_lo,
                               cfg.goal_y_hi - cfg.goal_y_lo], device=device)

    if verbose:
        print(f"[{tag}] training: {cfg.steps} steps, B={cfg.batch_size}, "
              f"d={cfg.repr_dim}, {sampler.n_ep} episodes, device={device}")

    loss_hist = []     # [[step, mean_loss, diag_acc], ...]
    eval_hist = []      # [[step, success_rate], ...]

    def run_eval(step):
        from seaquest_ccrl.evaluation.evaluate import evaluate  # local import: avoid cycle
        critic.eval()
        res = evaluate(critic, cfg, game, oracle, n_episodes=eval_episodes,
                       max_steps=eval_max_steps, device=device,
                       seed=cfg.seed, verbose=False)
        critic.train()
        eval_hist.append([step, res["success_rate"]])
        if verbose:
            print(f"[{tag}] step {step:6d}  EVAL success {res['success_rate']:.3f} "
                  f"({eval_episodes} eps)")

    # Mixed precision: the bottleneck is conv compute (84x84 x B=256). T4 tensor
    # cores give ~2-3x on fp16, so use AMP on CUDA; cudnn.benchmark picks fast
    # conv kernels. CPU path is unchanged (AMP disabled).
    use_amp = str(device).startswith("cuda")
    amp_device = "cuda" if use_amp else "cpu"
    if use_amp:
        torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Opt-in TensorBoard (lean v1): loss, diag_acc, logit_gap, grad_norm + periodic
    # action_shuffle_delta. When tb_logdir is None this is a no-op (identical behavior).
    writer = None
    SHUFFLE_EVERY = max(1, int(tb_shuffle_every))
    if tb_logdir:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(tb_logdir)

    critic.train()
    # accumulate loss on-device; .item() forces a CPU<->GPU sync, so do it only at
    # log time (per-step .item() serializes execution and throttles GPU throughput).
    running = torch.zeros((), device=device)
    arange_B = torch.arange(cfg.batch_size, device=device)
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        frames, actions, goals = sampler.sample(cfg.batch_size)
        # build the (B,B) target: diagonal + any in-batch goal within goal_radius px
        if goal_radius > 0:
            with torch.no_grad():
                raw = goals * _gspan + _glo               # de-normalize to pixels
                D = torch.cdist(raw, raw)                  # (B,B) pairwise px dist
                labels = (D <= goal_radius).float()        # near goals = positives
        else:
            labels = labels_eye
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=amp_device, dtype=torch.float16, enabled=use_amp):
            logits = critic(frames, actions, goals)       # (B,B)
            loss = bce(logits, labels).sum(dim=1).mean()  # NCE: sum candidates, mean anchors
        scaler.scale(loss).backward()
        # grad norm must be read from the SCALED grads (before scaler.step unscales them in
        # place), then divided by the loss scale -> true norm on both CPU (scale=1) and GPU/AMP.
        grad_norm_log = None
        if writer is not None and step % cfg.log_every == 0:
            scale = scaler.get_scale() if use_amp else 1.0
            gsq = sum(float((p.grad.detach().float().norm() ** 2).item())
                      for p in critic.parameters() if p.grad is not None)
            grad_norm_log = (gsq ** 0.5) / scale
        scaler.step(opt)
        scaler.update()

        running += loss.detach()
        if step % cfg.log_every == 0:
            with torch.no_grad():
                pick = logits.argmax(dim=1)
                if goal_radius > 0:                        # soft acc: pick within radius
                    raw = goals * _gspan + _glo
                    acc = (torch.norm(raw[pick] - raw, dim=1) <= goal_radius).float().mean().item()
                else:
                    acc = (pick == arange_B).float().mean().item()
            mean_loss = (running / cfg.log_every).item()
            running.zero_()
            loss_hist.append([step, mean_loss, acc])
            if writer is not None:
                with torch.no_grad():
                    lf = logits.float()                      # fp32: fp16 logits.sum() overflows -> inf
                    diag = lf.diagonal()
                    neg = (lf.sum() - diag.sum()) / (cfg.batch_size * cfg.batch_size - cfg.batch_size)
                    logit_gap = float((diag.mean() - neg).item())
                writer.add_scalar("train/loss", mean_loss, step)
                writer.add_scalar("train/diag_acc", acc, step)
                writer.add_scalar("train/logit_gap", logit_gap, step)
                if grad_norm_log is not None:
                    writer.add_scalar("train/grad_norm", grad_norm_log, step)
            if writer is not None and step % SHUFFLE_EVERY == 0:
                writer.add_scalar("diag/action_shuffle_delta",
                                  _action_shuffle_delta(critic, sampler, cfg, device), step)
            if verbose:
                rate = step / (time.time() - t0)
                print(f"[{tag}] step {step:6d}/{cfg.steps}  loss {mean_loss:.4f}"
                      f"  diag-acc {acc:.3f}  {rate:.1f} it/s")
        if eval_every and (step % eval_every == 0):
            run_eval(step)

    if eval_every and (not eval_hist or eval_hist[-1][0] != cfg.steps):
        run_eval(cfg.steps)   # ensure the curve ends at the final step

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    path = os.path.join(cfg.ckpt_dir, f"critic_{tag}.pt")
    torch.save({"state_dict": critic.state_dict(),
                "cfg": cfg.__dict__, "oracle": oracle}, path)
    with open(os.path.join(cfg.ckpt_dir, f"history_{tag}.json"), "w") as f:
        json.dump({"seed": cfg.seed, "oracle": oracle, "steps": cfg.steps,
                   "loss": loss_hist, "eval": eval_hist}, f, indent=2)
    if writer is not None:
        writer.flush(); writer.close()
    if verbose:
        print(f"[{tag}] saved -> {path}  ({time.time()-t0:.1f}s)")
    return path


def load_critic(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = TrainConfig(**ckpt["cfg"])
    critic = ContrastiveCritic(cfg.repr_dim, cfg.frame_size, cfg.nb_actions,
                               getattr(cfg, "frame_stack", 1)).to(device)
    critic.load_state_dict(ckpt["state_dict"])
    critic.eval()
    return critic, cfg, ckpt["oracle"]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--steps", type=int, default=DEFAULT.steps)
    args = ap.parse_args()
    cfg = TrainConfig(steps=args.steps)
    train(oracle=args.oracle, cfg=cfg)
