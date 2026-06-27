"""Train the offline BC-regularized actor on top of a FROZEN contrastive critic.

Objective (maximize), per Eysenbach et al. 2022 offline variant:
    (1-lambda) * E_{a~pi}[ f(s,a,g) ]  +  lambda * log pi(a_orig | s, g)
- critic term pulls pi toward high-value actions (uses the frozen critic);
- BC term pulls pi toward the demonstrated action a_orig (the navigation guardrail).

The actor sees the SAME view as its critic (naive=masked, oracle=unmasked), so a
naive actor must imitate a ghost-dependent demonstrated action from a ghost-masked
state -> it can't -> that's exactly where the confounding gap should appear.

Reuses an existing critic checkpoint (no critic retrain). __main__ trains a naive
and an oracle actor, evaluates both with the actor policy, and prints the gap.
"""
import os
import time
import argparse

import numpy as np
import torch
import torch.nn.functional as F

from seaquest_ccrl.training.config import TrainConfig
from seaquest_ccrl.training.dataset_sampler import HindsightSampler
from seaquest_ccrl.models.actor import GoalConditionedActor


def train_actor(critic, game, cfg: TrainConfig, oracle: bool, lam: float = 0.5,
                steps: int = 20000, device: str = "cpu", verbose: bool = True,
                root: str = None, ent_coef: float = 0.0):
    """ent_coef: entropy bonus coefficient. The original CRL actor is a continuous
    Gaussian policy with inherent entropy; the discrete categorical analog needs an
    explicit H(pi) term, else maximising a near-flat critic (lam=0) collapses the
    softmax to a constant action (vanishing gradient). Default 0.0 keeps prior behavior."""
    critic.eval()
    for p in critic.parameters():
        p.requires_grad_(False)
    rng = np.random.default_rng(cfg.seed + 777)
    sampler = HindsightSampler(game, oracle=oracle, cfg=cfg, device=device, rng=rng, root=root)
    actor = GoalConditionedActor(cfg.frame_size, cfg.nb_actions,
                                 getattr(cfg, "frame_stack", 1)).to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    use_amp = str(device).startswith("cuda")
    if use_amp:
        torch.backends.cudnn.benchmark = True
    # AMP is used ONLY for the frozen-critic inference (the 18 forwards = the compute
    # bottleneck). The ACTOR forward + policy loss run in FP32: fp16 policy-gradients
    # corrupted actor training on GPU (pure-BC stuck at chance while identical fp32 code
    # learns to BC-acc ~0.3+). No GradScaler needed once the trained path is fp32.
    A = cfg.nb_actions
    tag = "oracle" if oracle else "naive"
    t0 = time.time()
    running = torch.zeros((), device=device)

    for step in range(1, steps + 1):
        frames, a_orig, goals = sampler.sample(cfg.batch_size)
        with torch.no_grad(), torch.autocast(device_type="cuda" if use_amp else "cpu",
                                              dtype=torch.float16, enabled=use_amp):
            g_emb = critic.g_encoder(goals)                      # (B,d)
            f_all = torch.stack([
                (critic.sa_encoder(frames, torch.full((frames.shape[0],), a,
                                   device=device, dtype=torch.long)) * g_emb).sum(1)
                for a in range(A)], dim=1).float()               # (B,A) critic scores
        logits = actor(frames, goals)                        # (B,A)  FP32 actor forward
        logp = F.log_softmax(logits, dim=1)
        pi = logp.exp()
        critic_term = (pi * f_all).sum(1).mean()             # E_pi[f]
        bc_term = logp.gather(1, a_orig.view(-1, 1)).squeeze(1).mean()  # log pi(a_orig)
        ent_term = -(pi * logp).sum(1).mean()                # H(pi): max-ent bonus
        loss = -((1.0 - lam) * critic_term + lam * bc_term + ent_coef * ent_term)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running += loss.detach()
        if verbose and step % 500 == 0:
            with torch.no_grad():
                bc_acc = (logits.argmax(1) == a_orig).float().mean().item()
                ent_now = (-(pi * logp).sum(1).mean()).item()    # H(pi); ln(A) = uniform
            print(f"[actor:{tag}] step {step:6d}/{steps}  loss {(running/500).item():.4f}"
                  f"  BC-acc {bc_acc:.3f}  H {ent_now:.3f}  {step/(time.time()-t0):.1f} it/s")
            running.zero_()

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    path = os.path.join(cfg.ckpt_dir, f"actor_{tag}.pt")
    torch.save({"state_dict": actor.state_dict(), "cfg": cfg.__dict__,
                "oracle": oracle, "lambda": lam}, path)
    if verbose:
        print(f"[actor:{tag}] saved -> {path} ({time.time()-t0:.0f}s)")
    return actor


def load_actor(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = TrainConfig(**ck["cfg"])
    actor = GoalConditionedActor(cfg.frame_size, cfg.nb_actions,
                                 getattr(cfg, "frame_stack", 1)).to(device)
    actor.load_state_dict(ck["state_dict"])
    actor.eval()
    return actor, cfg, ck["oracle"]


def _run_pair(naive_ckpt, oracle_ckpt, game_name, lam, steps, eval_episodes,
              max_steps, device):
    from seaquest_ccrl.training.train_critic import load_critic
    from seaquest_ccrl.games import get_game
    from seaquest_ccrl.evaluation.policy import ActorGCPolicy
    from seaquest_ccrl.evaluation.evaluate import evaluate
    game = get_game(game_name)
    res = {}
    for oracle, ckpt in [(False, naive_ckpt), (True, oracle_ckpt)]:
        if not ckpt or not os.path.exists(ckpt):
            print(f"(skip {'oracle' if oracle else 'naive'}: missing {ckpt})"); continue
        critic, cfg, ck_oracle = load_critic(ckpt, device)
        cfg.ckpt_dir = os.path.dirname(os.path.abspath(ckpt))   # save actor next to its critic
        actor = train_actor(critic, game, cfg, oracle=ck_oracle, lam=lam,
                            steps=steps, device=device)
        pol = ActorGCPolicy(actor, cfg, device=device)
        r = evaluate(critic, cfg, game, ck_oracle, n_episodes=eval_episodes,
                     max_steps=max_steps, device=device, policy=pol, verbose=False)
        res["oracle" if ck_oracle else "naive"] = r["success_rate"]
        print(f"  [{'oracle' if ck_oracle else 'naive'}] actor success: {r['success_rate']:.3f} "
              f"({r['successes']}/{r['n_episodes']})")
    if "naive" in res and "oracle" in res:
        print(f"\n  ACTOR GAP (oracle - naive) = {res['oracle'] - res['naive']:+.3f}  "
              f"(naive {res['naive']:.3f}, oracle {res['oracle']:.3f}, lambda={lam})")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--naive-ckpt", required=True)
    ap.add_argument("--oracle-ckpt", required=True)
    ap.add_argument("--game", default="mspacman")
    ap.add_argument("--lambda-bc", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--eval-episodes", type=int, default=50)
    ap.add_argument("--max-steps", type=int, default=400)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _run_pair(args.naive_ckpt, args.oracle_ckpt, args.game, args.lambda_bc,
              args.steps, args.eval_episodes, args.max_steps, device)
