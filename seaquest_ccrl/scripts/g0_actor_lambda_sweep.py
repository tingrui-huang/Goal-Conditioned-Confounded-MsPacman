"""OFFLINE actor lambda-sweep + on-trajectory agreement preview (Colab GPU).

Why: so far we ONLY ever deployed the greedy argmax-critic policy (success_by_H=0.533
full-view). The ORIGINAL Contrastive RL (Eysenbach et al. 2022) deploys a LEARNED actor
pi(a|s,g) that maximises the critic, with an optional BC regulariser. We never ran it, so
we cannot yet say whether the (action-weak) critic, once parameterised through an actor,
supports more stable closed-loop control. This script trains that actor faithfully.

FAITHFULNESS: reuses train_actor.train_actor() verbatim -- the objective is exactly
    max_pi  (1-lam) * E_{a~pi}[ f(s,a,g) ]  +  lam * log pi(a_teacher | s, g)
with the critic FROZEN, same masked/unmasked view + frame_stack + hindsight-goal sampler
as the critic. We do NOT reimplement the loss. We only sweep lam:
    lam=0.0  -> pure critic-maximising actor (closest to the online original, no BC)
    lam=0.5  -> critic + BC (the offline-regularised configuration)
    lam=1.0  -> pure goal-conditioned BC (critic-FREE control baseline)
plus a small entropy bonus (--ent-coef, default 0.01). The original CRL actor is a
continuous Gaussian policy with inherent entropy; without an explicit H(pi) term the
discrete categorical lam=0 actor maximising this NEAR-FLAT critic (action-score range
~0.16) collapses the softmax to a single constant action (vanishing gradient) -- verified
on CPU/fp32, so it is an optimisation pathology, NOT an fp16 or code bug. The entropy
bonus restores a well-posed pure-critic actor; lam>0 is already protected by the BC term.

This file does the ENV-FREE half: train the three actors and measure, on held-out teacher
trajectory states (the sampler's hindsight (s, a_teacher, g) tuples), how each actor's
greedy action agrees with (a) the teacher action, (b) its vertical direction, and (c) the
argmax-critic action; plus the critic value E_pi[f] each actor actually attains. This
PREVIEWS the closed-loop outcome; the actual success-rate comparison is a separate,
env-dependent step (g0_closed_loop_eval.py with --actor-ckpt) run after the Docker rebuild.

Default critic = full-view (oracle) = un-confounded vanilla method; the masked arm can be
swept later with --critic-ckpt <masked>.
"""
import os, json, argparse
import numpy as np
import torch
import torch.nn.functional as F

from seaquest_ccrl.training.train_critic import load_critic
from seaquest_ccrl.training.train_actor import train_actor
from seaquest_ccrl.training.dataset_sampler import HindsightSampler
from seaquest_ccrl.games import get_game

# ALE vertical component: +1 = action moves UP, -1 = DOWN, 0 = neither
UP = {2, 6, 7, 10, 14, 15}
DOWN = {5, 8, 9, 13, 16, 17}
def vdir(a):
    return 1 if a in UP else (-1 if a in DOWN else 0)
_VDIR = None  # filled per nb_actions


@torch.no_grad()
def on_trajectory_agreement(actor, critic, sampler, cfg, device, n=4096):
    """Env-free preview on teacher-trajectory states with hindsight goals."""
    A = cfg.nb_actions
    vd = torch.tensor([vdir(a) for a in range(A)], device=device)
    fr, a_teacher, go = sampler.sample(n)
    # actor greedy action + entropy
    logits = actor(fr, go)
    pi = F.softmax(logits, 1)
    a_actor = logits.argmax(1)
    ent = -(pi * torch.log(pi.clamp_min(1e-9))).sum(1).mean()
    # argmax-critic action over the SAME states/goals
    g_emb = critic.g_encoder(go)
    f_all = torch.stack([(critic.sa_encoder(fr, torch.full((n,), a, device=device,
              dtype=torch.long)) * g_emb).sum(1) for a in range(A)], 1)   # (n,A)
    a_crit = f_all.argmax(1)
    # critic value attained by the actor's distribution vs the argmax ceiling
    val_actor = (pi * f_all).sum(1).mean()
    val_argmax = f_all.max(1).values.mean()
    # agreements
    top1_teacher = (a_actor == a_teacher).float().mean()
    vdir_teacher = (vd[a_actor] == vd[a_teacher]).float().mean()      # same up/down/neither
    top1_crit = (a_actor == a_crit).float().mean()
    return {
        "top1_agree_teacher": float(top1_teacher),
        "vdir_agree_teacher": float(vdir_teacher),
        "top1_agree_argmax_critic": float(top1_crit),
        "actor_entropy_nats": float(ent),                # ln(18)=2.89 = uniform
        "critic_value_actor_Epi_f": float(val_actor),
        "critic_value_argmax_f": float(val_argmax),
        "n": int(n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--critic-ckpt",
                    default="artifacts/seaquest/seaquest_stage_g0_fullview_train/critic_full_view.pt")
    ap.add_argument("--root", default="seaquest_ccrl/data/raw_hf")
    ap.add_argument("--lams", default="0.0,0.5,1.0")
    ap.add_argument("--ent-coef", type=float, default=0.01,
                    help="entropy bonus (max-ent); needed so lam=0 pure-critic doesn't "
                         "collapse the softmax to a constant action on this flat critic")
    ap.add_argument("--balance-grad", action="store_true",
                    help="per-step equalize the critic-term gradient norm to the BC term's, so "
                         "lam genuinely interpolates BC<->critic (else lam=0.5 is really ~9:1 BC)")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--out-dir", default="artifacts/seaquest/goal_control/actor_lambda_sweep")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    critic, cfg, oracle = load_critic(args.critic_ckpt, device)
    game = get_game("seaquest")
    view = "oracle_full_view" if oracle else "naive_masked"
    print(f"[actor-sweep] critic={args.critic_ckpt} view={view} fs={cfg.frame_stack} "
          f"device={device} steps={args.steps}")
    # frozen-critic argmax reference (the policy we ALREADY evaluated closed-loop)
    rng = np.random.default_rng(cfg.seed + 4242)
    eval_sampler = HindsightSampler(game, oracle=oracle, cfg=cfg, device=device, rng=rng,
                                    root=args.root)

    results = {"critic_ckpt": args.critic_ckpt, "view": view, "steps": args.steps,
               "frame_stack": cfg.frame_stack, "ent_coef": args.ent_coef,
               "balance_grad": bool(args.balance_grad), "lambda_sweep": {}}
    for lam in [float(x) for x in args.lams.split(",")]:
        print(f"\n=== training actor lambda={lam} "
              f"({'pure-critic' if lam==0 else 'pure-BC' if lam==1 else 'critic+BC'}) ===")
        # save each actor under its own subdir so ckpt_dir doesn't collide
        sub = os.path.join(args.out_dir, f"lam{lam:.2f}")
        os.makedirs(sub, exist_ok=True)
        cfg.ckpt_dir = sub
        actor = train_actor(critic, game, cfg, oracle=oracle, lam=lam,
                            steps=args.steps, device=device, verbose=True,
                            root=args.root, ent_coef=args.ent_coef, balance_grad=args.balance_grad)
        agree = on_trajectory_agreement(actor, critic, eval_sampler, cfg, device)
        # judge collapse by ENTROPY (ln(A)=uniform), NOT by argmax dominant-action share,
        # which is misleading for a near-uniform policy (a near-tie argmax looks "collapsed").
        agree["entropy_collapsed"] = bool(agree["actor_entropy_nats"] < 0.05)
        agree["ent_coef"] = args.ent_coef
        results["lambda_sweep"][f"{lam:.2f}"] = agree
        print(f"  [lam={lam}] {json.dumps(agree)}")

    out = os.path.join(args.out_dir, "on_trajectory_preview.json")
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nWROTE {out}")
    print("NOTE: this is the ENV-FREE preview. Closed-loop success-rate (the B/C/D verdict) "
          "is the separate g0_closed_loop_eval.py --actor-ckpt step after Docker rebuild.")


if __name__ == "__main__":
    main()
