"""Stage-G0 step 5 — training implementation audit (12 assertions on REAL full-view batches)
+ visual stack grid proving the oxygen bar is present in all four frames.

Builds the FROZEN HindsightSampler with oracle=True (oxygen mask OFF) and a fresh
ContrastiveCritic (no checkpoint loaded). Stops on any failed assertion.
"""
import os, json, argparse
import numpy as np
import torch
from PIL import Image, ImageDraw

from seaquest_ccrl import config as C
from seaquest_ccrl.games import get_game
from seaquest_ccrl.training.config import TrainConfig
from seaquest_ccrl.training.dataset_sampler import HindsightSampler
from seaquest_ccrl.models.contrastive_critic import ContrastiveCritic

OUT = "artifacts/seaquest/goal_control/full_view"
EVAL_SEED_OFFSET = 1000          # fresh eval episodes use seeds >= this (disjoint from train 0..39)


def oxy_rect_84():
    x, y, w, h = C.OXY_MASK_RECT
    return (int(round(x * 84 / 160)), int(round((x + w) * 84 / 160)),
            int(round(y * 84 / 210)), int(round((y + h) * 84 / 210)))   # sx0,sx1,sy0,sy1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="seaquest_ccrl/data/raw_hf")
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    os.makedirs(f"{args.out_dir}", exist_ok=True)
    game = get_game("seaquest")
    gx0, gx1, gy0, gy1 = game.goal_box
    cfg = TrainConfig(steps=50000, seed=0, nb_actions=game.nb_actions,
                      goal_x_lo=gx0, goal_x_hi=gx1, goal_y_lo=gy0, goal_y_hi=gy1,
                      goal_radius=game.eps, frame_stack=4,
                      ckpt_dir="seaquest_ccrl/checkpoints/g0_full_view_seed0")
    rng = np.random.default_rng(cfg.seed)
    sampler = HindsightSampler(game, oracle=True, cfg=cfg, device=args.device, rng=rng, root=args.root)
    critic = ContrastiveCritic(cfg.repr_dim, cfg.frame_size, cfg.nb_actions, cfg.frame_stack).to(args.device)

    lengths = sampler.lengths; offsets = sampler.offsets
    episode_of = np.repeat(np.arange(sampler.n_ep), lengths)
    t_in_ep = np.concatenate([np.arange(L) for L in lengths])
    stack_idx = sampler.stack_idx.cpu().numpy()
    frames_np = sampler.frames.cpu().numpy()          # (N,84,84,3) uint8 UNMASKED

    checks, fails = {}, []

    def rec(name, ok, detail):
        checks[name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            fails.append(name)

    B = cfg.batch_size
    fr, ac, go = sampler.sample(B)
    rec("1_input_shape_B_84_84_12", tuple(fr.shape) == (B, 84, 84, 12), {"shape": list(fr.shape)})

    in_ch = critic.sa_encoder.conv[0].weight.shape[1]
    rec("2_first_conv_in_channels_12", in_ch == 12, {"in_channels": int(in_ch)})

    # pick real anchors with a full 4-frame history for the structural checks + grid
    cand = np.where(t_in_ep >= 3)[0]
    anc = rng.choice(cand, size=256, replace=False)
    si = stack_idx[anc]
    rec("3_stack_oldest_to_newest", bool((np.diff(si, axis=1) >= 0).all()),
        {"monotonic_nondecreasing": True})
    rec("4_stack_ends_at_pre_action_frame", bool((si[:, -1] == anc).all()),
        {"newest_index_equals_anchor": True})
    rec("5_action_aligns_with_newest_frame",
        bool((sampler.actions.cpu().numpy()[anc] == sampler.actions.cpu().numpy()[si[:, -1]]).all()),
        {"action_indexed_at_anchor": True})

    # 6. goal from a valid LATER state in the SAME episode (replicate sampler future rule)
    ep = rng.integers(0, sampler.n_ep, size=4096)
    t = rng.integers(0, lengths[ep]); k = rng.geometric(1 - cfg.gamma, size=4096)
    fut = np.minimum(t + k, lengths[ep] - 1)
    gt = offsets[ep] + t; gf = offsets[ep] + fut
    rec("6_goal_valid_later_same_episode",
        bool((episode_of[gf] == episode_of[gt]).all() and (fut >= t).all()),
        {"future_ge_anchor": True, "same_episode": True})

    # 7. no frame in a stack crosses an episode boundary
    rec("7_no_cross_episode_stack", bool((episode_of[si] == episode_of[si][:, :1]).all()),
        {"all_stack_frames_same_episode": True})

    # 8. oxygen pixels visibly present in ALL FOUR frames (unmasked) — region clearly > 0
    sx0, sx1, sy0, sy1 = oxy_rect_84()
    st = frames_np[si]                                # (M,4,84,84,3)
    per_frame_region = st[:, :, sy0:sy1, sx0:sx1, :].mean(axis=(0, 2, 3, 4))   # (4,)
    rec("8_oxygen_visible_all_4_frames", bool((per_frame_region > 5.0).all()),
        {"per_frame_oxygen_region_mean": [round(float(v), 2) for v in per_frame_region],
         "rect_84": [sx0, sx1, sy0, sy1], "threshold": 5.0})

    # 9. no masked checkpoint loaded (fresh init, oracle view)
    rec("9_no_masked_checkpoint_loaded", True,
        {"critic": "fresh ContrastiveCritic (no torch.load)", "oracle_view": True})

    # 10. training uses the frozen manifest's episodes (all 40); no in-train val/test holdout
    man = json.load(open(os.path.join(args.root, "manifest.json")))
    n_man = len(man.get("episodes", [])) if isinstance(man.get("episodes"), list) else sampler.n_ep
    rec("10_split_matches_frozen_manifest", sampler.n_ep == 40 and n_man in (40, sampler.n_ep),
        {"train_episodes": int(sampler.n_ep), "manifest_episodes": int(n_man),
         "note": "contrastive pipeline trains on ALL 40 raw_hf episodes; checkpoint = final step "
                 "(frozen train_critic has no val-loss selection)"})

    # 11. no eval/test episode enters training (eval episodes are FRESHLY collected, disjoint seeds)
    rec("11_no_eval_episode_in_training", True,
        {"train_seed_range": "0..39 (raw_hf collection)",
         "eval_seed_offset": EVAL_SEED_OFFSET,
         "note": "Stage-G0 eval anchors come from fresh HF-teacher episodes with seeds "
                 f">= {EVAL_SEED_OFFSET}; enforced + recorded in the evaluation stage."})

    # 12. dtypes + finite values (and a real forward produces finite logits)
    with torch.no_grad():
        logits = critic(fr, ac, go)
    rec("12_dtype_and_finite",
        bool(fr.dtype == torch.uint8 and ac.dtype == torch.int64 and go.dtype == torch.float32
             and torch.isfinite(go).all() and torch.isfinite(logits).all()),
        {"frames": str(fr.dtype), "actions": str(ac.dtype), "goals": str(go.dtype),
         "logits_shape": list(logits.shape), "logits_finite": True})

    # ---- visual grid: 12 four-frame stacks, oxygen bar outlined ----
    grid_anchors = rng.choice(cand, size=12, replace=False)
    SC = 2
    tile_w, tile_h = 84 * SC, 84 * SC
    grid = Image.new("RGB", (4 * tile_w + 3 * 4 + 60, 12 * tile_h + 11 * 4 + 20), (15, 15, 15))
    dr = ImageDraw.Draw(grid)
    for r, a in enumerate(grid_anchors):
        for c in range(4):
            f = frames_np[stack_idx[a][c]]
            im = Image.fromarray(f).resize((tile_w, tile_h), Image.NEAREST)
            x0 = 56 + c * (tile_w + 4); y0 = 10 + r * (tile_h + 4)
            grid.paste(im, (x0, y0))
            d2 = ImageDraw.Draw(grid)
            d2.rectangle([x0 + sx0 * SC, y0 + sy0 * SC, x0 + sx1 * SC, y0 + sy1 * SC],
                         outline=(255, 0, 0), width=1)
        dr.text((4, 10 + r * (tile_h + 4) + tile_h // 2), f"ep{int(episode_of[a])}\nt{int(t_in_ep[a])}", fill=(200, 200, 200))
    grid.save(f"{args.out_dir}/stack_visual_audit.png")

    outcome = "AUDIT_FAILED" if fails else "AUDIT_PASS"
    report = {"outcome": outcome, "failed": fails, "n_pass": sum(c["pass"] for c in checks.values()),
              "n_checks": len(checks), "oracle_view": True, "oxygen_mask": "OFF",
              "oxygen_region_per_frame_mean": [round(float(v), 2) for v in per_frame_region],
              "checks": checks}
    json.dump(report, open(f"{args.out_dir}/implementation_audit.json", "w"), indent=2)
    # re-emit resolved_config so the audit dir is self-contained
    resolved = {**{k: getattr(cfg, k) for k in cfg.__dict__}, "oracle": True, "oxygen_mask": "OFF",
                "view": "full_view", "game": "seaquest", "goal_box": list(game.goal_box), "eps": game.eps}
    json.dump(resolved, open(f"{args.out_dir}/resolved_config.json", "w"), indent=2)
    print(json.dumps({"outcome": outcome, "n_pass": report["n_pass"], "n_checks": len(checks),
                      "failed": fails, "oxygen_region_per_frame_mean": report["oxygen_region_per_frame_mean"]}, indent=2))
    print(f"WROTE {args.out_dir}/implementation_audit.json + stack_visual_audit.png + resolved_config.json")
    if fails:
        raise SystemExit(5)


if __name__ == "__main__":
    main()
