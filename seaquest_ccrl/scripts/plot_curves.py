"""Plot training loss and eval success-rate vs step, across seeds.

Reads the per-(seed, tag) history files written by train() during a sweep run
that used --eval-every:

    {ckpt_base}/seed{S}/history_{naive,oracle}.json
        -> {"loss": [[step, mean_loss, diag_acc], ...],
            "eval": [[step, success_rate], ...]}

Produces a 2x2 figure: rows = {train loss, eval success}, cols = {naive, oracle},
one colored line per seed.

Run (after a sweep with --eval-every set):
    python -m seaquest_ccrl.scripts.plot_curves --ckpt-base "$WORK/ckpt" --seeds 0 1 2
"""
import os
import json
import glob
import argparse


def load_histories(ckpt_base, seeds):
    data = {}   # (seed, tag) -> dict
    for s in seeds:
        for tag in ("naive", "oracle"):
            p = os.path.join(ckpt_base, f"seed{s}", f"history_{tag}.json")
            if os.path.exists(p):
                data[(s, tag)] = json.load(open(p))
            else:
                print(f"(missing: {p})")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-base", default="seaquest_ccrl/checkpoints",
                    help="base dir holding seed*/history_*.json (point at $WORK/ckpt on Colab)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="seaquest_ccrl/figure/learning_curves.png")
    args = ap.parse_args()

    data = load_histories(args.ckpt_base, args.seeds)
    if not data:
        raise SystemExit(f"No history files under {args.ckpt_base!r}. "
                         f"Run the sweep with --eval-every first.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    color = {s: cmap(i) for i, s in enumerate(args.seeds)}
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

    for (s, tag), h in data.items():
        col = 0 if tag == "naive" else 1
        c = color[s]
        if h["loss"]:
            xs = [r[0] for r in h["loss"]]; ys = [r[1] for r in h["loss"]]
            ax[0, col].plot(xs, ys, color=c, label=f"seed {s}")
        if h["eval"]:
            xs = [r[0] for r in h["eval"]]; ys = [r[1] for r in h["eval"]]
            ax[1, col].plot(xs, ys, color=c, marker="o", ms=3, label=f"seed {s}")

    for col, tag in enumerate(("naive (masked)", "oracle (unmasked)")):
        ax[0, col].set_title(tag)
        ax[0, col].set_ylabel("train NCE loss")
        ax[1, col].set_ylabel("eval success rate")
        ax[1, col].set_xlabel("training step")
        ax[1, col].set_ylim(-0.02, 1.02)
        for row in (0, 1):
            ax[row, col].grid(alpha=0.3)
            ax[row, col].legend(fontsize=8)

    fig.suptitle("Level-2 learning curves (loss & goal-reaching) across seeds", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
