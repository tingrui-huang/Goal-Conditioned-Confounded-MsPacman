"""Shared matched-probe trainer for Stage-H0 (Colab side).

One small fixed CNN (oxy4_net.ProbeNet) over the four-frame state, with an optional
fixed-width `extra` vector (U and/or action one-hot). Matched conditions differ ONLY in
the `extra` content (zeros vs real-U vs shuffled-U); the frame pathway and capacity are
identical. Supports classification, multilabel (BCE) and regression. Saves RAW
predictions + per-row losses + episode ids (bootstrap inputs), never metrics only.

`extra` is z-normalised by its OWN train mean/std, so an all-zero condition stays
information-free and real/shuffled-U share the same scaling.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from seaquest_ccrl.probes.oxy4_net import ProbeNet


def _set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _norm_fit(E):
    if E is None or E.shape[1] == 0:
        return (lambda x: x), None
    mu = E.mean(0); sd = E.std(0) + 1e-6
    return (lambda x: ((x - mu) / sd).astype(np.float32)), {"mu": mu.tolist(), "sd": sd.tolist()}


def train_probe(data, view, gi_tr, gi_te, y_tr, y_te, task, out_dim,
                extra_tr=None, extra_te=None, epochs=12, lr=3e-4, batch=256,
                seed=0, device="cpu"):
    """task in {'clf','multilabel','reg'}. Returns dict with test preds, per-row loss,
    episode ids, metrics. y shapes: clf (N,), multilabel (N,out_dim), reg (N,out_dim)."""
    _set_seed(seed)
    extra_dim = 0 if extra_tr is None else extra_tr.shape[1]
    norm_e, estat = _norm_fit(extra_tr)

    if task == "reg":
        ymu = y_tr.mean(0); ysd = y_tr.std(0) + 1e-6
        yt = ((y_tr - ymu) / ysd).astype(np.float32)
    else:
        yt = y_tr

    model = ProbeNet(extra_dim=extra_dim, out_dim=out_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def batches(gi, y, E, shuffle):
        idx = np.random.permutation(len(gi)) if shuffle else np.arange(len(gi))
        for s in range(0, len(gi), batch):
            b = idx[s:s + batch]
            fr = data.stack(gi[b], view).to(device)
            ex = None if E is None else torch.as_tensor(norm_e(E[b]), device=device)
            yy = torch.as_tensor(np.asarray(y)[b], device=device)
            yield fr, ex, yy

    for _ in range(epochs):
        model.train()
        for fr, ex, yy in batches(gi_tr, yt, extra_tr, True):
            pred = model(fr, ex)
            if task == "reg":
                loss = Fnn.mse_loss(pred, yy.float())
            elif task == "multilabel":
                loss = Fnn.binary_cross_entropy_with_logits(pred, yy.float())
            else:
                loss = Fnn.cross_entropy(pred, yy.long())
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    preds, perrow = [], []
    with torch.no_grad():
        for fr, ex, yy in batches(gi_te, y_te, extra_te, False):
            out = model(fr, ex)
            yn = yy.cpu().numpy()
            if task == "reg":
                p = out.cpu().numpy() * ysd + ymu
                preds.append(p)
                perrow.append(((p - yn) ** 2).mean(1))
            elif task == "multilabel":
                logp = Fnn.logsigmoid(out).cpu().numpy()
                log1m = Fnn.logsigmoid(-out).cpu().numpy()
                preds.append(1.0 / (1.0 + np.exp(-out.cpu().numpy())))
                bce = -(yn * logp + (1 - yn) * log1m)
                perrow.append(bce.mean(1))
            else:
                logp = Fnn.log_softmax(out, 1).cpu().numpy()
                preds.append(np.exp(logp))
                perrow.append(-logp[np.arange(len(yn)), yn.astype(int)])
    P = np.concatenate(preds); per_row_loss = np.concatenate(perrow)
    ep_te = data.episode_of[gi_te]
    return {"pred": P, "per_row_loss": per_row_loss, "episode": ep_te, "y": np.asarray(y_te),
            "metrics": _metrics(task, P, np.asarray(y_te), out_dim),
            "extra_dim": extra_dim, "extra_norm": estat, "n_test": int(len(gi_te))}


def prior_loss(task, y_tr, y_te, out_dim):
    """Closed-form P0 (train-prior / non-image) per-row loss on test."""
    y_tr = np.asarray(y_tr); y_te = np.asarray(y_te)
    if task == "reg":
        mu = y_tr.mean(0)
        per_row = ((y_te - mu) ** 2).mean(1)
        return per_row, {"mse": float(per_row.mean())}
    if task == "multilabel":
        rate = np.clip(y_tr.mean(0), 1e-6, 1 - 1e-6)         # (out_dim,)
        bce = -(y_te * np.log(rate) + (1 - y_te) * np.log(1 - rate))
        per_row = bce.mean(1)
        return per_row, {"bce": float(per_row.mean())}
    freq = np.bincount(y_tr.astype(int), minlength=out_dim).astype(np.float64)
    freq = np.clip(freq / freq.sum(), 1e-9, 1.0)
    per_row = -np.log(freq[y_te.astype(int)])
    return per_row, {"log_loss": float(per_row.mean())}


def _metrics(task, P, y, out_dim):
    if task == "reg":
        err = P - y
        ss_res = (err ** 2).sum(0); ss_tot = ((y - y.mean(0)) ** 2).sum(0) + 1e-9
        return {"mse": float((err ** 2).mean()), "mae": float(np.abs(err).mean()),
                "r2": float((1 - ss_res / ss_tot).mean()),
                "r2_per_target": (1 - ss_res / ss_tot).tolist()}
    if task == "multilabel":
        pred = (P >= 0.5).astype(int)
        bce = -(y * np.log(np.clip(P, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - P, 1e-9, 1)))
        out = {"bce": float(bce.mean()), "subset_acc": float((pred == y).all(1).mean())}
        aurocs, auprcs, supp = [], [], []
        for c in range(P.shape[1]):
            supp.append(int(y[:, c].sum()))
            aurocs.append(_auroc(P[:, c], y[:, c]))
            auprcs.append(_auprc(P[:, c], y[:, c]))
        out["per_cell_support"] = supp
        out["auroc_per_cell"] = aurocs
        out["auprc_per_cell"] = auprcs
        out["macro_auroc"] = float(np.nanmean([a for a in aurocs if a is not None])) if any(
            a is not None for a in aurocs) else None
        return out
    pred = P.argmax(1)
    acc = float((pred == y).mean())
    nll = float(-np.log(np.clip(P[np.arange(len(y)), y.astype(int)], 1e-9, 1)).mean())
    f1s = []
    for c in range(out_dim):
        tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = ((pred != c) & (y == c)).sum()
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        f1s.append(2 * prec * rec / (prec + rec + 1e-9))
    out = {"accuracy": acc, "log_loss": nll, "macro_f1": float(np.mean(f1s))}
    if out_dim > 2:
        order = (-P).argsort(1)
        out["top1_acc"] = acc
        out["top3_acc"] = float(np.mean([y[i] in order[i, :3] for i in range(len(y))]))
    return out


def _auroc(s, y):
    pos = y == 1
    if not pos.any() or pos.all():
        return None
    order = np.argsort(s); ranks = np.empty_like(order, float); ranks[order] = np.arange(len(s))
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() - 1) / 2) / (pos.sum() * (~pos).sum()))


def _auprc(s, y):
    pos = y == 1
    if not pos.any():
        return None
    order = np.argsort(-s); yo = y[order]
    tp = np.cumsum(yo); fp = np.cumsum(1 - yo)
    prec = tp / (tp + fp); rec = tp / pos.sum()
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(trap(prec, rec)) if len(rec) > 1 else None


def boot_ci(values, episodes, n_boot=2000, seed=0):
    """Episode-level bootstrap CI of mean(values)."""
    values = np.asarray(values, float); episodes = np.asarray(episodes)
    uniq = np.unique(episodes); by = {e: values[episodes == e] for e in uniq}
    rng = np.random.RandomState(seed); ms = []
    for _ in range(n_boot):
        pk = rng.choice(uniq, size=len(uniq), replace=True)
        v = np.concatenate([by[e] for e in pk]); v = v[np.isfinite(v)]
        if len(v):
            ms.append(v.mean())
    lo, hi = float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))
    return {"mean": float(np.nanmean(values)), "ci95": [lo, hi],
            "lower_gt_0": bool(lo > 0), "upper_lt_0": bool(hi < 0)}


def single_episode_driven(values, episodes):
    """True if removing the single most-contributing episode flips the mean sign
    (a crude 'explained by one episode' guard for improvement metrics)."""
    values = np.asarray(values, float); episodes = np.asarray(episodes)
    if np.nanmean(values) == 0:
        return False
    uniq = np.unique(episodes)
    full = np.nanmean(values)
    for e in uniq:
        m = np.nanmean(values[episodes != e])
        if np.sign(m) != np.sign(full) or abs(m) < 0.2 * abs(full):
            return True
    return False
