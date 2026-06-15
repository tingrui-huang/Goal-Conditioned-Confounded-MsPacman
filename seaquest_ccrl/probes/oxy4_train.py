"""Generic supervised probe trainer + metrics + episode-level bootstrap.
Fixed modest budget (no extensive optimization). Normalization fit on TRAIN only.
Saves RAW predictions + per-row losses + episode ids (bootstrap inputs), not metrics only.
"""
import numpy as np
import torch
import torch.nn.functional as F

from seaquest_ccrl.probes.oxy4_net import ProbeNet


def _set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def train_probe(data, view, gi_tr, gi_va, gi_te, extra_fn, y_tr, y_va, y_te,
                task, out_dim, epochs=12, lr=3e-4, batch=256, seed=0, device="cpu"):
    """task in {'reg','clf'}. extra_fn(gidx)->(M,extra_dim) float (None if no extras).
    Returns dict with test predictions, per-row loss, metrics, test episode ids."""
    _set_seed(seed)
    Etr = extra_fn(gi_tr) if extra_fn else None
    extra_dim = 0 if Etr is None else Etr.shape[1]
    # normalize extras (oxygen col etc.) on train; one-hot cols have ~unit scale already
    if extra_dim:
        emu = Etr.mean(0); esd = Etr.std(0) + 1e-6
        norm_e = lambda E: ((E - emu) / esd).astype(np.float32)
    else:
        norm_e = lambda E: None
    # normalize regression targets on train
    if task == "reg":
        ymu = y_tr.mean(0); ysd = y_tr.std(0) + 1e-6
        yt = (y_tr - ymu) / ysd
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
            yy = torch.as_tensor(y[b], device=device)
            yield fr, ex, yy

    for ep in range(epochs):
        model.train()
        for fr, ex, yy in batches(gi_tr, yt, Etr, True):
            pred = model(fr, ex)
            if task == "reg":
                loss = F.mse_loss(pred, yy.float())
            else:
                loss = F.cross_entropy(pred, yy.long())
            opt.zero_grad(); loss.backward(); opt.step()

    # eval on test
    model.eval()
    Ete = extra_fn(gi_te) if extra_fn else None
    preds = []; perrow = []
    with torch.no_grad():
        for fr, ex, yy in batches(gi_te, y_te, Ete, False):
            out = model(fr, ex)
            if task == "reg":
                p = out.cpu().numpy() * (ysd if 'ysd' in dir() else 1.0) + (ymu if 'ymu' in dir() else 0.0)
                preds.append(p)
                perrow.append(((p - yy.cpu().numpy()) ** 2).mean(1))   # per-row MSE
            else:
                logp = F.log_softmax(out, 1).cpu().numpy()
                preds.append(np.exp(logp))
                perrow.append(-logp[np.arange(len(yy)), yy.cpu().numpy().astype(int)])  # per-row NLL
    P = np.concatenate(preds); per_row_loss = np.concatenate(perrow)
    ep_te = data.episode_of[gi_te]
    metrics = _metrics(task, P, y_te, out_dim)
    return {"pred": P, "per_row_loss": per_row_loss, "episode": ep_te, "y": y_te,
            "metrics": metrics, "extra_dim": extra_dim, "n_test": int(len(gi_te))}


def _metrics(task, P, y, out_dim):
    if task == "reg":
        err = P - y
        ss_res = (err ** 2).sum(0); ss_tot = ((y - y.mean(0)) ** 2).sum(0) + 1e-9
        return {"mse": float((err ** 2).mean()), "mae": float(np.abs(err).mean()),
                "r2": float((1 - ss_res / ss_tot).mean())}
    pred = P.argmax(1)
    acc = float((pred == y).mean())
    nll = float(-np.log(np.clip(P[np.arange(len(y)), y.astype(int)], 1e-9, 1)).mean())
    # macro-F1
    f1s = []
    for c in range(out_dim):
        tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum(); fn = ((pred != c) & (y == c)).sum()
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        f1s.append(2 * prec * rec / (prec + rec + 1e-9))
    out = {"accuracy": acc, "log_loss": nll, "macro_f1": float(np.mean(f1s))}
    if out_dim > 2:
        order = (-P).argsort(1)
        out["top3_acc"] = float(np.mean([y[i] in order[i, :3] for i in range(len(y))]))
    if out_dim == 2:
        # AUROC (binary)
        s = P[:, 1]; pos = y == 1
        if pos.any() and (~pos).any():
            order = np.argsort(s); ranks = np.empty_like(order, float); ranks[order] = np.arange(len(s))
            auc = (ranks[pos].sum() - pos.sum() * (pos.sum() - 1) / 2) / (pos.sum() * (~pos).sum())
            out["auroc"] = float(auc)
    return out


def boot_ci(values, episodes, n_boot=2000, seed=0):
    uniq = np.unique(episodes); by = {e: values[episodes == e] for e in uniq}
    rng = np.random.RandomState(seed); ms = []
    for _ in range(n_boot):
        pk = rng.choice(uniq, size=len(uniq), replace=True)
        v = np.concatenate([by[e] for e in pk]); v = v[np.isfinite(v)]
        if len(v):
            ms.append(v.mean())
    return {"mean": float(np.mean(values)), "ci95": [float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))],
            "ci_excludes_0": bool(np.percentile(ms, 2.5) > 0 or np.percentile(ms, 97.5) < 0)}


def paired_improvement_ci(loss_base, loss_plus, episodes, n_boot=2000, seed=0):
    """Per-row loss reduction from the '+oxygen' model (base - plus); CI over episodes.
    Positive mean => adding oxygen reduces held-out loss."""
    return boot_ci(loss_base - loss_plus, episodes, n_boot, seed)
