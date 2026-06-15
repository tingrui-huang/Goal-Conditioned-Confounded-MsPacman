"""Stage-S1 frozen evaluation (PyTorch). Representation metrics, action-shuffle
tests, same-state action sensitivity, and forced-branch alignment. Evaluation only;
never trains. Bootstrap CIs are EPISODE-level.
"""
import numpy as np
import torch


@torch.no_grad()
def _emb(model, states, actions_onehot, goals, device):
    s = torch.as_tensor(states, dtype=torch.float32, device=device)
    g = torch.as_tensor(goals, dtype=torch.float32, device=device)
    if model.use_action:
        a = torch.as_tensor(actions_onehot, dtype=torch.float32, device=device)
        sa = model.encode_sa(s, a)
    else:
        sa = model.encode_sa(s)
    gr = model.encode_g(g)
    return sa, gr


@torch.no_grad()
def diag_scores(model, states, actions_onehot, goals, device="cpu"):
    sa, gr = _emb(model, states, actions_onehot, goals, device)
    return ((sa * gr).sum(-1) / model._scale).cpu().numpy()


@torch.no_grad()
def retrieval_metrics(model, states, actions_onehot, goals, device="cpu", max_n=2048):
    n = min(len(states), max_n)
    sa, gr = _emb(model, states[:n], actions_onehot[:n], goals[:n], device)
    logits = (sa @ gr.t() / model._scale).cpu().numpy()  # [n,n]; col j = goal j
    # for each example i (row), rank of its true goal i among all goals
    ranks = []
    for i in range(n):
        row = logits[i]
        rank = int((row > row[i]).sum()) + 1
        ranks.append(rank)
    ranks = np.array(ranks)
    return {"n": n, "nce_test": _nce_np(logits),
            "top1_acc": float((ranks == 1).mean()), "top5_acc": float((ranks <= 5).mean()),
            "mean_rank": float(ranks.mean()), "mrr": float((1.0 / ranks).mean()),
            "chance_top1": 1.0 / n,
            "pos_logit_mean": float(np.diagonal(logits).mean()),
            "neg_logit_mean": float(logits[~np.eye(n, dtype=bool)].mean())}


def _nce_np(logits):
    n = logits.shape[0]
    lab = np.eye(n)
    z = logits
    # stable BCE-with-logits
    loss = np.maximum(z, 0) - z * lab + np.log1p(np.exp(-np.abs(z)))
    return float(loss.mean())


def _boot_ci(values_per_ep, n_boot=2000, seed=0):
    keys = list(values_per_ep.keys())
    rng = np.random.RandomState(seed)
    stats = []
    for _ in range(n_boot):
        s = rng.choice(keys, size=len(keys), replace=True)
        vals = np.concatenate([values_per_ep[k] for k in s])
        vals = vals[np.isfinite(vals)]
        if len(vals):
            stats.append(vals.mean())
    if not stats:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(stats)), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


@torch.no_grad()
def global_shuffle_delta(model, states, actions, goals, episode_ids, device="cpu", n_boot=2000, seed=0):
    """delta = f(s,a_true,g+) - f(s,a_shuffled,g+), shuffle actions across examples."""
    from critic import one_hot
    n = len(states)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    a_true = one_hot(actions, device=device); a_sh = one_hot(actions[perm], device=device)
    f_true = diag_scores(model, states, a_true.cpu().numpy(), goals, device)
    f_sh = diag_scores(model, states, a_sh.cpu().numpy(), goals, device)
    d = f_true - f_sh
    per_ep = {int(e): d[episode_ids == e] for e in np.unique(episode_ids)}
    m, lo, hi = _boot_ci(per_ep, n_boot, seed)
    return {"delta_global": m, "ci95": [lo, hi], "mean_raw": float(d.mean())}


@torch.no_grad()
def local_shuffle_delta(model, states, actions, goals, episode_ids, support_fn,
                        device="cpu", n_boot=2000, seed=0):
    """Replace true action with a locally-supported ALTERNATIVE (from support_fn)."""
    from critic import one_hot
    rng = np.random.RandomState(seed + 1)
    alt = np.array([support_fn(states[i], actions[i], rng) for i in range(len(states))])
    keep = alt >= 0
    a_true = one_hot(actions, device=device).cpu().numpy()
    a_alt = one_hot(np.where(keep, alt, actions), device=device).cpu().numpy()
    f_true = diag_scores(model, states, a_true, goals, device)
    f_alt = diag_scores(model, states, a_alt, goals, device)
    d = (f_true - f_alt)[keep]
    eids = episode_ids[keep]
    per_ep = {int(e): d[eids == e] for e in np.unique(eids)}
    m, lo, hi = _boot_ci(per_ep, n_boot, seed)
    return {"delta_local": m, "ci95": [lo, hi], "n_replaced": int(keep.sum()),
            "mean_raw": float(d.mean()) if keep.any() else None}


@torch.no_grad()
def zero_action_ablation(model, states, actions, goals, device="cpu"):
    from critic import one_hot
    a_true = one_hot(actions, device=device).cpu().numpy()
    a_zero = np.zeros_like(a_true)
    f_true = diag_scores(model, states, a_true, goals, device)
    f_zero = diag_scores(model, states, a_zero, goals, device)
    return {"mean_true": float(f_true.mean()), "mean_zero": float(f_zero.mean()),
            "degradation": float(f_true.mean() - f_zero.mean())}


@torch.no_grad()
def same_state_action_sensitivity(model, states, actions, goals, sem_cat, device="cpu", max_n=2048):
    """For each state, score all 18 actions against its true positive goal."""
    from critic import one_hot
    n = min(len(states), max_n)
    s = torch.as_tensor(states[:n], dtype=torch.float32, device=device)
    g = torch.as_tensor(goals[:n], dtype=torch.float32, device=device)
    gr = model.encode_g(g)  # [n,D]
    allA = one_hot(np.arange(18), device=device)  # [18,A]
    scores = np.zeros((n, 18))
    for a in range(18):
        aoh = allA[a:a + 1].repeat(n, 1)
        sa = model.encode_sa(s, aoh)
        scores[:, a] = ((sa * gr).sum(-1) / model._scale).cpu().numpy()
    std = scores.std(1); span = scores.max(1) - scores.min(1)
    true_rank = np.array([int((scores[i] > scores[i, actions[i]]).sum()) + 1 for i in range(n)])
    near_flat = float((std < 1e-3).mean())
    return {"n": n, "mean_std_across_actions": float(std.mean()),
            "mean_top_minus_bottom": float(span.mean()),
            "true_action_rank_mean": float(true_rank.mean()),
            "frac_states_near_identical": near_flat,
            "true_action_rank_hist": np.bincount(true_rank, minlength=19)[1:].tolist(),
            "action_score_matrix": scores.tolist()}


@torch.no_grad()
def forced_branch_alignment(model, anchor_states, future_goals, valid, sem_cat,
                            device="cpu", n_boot=2000, seed=0):
    """M[i,j] = f(anchor_state, action_i, future_goal_j) for locally-supported actions.
    future_goals: [n_anchor,18,goal_dim]; valid: [n_anchor,18]."""
    from critic import one_hot
    nA = anchor_states.shape[0]
    diag_margins = {}; top1 = []; top2 = []; chance = []; perm_better = []; pair_correct = []
    norm_mats = []; example_mats = []
    rng = np.random.RandomState(seed)
    s_all = torch.as_tensor(anchor_states, dtype=torch.float32, device=device)
    for i in range(nA):
        supp = np.where(valid[i] == 1)[0]
        K = len(supp)
        if K < 2:
            continue
        s = s_all[i:i + 1].repeat(K, 1)
        aoh = one_hot(supp, device=device)
        sa = model.encode_sa(s, aoh)  # [K,D]
        g = torch.as_tensor(future_goals[i, supp], dtype=torch.float32, device=device)
        gr = model.encode_g(g)  # [K,D]
        M = (sa @ gr.t() / model._scale).cpu().numpy()  # [K,K]; row=action, col=goal
        diag = np.diagonal(M)
        off = M[~np.eye(K, dtype=bool)]
        diag_margins[i] = diag - off.mean() if len(off) else diag
        # branch-action matching: for goal j, predicted action = argmax_i M[i,j]
        pred = M.argmax(0)
        top1.append((pred == np.arange(K)).mean())
        top2_hit = np.mean([int(np.argsort(-M[:, j])[:2].tolist().count(j) > 0) for j in range(K)])
        top2.append(top2_hit)
        chance.append(1.0 / K)
        # permutation test: shuffle goal columns, recompute top1
        pcount = 0
        for _ in range(200):
            cols = rng.permutation(K)
            pc = (M[:, cols].argmax(0) == np.arange(K)).mean()
            if pc >= (pred == np.arange(K)).mean():
                pcount += 1
        perm_better.append(pcount / 200.0)
        # pairwise ranking
        cnt = 0; tot = 0
        for a in range(K):
            for b in range(a + 1, K):
                tot += 1
                if M[a, a] > M[b, a] and M[b, b] > M[a, b]:
                    cnt += 1
        pair_correct.append(cnt / max(tot, 1))
        Mn = (M - M.min()) / (M.max() - M.min() + 1e-9)
        norm_mats.append(Mn if Mn.shape == (max(2, K), max(2, K)) else None)
        if len(example_mats) < 6:
            example_mats.append({"anchor": int(i), "supported_actions": supp.tolist(), "M": M.tolist()})
    if not diag_margins:
        return {"insufficient": True}
    m, lo, hi = _boot_ci({k: np.atleast_1d(v) for k, v in diag_margins.items()}, n_boot, seed)
    # aggregate normalized matrix over anchors with the modal K
    Ks = [m_.shape[0] for m_ in norm_mats if m_ is not None]
    aggK = max(set(Ks), key=Ks.count) if Ks else 0
    agg = np.nanmean([m_ for m_ in norm_mats if m_ is not None and m_.shape[0] == aggK], axis=0).tolist() if aggK else None
    return {"n_anchors_eligible": len(diag_margins),
            "diagonal_margin_mean": m, "diagonal_margin_ci95": [lo, hi],
            "top1_matching": float(np.mean(top1)), "top2_matching": float(np.mean(top2)),
            "chance_level": float(np.mean(chance)),
            "perm_test_pvalue": float(np.mean(perm_better)),
            "pairwise_ranking": float(np.mean(pair_correct)),
            "aggregate_matrix_K": aggK, "aggregate_normalized_matrix": agg,
            "example_matrices": example_mats}
