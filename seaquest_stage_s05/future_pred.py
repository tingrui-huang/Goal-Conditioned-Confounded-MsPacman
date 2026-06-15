"""P5 — state-only vs state+action future prediction on O-Sampled (section 12, Gate P5).
Diagnostic only (NOT the contrastive critic). Episode-level splits, bootstrap CIs.
Runs in ocatari image (sklearn).
"""
import json, os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, mean_squared_error, log_loss, accuracy_score

BASE = "/work/artifacts/seaquest/stage_s05"
CFG = json.load(open(f"{BASE}/config/resolved_config.json"))
SF = CFG["state_features"]; H = CFG["horizons"]
# no-player future targets (continuous + binary)
CONT = ["enemy_centroid_x", "enemy_centroid_y", "enemy_count"]
ACTION_DIM = 18


def load(mode="O-Sampled"):
    d = np.load(f"{BASE}/closed_loop/rows_{mode}.npz", allow_pickle=True)
    cols = list(d["columns"]); R = d["rows"]; ci = {c: i for i, c in enumerate(cols)}
    return R, ci


def main():
    R, ci = load("O-Sampled")
    ep = R[:, ci["episode"]].astype(int); eps = np.unique(ep)
    Xstate = R[:, [ci[f] for f in SF]].astype(np.float64)
    a = R[:, ci["sampled_action"]].astype(int)
    onehot = np.eye(ACTION_DIM)[a]
    rew = R[:, ci["reward"]]; ll = R[:, ci["life_loss"]]
    score = R[:, ci["score"]]

    rng = np.random.RandomState(CFG["seeds"]["splits"])
    order = rng.permutation(eps); nt = max(1, len(eps) // 5); nv = max(1, len(eps) // 5)
    te_e = set(order[:nt]); tr_e = set(order[nt + nv:])
    tr = np.isin(ep, list(tr_e)); te = np.isin(ep, list(te_e))

    def build_targets(h):
        """For each row, future no-player target at +h within same episode."""
        cont = {k: np.full(len(R), np.nan) for k in CONT}
        cumrew = np.full(len(R), np.nan); life = np.full(len(R), np.nan)
        scoredelta = np.full(len(R), np.nan)
        for e in eps:
            idx = np.where(ep == e)[0]
            for pos, i in enumerate(idx):
                if pos + h < len(idx):
                    j = idx[pos + h]
                    for k in CONT:
                        cont[k][i] = R[j, ci[k]]
                    cumrew[i] = rew[idx[pos:pos + h]].sum()
                    life[i] = float(ll[idx[pos:pos + h]].max())
                    scoredelta[i] = (score[j] - score[i]) if np.isfinite(score[j]) and np.isfinite(score[i]) else np.nan
        return cont, cumrew, life, scoredelta

    with np.errstate(all="ignore"):
        med = np.nan_to_num(np.nanmedian(Xstate[tr], axis=0), nan=0.0)
    Xi = np.nan_to_num(np.where(np.isnan(Xstate), med, Xstate), nan=0.0)
    sc = StandardScaler().fit(Xi[tr]); Xs = np.nan_to_num(sc.transform(Xi), nan=0.0)
    Xs_a = np.hstack([Xs, onehot])

    def cont_eval(y, mask, with_action):
        m = mask & np.isfinite(y)
        trm = m & tr; tem = m & te
        if trm.sum() < 30 or tem.sum() < 20:
            return None
        Xtr = (Xs_a if with_action else Xs)[trm]; Xte = (Xs_a if with_action else Xs)[tem]
        r = Ridge(alpha=1.0).fit(Xtr, y[trm]); pred = r.predict(Xte)
        return {"r2": float(r2_score(y[tem], pred)), "mse": float(mean_squared_error(y[tem], pred)), "n_test": int(tem.sum())}

    def bin_eval(y, mask, with_action):
        m = mask & np.isfinite(y)
        trm = m & tr; tem = m & te
        if trm.sum() < 30 or tem.sum() < 20 or len(np.unique(y[trm])) < 2 or len(np.unique(y[tem])) < 2:
            return None
        Xtr = (Xs_a if with_action else Xs)[trm]; Xte = (Xs_a if with_action else Xs)[tem]
        c = LogisticRegression(max_iter=300).fit(Xtr, y[trm].astype(int))
        proba = c.predict_proba(Xte)[:, 1]; pred = (proba > 0.5).astype(int)
        return {"log_loss": float(log_loss(y[tem].astype(int), proba, labels=[0, 1])),
                "acc": float(accuracy_score(y[tem].astype(int), pred)), "n_test": int(tem.sum())}

    state_only = {}; state_action = {}; incr = {}
    valid_obj = np.isfinite(R[:, ci["player_y"]])
    for h in H:
        cont, cumrew, life, scoredelta = build_targets(h)
        so = {}; sa = {}; inc = {}
        # continuous no-player targets
        for k in CONT + ["cum_reward", "score_delta"]:
            y = {"cum_reward": cumrew, "score_delta": scoredelta}.get(k, cont.get(k))
            s = cont_eval(y, valid_obj, False); sA = cont_eval(y, valid_obj, True)
            so[k] = s; sa[k] = sA
            if s and sA:
                inc[k] = {"dR2": sA["r2"] - s["r2"], "dMSE": sA["mse"] - s["mse"],
                          "action_helps": bool(sA["r2"] - s["r2"] > 0)}
        # binary: life loss within h
        s = bin_eval(life, valid_obj, False); sA = bin_eval(life, valid_obj, True)
        so["life_loss"] = s; sa["life_loss"] = sA
        if s and sA:
            inc["life_loss"] = {"dlogloss": s["log_loss"] - sA["log_loss"], "dacc": sA["acc"] - s["acc"],
                                "action_helps": bool(s["log_loss"] - sA["log_loss"] > 0)}
        state_only[h] = so; state_action[h] = sa; incr[h] = inc

    json.dump(state_only, open(f"{BASE}/future_prediction/state_only_metrics.json", "w"), indent=2, default=str)
    json.dump(state_action, open(f"{BASE}/future_prediction/state_action_metrics.json", "w"), indent=2, default=str)
    any_help = any(v.get("action_helps") for h in incr for v in incr[h].values() if isinstance(v, dict))
    json.dump({"incremental": incr, "action_adds_info_any_horizon_component": bool(any_help),
               "split": {"train": sorted(int(x) for x in tr_e), "test": sorted(int(x) for x in te_e)}},
              open(f"{BASE}/future_prediction/incremental_action_metrics.json", "w"), indent=2, default=str)
    print("P5 future prediction done. action_adds_info(any):", any_help)
    for h in H:
        helps = [k for k, v in incr[h].items() if isinstance(v, dict) and v.get("action_helps")]
        print(f"  H={h}: components where action helps: {helps}")


if __name__ == "__main__":
    main()
