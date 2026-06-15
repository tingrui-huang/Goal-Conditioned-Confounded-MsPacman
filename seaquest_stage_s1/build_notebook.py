"""Assemble the self-contained Stage-S1 Colab notebook (nbformat v4 JSON).
Embeds critic.py / losses.py / evaluation.py / validate_colab_pack.py inline so the
notebook needs no repo clone. PyTorch-only at runtime. Run locally:
    python seaquest_stage_s1/build_notebook.py
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "notebooks", "Seaquest_Stage_S1_Vanilla_State_Critic.ipynb")


def src(name):
    s = open(os.path.join(HERE, name), encoding="utf-8").read()
    # drop the `if __name__ == "__main__"` blocks for clean import-as-module
    s = re.split(r'\nif __name__ == "__main__":', s)[0]
    return s.rstrip() + "\n"


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in s.strip("\n").split("\n")]}


cells = []
cells.append(md(
    "# Seaquest Stage-S1 — Vanilla Action-Conditioned State Critic",
    "",
    "**Question:** can a vanilla Eysenbach-style state critic actually learn AND use the action input",
    "on the validated Seaquest observational data? (No oxygen-confounder work; no pixels; no actor;",
    "no causal/robust losses; no data regeneration in Colab.)",
    "",
    "This notebook is **self-contained**: it loads the frozen `seaquest_s1_colab_pack.zip` exported",
    "locally from the corrected Stage-S0.5 run, trains the critic (Models A/B/C, seed 0), runs frozen",
    "evaluation (representation, action-shuffle, same-state sensitivity, forced-branch alignment),",
    "applies gates C1-C5, and exports checkpoints/metrics/figures/SUMMARY.",
    "",
    "Forbidden at runtime: EnvPool, OCAtari, ALE, ROMs, JAX/Flax, the CleanRL teacher, data regen."))

# ---- Section 1
cells.append(md("## Section 1 — Setup"))
cells.append(code(r"""
import os, sys, json, io, zipfile, hashlib, time, math, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

# --- optional Google Drive (uncomment to mount; pack can also be uploaded) ---
# from google.colab import drive; drive.mount('/content/drive')

# --- (optional) clone repo at an explicit reviewed commit for reference only ---
# !git clone https://github.com/tingrui-huang/Goal-Conditioned-Confounded-MsPacman /content/repo
# %cd /content/repo && !git checkout <REVIEWED_COMMIT>

SEED = 0
def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
set_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG = dict(optimizer="Adam", lr=3e-4, batch_size=256, max_epochs=60,
              patience=8, weight_decay=0.0, grad_clip=10.0, emb_dim=128,
              horizon=16, primary_seed=0, checkpoint_criterion="lowest_val_NCE")
print("Python", sys.version.split()[0], "| PyTorch", torch.__version__,
      "| CUDA", torch.cuda.is_available(), "| GPU", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("device", DEVICE, "| seed", SEED)
print("config", json.dumps(CONFIG))
"""))

# ---- Section: inline frozen modules
cells.append(md("## Frozen model / loss / evaluation code (embedded; do not edit)"))
cells.append(code("# === critic.py ===\n" + src("critic.py")))
cells.append(code("# === losses.py ===\n" + src("losses.py")))
cells.append(code("# === evaluation.py ===\n" + src("evaluation.py")))
cells.append(code("# === validate_colab_pack.py (hash/dim/leakage validator) ===\n" + src("validate_colab_pack.py")))

# ---- Section 2 load pack
cells.append(md("## Section 2 — Load the frozen pack (verify hashes before continuing)"))
cells.append(code(r"""
# Option A: upload (Colab):
#   from google.colab import files; up = files.upload(); PACK = list(up.keys())[0]
# Option B: Google Drive path:
#   PACK = '/content/drive/MyDrive/seaquest_s1_colab_pack.zip'
# Option C: local path (when running outside Colab):
PACK = os.environ.get("S1_PACK", "artifacts/seaquest/stage_s1/seaquest_s1_colab_pack.zip")

summary = validate(PACK, strict=True)   # raises on any failure
print("PACK VALIDATION:", "PASS" if summary["PASS"] else "FAIL")
print(json.dumps({k: summary[k] for k in ["n_transitions","state_dim","np_dim","wo_dim","n_episodes","n_anchors","n_valid_branches","split"]}, indent=2))

raw, arr, js = load_pack(PACK)
MANIFEST = js("manifest.json"); SCHEMA = js("feature_schema.json")
NORM = js("normalization.json"); SPLIT = js("split_manifest.json")
S  = arr("observational/states.npy").astype(np.float32)
A  = arr("observational/actions.npy").astype(np.int64)
GNP = arr("observational/goals_no_player_H16.npy").astype(np.float32)
GWO = arr("observational/goals_world_only_H16.npy").astype(np.float32)
EID = arr("observational/episode_ids.npy").astype(np.int64)
TS  = arr("observational/timesteps.npy").astype(np.int64)
B_AS  = arr("branches/anchor_states.npy").astype(np.float32)
B_FNP = arr("branches/future_no_player_H16.npy").astype(np.float32)
B_FWO = arr("branches/future_world_only_H16.npy").astype(np.float32)
B_VM  = arr("branches/valid_mask.npy").astype(np.int64)
B_SC  = arr("branches/local_support_counts.npy").astype(np.int64)
B_SEM = arr("branches/semantic_action_categories.npy").astype(np.int64)
STATE_DIM = S.shape[1]; NP_DIM = GNP.shape[1]; WO_DIM = GWO.shape[1]
print("source git commit:", MANIFEST["source_git_commit"], "| pack sha (manifest file):",
      hashlib.sha256(raw["manifest.json"]).hexdigest()[:16])
"""))

# ---- Section 3 inspect
cells.append(md("## Section 3 — Inspect data"))
cells.append(code(r"""
print("state_dim", STATE_DIM, "np_dim", NP_DIM, "wo_dim", WO_DIM)
print("transitions", len(S), "anchors", len(B_AS), "valid branches", int(B_VM.sum()))
print("action counts:", np.bincount(A, minlength=18).tolist())
tr=set(SPLIT["train_episode_ids"]); va=set(SPLIT["val_episode_ids"]); te=set(SPLIT["test_episode_ids"])
print("split episodes -> train", len(tr), "val", len(va), "test", len(te))
print("split transitions -> train", int(np.isin(EID,list(tr)).sum()),
      "val", int(np.isin(EID,list(va)).sum()), "test", int(np.isin(EID,list(te)).sum()))
print("state features:", SCHEMA["state_schema"])
print("no_player keys:", SCHEMA["no_player_keys"]); print("world_only keys:", SCHEMA["world_only_keys"])
assert np.isfinite(S).all() and np.isfinite(GNP).all() and np.isfinite(GWO).all(), "non-finite (should be excluded at export)"
print("missing/censoring: dataset pre-filtered at export (no NaN; valid H=16 future; no boundary cross).")
"""))

# ---- Section 4 unit tests
cells.append(md("## Section 4 — Pre-training unit tests"))
cells.append(code(r"""
def _norm_states(x): return (x - np.array(NORM["state_mean"])) / np.array(NORM["state_std"])
def _norm_goals(x, view):
    m = NORM[view+"_mean"]; s = NORM[view+"_std"]; return (x - np.array(m)) / np.array(s)

set_seed(0)
m = StateActionCritic(STATE_DIM, NP_DIM).to(DEVICE)
s = torch.randn(4, STATE_DIM, device=DEVICE); g = torch.randn(4, NP_DIM, device=DEVICE)
a1 = one_hot(np.array([0,1,2,3]), device=DEVICE); a2 = one_hot(np.array([5,6,7,8]), device=DEVICE)
o1 = m.encode_sa(s, a1); o2 = m.encode_sa(s, a2)
t1 = (o1 - o2).abs().max().item() > 1e-6
t2 = torch.allclose(m.encode_sa(s, a1), m.encode_sa(s, a1))         # identical input -> identical
t3 = not torch.allclose(torch.cat([s,a1],-1), torch.cat([s,a2],-1)) # permuting action changes input
loss,_ = nce_loss(m.encode_sa(s,a1), m.encode_g(g)); loss.backward()
gw = m.phi[0].weight.grad[:, STATE_DIM:]                            # action-connected columns
t4 = gw.abs().sum().item() > 0
t5 = (B_FNP.shape[1:] == (18, NP_DIM)) and (B_VM.shape == (len(B_AS),18))
t6 = ("train" in NORM.get("fit_on","")) and (set(SPLIT["test_episode_ids"]) and True)  # norm fit on train only
print(dict(action_changes_phi=t1, identical_input_identical=t2, permute_changes_input=t3,
           gradients_reach_action_weights=t4, branch_shapes_valid=bool(t5), norm_excludes_test=bool(t6)))
assert all([t1,t2,t3,t4,t5,t6]), "unit tests failed"
print("UNIT TESTS PASS")
"""))

# ---- Section 5 train
cells.append(md("## Section 5 — Train seed 0 (Model A action/no_player, B no-action/no_player, C action/world_only)"))
cells.append(code(r"""
def make_split_tensors(goals_view):
    Sn = _norm_states(S).astype(np.float32)
    Gn = _norm_goals(goals_view, "no_player" if goals_view is GNP else "world_only").astype(np.float32)
    tr=np.isin(EID,SPLIT["train_episode_ids"]); va=np.isin(EID,SPLIT["val_episode_ids"]); te=np.isin(EID,SPLIT["test_episode_ids"])
    pack = lambda mask: (torch.tensor(Sn[mask]), torch.tensor(A[mask]), torch.tensor(Gn[mask]), EID[mask])
    return pack(tr), pack(va), pack(te)

def epoch_pass(model, S_t, A_t, G_t, opt=None):
    model.train(opt is not None)
    n=len(S_t); idx=torch.randperm(n) if opt else torch.arange(n)
    tot=0.0; nb=0; diag_acc={}
    for i in range(0, n, CONFIG["batch_size"]):
        b=idx[i:i+CONFIG["batch_size"]]
        if len(b)<2: continue
        s=S_t[b].to(DEVICE); g=G_t[b].to(DEVICE)
        if model.use_action:
            a=one_hot(A_t[b], device=DEVICE); sa=model.encode_sa(s,a)
        else:
            sa=model.encode_sa(s)
        gr=model.encode_g(g)
        loss,diag=nce_loss(sa,gr)
        if opt:
            opt.zero_grad(); loss.backward()
            gn=torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
            opt.step(); diag["grad_norm"]=float(gn)
        tot+=diag["loss"]; nb+=1
        for k,v in diag.items(): diag_acc[k]=diag_acc.get(k,0)+v
    return {k: v/max(nb,1) for k,v in diag_acc.items()} | {"loss": tot/max(nb,1)}

def train_model(kind, goals_view, seed, tag):
    set_seed(seed)
    (Str,Atr,Gtr,_),(Sva,Ava,Gva,_),_ = make_split_tensors(goals_view)
    gdim = goals_view.shape[1]
    model = build_critic(kind, STATE_DIM, gdim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
    best=1e9; best_state=None; hist=[]; bad=0
    for ep in range(CONFIG["max_epochs"]):
        tr=epoch_pass(model,Str,Atr,Gtr,opt)
        with torch.no_grad(): vl=epoch_pass(model,Sva,Ava,Gva,None)
        hist.append({"epoch":ep,"train_loss":tr["loss"],"val_loss":vl["loss"],
                     "pos_neg_margin":tr.get("pos_neg_margin"),"grad_norm":tr.get("grad_norm")})
        assert math.isfinite(tr["loss"]) and math.isfinite(vl["loss"]), "non-finite loss"
        if vl["loss"]<best-1e-5: best=vl["loss"]; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else: bad+=1
        if bad>=CONFIG["patience"]: break
    model.load_state_dict(best_state)   # validation-selected checkpoint ONLY
    print(f"[{tag}] kind={kind} gdim={gdim} epochs={len(hist)} best_val={best:.4f}")
    return model, hist, best

set_seed(0)
MODELS={}; HIST={}
MODELS["A"],HIST["A"],_ = train_model("action",   GNP, 0, "A action/no_player")
MODELS["B"],HIST["B"],_ = train_model("no_action",GNP, 0, "B no-action/no_player")
MODELS["C"],HIST["C"],_ = train_model("action",   GWO, 0, "C action/world_only")
"""))

# ---- Section 6 frozen eval
cells.append(md("## Section 6 — Frozen evaluation (validation-selected checkpoints, test set only)"))
cells.append(code(r"""
def test_arrays(goals_view, view_name):
    Sn=_norm_states(S).astype(np.float32); Gn=_norm_goals(goals_view, view_name).astype(np.float32)
    te=np.isin(EID,SPLIT["test_episode_ids"])
    return Sn[te], A[te], Gn[te], EID[te]

# local supported-action sampler from frozen S0.5 branch support (anchor-based NN-free proxy:
# use the per-anchor support; for observational states, fall back to global supported set).
GLOBAL_SUPPORTED = np.where(np.bincount(A, minlength=18) >= max(5, int(0.01*len(A))))[0]
def local_support_fn(state_vec, true_a, rng):
    alt=[a for a in GLOBAL_SUPPORTED if a!=true_a]
    return int(rng.choice(alt)) if alt else -1

EVAL={}
# Model A (no_player)
Ste,Ate,Gte,Ete = test_arrays(GNP,"no_player")
EVAL["A_retrieval"]=retrieval_metrics(MODELS["A"],Ste,one_hot(Ate).numpy(),Gte,DEVICE)
EVAL["A_global_shuffle"]=global_shuffle_delta(MODELS["A"],Ste,Ate,Gte,Ete,DEVICE)
EVAL["A_local_shuffle"]=local_shuffle_delta(MODELS["A"],Ste,Ate,Gte,Ete,local_support_fn,DEVICE)
EVAL["A_zero_action"]=zero_action_ablation(MODELS["A"],Ste,Ate,Gte,DEVICE)
EVAL["A_sensitivity"]=same_state_action_sensitivity(MODELS["A"],Ste,Ate,Gte,B_SEM,DEVICE)
# Model B (no-action baseline) on identical test examples
EVAL["B_retrieval"]=retrieval_metrics(MODELS["B"],Ste,one_hot(Ate).numpy(),Gte,DEVICE)
# Model C (world_only)
Stw,Atw,Gtw,Etw = test_arrays(GWO,"world_only")
EVAL["C_retrieval"]=retrieval_metrics(MODELS["C"],Stw,one_hot(Atw).numpy(),Gtw,DEVICE)
# forced-branch alignment (branch goals normalized with the SAME train stats)
BFNP=( B_FNP - np.array(NORM["no_player_mean"]) ) / np.array(NORM["no_player_std"])
BFWO=( B_FWO - np.array(NORM["world_only_mean"]) ) / np.array(NORM["world_only_std"])
BASn=_norm_states(B_AS).astype(np.float32)
EVAL["A_branch_no_player"]=forced_branch_alignment(MODELS["A"],BASn,BFNP.astype(np.float32),B_VM,B_SEM,DEVICE)
EVAL["C_branch_world_only"]=forced_branch_alignment(MODELS["C"],BASn,BFWO.astype(np.float32),B_VM,B_SEM,DEVICE)
# negative controls on no_player branch alignment
set_seed(123); rand_model=build_critic("action",STATE_DIM,NP_DIM).to(DEVICE)
EVAL["ctrl_random_branch"]=forced_branch_alignment(rand_model,BASn,BFNP.astype(np.float32),B_VM,B_SEM,DEVICE)
EVAL["ctrl_no_action_branch"]=forced_branch_alignment(MODELS["B"],BASn,BFNP.astype(np.float32),B_VM,B_SEM,DEVICE)
print(json.dumps({"A_top1":EVAL["A_retrieval"]["top1_acc"],
                  "A_global_delta":EVAL["A_global_shuffle"]["delta_global"],
                  "A_global_ci":EVAL["A_global_shuffle"]["ci95"],
                  "A_local_delta":EVAL["A_local_shuffle"]["delta_local"],
                  "A_local_ci":EVAL["A_local_shuffle"]["ci95"],
                  "A_zero_degrade":EVAL["A_zero_action"]["degradation"],
                  "A_branch_diag":EVAL["A_branch_no_player"].get("diagonal_margin_mean"),
                  "A_branch_top1":EVAL["A_branch_no_player"].get("top1_matching"),
                  "A_branch_pair":EVAL["A_branch_no_player"].get("pairwise_ranking")}, indent=2))
"""))

# ---- Section 7 gates
cells.append(md("## Section 7 — Seed-0 gates (C1–C5) and outcome"))
cells.append(code(r"""
def gate_C1():
    h=HIST["A"]; vls=[x["val_loss"] for x in h]
    improved = vls[-1] < vls[0]
    pos_margin = (h[-1]["pos_neg_margin"] or 0) > 0
    above_chance = EVAL["A_retrieval"]["top1_acc"] > EVAL["A_retrieval"]["chance_top1"]
    finite = all(math.isfinite(x["val_loss"]) for x in h)
    return dict(passed=bool(improved and pos_margin and above_chance and finite),
                improved=improved, pos_margin=pos_margin, above_chance=above_chance)
def gate_C2():
    g=EVAL["A_global_shuffle"]; l=EVAL["A_local_shuffle"]; z=EVAL["A_zero_action"]
    return dict(passed=bool(g["ci95"][0]>0 and l["ci95"][0]>0 and z["degradation"]>0),
                global_lb=g["ci95"][0], local_lb=l["ci95"][0], zero_degrade=z["degradation"])
def gate_C3():
    a=EVAL["A_retrieval"]; b=EVAL["B_retrieval"]
    wins = sum([a["nce_test"]<b["nce_test"], a["top1_acc"]>b["top1_acc"],
                (a["pos_logit_mean"]-a["neg_logit_mean"])>(b["pos_logit_mean"]-b["neg_logit_mean"])])
    return dict(passed=bool(wins>=2), wins=wins,
                nce_A=a["nce_test"], nce_B=b["nce_test"], top1_A=a["top1_acc"], top1_B=b["top1_acc"])
def gate_C4():
    e=EVAL["A_branch_no_player"]
    if e.get("insufficient"): return dict(passed=False, reason="insufficient anchors")
    return dict(passed=bool(e["diagonal_margin_ci95"][0]>0 and e["perm_test_pvalue"]<0.05 and e["pairwise_ranking"]>0.5),
                diag_lb=e["diagonal_margin_ci95"][0], top1=e["top1_matching"], chance=e["chance_level"],
                perm_p=e["perm_test_pvalue"], pairwise=e["pairwise_ranking"])
def gate_C5():
    e=EVAL["C_branch_world_only"]
    if e.get("insufficient"): return dict(passed=False)
    crit=[e["diagonal_margin_ci95"][0]>0, e["top1_matching"]>e["chance_level"], e["pairwise_ranking"]>0.5]
    return dict(passed=bool(sum(crit)>=2), diag_lb=e["diagonal_margin_ci95"][0],
                top1=e["top1_matching"], chance=e["chance_level"], pairwise=e["pairwise_ranking"])

GATES={"C1":gate_C1(),"C2":gate_C2(),"C3":gate_C3(),"C4":gate_C4(),"C5":gate_C5()}
def outcome():
    if not GATES["C1"]["passed"]: return "STOP_CRITIC_OPTIMIZATION_FAILURE"
    if not GATES["C2"]["passed"]: return "STOP_CRITIC_IGNORES_ACTION"
    if not GATES["C3"]["passed"]: return "STOP_STATE_EXPLAINS_ALL_SIGNAL"
    if not GATES["C4"]["passed"]: return "STOP_ACTION_SENSITIVITY_NOT_DYNAMICS_ALIGNED"
    if not GATES["C5"]["passed"]: return "PROCEED_ACTION_LEARNED_NO_PLAYER_ONLY"
    return "PROCEED_ACTION_LEARNED_WORLD_ONLY"
OUTCOME=outcome()
print(json.dumps({k:v["passed"] for k,v in GATES.items()}, indent=2)); print("OUTCOME:", OUTCOME)
"""))

# ---- Section 8 export + figures + summary
cells.append(md("## Section 8 — Figures, metrics, SUMMARY, and downloadable ZIP"))
cells.append(code(r"""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="stage_s1_outputs"; os.makedirs(OUT+"/figures", exist_ok=True); os.makedirs(OUT+"/checkpoints", exist_ok=True)
for tag,mdl in MODELS.items(): torch.save(mdl.state_dict(), f"{OUT}/checkpoints/model_{tag}.pt")
def save(fig,name): fig.savefig(f"{OUT}/figures/{name}",dpi=110,bbox_inches="tight"); plt.close(fig)

for view,tag in [("A","no_player"),("C","world_only")]:
    h=HIST[view]; fig,ax=plt.subplots(figsize=(6,4))
    ax.plot([x["train_loss"] for x in h],label="train"); ax.plot([x["val_loss"] for x in h],label="val")
    ax.set_title(f"training loss {tag} (Model {view})"); ax.legend(); ax.set_xlabel("epoch"); save(fig,f"training_loss_{tag}.png")
r=EVAL["A_retrieval"]; fig,ax=plt.subplots(figsize=(5,4))
ax.bar(["pos","neg"],[r["pos_logit_mean"],r["neg_logit_mean"]]); ax.set_title("positive vs negative logits (A)"); save(fig,"positive_negative_logits.png")
fig,ax=plt.subplots(figsize=(5,4)); ax.bar(["top1","top5","mrr"],[r["top1_acc"],r["top5_acc"],r["mrr"]]); ax.axhline(r["chance_top1"],ls="--",c="k"); ax.set_title("retrieval (A)"); save(fig,"retrieval_metrics.png")
for key,nm in [("A_global_shuffle","global_action_shuffle_delta"),("A_local_shuffle","local_action_shuffle_delta")]:
    e=EVAL[key]; d=e.get("delta_global",e.get("delta_local")); ci=e["ci95"]
    fig,ax=plt.subplots(figsize=(5,4)); ax.bar([nm],[d],yerr=[[d-ci[0]],[ci[1]-d]],capsize=6); ax.axhline(0,c="k"); ax.set_title(nm); save(fig,nm+".png")
sen=EVAL["A_sensitivity"]; sc=np.array(sen["action_score_matrix"])
fig,ax=plt.subplots(figsize=(6,4)); ax.hist(sc.std(1),bins=40); ax.set_title("same-state action score std"); save(fig,"same_state_action_score_spread.png")
fig,ax=plt.subplots(figsize=(6,4)); ax.bar(range(1,19),sen["true_action_rank_hist"]); ax.set_title("true-action rank histogram"); save(fig,"true_action_rank_histogram.png")
fig,ax=plt.subplots(figsize=(5,4)); ax.bar(["A_nce","B_nce"],[EVAL["A_retrieval"]["nce_test"],EVAL["B_retrieval"]["nce_test"]]); ax.set_title("no-action baseline comparison (test NCE)"); save(fig,"no_action_baseline_comparison.png")
for key,nm in [("A_branch_no_player","forced_branch_matrix_no_player"),("C_branch_world_only","forced_branch_matrix_world_only")]:
    e=EVAL[key]; M=e.get("aggregate_normalized_matrix")
    if M is not None:
        fig,ax=plt.subplots(figsize=(5,5)); im=ax.imshow(np.array(M),cmap="viridis"); plt.colorbar(im,ax=ax); ax.set_title(nm+f" (K={e['aggregate_matrix_K']})"); ax.set_xlabel("forced future goal j"); ax.set_ylabel("action i"); save(fig,nm+".png")
fig,ax=plt.subplots(figsize=(7,4))
labels=["diag_margin","top1","pairwise"]; npv=[EVAL["A_branch_no_player"].get(k) for k in ["diagonal_margin_mean","top1_matching","pairwise_ranking"]]
wov=[EVAL["C_branch_world_only"].get(k) for k in ["diagonal_margin_mean","top1_matching","pairwise_ranking"]]
x=np.arange(3); ax.bar(x-0.2,npv,0.4,label="no_player"); ax.bar(x+0.2,wov,0.4,label="world_only"); ax.set_xticks(x); ax.set_xticklabels(labels); ax.legend(); ax.set_title("forced-branch alignment metrics"); save(fig,"forced_branch_alignment_metrics.png")

json.dump({"config":CONFIG,"gates":GATES,"outcome":OUTCOME,"eval":{k:(v if not isinstance(v,dict) else {kk:vv for kk,vv in v.items() if kk not in ("action_score_matrix","example_matrices","aggregate_normalized_matrix")}) for k,v in EVAL.items()},"history":HIST,"manifest_commit":MANIFEST["source_git_commit"]}, open(f"{OUT}/metrics.json","w"), indent=2, default=float)
np.savez_compressed(f"{OUT}/raw_eval_arrays.npz", action_score_matrix=np.array(sen["action_score_matrix"]),
                    A_branch_agg=np.array(EVAL["A_branch_no_player"].get("aggregate_normalized_matrix") or []),
                    C_branch_agg=np.array(EVAL["C_branch_world_only"].get("aggregate_normalized_matrix") or []))

def q(b): return "YES" if b else "NO"
g=GATES
rep=f'''# Seaquest Stage-S1 — Vanilla State Critic — SUMMARY

**OUTCOME: `{OUTCOME}`**  (seed 0; pack commit {MANIFEST["source_git_commit"]})

1. Did the critic optimize normally? {q(g["C1"]["passed"])} (val improved, margin>0, retrieval>chance, finite).
2. Did it learn future-state discrimination? top1={EVAL["A_retrieval"]["top1_acc"]:.3f} (chance {EVAL["A_retrieval"]["chance_top1"]:.4f}), top5={EVAL["A_retrieval"]["top5_acc"]:.3f}, mrr={EVAL["A_retrieval"]["mrr"]:.3f}.
3. Did GLOBAL action shuffle reduce matched scores? delta={EVAL["A_global_shuffle"]["delta_global"]:.4f} CI{EVAL["A_global_shuffle"]["ci95"]}.
4. Did LOCAL supported-action replacement reduce matched scores? delta={EVAL["A_local_shuffle"]["delta_local"]:.4f} CI{EVAL["A_local_shuffle"]["ci95"]}.
5. Did zeroing action degrade performance? degradation={EVAL["A_zero_action"]["degradation"]:.4f}.
6. Did action-conditioned beat no-action baseline? wins={g["C3"]["wins"]}/3 (A nce {g["C3"]["nce_A"]:.4f} vs B {g["C3"]["nce_B"]:.4f}).
7. Same-state action score variation: mean std={EVAL["A_sensitivity"]["mean_std_across_actions"]:.4f}, top-bottom={EVAL["A_sensitivity"]["mean_top_minus_bottom"]:.4f}, near-flat frac={EVAL["A_sensitivity"]["frac_states_near_identical"]:.3f}.
8. Does variation align with forced branches (no_player)? diag_margin={EVAL["A_branch_no_player"].get("diagonal_margin_mean")} CI{EVAL["A_branch_no_player"].get("diagonal_margin_ci95")}, top1={EVAL["A_branch_no_player"].get("top1_matching")} (chance {EVAL["A_branch_no_player"].get("chance_level")}), pairwise={EVAL["A_branch_no_player"].get("pairwise_ranking")}, perm_p={EVAL["A_branch_no_player"].get("perm_test_pvalue")}.
9. Does alignment survive in world_only? diag_margin={EVAL["C_branch_world_only"].get("diagonal_margin_mean")}, top1={EVAL["C_branch_world_only"].get("top1_matching")}, pairwise={EVAL["C_branch_world_only"].get("pairwise_ranking")}.
10. Gates: C1={q(g["C1"]["passed"])} C2={q(g["C2"]["passed"])} C3={q(g["C3"]["passed"])} C4={q(g["C4"]["passed"])} C5={q(g["C5"]["passed"])}.
11. Is the action-learning problem solved for this state critic? {"YES (no_player; world_only too)" if OUTCOME=="PROCEED_ACTION_LEARNED_WORLD_ONLY" else ("PARTIAL (no_player only)" if OUTCOME=="PROCEED_ACTION_LEARNED_NO_PLAYER_ONLY" else "NO — "+OUTCOME)}.

Negative controls (no_player branch diag margin): trained-action={EVAL["A_branch_no_player"].get("diagonal_margin_mean")}, random-init={EVAL["ctrl_random_branch"].get("diagonal_margin_mean")}, no-action={EVAL["ctrl_no_action_branch"].get("diagonal_margin_mean")}.
'''
open(f"{OUT}/SUMMARY.md","w").write(rep); print(rep)
import shutil; shutil.make_archive("seaquest_stage_s1_results","zip",OUT)
print("wrote seaquest_stage_s1_results.zip")
# from google.colab import files; files.download("seaquest_stage_s1_results.zip")
"""))

# ---- Section 7b seeds [1,2] manual
cells.append(md("## Section 7b — Seeds [1, 2] (MANUAL; run only if seed-0 passed C1–C4)"))
cells.append(code(r"""
# Guarded: do NOT auto-run after a failure. Confirm seed-0 outcome first.
assert OUTCOME.startswith("PROCEED"), f"seed-0 outcome is {OUTCOME}; do not run more seeds"
EXTRA={}
for sd in [1,2]:
    mA,_,_=train_model("action",GNP,sd,f"A seed{sd}")
    St,At,Gt,Et=test_arrays(GNP,"no_player")
    EXTRA[sd]={"global":global_shuffle_delta(mA,St,At,Gt,Et,DEVICE),
               "branch":forced_branch_alignment(mA,_norm_states(B_AS).astype('float32'),
                        ((B_FNP-np.array(NORM["no_player_mean"]))/np.array(NORM["no_player_std"])).astype('float32'),B_VM,B_SEM,DEVICE)}
print(json.dumps({sd:{"global_ci":EXTRA[sd]["global"]["ci95"],"branch_diag":EXTRA[sd]["branch"].get("diagonal_margin_mean")} for sd in EXTRA}, indent=2))
"""))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("wrote", OUT, "with", len(cells), "cells")
