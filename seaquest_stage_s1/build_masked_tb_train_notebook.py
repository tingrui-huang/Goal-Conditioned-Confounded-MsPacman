"""Build notebooks/Seaquest_Masked_4Frame_Train_TB.ipynb — re-train the MASKED four-frame
critic on Colab GPU with live TensorBoard. THIN: every step is one committed script.
Lean TB v1 (train/loss, train/diag_acc, train/logit_gap, train/grad_norm, diag/action_shuffle_delta).
Gates: TB smoke -> full 50k train (live TB) -> action-use gate -> persist checkpoint + TB to Drive.
"""
import json, os
OUT = os.path.join(os.path.dirname(__file__), "..", "notebooks", "Seaquest_Masked_4Frame_Train_TB.ipynb")


def md(*L): return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in L]}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": [l + "\n" for l in s.strip("\n").split("\n")]}

CK = "/content/ckpt_masked"; TB = "/content/tb"
cells = []
cells.append(md(
    "# Seaquest — Masked Four-Frame Critic re-train (Colab GPU) + TensorBoard",
    "Re-creates the authoritative **masked/naive** four-frame critic (oxygen bar masked), platform-",
    "matched with the full-view critic. Live TensorBoard (lean v1): `train/loss`, `train/diag_acc`,",
    "`train/logit_gap`, `train/grad_norm`, `diag/action_shuffle_delta`. **Smoke gate first**, then",
    "the 50k run. NOTE: `train/diag_acc` = contrastive learning, NOT goal-reaching success (that is",
    "measured later by the local closed-loop eval -> `eval/*`). **Use:** TOKEN + Drive `raw_hf.zip`."))

cells.append(code("import torch\nprint('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"))

cells.append(code(r"""
# 1. Clone repo at the reviewed commit.
TOKEN = 'PASTE_YOUR_GITHUB_TOKEN_HERE'
import os, subprocess
D = '/content/Goal-Conditioned-Confounded-MsPacman'
if not os.path.isdir(D):
    subprocess.run(['git','clone',f'https://{TOKEN}@github.com/tingrui-huang/Goal-Conditioned-Confounded-MsPacman.git',D], check=True)
%cd /content/Goal-Conditioned-Confounded-MsPacman
!git pull -q && git log --oneline -1
"""))

cells.append(code(r"""
# 2. Load raw_hf from Drive (mount + unzip + auto-locate traj_*.npz).
import glob, os, zipfile
from google.colab import drive
drive.mount('/content/drive')
ZIP = '/content/drive/MyDrive/raw_hf.zip'     # <-- EDIT if elsewhere
assert os.path.exists(ZIP), f'zip not found at {ZIP}'
EXTRACT = '/content/raw_hf_extracted'
if not glob.glob(EXTRACT + '/**/traj_0000.npz', recursive=True):
    with zipfile.ZipFile(ZIP) as z: z.extractall(EXTRACT)
DATA_ROOT = os.path.dirname(glob.glob(EXTRACT + '/**/traj_0000.npz', recursive=True)[0])
n = len(glob.glob(DATA_ROOT + '/traj_*.npz'))
print('DATA_ROOT =', DATA_ROOT, '| trajectories:', n); assert n == 40
"""))

cells.append(code(f"# 3. Live TensorBoard (refreshes during training).\n"
                  f"%load_ext tensorboard\n"
                  f"%tensorboard --logdir {TB}"))

cells.append(code(f'# 4. TB SMOKE GATE (1.2k steps) — events update / finite / save+reload / action_shuffle / files.\n'
                  f'!python -m seaquest_ccrl.scripts.g0_tb_smoke --root "$DATA_ROOT" --steps 1200 --shuffle-every 400\n'
                  f"import json; print(json.dumps(json.load(open('artifacts/_tb_smoke/tb_smoke_gate.json'))['outcome'], indent=2))"))

cells.append(code(f'# 5. Full masked 50k train with live TensorBoard + provenance. (Watch cell 3.)\n'
                  f'!python -u -m seaquest_ccrl.scripts.run_hf_4frame --root "$DATA_ROOT" --steps 50000 --seed 0 '
                  f'--ckpt-dir {CK} --tb-logdir {TB}'))

cells.append(code(f'# 6. Action-use gate (confirm the masked critic uses the action, like Phase 1).\n'
                  f'!python -m seaquest_ccrl.scripts.eval_hf_action_use --ckpt {CK}/critic_naive.pt --root "$DATA_ROOT" '
                  f'--out {CK}/action_use_diag.json\n'
                  f'!python -m seaquest_ccrl.scripts.gate_hf_4frame --ckpt {CK}/critic_naive.pt '
                  f'--diag {CK}/action_use_diag.json --ckpt-dir {CK} --out-dir {CK} --zip seaquest_masked_4frame\n'
                  f"import json; print(json.load(open('{CK}/action_use_gate.json'))['outcome'])"))

cells.append(code(rf"""
# 7. Persist checkpoint + TensorBoard events + provenance to Drive + download.
import shutil, os
os.makedirs('/content/masked_out', exist_ok=True)
for p in ['{CK}/critic_naive.pt','{CK}/history_naive.json','{CK}/run_provenance.json',
          '{CK}/action_use_diag.json','{CK}/action_use_gate.json']:
    if os.path.exists(p): shutil.copy(p, '/content/masked_out')
shutil.copytree('{TB}', '/content/masked_out/tb', dirs_exist_ok=True)
shutil.make_archive('seaquest_masked_4frame_train', 'zip', '/content/masked_out')
try:
    from google.colab import drive; drive.mount('/content/drive')
    shutil.copy('seaquest_masked_4frame_train.zip', '/content/drive/MyDrive'); print('copied to Drive')
except Exception as e: print('Drive optional:', e)
try:
    from google.colab import files; files.download('seaquest_masked_4frame_train.zip')
except Exception: pass
"""))

cells.append(md(
    "## Next — local closed-loop eval",
    "Download `critic_naive.pt`, drop it at `artifacts/seaquest/oxygen_4frame/naive_critic_authoritative/critic_naive.pt`,",
    "then run the LOCAL Docker masked eval (`g0_closed_loop_eval --critic-ckpt …`). Finally",
    "`g0_eval_tb_log.py` logs `eval/full_view_*` vs `eval/masked_*` (the real goal-reaching comparison)."))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, "w", encoding="utf-8"), indent=1)
print("wrote", os.path.normpath(OUT), "with", len(cells), "cells")
