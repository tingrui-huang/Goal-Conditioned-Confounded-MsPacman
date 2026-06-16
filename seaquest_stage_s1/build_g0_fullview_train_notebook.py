"""Build notebooks/Seaquest_StageG0_FullView_Train.ipynb — Stage-G0 TRAINING half, THIN.

Pre-training gates + fresh full-view four-frame critic + action-use sanity. Each step is one
committed, locally-tested script. STOPS after the action-use sanity gate (closed-loop rollout
evaluation is a separate local-ALE stage). config drift -> impl audit (12) + visual grid ->
train (oracle=True, 50k) -> action-use sanity.
"""
import json, os
OUT = os.path.join(os.path.dirname(__file__), "..", "notebooks", "Seaquest_StageG0_FullView_Train.ipynb")


def md(*L): return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in L]}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": [l + "\n" for l in s.strip("\n").split("\n")]}

G = "artifacts/seaquest/goal_control/full_view"
cells = []
cells.append(md(
    "# Seaquest Stage-G0 — Full-View Four-Frame Critic (training half)",
    "Train a FRESH full-view (oxygen mask OFF) four-frame contrastive critic with the FROZEN",
    "pipeline — the ONLY change vs the masked four-frame critic is `oracle=True`. Gates: config",
    "drift → 12-point implementation audit + visual grid → train (50k) → action-use sanity.",
    "STOPS before closed-loop rollout (that runs locally on ALE/OCAtari). **Use:** TOKEN + Drive `raw_hf.zip`."))

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

cells.append(code('# 3. Step 4 — config diff vs masked critic (FULL_VIEW_CONFIG_DRIFT gate).\n'
                  '!python -m seaquest_ccrl.scripts.g0_config_diff'))
cells.append(code('# 4. Step 5 — implementation audit (12 assertions on real full-view batches) + visual grid.\n'
                  '!python -m seaquest_ccrl.scripts.g0_train_audit --root "$DATA_ROOT"\n'
                  'from IPython.display import Image, display\n'
                  f'display(Image("{G}/stack_visual_audit.png"))'))
cells.append(code('# 5. Steps 6-7 — train fresh full-view critic (oracle=True, 50k) + action-use sanity gate.\n'
                  '!python -m seaquest_ccrl.scripts.g0_train_fullview --root "$DATA_ROOT" --steps 50000\n'
                  f"import json; print(json.dumps(json.load(open('{G}/action_use_gate.json')), indent=2))"))

cells.append(code(rf"""
# 6. Persist + download the training artifacts (checkpoint, history, audits, diagnostics).
import shutil
shutil.make_archive('seaquest_stage_g0_fullview_train', 'zip', '{G}')
try:
    from google.colab import drive; drive.mount('/content/drive')
    shutil.copy('seaquest_stage_g0_fullview_train.zip', '/content/drive/MyDrive'); print('copied to Drive')
except Exception as e: print('Drive optional:', e)
try:
    from google.colab import files; files.download('seaquest_stage_g0_fullview_train.zip')
except Exception: pass
"""))

cells.append(md(
    "## STOP — training half only",
    "Send `action_use_gate.json` + `implementation_audit.json` + `config_diff_vs_masked.json`.",
    "Closed-loop goal-reaching evaluation (the PRIMARY Stage-G0 gate) runs in the validated local",
    "ALE/OCAtari environment with clone/restore — a separate stage, built after this passes review."))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, "w", encoding="utf-8"), indent=1)
print("wrote", os.path.normpath(OUT), "with", len(cells), "cells")
