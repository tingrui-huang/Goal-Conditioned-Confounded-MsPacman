"""Build notebooks/Seaquest_Oxygen_4Frame_DupRobustness.ipynb — narrow duplicate-robustness
audit, THIN. Re-hash the EXACT V1/V4 model-input tensors, inventory cross-split duplicates +
label consistency (model-independent), then recompute S0-S3 metrics from the ALREADY-SAVED
source-audit predictions (NO retraining). One outcome via the predeclared >=0.15 R2-drop rule.
"""
import json, os
OUT = os.path.join(os.path.dirname(__file__), "..", "notebooks",
                   "Seaquest_Oxygen_4Frame_DupRobustness.ipynb")


def md(*L): return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in L]}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": [l + "\n" for l in s.strip("\n").split("\n")]}

SA = "artifacts/seaquest/oxygen_4frame/leakage/source_audit"
DR = f"{SA}/duplicate_robustness"
cells = []
cells.append(md(
    "# Seaquest Oxygen 4-Frame — Cross-Split Duplicate Robustness (V1 & V4)",
    "Did exact-duplicate visual inputs across train/test inflate the leakage R² (V1=0.8514,",
    "V4=0.8180)? Re-hash the EXACT model inputs (V1 newest masked 84×84×3, V4 four-frame",
    "gameplay-crop 84×84×12, SHA256), inventory cross-split duplicates + label consistency,",
    "then recompute S0–S3 metrics from the SAVED predictions (no retraining). **Use:** TOKEN +",
    "Drive `raw_hf.zip` + your prior `seaquest_oxygen_4frame_phase21b.zip` (holds V1/V4 predictions)."))

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

cells.append(code(rf"""
# 3. Restore the source-audit V1/V4 predictions (needed for the S0-S3 metric recompute).
import os, zipfile
PRIOR = '/content/drive/MyDrive/seaquest_oxygen_4frame_phase21b.zip'   # <-- EDIT if elsewhere
os.makedirs('{SA}', exist_ok=True)
if os.path.exists(PRIOR):
    with zipfile.ZipFile(PRIOR) as z: z.extractall('{SA}')
    print('restored source-audit outputs (predictions) from', PRIOR)
else:
    print('prior phase21b ZIP not found -> metrics will be PENDING_PREDICTIONS')
import glob; print('predictions present:', sorted(os.path.basename(p) for p in glob.glob('{SA}/predictions/*.npz')))
"""))

cells.append(code('# 4. Steps 1-4 — re-hash exact V1/V4 inputs; duplicate inventory + label consistency + figures.\n'
                  '!python -m seaquest_ccrl.scripts.oxy4_dup_inventory --root "$DATA_ROOT"'))
cells.append(code('# 5. Steps 5-7 — recompute S0-S3 metrics from saved predictions; one outcome.\n'
                  '!python -m seaquest_ccrl.scripts.oxy4_dup_metrics'))
cells.append(code(f"# 6. Show the summary + decision.\n"
                  f"print(open('{DR}/SUMMARY.md').read())\n"
                  f"import json; print(json.dumps(json.load(open('{DR}/decision.json')), indent=2))\n"
                  f"from IPython.display import Image, display\n"
                  f"for V in ('V1','V4'):\n"
                  f"    p=f'{DR}/figures/top20_shared_{{V}}.png'\n"
                  f"    import os\n"
                  f"    if os.path.exists(p): display(Image(p))"))

cells.append(code(rf"""
# 7. Persist + download the duplicate-robustness outputs.
import shutil
shutil.make_archive('seaquest_oxygen_4frame_dup_robustness', 'zip', '{DR}')
try:
    from google.colab import drive; drive.mount('/content/drive')
    shutil.copy('seaquest_oxygen_4frame_dup_robustness.zip', '/content/drive/MyDrive'); print('copied to Drive')
except Exception as e: print('Drive optional:', e)
try:
    from google.colab import files; files.download('seaquest_oxygen_4frame_dup_robustness.zip')
except Exception: pass
"""))

cells.append(md(
    "## STOP — narrow robustness audit only",
    "Send `SUMMARY.md` + `decision.json`. Do **not** retrain, change the confounder, or resume",
    "Phase 2.2 / 2.3 / oracle. A stricter V1/V4 rerun is prepared but executes only on explicit",
    "approval if the outcome is `DUPLICATES_MATERIALLY_INFLATE_RESULTS` / `DUPLICATE_FREE_SUBSET_INSUFFICIENT`."))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, "w", encoding="utf-8"), indent=1)
print("wrote", os.path.normpath(OUT), "with", len(cells), "cells")
