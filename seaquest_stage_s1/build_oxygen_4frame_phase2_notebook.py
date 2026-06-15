"""Build notebooks/Seaquest_Oxygen_4Frame_Phase2.ipynb — PHASE 2 ONLY, THIN.
Every step is one committed, locally-tested script: assertions -> leakage (2.1) ->
U->A (2.2) -> U->future (2.3, H=16/32/64) -> qualification. STOPS after the qualification
report (no Phase 3 / oracle / anchors / paired eval).
"""
import json, os
OUT = os.path.join(os.path.dirname(__file__), "..", "notebooks", "Seaquest_Oxygen_4Frame_Phase2.ipynb")


def md(*L): return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in L]}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": [l + "\n" for l in s.strip("\n").split("\n")]}

B = "artifacts/seaquest/oxygen_4frame"
cells = []
cells.append(md(
    "# Seaquest Oxygen 4-Frame Study — Phase 2 (oxygen qualification probes)",
    "Three supervised probes on the FROZEN `raw_hf` masked four-frame state — leakage, U→A,",
    "U→future (H=16/32/64) — then one qualification outcome. Each step is one committed,",
    "locally-tested script (no inline logic). Does NOT retrain the critic, build the oracle,",
    "or do anchors/paired eval. **Use:** TOKEN (Cell 1) + Drive raw_hf.zip (Cell 2), Run-all."))

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
hits = glob.glob(EXTRACT + '/**/traj_0000.npz', recursive=True)
assert hits, f'no traj_*.npz inside {ZIP}'
DATA_ROOT = os.path.dirname(hits[0])
n = len(glob.glob(DATA_ROOT + '/traj_*.npz'))
print('DATA_ROOT =', DATA_ROOT, '| trajectories:', n)
assert n == 40, f'expected 40 traj, found {n}'
"""))

cells.append(code('# 3. Pre-probe assertions (shape (B,84,84,12), masking, alignment, no boundary cross, no future-oxygen target, frozen split seed 2606).\n'
                  '!python -m seaquest_ccrl.scripts.oxy4_assertions --root "$DATA_ROOT"'))
cells.append(code('# 4. Phase 2.1 — four-frame oxygen leakage (masked vs visible vs trivial baseline).\n'
                  '!python -m seaquest_ccrl.scripts.oxy4_probe_leakage --root "$DATA_ROOT"'))
cells.append(code('# 5. Phase 2.2 — conditional U->A (masked state vs +oxygen; exact-18 + semantic-12).\n'
                  '!python -m seaquest_ccrl.scripts.oxy4_probe_action --root "$DATA_ROOT"'))
cells.append(code('# 6. Phase 2.3 — conditional U->future at H=16,32,64 (state+action vs +oxygen).\n'
                  '!python -m seaquest_ccrl.scripts.oxy4_probe_future --root "$DATA_ROOT" --horizons 16,32,64'))
cells.append(code('# 7. Oxygen qualification report (one outcome).\n'
                  '!python -m seaquest_ccrl.scripts.oxy4_qualify\n'
                  f"import json; print(json.dumps(json.load(open('{B}/oxygen_qualification.json')), indent=2))"))

cells.append(code(rf"""
# 8. Persist + download Phase-2 outputs (predictions, bootstrap inputs, metrics, figures).
import shutil, os, glob
shutil.make_archive('seaquest_oxygen_4frame_phase2', 'zip', '{B}')
try:
    from google.colab import drive; drive.mount('/content/drive')
    dst='/content/drive/MyDrive/seaquest_oxygen_4frame_phase2'; os.makedirs(dst, exist_ok=True)
    shutil.copy('seaquest_oxygen_4frame_phase2.zip', dst); print('copied to Drive')
except Exception as e: print('Drive optional:', e)
try:
    from google.colab import files; files.download('seaquest_oxygen_4frame_phase2.zip')
except Exception: pass
"""))

cells.append(md(
    "## STOP — Phase 2 only",
    "Send `oxygen_qualification.json` for review. Do **not** start Phase 3 (oracle critic) /",
    "anchors / paired evaluation until the qualification outcome is approved."))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, "w", encoding="utf-8"), indent=1)
print("wrote", os.path.normpath(OUT), "with", len(cells), "cells")
