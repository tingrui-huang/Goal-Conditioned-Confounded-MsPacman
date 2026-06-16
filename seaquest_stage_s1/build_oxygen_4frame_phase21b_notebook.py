"""Build notebooks/Seaquest_Oxygen_4Frame_Phase21b_LeakageAudit.ipynb — PHASE 2.1b ONLY, THIN.

Leakage SOURCE audit: preserve original 2.1 -> implementation audit -> region figure ->
matched visual probes V1-V4 + proxy baselines P1-P3 + sanity controls -> one primary
outcome + 9 decision answers. Each step is one committed, locally-tested script. Does NOT
run Phase 2.2/2.3, does NOT train the oracle, does NOT modify raw_hf / the naive critic /
the original 2.1 artifacts.
"""
import json, os
OUT = os.path.join(os.path.dirname(__file__), "..", "notebooks",
                   "Seaquest_Oxygen_4Frame_Phase21b_LeakageAudit.ipynb")


def md(*L): return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in L]}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": [l + "\n" for l in s.strip("\n").split("\n")]}

B = "artifacts/seaquest/oxygen_4frame"
SA = f"{B}/leakage/source_audit"
cells = []
cells.append(md(
    "# Seaquest Oxygen 4-Frame — Phase 2.1b: Leakage SOURCE Audit",
    "The masked four-frame oxygen probe was unexpectedly strong (R²≈0.901). Before any U→A /",
    "U→future / oracle work, find WHERE the residual oxygen signal comes from: matched visual",
    "probes **V1** (newest masked frame) · **V2** (4 masked frames) · **V3** (bottom-HUD masked) ·",
    "**V4** (gameplay-only crop), non-image proxies **P1–P3** (timestep / player-y), and sanity",
    "controls. Each step is one committed script. **Use:** TOKEN (Cell 1) + Drive `raw_hf.zip`."))

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

cells.append(code(rf"""
# 3. (optional) Restore the ORIGINAL Phase-2 outputs from Drive so the 0.901 leakage
#    artifacts can be preserved verbatim, then preserve them (step 1). Missing files are
#    documented, not silently skipped.
import os, glob, zipfile
PRIOR = '/content/drive/MyDrive/seaquest_oxygen_4frame_phase2.zip'   # <-- EDIT if elsewhere
if os.path.exists(PRIOR):
    os.makedirs('{B}', exist_ok=True)
    with zipfile.ZipFile(PRIOR) as z: z.extractall('{B}')
    print('restored original Phase-2 outputs from', PRIOR)
else:
    print('prior Phase-2 ZIP not found; original 2.1 files will be marked UNAVAILABLE')
!python -m seaquest_ccrl.scripts.oxy4_audit_preserve
"""))

cells.append(code('# 4. Step 2 — implementation audit (10 checks vs actual tensors+code). STOPS on a bug.\n'
                  '!python -m seaquest_ccrl.scripts.oxy4_audit_implementation --root "$DATA_ROOT"'))
cells.append(code('# 5. Step 3 — define + visualize raw-frame regions (5-panel figure + coordinates).\n'
                  '!python -m seaquest_ccrl.scripts.oxy4_audit_regions --root "$DATA_ROOT"\n'
                  'from IPython.display import Image, display\n'
                  f'display(Image("{SA}/figures/regions_5panel.png"))'))
cells.append(code('# 6. Steps 4-7 — matched visual probes V1-V4 + proxies P1-P3 + sanity controls.\n'
                  '!python -m seaquest_ccrl.scripts.oxy4_audit_probes --root "$DATA_ROOT"'))
cells.append(code('# 7. Steps 8-9 — one primary leakage-source outcome + the 9 decision answers.\n'
                  '!python -m seaquest_ccrl.scripts.oxy4_audit_report\n'
                  f"import json; print(json.dumps(json.load(open('{SA}/leakage_source_report.json')), indent=2))"))

cells.append(code(rf"""
# 8. Persist + download the source-audit outputs (predictions, metrics, figures, report).
import shutil, os
shutil.make_archive('seaquest_oxygen_4frame_phase21b', 'zip', '{B}/leakage/source_audit')
try:
    from google.colab import drive; drive.mount('/content/drive')
    dst='/content/drive/MyDrive'; shutil.copy('seaquest_oxygen_4frame_phase21b.zip', dst); print('copied to Drive')
except Exception as e: print('Drive optional:', e)
try:
    from google.colab import files; files.download('seaquest_oxygen_4frame_phase21b.zip')
except Exception: pass
"""))

cells.append(md(
    "## STOP — Phase 2.1b only",
    "Send `leakage_source_report.json` for review. Do **not** resume Phase 2.2 / 2.3, change the",
    "observation mask, or train the oracle until the leakage-source outcome is approved."))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, "w", encoding="utf-8"), indent=1)
print("wrote", os.path.normpath(OUT), "with", len(cells), "cells")
