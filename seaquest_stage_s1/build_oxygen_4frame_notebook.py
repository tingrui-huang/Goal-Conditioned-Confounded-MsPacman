"""Build notebooks/Seaquest_Oxygen_4Frame_Study.ipynb — PHASE 1 ONLY, THIN.

Mirrors the simple HF-Expert notebook: every step is a single `!python -m ...` call into
COMMITTED, locally-tested repo scripts (audit_hf_4frame_stacks, run_hf_4frame,
eval_hf_action_use, gate_hf_4frame). No heavy inline logic -> minimal Colab debugging.
Only scientific change vs the single-frame run: frame_stack 1 -> 4. STOPS at the gate.
"""
import json, os
OUT = os.path.join(os.path.dirname(__file__), "..", "notebooks", "Seaquest_Oxygen_4Frame_Study.ipynb")


def md(*L): return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in L]}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": [l + "\n" for l in s.strip("\n").split("\n")]}

CKPT = "seaquest_ccrl/checkpoints/hf_4frame_seed0/critic_naive.pt"
CKDIR = "seaquest_ccrl/checkpoints/hf_4frame_seed0"
DIAG = "artifacts/seaquest/oxygen_4frame/naive_critic/action_use_diag.json"

cells = []
cells.append(md(
    "# Seaquest Oxygen 4-Frame Study — Phase 1 (four-frame naive critic + action-use gate)",
    "Train the **unchanged** committed `seaquest_ccrl` contrastive critic on `raw_hf`, the **only**",
    "change vs the successful single-frame run being **frame_stack 1 → 4**. Every step below is one",
    "committed script (hard-asserts nb_actions=18 / frame_stack=4 / first-conv in_channels=12; refuses",
    "MsPacman paths). **Use:** set TOKEN (Cell 1) + DATA_ROOT (Cell 2), then Run-all. Stops at the gate."))

cells.append(code("import torch\nprint('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"))

cells.append(code(r"""
# 1. Clone the repo at the reviewed commit (commit run_hf_4frame.py + the 3 Phase-1 scripts first).
TOKEN = 'PASTE_YOUR_GITHUB_TOKEN_HERE'
import os, subprocess
D = '/content/Goal-Conditioned-Confounded-MsPacman'
if not os.path.isdir(D):
    subprocess.run(['git','clone',f'https://{TOKEN}@github.com/tingrui-huang/Goal-Conditioned-Confounded-MsPacman.git',D], check=True)
%cd /content/Goal-Conditioned-Confounded-MsPacman
!git pull -q && git log --oneline -1
"""))

cells.append(code(r"""
# 2. Point at the raw_hf you uploaded (folder with traj_*.npz + manifest.json).
import glob
DATA_ROOT = '/content/raw_hf'      # <-- EDIT
n = len(glob.glob(DATA_ROOT + '/traj_*.npz'))
print('DATA_ROOT =', DATA_ROOT, '| trajectories:', n)
assert n == 40, f'expected 40 traj, found {n}'
"""))

cells.append(code('# 3. Four-frame stack assertions + masking check + visual grid (committed script).\n'
                  '!python -m seaquest_ccrl.scripts.audit_hf_4frame_stacks --root "$DATA_ROOT"'))

cells.append(code('# 4. Train the four-frame NAIVE critic (committed runner; identical to single-frame except frame_stack=4).\n'
                  f'!python -m seaquest_ccrl.scripts.run_hf_4frame --root "$DATA_ROOT" --steps 50000 --seed 0 --ckpt-dir {CKDIR}'))

cells.append(code('# 5. Action-use diagnostic (committed; evaluation only).\n'
                  f'!python -m seaquest_ccrl.scripts.eval_hf_action_use --ckpt {CKPT} --root "$DATA_ROOT" --out {DIAG}'))

cells.append(code('# 6. Action gate + persist everything (checkpoint, resolved config, history, diag, stack audit, hashes -> ZIP).\n'
                  f'!python -m seaquest_ccrl.scripts.gate_hf_4frame --ckpt {CKPT} --ckpt-dir {CKDIR} --diag {DIAG}\n'
                  "import json; print(json.dumps(json.load(open('artifacts/seaquest/oxygen_4frame/naive_critic/action_use_gate.json')), indent=2))"))

cells.append(code(r"""
# 7. (optional) save to Drive + download the Phase-1 ZIP.
import shutil, os
try:
    from google.colab import drive; drive.mount('/content/drive')
    dst='/content/drive/MyDrive/seaquest_oxygen_4frame_phase1'; os.makedirs(dst, exist_ok=True)
    for p in __import__('glob').glob('artifacts/seaquest/oxygen_4frame/naive_critic/*'): shutil.copy(p, dst)
    print('copied to Drive:', dst)
except Exception as e: print('Drive optional:', e)
try:
    from google.colab import files; files.download('seaquest_oxygen_4frame_phase1.zip')
except Exception: pass
"""))

cells.append(md(
    "## STOP — Phase 1 only",
    "Oxygen probes (Phase 2), oracle (Phase 3), and the paired Docker eval (Phases 4–5) are **not** here.",
    "Send `action_use_gate.json` for review. If `STOP_4FRAME_CRITIC_DOES_NOT_USE_ACTION`, the oxygen study stops."))

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}, "accelerator": "GPU"}, "nbformat": 4, "nbformat_minor": 5}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(nb, open(OUT, "w", encoding="utf-8"), indent=1)
print("wrote", os.path.normpath(OUT), "with", len(cells), "cells")
