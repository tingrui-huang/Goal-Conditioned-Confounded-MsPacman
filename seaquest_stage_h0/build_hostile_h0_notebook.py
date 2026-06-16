"""Generate the THIN Colab notebook for Stage-H0 qualification.

Every notebook cell calls a committed, locally tested module. No scientific logic is
inlined in the notebook. Also provides `synthetic_smoke()` — a non-scientific engineering
check of the qualification wiring on MOCK metrics (it never writes the real qualification
report and never emits a qualification claim).
"""
import os, json, argparse


def _code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


def _md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def build_notebook(path):
    cells = [
        _md("# Seaquest Stage-H0 — Hostile-Field Qualification (Colab, PyTorch only)\n"
            "\n"
            "**Do NOT install/run ALE, ROMs, OCAtari, EnvPool, JAX, Flax, or the HF "
            "teacher here.** This notebook only loads the FROZEN `raw_hf` + hostile "
            "metadata pack (produced locally in Docker), verifies every raw SHA256, "
            "builds the hostile-removed four-frame states, runs the three probes and "
            "writes the qualification report. It is THIN: every cell calls a committed "
            "module under `seaquest_ccrl/hostile` / `seaquest_ccrl/scripts`."),
        _code("# 0. environment (torch only) + repo on path\n"
              "#    The Stage-H0 code lives on the 'seaquest-stage-h0' BRANCH (not main).\n"
              "import sys, os, subprocess\n"
              "REPO = os.environ.get('H0_REPO', '/content/Goal-Conditioned-Confounded-MsPacman')\n"
              "BRANCH = os.environ.get('H0_BRANCH', 'seaquest-stage-h0')\n"
              "REPO_URL = os.environ.get('H0_REPO_URL',\n"
              "    'https://github.com/tingrui-huang/Goal-Conditioned-Confounded-MsPacman.git')\n"
              "if not os.path.isdir(os.path.join(REPO, 'seaquest_stage_h0')):\n"
              "    if not os.path.isdir(os.path.join(REPO, '.git')):\n"
              "        subprocess.run(['git', 'clone', '-b', BRANCH, REPO_URL, REPO], check=True)\n"
              "    else:  # repo cloned on the wrong branch (e.g. main) -> switch\n"
              "        subprocess.run(['git', '-C', REPO, 'fetch', 'origin', BRANCH], check=True)\n"
              "        subprocess.run(['git', '-C', REPO, 'checkout', BRANCH], check=True)\n"
              "sys.path.insert(0, REPO)\n"
              "assert os.path.isdir(os.path.join(REPO, 'seaquest_stage_h0')), (\n"
              "    f'seaquest_stage_h0 not found under {REPO!r}; clone branch {BRANCH!r}')\n"
              "RAW = os.environ.get('H0_RAW', f'{REPO}/seaquest_ccrl/data/raw_hf')\n"
              "META = os.environ.get('H0_META', f'{REPO}/seaquest_ccrl/data/hostile_h0_metadata')\n"
              "OUT = os.environ.get('H0_OUT', f'{REPO}/artifacts/seaquest/hostile_h0')\n"
              "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\n"
              "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
              "print('repo OK:', REPO, '| branch:', BRANCH)\n"
              "# NOTE: raw_hf frames + hostile_h0_metadata are DATA (untracked, not in the repo).\n"
              "# Upload them to Drive and point H0_RAW / H0_META at those paths."),
        _code("# 1. verify every raw SHA256 + pack self-consistency (no teacher/ALE)\n"
              "from seaquest_stage_h0.validate_recollection_parity import validate\n"
              "pv = validate(RAW, META, os.path.join(OUT, 'recollection_parity.json'))\n"
              "assert pv['ok'], 'HOSTILE_RECOLLECTION pack invalid — see recollection_parity.json'\n"
              "pv['totals']"),
        _code("# 2. build hostile-removed four-frame states (verifies removal invariants per frame)\n"
              "#    uses the RESOLVED palette (observed RGB) from the local object-identity audit\n"
              "from seaquest_ccrl.hostile.data import HostileH0Data\n"
              "from seaquest_ccrl.hostile import removal as RM\n"
              "import os, json\n"
              "pal_path = os.path.join(OUT, 'schema', 'resolved_palette.json')\n"
              "palettes = RM.load_resolved_palettes(pal_path) if os.path.exists(pal_path) else None\n"
              "tol = json.load(open(pal_path))['resolved_tol'] if os.path.exists(pal_path) else RM.DEFAULT_TOL\n"
              "data = HostileH0Data(RAW, META, device=DEVICE, load_visible=True, verify_sha=True,\n"
              "                     palettes=palettes, tol=tol)\n"
              "print('N', data.N, 'episodes', data.n_ep, 'palette', 'resolved' if palettes else 'RAM-default')"),
        _code("# 3. support / data density (Section 12)\n"
              "import json\n"
              "sup = data.support_summary()\n"
              "json.dump(sup, open(os.path.join(OUT, 'support.json'), 'w'), indent=2)\n"
              "sup['enemy']['pass'], sup['missile']['pass'], sup['joint']['pass']"),
        _code("# 4. Probe A — hiddenness (P0 / PV / PM)\n"
              "from seaquest_ccrl.scripts import h0_probe_hiddenness as PA\n"
              "hidden = PA.run(data, os.path.join(OUT, 'hiddenness'), device=DEVICE)\n"
              "{c: hidden[c]['recovery_ci'] for c in hidden}"),
        _code("# 5. Probe B — conditional U -> action\n"
              "from seaquest_ccrl.scripts import h0_probe_action as PB\n"
              "action = PB.run(data, os.path.join(OUT, 'action'), device=DEVICE)\n"
              "{c: (action[c]['improvement_mean'], action[c]['improvement_ci']) for c in action}"),
        _code("# 6. Probe C — conditional U -> future\n"
              "from seaquest_ccrl.scripts import h0_probe_future as PC\n"
              "future = PC.run(data, os.path.join(OUT, 'future'), device=DEVICE)\n"
              "{c: list(future[c]['per_horizon'].keys()) for c in future}"),
        _code("# 7. assemble component results -> qualification report\n"
              "from seaquest_ccrl.scripts import h0_qualify as Q\n"
              "def hidden_in(c):\n"
              "    h = hidden.get(c);\n"
              "    return None if h is None else {'recovery_ci': h['recovery_ci'],\n"
              "        'visible_better_than_prior': h['visible_better_than_prior'],\n"
              "        'masked_nearest_r2': h['masked_nearest_r2'], 'adequate_support': h['adequate_support']}\n"
              "results = {\n"
              "  'recollection': {'identical': True},  # asserted at Docker collection time\n"
              "  'object_schema': json.load(open(os.path.join(OUT,'schema','object_identity_audit.json'))),\n"
              "  'removal': json.load(open(os.path.join(OUT,'removal','removal_audit.json'))),\n"
              "  'support': {c: sup[c] for c in ('enemy','missile','joint')},\n"
              "  'hiddenness': {c: hidden_in(c) for c in ('enemy','missile')},\n"
              "  'action': {c: action.get(c) for c in ('enemy','missile','joint')},\n"
              "  'future': {c: future.get(c) for c in ('enemy','missile','joint')},\n"
              "}\n"
              "# joint hiddenness defaults to enemy (joint removal == enemy+missile removed)\n"
              "results['hiddenness']['joint'] = hidden_in('enemy')\n"
              "rep = Q.decide(results)\n"
              "Q.write_report(rep, os.path.join(OUT,'hostile_qualification.json'), os.path.join(OUT,'hostile_qualification.md'))\n"
              "print('FINAL OUTCOME:', rep['final_outcome'], '(', rep['failure_kind'], ')')\n"
              "rep['components']"),
        _md("## Pack + download all Stage-H0 results\n"
            "Zips everything under `OUT` (qualification report, per-probe raw "
            "predictions/losses, support, and the copied audit JSONs) and triggers a "
            "browser download in Colab."),
        _code("# 8. pack + download all results\n"
              "import shutil, zipfile, os\n"
              "stamp = os.environ.get('H0_STAMP', 'results')\n"
              "pack_path = shutil.make_archive(f'/content/seaquest_hostile_h0_{stamp}', 'zip', OUT)\n"
              "size_mb = round(os.path.getsize(pack_path) / 1e6, 2)\n"
              "with zipfile.ZipFile(pack_path) as z:\n"
              "    names = z.namelist()\n"
              "print(f'packed {pack_path}  ({size_mb} MB, {len(names)} files)')\n"
              "for n in sorted(names):\n"
              "    if n.endswith(('.json', '.md', '.png', '.csv')):\n"
              "        print('  ', n)\n"
              "try:\n"
              "    from google.colab import files\n"
              "    files.download(pack_path)\n"
              "except Exception as e:\n"
              "    print('(not in Colab / download unavailable):', e, '-> grab', pack_path)"),
        _md("## Smoke (engineering only — NOT a scientific result)\n"
            "Runs the qualification wiring on MOCK metrics; writes to a `smoke/` dir, "
            "never the real report."),
        _code("from seaquest_stage_h0.build_hostile_h0_notebook import synthetic_smoke\n"
              "synthetic_smoke(os.path.join(OUT, 'smoke'))"),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(nb, open(path, "w"), indent=1)
    print(f"[notebook] wrote {path} ({len(cells)} cells)")
    return path


def synthetic_smoke(out_dir):
    """Mock-metric end-to-end check of qualification logic. Never a scientific claim."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from seaquest_ccrl.scripts import h0_qualify as Q
    os.makedirs(out_dir, exist_ok=True)
    mock = {
        "recollection": {"identical": True},
        "object_schema": {"pass": True}, "removal": {"pass": True},
        "support": {"enemy": {"pass": True}, "missile": {"pass": True}, "joint": {"pass": True}},
        "hiddenness": {
            "enemy": {"recovery_ci": [0.10, 0.42], "visible_better_than_prior": True,
                      "masked_nearest_r2": 0.20, "adequate_support": True},
            "missile": {"recovery_ci": [0.05, 0.30], "visible_better_than_prior": True,
                        "masked_nearest_r2": 0.10, "adequate_support": True},
            "joint": {"recovery_ci": [0.10, 0.42], "visible_better_than_prior": True,
                      "masked_nearest_r2": 0.20, "adequate_support": True}},
        "action": {c: {"improvement_mean": 0.02, "improvement_ci": [0.005, 0.04],
                       "shuffled_mean": 0.0005, "shuffled_ci": [-0.001, 0.002], "one_episode": False}
                   for c in ("enemy", "missile", "joint")},
        "future": {c: {"per_horizon": {
            "4": {"target": "displacement_y", "mse_red_mean": 1.2, "mse_red_ci": [0.3, 2.0]},
            "32": {"target": "future_player_y", "mse_red_mean": 0.8, "mse_red_ci": [0.1, 1.5]}},
            "shuffled_reproduces": False} for c in ("enemy", "missile", "joint")},
    }
    rep = Q.decide(mock)
    Q.write_report(rep, os.path.join(out_dir, "SMOKE_qualification.json"),
                   os.path.join(out_dir, "SMOKE_qualification.md"))
    print(f"[SMOKE — NOT A RESULT] qualification wiring produced: {rep['final_outcome']}")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="notebooks/Seaquest_Hostile_H0_Qualification.ipynb")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        synthetic_smoke("artifacts/seaquest/hostile_h0/smoke")
    else:
        build_notebook(args.out)


if __name__ == "__main__":
    main()
