"""Generate 4 per-mode agentic notebooks (a2/a3/fusion/a1) for parallel Kaggle runs.

Each notebook: clone -> install -> paths -> manifest -> cache frozen features ->
run the full rung ladder for its mode -> zip results + per-image preds + feature cache.
Run:  python _gen_agentic_4nb.py
"""
import json
from pathlib import Path

REPO_URL = "https://github.com/mabdullahi7780/dl-project-codebase.git"
BRANCH = "cleaned-repo"
NB_DIR = Path(__file__).parent / "notebooks"

MODES = {
    "a2":     {"rungs": "1 2 3 4 5 6", "grid": 0},
    "a3":     {"rungs": "1 2 3 5 6",   "grid": 0},
    "fusion": {"rungs": "1 2 3 4 5 6", "grid": 0},
    "a1":     {"rungs": "1 3 4 6",     "grid": 7},
}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def build(mode, rungs, grid):
    feat_tag = f"grid{grid}" if grid else "cls"
    cache_args = f'"--patch-grid", "{grid}", ' if grid else ""
    cells = [
        md(f"# TB Portals — **Agentic 6-Rung Pipeline**: mode `{mode}`\n"
           f"Frozen RAD-DINO features + rungs `{rungs}`. Run end-to-end; download `agentic_{mode}.zip`.\n"
           "Attach only **tb-portals-cxr-pngs**; Internet **ON**; GPU T4."),

        md("## 0 — Clone  *(restart kernel after any pull that changed .py)*"),
        code('import os, sys, subprocess\n'
             f'REPO_URL = "{REPO_URL}"\n'
             'REPO_DIR = "/kaggle/working/dl-project-codebase"\n'
             f'BRANCH   = "{BRANCH}"\n'
             'if os.path.isdir(REPO_DIR):\n'
             '    subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=True)\n'
             'else:\n'
             '    subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO_DIR], check=True)\n'
             'for _p in (REPO_DIR, REPO_DIR + "/scripts"):\n'
             '    if _p not in sys.path: sys.path.insert(0, _p)\n'
             'print("repo ready at", REPO_DIR)\n'
             '# After a git pull that changed .py modules, RESTART the kernel so Python reloads them.'),

        md("## Install deps (transformers for RAD-DINO; torchxrayvision = fallback backbone)"),
        code('import sys, subprocess\n'
             'subprocess.run([sys.executable, "-m", "pip", "install", "-q",\n'
             '                "transformers", "torchxrayvision", "pydicom", "pylibjpeg", "pylibjpeg-libjpeg"], check=False)\n'
             'print("deps installed")'),

        md("## Paths"),
        code('import os\n'
             'WORK          = "/kaggle/working"\n'
             'REPO_DIR      = "/kaggle/working/dl-project-codebase"\n'
             'DATASET       = "/kaggle/input/datasets/mabdullahi454/tb-portals-cxr-pngs"\n'
             'KAGGLE_EXPORT = f"{DATASET}/kaggle_export"\n'
             'PAPER_MANIFEST = f"{WORK}/tbportals_manifest_paper.csv"\n'
             f'MODE          = "{mode}"\n'
             'BACKBONE      = "rad-dino"      # rad-dino (primary) | txrv (fallback) | densenet (control)\n'
             f'PATCH_GRID    = {grid}          # 0 = CLS vector; >0 = GxG patch grid (a1 spatial head)\n'
             f'FEATURES      = f"{{WORK}}/features_{{BACKBONE}}_{feat_tag}.npz"\n'
             'print("MODE:", MODE, "| KAGGLE_EXPORT:", os.path.isdir(KAGGLE_EXPORT), "| features ->", FEATURES)'),

        md("## 1 — Build the 5,010-image manifest (Kantipudi Table 1)"),
        code('import sys, pandas as pd\n'
             'from pathlib import Path\n'
             'if REPO_DIR + "/scripts" not in sys.path: sys.path.insert(0, REPO_DIR + "/scripts")\n'
             'from build_paper_manifest import subsample, PAPER_TOTAL\n'
             'raw = pd.read_csv(f"{KAGGLE_EXPORT}/manifest.csv",\n'
             '                  dtype={"image_id": str, "patient_id": str, "country": str})\n'
             'raw["image_path"] = raw["image_path"].apply(\n'
             '    lambda p: p if str(p).startswith("/") else f"{KAGGLE_EXPORT}/{p}")\n'
             'paper_df = subsample(raw, seed=42)\n'
             'paper_df["image_id"] = paper_df["image_path"].apply(lambda p: Path(str(p)).stem)\n'
             'paper_df.to_csv(PAPER_MANIFEST, index=False)\n'
             'print(f"Paper manifest: {len(paper_df)} images (target {PAPER_TOTAL}) -> {PAPER_MANIFEST}")'),

        md("## 2 — Cache frozen features (~10–20 min, one-time; idempotent)"),
        code('import os, sys\n'
             'if REPO_DIR + "/scripts" not in sys.path: sys.path.insert(0, REPO_DIR + "/scripts")\n'
             'if os.path.isfile(FEATURES):\n'
             '    print("features already cached ->", FEATURES)\n'
             'else:\n'
             '    from cache_features import main as cache_main\n'
             '    cache_main(["--manifest", PAPER_MANIFEST, "--out", FEATURES,\n'
             f'                "--backbone", BACKBONE, {cache_args}"--batch-size", "32"])'),

        md(f"## 3 — Run the rung ladder for mode `{mode}` (rungs {rungs}, 3 seeds)"),
        code('from src.training.train_agentic import main as agentic_main\n'
             'agentic_main(["--features", FEATURES, "--manifest", PAPER_MANIFEST,\n'
             '              "--mode", MODE, "--out-dir", f"{WORK}/agentic_{MODE}",\n'
             f'              "--rungs", {", ".join(repr(r) for r in rungs.split())},\n'
             '              "--held-outs", "Romania", "Moldova", "Kazakhstan",\n'
             '              "--seeds", "0", "1", "2"])'),

        md("## 4 — Save (download → baseline_runs/agentic/)"),
        code('import os, shutil\n'
             'src = f"{WORK}/agentic_{MODE}"\n'
             '# keep the feature cache alongside results so it can be re-attached to skip caching\n'
             'try: shutil.copy(FEATURES, f"{src}/{os.path.basename(FEATURES)}")\n'
             'except Exception as e: print("cache copy skipped:", e)\n'
             'zip_path = shutil.make_archive(f"{WORK}/agentic_{MODE}", "zip", src)\n'
             'print("Saved ->", zip_path)\n'
             'print("Download it; drop results_agentic_*.csv + preds_*.csv into baseline_runs/agentic/.")'),
    ]
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


for mode, spec in MODES.items():
    nb = build(mode, spec["rungs"], spec["grid"])
    out = NB_DIR / f"tbportals_agentic_{mode}.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)
