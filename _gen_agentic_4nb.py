"""Generate 4 per-mode agentic notebooks (a2/a3/fusion/a1) for parallel Kaggle runs.

Each notebook caches RAD-DINO features in TWO forms (when applicable):
  * CLS [N, 768]                 - for the ALP/Timika heads.
  * patch grid 7x7 [N, 49, 768]  - for the spatial cavity head (A1 uses grid for ALP too).

Then runs the full rung ladder (rungs 1-6 + agentic_best + agentic_best_tta +
rung4b_spatial_cavity + agentic_best_spatcav + stacked_legacy), 5 seeds, M=10
ensemble, and pickles trained heads for local figure generation.

Run:  python _gen_agentic_4nb.py
"""
import json
from pathlib import Path

REPO_URL = "https://github.com/mabdullahi7780/dl-project-codebase.git"
BRANCH = "cleaned-repo"
NB_DIR = Path(__file__).parent / "notebooks"

# (mode -> rungs)  rung4 dropped for a2/fusion (negative result); kept on a1 for ablation
# completeness. a3 has no cavity head so r4 is auto-skipped.
MODES = {
    "a2":     {"rungs": "1 2 3 5 6 7", "primary_grid": False},
    "a3":     {"rungs": "1 2 3 5 6 7", "primary_grid": False},
    "fusion": {"rungs": "1 2 3 5 6 7", "primary_grid": False},
    "a1":     {"rungs": "1 3 6 7",     "primary_grid": True},
}
SEEDS = "0 1 2 3 4"     # 5 seeds for the headline
ENSEMBLE_M = 10
PATCH_GRID = 7


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def build(mode, rungs, primary_grid):
    """One notebook for one mode."""
    if primary_grid:
        # A1: primary = grid; spatial cavity reuses the same grid (no aux cache needed).
        cache_primary_args = f'"--patch-grid", "{PATCH_GRID}", '
        features_primary_path = f'f"{{WORK}}/features_{{BACKBONE}}_grid{PATCH_GRID}.npz"'
        features_grid_block = "# (a1) primary cache IS the grid; no separate grid cache needed."
        cache_grid_block = "# (a1) skipped\nprint('a1 mode: grid is the primary cache, skipping aux grid cache.')"
        train_grid_arg = '""'  # no --features-grid for a1
    else:
        cache_primary_args = ""
        features_primary_path = 'f"{WORK}/features_{BACKBONE}_cls.npz"'
        features_grid_block = (
            f'FEATURES_GRID = f"{{WORK}}/features_{{BACKBONE}}_grid{PATCH_GRID}.npz"\n'
            f'print("aux grid features ->", FEATURES_GRID)'
        )
        cache_grid_block = (
            'if os.path.isfile(FEATURES_GRID):\n'
            '    print("grid features already cached ->", FEATURES_GRID)\n'
            'else:\n'
            '    from cache_features import main as cache_main\n'
            '    cache_main(["--manifest", PAPER_MANIFEST, "--out", FEATURES_GRID,\n'
            f'                "--backbone", BACKBONE, "--patch-grid", "{PATCH_GRID}", "--batch-size", "32"])'
        )
        train_grid_arg = 'FEATURES_GRID'

    cells = [
        md(f"# TB Portals — **Agentic 6-Rung Pipeline (headline run)**: mode `{mode}`\n"
           f"Caches RAD-DINO {'patch-grid' if primary_grid else 'CLS+patch-grid'} features, "
           f"runs rungs `{rungs}` + agentic_best variants + spatial-cavity probe, "
           f"**5 seeds × M={ENSEMBLE_M}**, saves heads. Download `agentic_{mode}.zip`.\n"
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
             'print("repo ready at", REPO_DIR)'),

        md("## Install deps"),
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
             'BACKBONE      = "rad-dino"\n'
             f'FEATURES      = {features_primary_path}\n'
             f'{features_grid_block}\n'
             'print("MODE:", MODE, "| KAGGLE_EXPORT:", os.path.isdir(KAGGLE_EXPORT), "| primary features ->", FEATURES)'),

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

        md("## 2 — Cache primary features (one-time; idempotent)"),
        code('import os, sys\n'
             'if REPO_DIR + "/scripts" not in sys.path: sys.path.insert(0, REPO_DIR + "/scripts")\n'
             'if os.path.isfile(FEATURES):\n'
             '    print("primary features already cached ->", FEATURES)\n'
             'else:\n'
             '    from cache_features import main as cache_main\n'
             '    cache_main(["--manifest", PAPER_MANIFEST, "--out", FEATURES,\n'
             f'                "--backbone", BACKBONE, {cache_primary_args}"--batch-size", "32"])'),

        md("## 2b — Cache patch-grid features (for the spatial cavity head)"),
        code(cache_grid_block),

        md(f"## 3 — Run the rung ladder for mode `{mode}` (rungs {rungs}, 5 seeds, M={ENSEMBLE_M})"),
        code('from src.training.train_agentic import main as agentic_main\n'
             'args = ["--features", FEATURES, "--manifest", PAPER_MANIFEST,\n'
             '        "--mode", MODE, "--out-dir", f"{WORK}/agentic_{MODE}",\n'
             f'        "--rungs", {", ".join(repr(r) for r in rungs.split())},\n'
             f'        "--seeds", {", ".join(repr(s) for s in SEEDS.split())},\n'
             f'        "--ensemble-m", "{ENSEMBLE_M}",\n'
             '        "--held-outs", "Romania", "Moldova", "Kazakhstan",\n'
             '        "--save-heads"]\n'
             + ('args += ["--features-grid", FEATURES_GRID]\n' if not primary_grid else '')
             + 'agentic_main(args)'),

        md("## 4 — Save (download → baseline_runs/agentic_runs/)"),
        code('import os, shutil\n'
             'src = f"{WORK}/agentic_{MODE}"\n'
             'try: shutil.copy(FEATURES, f"{src}/{os.path.basename(FEATURES)}")\n'
             'except Exception as e: print("primary cache copy skipped:", e)\n'
             + ('try: shutil.copy(FEATURES_GRID, f"{src}/{os.path.basename(FEATURES_GRID)}")\n'
                'except Exception as e: print("grid cache copy skipped:", e)\n' if not primary_grid else '')
             + 'zip_path = shutil.make_archive(f"{WORK}/agentic_{MODE}", "zip", src)\n'
             'print("Saved ->", zip_path)\n'
             'print("Contents include: results_agentic_*.csv, preds_*.csv, heads/*.pt, features_*.npz")'),
    ]
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


for mode, spec in MODES.items():
    nb = build(mode, spec["rungs"], spec["primary_grid"])
    out = NB_DIR / f"tbportals_agentic_{mode}.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)
