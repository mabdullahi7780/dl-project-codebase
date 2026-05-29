"""One-shot fixer for the new Kaggle notebooks' broken imports and API usage.

Bugs the original notebooks had:
- import path: src.data.tbportals_dataset (correct: src.data.tbportals)
- function kwargs: held_out= / val_frac= (correct: held_out_country= / val_fraction=)
- return value: treated as dict[ids] (correct: tuple of DataFrames)
- _set_seed import that does not exist (just remove)
"""
from __future__ import annotations
import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

REPLACEMENTS = [
    # 1. Import path -- the dataset module does NOT export make_country_split.
    ("from src.data.tbportals_dataset import _load_image, make_country_split",
     "from src.data.tbportals_dataset import _load_image\n"
     "from src.data.tbportals import make_country_split"),
    ("from src.data.tbportals_dataset import make_country_split",
     "from src.data.tbportals import make_country_split"),
    # 2. _set_seed does not exist in train_agentic. Drop the line.
    ("from src.training.train_agentic import _set_seed\n",
     ""),
    ("from src.training.train_agentic import _set_seed",
     ""),
    # 3. Function-signature kwarg renames.
    ("held_out=country",       "held_out_country=country"),
    ("held_out=country,",      "held_out_country=country,"),
    ("val_frac=0.2",           "val_fraction=0.2"),
]

# Specific (notebook-by-notebook) tuple-unpack rewrites: the previous code
# treated the return as a dict-of-id-lists; rewrite to unpack the tuple of
# DataFrames that make_country_split actually returns.
TUPLE_REWRITES = {
    "tbportals_backbones_comparison.ipynb": [
        ("    split = make_country_split(manifest, held_out_country=country, val_fraction=0.2, seed=0)\n"
         "    tr_df = manifest[manifest['image_id'].isin(split['train_ids'])].copy()\n"
         "    te_df = manifest[manifest['image_id'].isin(split['test_ids'])].copy()\n",
         "    tr_df, _, te_df = make_country_split(manifest, held_out_country=country, val_fraction=0.2, seed=0)\n"
         "    tr_df = tr_df.copy(); te_df = te_df.copy()\n"),
    ],
    "tbportals_cavity_attention.ipynb": [
        ("    split = make_country_split(manifest, held_out_country=country, val_fraction=0.2, seed=0)\n"
         "    test_ids = [i for i in split['test_ids'] if i in grid_feats]\n",
         "    _, _, test_df = make_country_split(manifest, held_out_country=country, val_fraction=0.2, seed=0)\n"
         "    test_ids = [i for i in test_df['image_id'].astype(str).values if i in grid_feats]\n"),
    ],
    "tbportals_alp_spatial_maps.ipynb": [
        ("    split = make_country_split(manifest, held_out_country=country, val_fraction=0.2, seed=0)\n"
         "    test_ids = [i for i in split['test_ids'] if i in grid_feats]\n",
         "    _, _, test_df = make_country_split(manifest, held_out_country=country, val_fraction=0.2, seed=0)\n"
         "    test_ids = [i for i in test_df['image_id'].astype(str).values if i in grid_feats]\n"),
    ],
    "tbportals_published_cavity_baselines.ipynb": [
        ("        split = make_country_split(paper_df, held_out_country=country, val_fraction=0.2, seed=seed)\n"
         "        val_df = paper_df[paper_df['image_id'].isin(split['val_ids'])].dropna(subset=['txv_lesion_prob','chexzero_cavity_prob'])\n"
         "        test_df= paper_df[paper_df['image_id'].isin(split['test_ids'])].dropna(subset=['txv_lesion_prob','chexzero_cavity_prob']).copy()\n",
         "        _, val_df, test_df = make_country_split(paper_df, held_out_country=country, val_fraction=0.2, seed=seed)\n"
         "        val_df = val_df.dropna(subset=['txv_lesion_prob','chexzero_cavity_prob'])\n"
         "        test_df= test_df.dropna(subset=['txv_lesion_prob','chexzero_cavity_prob']).copy()\n"),
    ],
    "tbportals_shift_robust_baselines.ipynb": [
        ("        split = make_country_split(paper_df, held_out_country=country, val_fraction=0.2, seed=seed)\n"
         "        tr_df = paper_df[paper_df['image_id'].isin(split['train_ids'])]\n"
         "        te_df = paper_df[paper_df['image_id'].isin(split['test_ids'])]\n",
         "        tr_df, _, te_df = make_country_split(paper_df, held_out_country=country, val_fraction=0.2, seed=seed)\n"),
    ],
}


def fix_notebook(path: Path) -> int:
    nb = json.loads(path.read_text(encoding="utf-8"))
    changed = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        new = src
        for old, repl in REPLACEMENTS:
            new = new.replace(old, repl)
        for old, repl in TUPLE_REWRITES.get(path.name, []):
            new = new.replace(old, repl)
        if new != src:
            cell["source"] = [line + "\n" for line in new.split("\n")[:-1]] + (
                [new.split("\n")[-1]] if new.split("\n")[-1] else []
            )
            # cleaner: just keep lines as written
            cell["source"] = new.splitlines(keepends=True)
            changed += 1
    if changed:
        path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return changed


def main():
    targets = [
        "tbportals_backbones_comparison.ipynb",
        "tbportals_cavity_attention.ipynb",
        "tbportals_alp_spatial_maps.ipynb",
        "tbportals_published_cavity_baselines.ipynb",
        "tbportals_shift_robust_baselines.ipynb",
    ]
    for name in targets:
        path = NB_DIR / name
        if not path.exists():
            print(f"MISSING: {path}")
            continue
        n = fix_notebook(path)
        print(f"{name}: fixed {n} cell(s)")


if __name__ == "__main__":
    main()
