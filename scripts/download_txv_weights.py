"""Download TorchXRayVision DenseNet121 weights locally for offline Kaggle use.

Triggers torchxrayvision's built-in download of ``densenet121-res224-all``
(~30 MB) into ``~/.torchxrayvision/models_data/`` and then copies whatever
landed there into a clean staging directory you can zip + upload to Kaggle as
a dataset called ``torchxrayvision-weights``.

The Component 2 Kaggle bootstrap (``scripts/kaggle_component2_finetune.py``)
already checks for ``/kaggle/input/datasets/iahmedhabib/torchxrayvision-weights``
and symlinks it into ``~/.torchxrayvision/models_data/`` if present, so no
notebook changes are needed once this dataset is attached.

Usage::

    python scripts/download_txv_weights.py
    # -> prints the staging path, then:
    #    1) zip -r txv_weights.zip txv_weights_stage/
    #    2) upload to Kaggle as a new dataset named `torchxrayvision-weights`
    #    3) attach it to the Component 2 notebook and re-run

Requires ``torchxrayvision`` installed locally (``pip install torchxrayvision``).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


STAGE_DIR = Path(__file__).resolve().parent.parent / "txv_weights_stage"
DEFAULT_CACHE = Path.home() / ".torchxrayvision" / "models_data"
WEIGHTS_TAG = "densenet121-res224-all"


def main() -> None:
    try:
        import torchxrayvision as xrv  # type: ignore
    except ModuleNotFoundError:
        print("torchxrayvision is not installed. Run: pip install torchxrayvision")
        sys.exit(1)

    print(f"Triggering download of {WEIGHTS_TAG} (~30 MB)...")
    xrv.models.DenseNet(weights=WEIGHTS_TAG)
    print(f"Download complete. Files cached to: {DEFAULT_CACHE}")

    if not DEFAULT_CACHE.exists():
        print(f"ERROR: expected cache dir {DEFAULT_CACHE} does not exist.")
        sys.exit(2)

    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    STAGE_DIR.mkdir(parents=True, exist_ok=False)

    copied = 0
    for src in DEFAULT_CACHE.iterdir():
        if src.is_file():
            shutil.copy2(src, STAGE_DIR / src.name)
            copied += 1
            print(f"  staged: {src.name}  ({src.stat().st_size // (1024*1024)} MB)")

    print()
    print(f"Staged {copied} file(s) to: {STAGE_DIR}")
    print()
    print("Next steps:")
    print(f"  1) cd {STAGE_DIR.parent}")
    print(f"  2) zip -r txv_weights.zip {STAGE_DIR.name}")
    print("  3) Upload txv_weights.zip to Kaggle as a new dataset")
    print("     -> kaggle.com/datasets → 'New Dataset' → drop the zip")
    print("     -> Title/slug: 'torchxrayvision-weights' (under your iahmedhabib account)")
    print("  4) Attach it to the Component 2 notebook; the bootstrap auto-detects it.")


if __name__ == "__main__":
    main()
