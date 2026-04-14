"""Kaggle-ready Component 1 (DANN) fine-tune bootstrap.

Intended use: paste the cell at the bottom of this file into a Kaggle
notebook. The script auto-detects the Kaggle dataset mounts, wires up
dataset / checkpoint paths via env vars, writes an override YAML with
``epochs: 8``, and invokes the existing trainer in-process.

Expected Kaggle inputs (attach these to the notebook):

    /kaggle/input/tb-cxr-baseline-codes/   # this repo
    /kaggle/input/montgomery/              # Montgomery CXR + ManualMask
    /kaggle/input/shehzhenn/               # Shenzhen CXR + mask
    /kaggle/input/tbx-11/ or tbx11/        # TBX11K (imgs + optional lists)
    /kaggle/input/sam_vit_h_4b8939/        # SAM ViT-H checkpoint (.pth)

NIH CXR14 is not expected on Kaggle. The trainer skips it automatically
when the root path does not exist (sampling weight is still honoured for
the three remaining domains).

Run from a Kaggle cell:

    !pip -q install segment_anything pyyaml safetensors tqdm
    !cp -r /kaggle/input/tb-cxr-baseline-codes /kaggle/working/repo
    !python /kaggle/working/repo/scripts/kaggle_component1_finetune.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import yaml


EPOCHS = 8
BATCH_SIZE = 2          # SAM ViT-H @ 1024 is heavy; start at 2 on T4, drop to 1 if OOM.
NUM_WORKERS = 2
LIMIT_PER_DOMAIN = None  # set to e.g. 200 for a fast smoke run

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def _first_existing(*candidates: Path) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_dataset_root(*slug_candidates: str) -> Path | None:
    if not KAGGLE_INPUT.exists():
        return None
    for slug in slug_candidates:
        path = KAGGLE_INPUT / slug
        if path.exists():
            return path
    # Fall back to a loose glob if the slug was renamed.
    for slug in slug_candidates:
        matches = sorted(KAGGLE_INPUT.glob(f"{slug}*"))
        if matches:
            return matches[0]
    return None


def _find_sam_checkpoint(root: Path | None) -> Path | None:
    if root is None:
        return None
    for pattern in ("*.pth", "*.pt", "**/*.pth", "**/*.pt"):
        hits = sorted(root.glob(pattern))
        if hits:
            return hits[0]
    return None


def _find_repo_root() -> Path:
    """Locate the tb-cxr-baseline-code repo root (contains ``src/``)."""

    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "src" / "training" / "train_component1_dann.py").is_file():
            return parent

    for candidate in (
        KAGGLE_WORKING / "repo",
        KAGGLE_WORKING / "tb-cxr-baseline-codes",
        KAGGLE_WORKING / "dl-project-codebase",
        KAGGLE_INPUT / "tb-cxr-baseline-codes",
    ):
        if (candidate / "src" / "training" / "train_component1_dann.py").is_file():
            return candidate

    raise FileNotFoundError(
        "Could not locate the tb-cxr-baseline-code repo. Copy it to /kaggle/working/repo first."
    )


def _detect_tbx_images_root(tbx_root: Path) -> Path:
    """TBX11K tar layouts vary; return the directory that actually holds images."""

    for sub in ("imgs", "TBX11K/imgs", "images", "tbx11k/imgs"):
        candidate = tbx_root / sub
        if candidate.exists():
            return candidate
    return tbx_root


def _detect_tbx_list(tbx_root: Path) -> str | None:
    for sub in ("lists", "TBX11K/lists"):
        lists_dir = tbx_root / sub
        if not lists_dir.exists():
            continue
        for name in ("all_trainval.txt", "trainval.txt", "train.txt"):
            if (lists_dir / name).is_file():
                return name
    return None


def _write_override_config(
    repo_root: Path,
    *,
    sam_ckpt: Path | None,
    save_dir: Path,
) -> Path:
    src_config = repo_root / "configs" / "component1_dann.yaml"
    with src_config.open("r", encoding="utf-8") as handle:
        cfg_root = yaml.safe_load(handle)

    cfg = cfg_root["component1_dann"]

    if sam_ckpt is None or not sam_ckpt.is_file():
        raise FileNotFoundError(
            "SAM ViT-H checkpoint not found. Attach the `sam_vit_h_4b8939` Kaggle "
            "dataset (must contain sam_vit_h_4b8939.pth) — Component 1 fine-tuning "
            "requires the real backbone, the mock encoder is not acceptable here."
        )
    cfg["encoder"]["backend"] = "segment_anything"
    cfg["encoder"]["checkpoint_path"] = str(sam_ckpt)

    training = cfg["training"]
    training["epochs"] = EPOCHS
    training["batch_size"] = BATCH_SIZE
    training["num_workers"] = NUM_WORKERS
    training["grl_ramp_epochs"] = max(1, EPOCHS // 2)
    training["limit_per_domain"] = LIMIT_PER_DOMAIN
    training["save_dir"] = str(save_dir)
    training["adapter_save_name"] = "component1_adapters.safetensors"
    training["save_full_checkpoint"] = True
    training["save_name"] = "component1_dann_full.pt"
    training["device"] = None

    # NIH archives are not on Kaggle; drop the split reference so the trainer
    # takes the graceful skip path when the root is absent.
    cfg["data"]["nih_split"] = None
    cfg["data"]["manifest_cache"] = str(save_dir / "nih_index_cache.json")
    cfg["data"]["domain_sampling_weights"] = {
        "montgomery": 100.0,
        "shenzhen": 20.0,
        "tbx11k": 2.0,
        "nih_cxr14": 1.0,
    }

    override = save_dir / "component1_dann.kaggle.yaml"
    override.parent.mkdir(parents=True, exist_ok=True)
    with override.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg_root, handle, sort_keys=False)
    return override


def _write_paths_override(
    repo_root: Path,
    *,
    montgomery: Path | None,
    shenzhen: Path | None,
    tbx11k: Path | None,
    nih: Path | None,
    save_dir: Path,
) -> Path:
    paths_cfg = {
        "project_root": str(repo_root),
        "external_data_root": str(KAGGLE_INPUT if KAGGLE_INPUT.exists() else repo_root),
        "datasets": {
            "montgomery": str(montgomery) if montgomery else str(save_dir / "_missing_montgomery"),
            "shenzhen": str(shenzhen) if shenzhen else str(save_dir / "_missing_shenzhen"),
            "tbx11k": str(tbx11k) if tbx11k else str(save_dir / "_missing_tbx11k"),
            "nih_cxr14": str(nih) if nih else str(save_dir / "_missing_nih"),
        },
        "artifacts": {
            "notebook_cache": str(save_dir / "notebook_cache"),
            "processed": str(save_dir / "processed"),
            "reports": str(save_dir / "reports"),
        },
    }
    override = save_dir / "paths.kaggle.yaml"
    with override.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(paths_cfg, handle, sort_keys=False)
    return override


def main() -> None:
    repo_root = _find_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # --- Resolve Kaggle dataset mounts ------------------------------------
    montgomery = _find_dataset_root("montgomery")
    shenzhen = _find_dataset_root("shehzhenn", "shenzhen", "shehzhen")
    tbx_root = _find_dataset_root("tbx-11", "tbx11", "tbx-11k", "tbx11k")
    sam_root = _find_dataset_root("sam_vit_h_4b8939", "sam-vit-h", "sam_vit_h")

    tbx_images = _detect_tbx_images_root(tbx_root) if tbx_root is not None else None
    tbx_list_name = _detect_tbx_list(tbx_root) if tbx_root is not None else None
    sam_ckpt = _find_sam_checkpoint(sam_root)

    print("Resolved Kaggle mounts:")
    print(f"  montgomery : {montgomery}")
    print(f"  shenzhen   : {shenzhen}")
    print(f"  tbx11k     : {tbx_images} (list={tbx_list_name})")
    print(f"  sam_vit_h  : {sam_ckpt}")

    save_dir = KAGGLE_WORKING / "component1_runs" if KAGGLE_WORKING.exists() else repo_root / "outputs" / "component1_runs"
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Write the two override YAMLs -------------------------------------
    override_cfg = _write_override_config(repo_root, sam_ckpt=sam_ckpt, save_dir=save_dir)
    paths_cfg = _write_paths_override(
        repo_root,
        montgomery=montgomery,
        shenzhen=shenzhen,
        tbx11k=tbx_images,
        nih=None,
        save_dir=save_dir,
    )

    # The trainer's TBX11K loader reads ``data.tbx_list``; patch it if we
    # found a different list name.
    if tbx_list_name is not None:
        with override_cfg.open("r", encoding="utf-8") as handle:
            cfg_root = yaml.safe_load(handle)
        cfg_root["component1_dann"]["data"]["tbx_list"] = tbx_list_name
        with override_cfg.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg_root, handle, sort_keys=False)

    # --- Invoke the existing trainer --------------------------------------
    os.chdir(repo_root)
    sys.argv = [
        "train_component1_dann.py",
        "--config", str(override_cfg),
        "--paths", str(paths_cfg),
    ]
    print(f"\nLaunching Component 1 DANN fine-tune for {EPOCHS} epochs...")
    print(f"  config : {override_cfg}")
    print(f"  paths  : {paths_cfg}")
    print(f"  cwd    : {Path.cwd()}\n")

    from src.training.train_component1_dann import main as train_main  # noqa: E402

    train_main()

    adapters = save_dir / "component1_adapters.safetensors"
    if adapters.exists():
        print(f"\nAdapters saved to: {adapters}")
    else:
        print("\nTraining finished but adapter file was not found; check training output above.")

    # Persist run artefacts somewhere easy to download from the Kaggle UI.
    if KAGGLE_WORKING.exists():
        mirror = KAGGLE_WORKING / "component1_artifacts"
        mirror.mkdir(parents=True, exist_ok=True)
        for name in ("component1_adapters.safetensors", "component1_dann_full.pt"):
            src = save_dir / name
            if src.exists():
                shutil.copy2(src, mirror / name)
        print(f"Artefacts mirrored to: {mirror}")


if __name__ == "__main__":
    main()
