"""Kaggle-ready Component 1 (MedSAM ViT-B + LoRA + DANN) fine-tune bootstrap.

Intended use: paste the cell at the bottom of this file into a Kaggle
notebook. The script pins the exact Kaggle dataset mount paths we use,
writes override config + paths YAMLs, and invokes the existing trainer
in-process so we do not duplicate any training logic.

Component 1 now uses MedSAM ViT-B (not SAM ViT-H) so the whole pipeline
trains comfortably on a Kaggle free-tier T4 16 GB.

Expected Kaggle inputs (attach these to the notebook with exactly these
mount paths — see /kaggle/input after attaching):

    /kaggle/input/datasets/iahmedhabib/montgomery/montgomery
    /kaggle/input/datasets/iahmedhabib/shehzhenn/shenzhen
    /kaggle/input/datasets/usmanshams/tbx-11/TBX11K
    /kaggle/input/datasets/organizations/nih-chest-xrays/data
    /kaggle/input/datasets/iahmedhabib/medsam-vit-b/medsam_vit_b.pth

Typical Kaggle cells:

    !pip -q install segment_anything pyyaml safetensors tqdm
    !git clone https://github.com/<you>/dl-project-codebase /kaggle/working/repo
    # (or copy from an attached Kaggle dataset of the repo)

    # smoke run (tiny manifest, 1 epoch, ~5 min):
    !python /kaggle/working/repo/scripts/kaggle_component1_finetune.py --mode smoke

    # dry-run (prints manifest counts, no training):
    !python /kaggle/working/repo/scripts/kaggle_component1_finetune.py --mode dry

    # full run:
    !python /kaggle/working/repo/scripts/kaggle_component1_finetune.py --mode full

    # resume after a Kaggle session kill:
    !python /kaggle/working/repo/scripts/kaggle_component1_finetune.py \
        --mode full \
        --resume /kaggle/working/component1_runs/last_component1_snapshot.pt
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import yaml


KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

# Exact Kaggle dataset mount paths used by this repo. If Kaggle renames
# any of these, update this block (and nothing else).
MONTGOMERY_PATH = KAGGLE_INPUT / "datasets/iahmedhabib/montgomery/montgomery"
SHENZHEN_PATH = KAGGLE_INPUT / "datasets/iahmedhabib/shehzhenn/shenzhen"
TBX11K_PATH = KAGGLE_INPUT / "datasets/usmanshams/tbx-11/TBX11K"
NIH_PATH = KAGGLE_INPUT / "datasets/organizations/nih-chest-xrays/data"
MEDSAM_CKPT_PATH = KAGGLE_INPUT / "datasets/iahmedhabib/medsam-vit-b/medsam_vit_b.pth"


MODE_PRESETS: dict[str, dict[str, object]] = {
    # Fast sanity check: a handful of images per domain, 1 epoch.
    # bs=1 so smoke succeeds even if gradient checkpointing is disabled.
    "smoke": {
        "epochs": 1,
        "batch_size": 1,
        "num_workers": 2,
        "limit_per_domain": 8,
    },
    # Real but short: check GPU + losses behave. ~20-30 min on T4.
    "short": {
        "epochs": 2,
        "batch_size": 1,
        "num_workers": 2,
        "limit_per_domain": 200,
    },
    # Full fine-tune. MedSAM ViT-B at 1024^2 with activation checkpointing
    # fits bs=2 on a T4 16 GB. Drop to 1 if OOM; raise to 4 only on A100.
    "full": {
        "epochs": 8,
        "batch_size": 2,
        "num_workers": 2,
        "limit_per_domain": None,
    },
    # No training — just print manifest counts per domain.
    "dry": {
        "epochs": 1,
        "batch_size": 2,
        "num_workers": 2,
        "limit_per_domain": None,
    },
}


def _find_repo_root() -> Path:
    """Locate the dl-project-codebase repo root (must contain ``src/``)."""

    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "src" / "training" / "train_component1_dann.py").is_file():
            return parent

    for candidate in (
        KAGGLE_WORKING / "repo",
        KAGGLE_WORKING / "dl-project-codebase",
        KAGGLE_INPUT / "dl-project-codebase",
    ):
        if (candidate / "src" / "training" / "train_component1_dann.py").is_file():
            return candidate

    raise FileNotFoundError(
        "Could not locate the dl-project-codebase repo. Clone it into "
        "/kaggle/working/repo first."
    )


def _detect_tbx_images_root(tbx_root: Path) -> Path:
    """TBX11K layouts vary; return the directory that actually holds images."""

    for sub in ("imgs", "TBX11K/imgs", "images"):
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


def _check_mount(label: str, path: Path, *, required: bool) -> Path | None:
    if path.exists():
        print(f"  {label:<11}: {path}")
        return path
    if required:
        raise FileNotFoundError(
            f"Expected Kaggle mount for {label} at {path} but it is missing. "
            f"Attach the right Kaggle dataset to the notebook."
        )
    print(f"  {label:<11}: MISSING (optional) at {path}")
    return None


def _write_override_config(
    repo_root: Path,
    *,
    medsam_ckpt: Path,
    save_dir: Path,
    preset: dict[str, object],
    tbx_list_name: str | None,
) -> Path:
    src_config = repo_root / "configs" / "component1_dann.yaml"
    with src_config.open("r", encoding="utf-8") as handle:
        cfg_root = yaml.safe_load(handle)

    cfg = cfg_root["component1_dann"]

    cfg["encoder"]["backend"] = "segment_anything"
    cfg["encoder"]["checkpoint_path"] = str(medsam_ckpt)

    training = cfg["training"]
    training["epochs"] = int(preset["epochs"])
    training["batch_size"] = int(preset["batch_size"])
    training["num_workers"] = int(preset["num_workers"])
    training["grl_ramp_epochs"] = max(1, int(preset["epochs"]) // 2)
    training["limit_per_domain"] = preset["limit_per_domain"]
    training["save_dir"] = str(save_dir)
    training["adapter_save_name"] = "component1_adapters.safetensors"
    training["save_full_checkpoint"] = True
    training["save_name"] = "component1_dann_full.pt"
    training["save_every"] = 1
    training["device"] = None
    training["amp"] = True

    data = cfg["data"]
    data["manifest_cache"] = str(save_dir / "nih_index_cache.json")
    # Let the trainer try both Data_Entry_2017_v2020.csv and Data_Entry_2017.csv.
    data["nih_metadata_csv"] = None
    data["nih_split"] = None
    if tbx_list_name is not None:
        data["tbx_list"] = tbx_list_name
    data["domain_sampling_weights"] = {
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
    montgomery: Path,
    shenzhen: Path,
    tbx11k: Path,
    nih: Path | None,
    save_dir: Path,
) -> Path:
    paths_cfg = {
        "project_root": str(repo_root),
        "external_data_root": str(KAGGLE_INPUT),
        "datasets": {
            "montgomery": str(montgomery),
            "shenzhen": str(shenzhen),
            "tbx11k": str(tbx11k),
            "nih_cxr14": str(nih) if nih is not None else str(save_dir / "_missing_nih"),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle Component 1 fine-tune bootstrap.")
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_PRESETS),
        default="full",
        help="Preset: smoke / short / full / dry (default: full).",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to last_component1_snapshot.pt to resume from.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = MODE_PRESETS[args.mode]

    # Reduce CUDA fragmentation on T4 so the LoRA-through-frozen-backbone
    # backward pass has room to work. Must be set before any torch import
    # that initializes CUDA — setting it here, before the trainer import,
    # is early enough.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    repo_root = _find_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    print(f"Mode: {args.mode}  preset={preset}")
    print("Kaggle mounts:")
    montgomery = _check_mount("montgomery", MONTGOMERY_PATH, required=True)
    shenzhen = _check_mount("shenzhen", SHENZHEN_PATH, required=True)
    tbx_root = _check_mount("tbx11k", TBX11K_PATH, required=True)
    nih_root = _check_mount("nih_cxr14", NIH_PATH, required=False)
    medsam_ckpt = _check_mount("medsam_vit_b", MEDSAM_CKPT_PATH, required=True)
    assert montgomery is not None and shenzhen is not None and tbx_root is not None
    assert medsam_ckpt is not None

    tbx_images = _detect_tbx_images_root(tbx_root)
    tbx_list_name = _detect_tbx_list(tbx_root)
    print(f"  tbx_images : {tbx_images} (list={tbx_list_name})")

    # The trainer's _build_tbx11k_samples expects the TBX11K *root* (the
    # directory that contains both ``imgs/`` and ``lists/``), not the
    # ``imgs/`` subdir. Some Kaggle mirrors nest TBX11K one level deep,
    # so pick whichever of ``tbx_root`` or ``tbx_root/TBX11K`` actually
    # holds the ``imgs/`` subtree.
    if (tbx_root / "imgs").exists():
        tbx_dataset_root = tbx_root
    elif (tbx_root / "TBX11K" / "imgs").exists():
        tbx_dataset_root = tbx_root / "TBX11K"
    else:
        tbx_dataset_root = tbx_root
    print(f"  tbx_root   : {tbx_dataset_root}")

    save_dir = KAGGLE_WORKING / "component1_runs" if KAGGLE_WORKING.exists() else repo_root / "outputs" / "component1_runs"
    save_dir.mkdir(parents=True, exist_ok=True)

    override_cfg = _write_override_config(
        repo_root,
        medsam_ckpt=medsam_ckpt,
        save_dir=save_dir,
        preset=preset,
        tbx_list_name=tbx_list_name,
    )
    paths_cfg = _write_paths_override(
        repo_root,
        montgomery=montgomery,
        shenzhen=shenzhen,
        tbx11k=tbx_dataset_root,
        nih=nih_root,
        save_dir=save_dir,
    )

    os.chdir(repo_root)

    trainer_argv: list[str] = [
        "train_component1_dann.py",
        "--config", str(override_cfg),
        "--paths", str(paths_cfg),
    ]
    if args.mode == "dry":
        trainer_argv.append("--dry-run")
    if args.resume:
        trainer_argv.extend(["--resume", args.resume])

    sys.argv = trainer_argv
    print(f"\nLaunching Component 1 trainer ({args.mode})")
    print(f"  config : {override_cfg}")
    print(f"  paths  : {paths_cfg}")
    print(f"  cwd    : {Path.cwd()}\n")

    from src.training.train_component1_dann import main as train_main  # noqa: E402

    train_main()

    if args.mode == "dry":
        return

    adapters = save_dir / "component1_adapters.safetensors"
    if adapters.exists():
        print(f"\nAdapters saved to: {adapters}")
    else:
        print("\nTraining finished but adapter file was not found; check training output above.")

    if KAGGLE_WORKING.exists():
        mirror = KAGGLE_WORKING / "component1_artifacts"
        mirror.mkdir(parents=True, exist_ok=True)
        for name in (
            "component1_adapters.safetensors",
            "component1_dann_full.pt",
            "component1_training_history.jsonl",
            "last_component1_snapshot.pt",
        ):
            src = save_dir / name
            if src.exists():
                shutil.copy2(src, mirror / name)
        print(f"Artefacts mirrored to: {mirror}")


if __name__ == "__main__":
    main()
