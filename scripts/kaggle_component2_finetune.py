"""Kaggle-ready Component 2 (TorchXRayVision DenseNet121 routing head) bootstrap.

Pins the exact Kaggle dataset mount paths used by this repo, writes override
YAMLs, then invokes the existing trainer in-process without duplicating any
training logic.

Component 2 uses a frozen TorchXRayVision DenseNet121 backbone + a trainable
256-dim supervised-contrastive routing head. NIH CXR14 is excluded from training
by the default config (`data.exclude_datasets: [nih_cxr14]`), so only the three
TB datasets need to be mounted.

Expected Kaggle inputs (attach these with exactly these mount paths):

    /kaggle/input/datasets/iahmedhabib/montgomery/montgomery
    /kaggle/input/datasets/iahmedhabib/shehzhenn/shenzhen
    /kaggle/input/datasets/usmanshams/tbx-11/TBX11K

Optional (only needed if the TXV weights cache is not pre-populated in the
Kaggle image — if internet is enabled on the notebook torchxrayvision will
fetch them on first use automatically):

    /kaggle/input/datasets/iahmedhabib/torchxrayvision-weights

Typical Kaggle cells::

    !pip -q install torchxrayvision pyyaml tqdm
    !python /kaggle/working/repo/scripts/kaggle_component2_finetune.py --mode smoke
    !python /kaggle/working/repo/scripts/kaggle_component2_finetune.py --mode full
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

MONTGOMERY_PATH = KAGGLE_INPUT / "datasets/iahmedhabib/montgomery/montgomery"
SHENZHEN_PATH = KAGGLE_INPUT / "datasets/iahmedhabib/shehzhenn/shenzhen"
TBX11K_PATH = KAGGLE_INPUT / "datasets/usmanshams/tbx-11/TBX11K"
NIH_PATH = KAGGLE_INPUT / "datasets/organizations/nih-chest-xrays/data"
TXV_WEIGHTS_CACHE = KAGGLE_INPUT / "datasets/iahmedhabib/torchxrayvision-weights"


MODE_PRESETS: dict[str, dict[str, object]] = {
    "smoke": {
        "epochs": 1,
        "batch_size": 2,
        "num_workers": 2,
        "limit_per_domain": 8,
    },
    "short": {
        "epochs": 2,
        "batch_size": 8,
        "num_workers": 2,
        "limit_per_domain": 200,
    },
    "full": {
        "epochs": 10,
        "batch_size": 32,
        "num_workers": 2,
        "limit_per_domain": 1000,
    },
    "dry": {
        "epochs": 1,
        "batch_size": 2,
        "num_workers": 2,
        "limit_per_domain": None,
    },
}


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "src" / "training" / "train_component2_txv.py").is_file():
            return parent
    for candidate in (
        KAGGLE_WORKING / "repo",
        KAGGLE_WORKING / "dl-project-codebase",
        KAGGLE_INPUT / "dl-project-codebase",
    ):
        if (candidate / "src" / "training" / "train_component2_txv.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not locate the dl-project-codebase repo. Clone it into "
        "/kaggle/working/repo first."
    )


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
        print(f"  {label:<14}: {path}")
        return path
    if required:
        raise FileNotFoundError(
            f"Expected Kaggle mount for {label} at {path} but it is missing. "
            f"Attach the right Kaggle dataset to the notebook."
        )
    print(f"  {label:<14}: MISSING (optional) at {path}")
    return None


def _write_component2_config(
    repo_root: Path,
    *,
    save_dir: Path,
    preset: dict[str, object],
) -> Path:
    src_config = repo_root / "configs" / "component2_txv.yaml"
    with src_config.open("r", encoding="utf-8") as handle:
        cfg_root = yaml.safe_load(handle)

    cfg = cfg_root["component2_txv"]
    training = cfg["training"]
    training["epochs"] = int(preset["epochs"])
    training["batch_size"] = int(preset["batch_size"])
    training["num_workers"] = int(preset["num_workers"])
    training["limit_per_domain"] = preset["limit_per_domain"]
    training["save_dir"] = str(save_dir)
    training["save_name"] = "component2_routing_head.pt"
    training["device"] = None

    out = save_dir / "component2_txv.kaggle.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg_root, handle, sort_keys=False)
    return out


def _write_component1_manifest_config(
    repo_root: Path,
    *,
    save_dir: Path,
    tbx_list_name: str | None,
) -> Path:
    """Minimal component1_dann override used only for manifest building."""
    src_config = repo_root / "configs" / "component1_dann.yaml"
    with src_config.open("r", encoding="utf-8") as handle:
        cfg_root = yaml.safe_load(handle)

    cfg = cfg_root["component1_dann"]
    data = cfg["data"]
    data["manifest_cache"] = str(save_dir / "nih_index_cache.json")
    data["nih_metadata_csv"] = None
    data["nih_split"] = None
    if tbx_list_name is not None:
        data["tbx_list"] = tbx_list_name
    data["domain_sampling_weights"] = {
        "montgomery": 1.0,
        "shenzhen": 1.0,
        "tbx11k": 1.0,
        "nih_cxr14": 1.0,
    }

    out = save_dir / "component1_dann.kaggle.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg_root, handle, sort_keys=False)
    return out


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
    out = save_dir / "paths.kaggle.yaml"
    with out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(paths_cfg, handle, sort_keys=False)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle Component 2 fine-tune bootstrap.")
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_PRESETS),
        default="full",
        help="Preset: smoke / short / full / dry (default: full).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = MODE_PRESETS[args.mode]

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if TXV_WEIGHTS_CACHE.exists():
        target = Path.home() / ".torchxrayvision" / "models_data"
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            try:
                target.symlink_to(TXV_WEIGHTS_CACHE)
                print(f"Linked TXV weights cache: {target} -> {TXV_WEIGHTS_CACHE}")
            except OSError as exc:
                print(f"Could not link TXV weights cache ({exc}); falling back to download.")

    repo_root = _find_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    print(f"Mode: {args.mode}  preset={preset}")
    print("Kaggle mounts:")
    montgomery = _check_mount("montgomery", MONTGOMERY_PATH, required=True)
    shenzhen = _check_mount("shenzhen", SHENZHEN_PATH, required=True)
    tbx_root = _check_mount("tbx11k", TBX11K_PATH, required=True)
    nih_root = _check_mount("nih_cxr14", NIH_PATH, required=False)
    assert montgomery is not None and shenzhen is not None and tbx_root is not None

    if (tbx_root / "imgs").exists():
        tbx_dataset_root = tbx_root
    elif (tbx_root / "TBX11K" / "imgs").exists():
        tbx_dataset_root = tbx_root / "TBX11K"
    else:
        tbx_dataset_root = tbx_root
    tbx_list_name = _detect_tbx_list(tbx_root)
    print(f"  tbx_root      : {tbx_dataset_root} (list={tbx_list_name})")

    save_dir = (
        KAGGLE_WORKING / "component2_runs"
        if KAGGLE_WORKING.exists()
        else repo_root / "outputs" / "component2_runs"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    component2_cfg = _write_component2_config(
        repo_root,
        save_dir=save_dir,
        preset=preset,
    )
    component1_cfg = _write_component1_manifest_config(
        repo_root,
        save_dir=save_dir,
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
        "train_component2_txv.py",
        "--config", str(component2_cfg),
        "--paths", str(paths_cfg),
        "--component1_config", str(component1_cfg),
    ]
    if args.mode == "dry":
        trainer_argv.append("--dry-run")

    sys.argv = trainer_argv
    print(f"\nLaunching Component 2 trainer ({args.mode})")
    print(f"  config : {component2_cfg}")
    print(f"  paths  : {paths_cfg}")
    print(f"  cwd    : {Path.cwd()}\n")

    from src.training.train_component2_txv import main as train_main  # noqa: E402

    train_main()

    if args.mode == "dry":
        return

    routing_head = save_dir / "component2_routing_head.pt"
    if routing_head.exists():
        print(f"\nRouting head saved to: {routing_head}")
    else:
        print("\nTraining finished but routing head file was not found; check training output above.")

    if KAGGLE_WORKING.exists():
        mirror = KAGGLE_WORKING / "component2_artifacts"
        mirror.mkdir(parents=True, exist_ok=True)
        for name in (
            "component2_routing_head.pt",
            "component2_txv.kaggle.yaml",
            "paths.kaggle.yaml",
        ):
            src = save_dir / name
            if src.exists():
                shutil.copy2(src, mirror / name)
        print(f"Artefacts mirrored to: {mirror}")


if __name__ == "__main__":
    main()
