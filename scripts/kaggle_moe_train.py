"""Kaggle-ready MoE (C3 + C5 + C6) training bootstrap.

Pinned for the same Kaggle dataset mounts used by the Component 1 trainer.
Runs all three MoE training phases sequentially:

    Phase 1 — pretrain each of the 4 expert decoders independently
    Phase 2 — joint MoE training (gate + experts + fusion)
    Phase 3 — boundary critic training (Component 7 ResNet18)

Optional:
    Phase 0 — pre-compute and cache image embeddings to /kaggle/working

Typical Kaggle cells::

    !pip -q install segment_anything pyyaml safetensors tqdm scipy
    !git clone https://github.com/<you>/dl-project-codebase /kaggle/working/repo

    # smoke run (tiny manifest, 1 epoch each phase, ~5 min):
    !python /kaggle/working/repo/scripts/kaggle_moe_train.py --mode smoke

    # full run:
    !python /kaggle/working/repo/scripts/kaggle_moe_train.py --mode full

    # only phase 1 (expert pretraining):
    !python /kaggle/working/repo/scripts/kaggle_moe_train.py --mode full --phase pretrain
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
MEDSAM_CKPT_PATH = KAGGLE_INPUT / "datasets/iahmedhabib/medsam-vit-b/medsam_vit_b.pth"

# Optional inputs: previously trained C1 adapters and C4 decoder
C1_ADAPTER_PATH = KAGGLE_INPUT / "datasets/iahmedhabib/component1-artifacts/component1_adapters.safetensors"
C4_DECODER_PATH = KAGGLE_INPUT / "datasets/iahmedhabib/component4-artifacts/component4_mask_decoder.pt"


MODE_PRESETS: dict[str, dict[str, object]] = {
    "smoke": {
        "limit_per_domain": 8,
        "pretrain_epochs": 1,
        "joint_epochs": 1,
        "critic_epochs": 1,
        "batch_size": 2,
    },
    "short": {
        "limit_per_domain": 100,
        "pretrain_epochs": 3,
        "joint_epochs": 3,
        "critic_epochs": 2,
        "batch_size": 4,
    },
    "full": {
        "limit_per_domain": 500,
        "pretrain_epochs": 10,
        "joint_epochs": 15,
        "critic_epochs": 5,
        "batch_size": 4,
    },
}

ALL_PHASES = ("cache", "pretrain", "joint", "critic")


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "src" / "training" / "train_moe_joint.py").is_file():
            return parent
    for candidate in (
        KAGGLE_WORKING / "repo",
        KAGGLE_WORKING / "dl-project-codebase",
        KAGGLE_INPUT / "dl-project-codebase",
    ):
        if (candidate / "src" / "training" / "train_moe_joint.py").is_file():
            return candidate
    raise FileNotFoundError("Could not locate dl-project-codebase repo.")


def _check_mount(label: str, path: Path, *, required: bool) -> Path | None:
    if path.exists():
        print(f"  {label:<14}: {path}")
        return path
    if required:
        raise FileNotFoundError(f"Expected Kaggle mount for {label} at {path}")
    print(f"  {label:<14}: MISSING (optional) at {path}")
    return None


def _write_moe_config(
    repo_root: Path,
    *,
    save_dir: Path,
    medsam_ckpt: Path,
    c1_adapter: Path | None,
    c4_decoder: Path | None,
    preset: dict[str, object],
) -> Path:
    src = repo_root / "configs" / "moe.yaml"
    with src.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["component1"]["checkpoint_path"] = str(medsam_ckpt)
    cfg["component4"]["checkpoint_path"] = str(medsam_ckpt)
    if c1_adapter is not None:
        cfg["component1"]["adapter_path"] = str(c1_adapter)
    else:
        cfg["component1"]["adapter_path"] = None
    if c4_decoder is not None:
        cfg["component4"]["decoder_checkpoint_path"] = str(c4_decoder)
    else:
        cfg["component4"]["decoder_checkpoint_path"] = None

    cfg["moe_training"]["save_dir"] = str(save_dir)
    cfg["moe_training"]["pretrain"]["epochs"] = int(preset["pretrain_epochs"])
    cfg["moe_training"]["pretrain"]["batch_size"] = int(preset["batch_size"])
    cfg["moe_training"]["joint"]["epochs"] = int(preset["joint_epochs"])
    cfg["moe_training"]["joint"]["batch_size"] = int(preset["batch_size"])
    cfg["moe_training"]["boundary_critic"]["epochs"] = int(preset["critic_epochs"])

    out = save_dir / "moe.kaggle.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
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
    cfg = {
        "project_root": str(repo_root),
        "external_data_root": str(KAGGLE_INPUT),
        "datasets": {
            "montgomery": str(montgomery),
            "shenzhen": str(shenzhen),
            "tbx11k": str(tbx11k),
            "nih_cxr14": str(nih) if nih else str(save_dir / "_missing_nih"),
        },
        "artifacts": {
            "notebook_cache": str(save_dir / "notebook_cache"),
            "processed": str(save_dir / "processed"),
            "reports": str(save_dir / "reports"),
        },
    }
    out = save_dir / "paths.kaggle.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kaggle MoE training bootstrap.")
    p.add_argument("--mode", choices=sorted(MODE_PRESETS), default="full")
    p.add_argument(
        "--phase",
        choices=ALL_PHASES + ("all",),
        default="all",
        help="Run a specific phase or 'all' (cache → pretrain → joint → critic).",
    )
    p.add_argument("--cache-dir", default=None, help="Reuse an existing cache directory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    preset = MODE_PRESETS[args.mode]

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    repo_root = _find_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    print(f"Mode: {args.mode}  preset={preset}  phase={args.phase}")
    print("Kaggle mounts:")
    montgomery = _check_mount("montgomery", MONTGOMERY_PATH, required=True)
    shenzhen = _check_mount("shenzhen", SHENZHEN_PATH, required=True)
    tbx_root = _check_mount("tbx11k", TBX11K_PATH, required=True)
    nih_root = _check_mount("nih_cxr14", NIH_PATH, required=False)
    medsam_ckpt = _check_mount("medsam_vit_b", MEDSAM_CKPT_PATH, required=True)
    c1_adapter = _check_mount("c1_adapter", C1_ADAPTER_PATH, required=False)
    c4_decoder = _check_mount("c4_decoder", C4_DECODER_PATH, required=False)

    assert all([montgomery, shenzhen, tbx_root, medsam_ckpt])

    save_dir = (KAGGLE_WORKING / "moe_runs") if KAGGLE_WORKING.exists() else (repo_root / "outputs" / "moe_runs")
    save_dir.mkdir(parents=True, exist_ok=True)

    config_path = _write_moe_config(
        repo_root,
        save_dir=save_dir,
        medsam_ckpt=medsam_ckpt,
        c1_adapter=c1_adapter,
        c4_decoder=c4_decoder,
        preset=preset,
    )
    paths_path = _write_paths_override(
        repo_root,
        montgomery=montgomery,
        shenzhen=shenzhen,
        tbx11k=tbx_root,
        nih=nih_root,
        save_dir=save_dir,
    )

    os.chdir(repo_root)

    cache_dir = Path(args.cache_dir) if args.cache_dir else (save_dir / "embedding_cache")

    phases_to_run = ALL_PHASES if args.phase == "all" else (args.phase,)

    # ---- Phase 0: cache embeddings ----
    if "cache" in phases_to_run and not cache_dir.exists():
        print(f"\n=== Phase 0: cache embeddings → {cache_dir} ===")
        sys.argv = [
            "cache_moe_embeddings.py",
            "--config", str(config_path),
            "--paths", str(paths_path),
            "--output", str(cache_dir),
            "--limit-per-domain", str(preset["limit_per_domain"]),
        ]
        from scripts.cache_moe_embeddings import main as cache_main
        cache_main()
    elif "cache" in phases_to_run:
        print(f"\n=== Phase 0: cache exists at {cache_dir}, skipping ===")

    # ---- Phase 1: expert pretraining ----
    if "pretrain" in phases_to_run:
        print(f"\n=== Phase 1: expert pretraining ===")
        sys.argv = [
            "train_experts.py",
            "--config", str(config_path),
            "--cache-dir", str(cache_dir),
            "--all",
        ]
        from src.training.train_experts import main as experts_main
        experts_main()

    # ---- Phase 2: joint MoE training ----
    if "joint" in phases_to_run:
        print(f"\n=== Phase 2: joint MoE training ===")
        sys.argv = [
            "train_moe_joint.py",
            "--config", str(config_path),
            "--cache-dir", str(cache_dir),
        ]
        from src.training.train_moe_joint import main as joint_main
        joint_main()

    # ---- Phase 3: boundary critic ----
    if "critic" in phases_to_run:
        print(f"\n=== Phase 3: boundary critic ===")
        sys.argv = [
            "train_boundary_critic.py",
            "--config", str(config_path),
            "--cache-dir", str(cache_dir),
        ]
        from src.training.train_boundary_critic import main as critic_main
        critic_main()

    # Mirror artifacts to /kaggle/working/moe_artifacts/
    if KAGGLE_WORKING.exists():
        mirror = KAGGLE_WORKING / "moe_artifacts"
        mirror.mkdir(parents=True, exist_ok=True)
        for f in save_dir.glob("*.pt"):
            shutil.copy2(f, mirror / f.name)
        for f in save_dir.glob("*.jsonl"):
            shutil.copy2(f, mirror / f.name)
        print(f"\nArtefacts mirrored to: {mirror}")


if __name__ == "__main__":
    main()
