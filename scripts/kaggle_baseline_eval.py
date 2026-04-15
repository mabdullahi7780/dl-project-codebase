"""Kaggle-ready baseline evaluation bootstrap.

Mirrors ``scripts/kaggle_component1_finetune.py`` but for the baseline
*evaluation* (no training). Pins the same Kaggle dataset mount paths,
writes override ``baseline.yaml`` + ``paths.yaml``, then calls
``src.evaluation.baseline_eval.run_baseline_evaluation`` in-process.

Expected Kaggle inputs (same as the fine-tune bootstrap):

    /kaggle/input/datasets/iahmedhabib/montgomery/montgomery
    /kaggle/input/datasets/iahmedhabib/shehzhenn/shenzhen
    /kaggle/input/datasets/usmanshams/tbx-11/TBX11K
    /kaggle/input/datasets/organizations/nih-chest-xrays/data
    /kaggle/input/datasets/iahmedhabib/medsam-vit-b/medsam_vit_b.pth

Optional (when fine-tuned artifacts have been downloaded back into the
repo's root after training):

    /kaggle/working/repo/checkpoints/component1/component1_adapters.safetensors
    /kaggle/working/repo/checkpoints/component4/component4_mask_decoder.pt

Typical Kaggle cells:

    !python /kaggle/working/repo/scripts/kaggle_baseline_eval.py --mode full

    # dry: just build manifests and print counts, no GPU work
    !python /kaggle/working/repo/scripts/kaggle_baseline_eval.py --mode dry

    # with finetuned component 1 adapters (after they've been downloaded)
    !python /kaggle/working/repo/scripts/kaggle_baseline_eval.py \
        --mode full \
        --component1-adapter /kaggle/working/repo/checkpoints/component1/component1_adapters.safetensors
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


MODE_PRESETS: dict[str, dict[str, object]] = {
    # Fast sanity: 4 images/domain, just to check the plumbing end-to-end.
    "smoke": {"limit_per_domain": 4},
    # Baseline paper numbers: 200/domain, ~20-40 min on T4.
    "full": {"limit_per_domain": 200},
    # Manifest-only: build splits, no pipeline forward.
    "dry": {"limit_per_domain": 200},
}


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "src" / "evaluation" / "baseline_eval.py").is_file():
            return parent
    for candidate in (
        KAGGLE_WORKING / "repo",
        KAGGLE_WORKING / "dl-project-codebase",
        KAGGLE_INPUT / "dl-project-codebase",
    ):
        if (candidate / "src" / "evaluation" / "baseline_eval.py").is_file():
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


def _resolve_tbx_dataset_root(tbx_root: Path) -> Path:
    """Pick the directory that actually holds ``imgs/`` and ``lists/``."""
    if (tbx_root / "imgs").exists():
        return tbx_root
    if (tbx_root / "TBX11K" / "imgs").exists():
        return tbx_root / "TBX11K"
    return tbx_root


def _check_mount(label: str, path: Path, *, required: bool) -> Path | None:
    if path.exists():
        print(f"  {label:<12}: {path}")
        return path
    if required:
        raise FileNotFoundError(
            f"Expected Kaggle mount for {label} at {path} but it is missing. "
            f"Attach the right Kaggle dataset to the notebook."
        )
    print(f"  {label:<12}: MISSING (optional) at {path}")
    return None


def _write_override_baseline_config(
    repo_root: Path,
    *,
    medsam_ckpt: Path,
    component1_adapter: Path | None,
    component4_decoder_ckpt: Path | None,
    save_dir: Path,
) -> Path:
    src_config = repo_root / "configs" / "baseline.yaml"
    with src_config.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    cfg.setdefault("runtime", {})["device"] = None

    component1 = cfg.setdefault("component1", {})
    component1["backend"] = "auto"
    component1["checkpoint_path"] = str(medsam_ckpt)
    component1["adapter_path"] = str(component1_adapter) if component1_adapter is not None else None

    component4 = cfg.setdefault("component4", {})
    component4["backend"] = "auto"
    component4["checkpoint_path"] = str(medsam_ckpt)
    component4["model_type"] = "vit_b"
    component4["decoder_checkpoint_path"] = (
        str(component4_decoder_ckpt) if component4_decoder_ckpt is not None else None
    )

    override = save_dir / "baseline.kaggle.yaml"
    override.parent.mkdir(parents=True, exist_ok=True)
    with override.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
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
    parser = argparse.ArgumentParser(description="Kaggle baseline evaluation bootstrap.")
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_PRESETS),
        default="full",
        help="Preset: smoke / full / dry (default: full).",
    )
    parser.add_argument(
        "--component1-adapter",
        default=None,
        help="Path to a component1_adapters.safetensors file (LoRA+DANN).",
    )
    parser.add_argument(
        "--component4-decoder",
        default=None,
        help="Path to a fine-tuned component 4 mask decoder .pt.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: /kaggle/working/outputs/baseline_eval or outputs/baseline_eval locally).",
    )
    parser.add_argument(
        "--limit-per-domain",
        type=int,
        default=None,
        help="Override preset limit_per_domain.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = dict(MODE_PRESETS[args.mode])
    if args.limit_per_domain is not None:
        preset["limit_per_domain"] = args.limit_per_domain

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

    tbx_dataset_root = _resolve_tbx_dataset_root(tbx_root)
    tbx_list_name = _detect_tbx_list(tbx_root)
    print(f"  tbx_root    : {tbx_dataset_root} (list={tbx_list_name})")

    adapter_path: Path | None = None
    if args.component1_adapter:
        adapter_path = Path(args.component1_adapter)
        if not adapter_path.is_file():
            raise FileNotFoundError(
                f"--component1-adapter given but not a file: {adapter_path}"
            )
        print(f"  c1_adapter  : {adapter_path}")
    else:
        # Auto-detect a typical download location inside the cloned repo.
        for candidate in (
            repo_root / "checkpoints" / "component1" / "component1_adapters.safetensors",
            repo_root / "checkpoints" / "component1" / "component1_adapters.pt",
            repo_root / "component1_adapters.safetensors",
        ):
            if candidate.is_file():
                adapter_path = candidate
                print(f"  c1_adapter  : {adapter_path} (auto-detected)")
                break
        if adapter_path is None:
            print("  c1_adapter  : NOT FOUND — evaluating encoder WITHOUT LoRA+DANN adapters")

    decoder_ckpt: Path | None = None
    if args.component4_decoder:
        decoder_ckpt = Path(args.component4_decoder)
        if not decoder_ckpt.is_file():
            raise FileNotFoundError(
                f"--component4-decoder given but not a file: {decoder_ckpt}"
            )
        print(f"  c4_decoder  : {decoder_ckpt}")
    else:
        for candidate in (
            repo_root / "checkpoints" / "component4" / "component4_mask_decoder.pt",
        ):
            if candidate.is_file():
                decoder_ckpt = candidate
                print(f"  c4_decoder  : {decoder_ckpt} (auto-detected)")
                break
        if decoder_ckpt is None:
            print("  c4_decoder  : NOT FOUND — using stock MedSAM decoder")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif KAGGLE_WORKING.exists():
        output_dir = KAGGLE_WORKING / "outputs" / "baseline_eval"
    else:
        output_dir = repo_root / "outputs" / "baseline_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    override_cfg = _write_override_baseline_config(
        repo_root,
        medsam_ckpt=medsam_ckpt,
        component1_adapter=adapter_path,
        component4_decoder_ckpt=decoder_ckpt,
        save_dir=output_dir,
    )
    paths_cfg = _write_paths_override(
        repo_root,
        montgomery=montgomery,
        shenzhen=shenzhen,
        tbx11k=tbx_dataset_root,
        nih=nih_root,
        save_dir=output_dir,
    )

    os.chdir(repo_root)

    from src.evaluation.baseline_eval import (  # noqa: E402
        build_eval_manifest,
        make_test_splits,
        run_baseline_evaluation,
    )

    if args.mode == "dry":
        # Manifest + split counts, no pipeline forward.
        samples = build_eval_manifest(
            montgomery_root=montgomery,
            shenzhen_root=shenzhen,
            tbx11k_root=tbx_dataset_root,
            nih_root=nih_root,
            tbx_list_name=tbx_list_name,
            nih_cache_path=output_dir / "_nih_index_cache.json",
        )
        splits = make_test_splits(
            samples,
            limit_per_domain=int(preset["limit_per_domain"]),
            cache_path=output_dir / "test_splits.json",
        )
        print("\nManifest counts:")
        for dom, items in samples.items():
            print(f"  {dom:<11}: {len(items)} total")
        print("Test split counts:")
        for dom, items in splits.items():
            labelled = sum(1 for e in items if e.tb_label is not None)
            print(f"  {dom:<11}: {len(items)} (TB-labelled: {labelled})")
        return

    run_baseline_evaluation(
        baseline_config_path=override_cfg,
        paths_config_path=paths_cfg,
        output_dir=output_dir,
        limit_per_domain=int(preset["limit_per_domain"]),
        tbx_list_name=tbx_list_name,
        repo_root=repo_root,
    )

    # Mirror key files into /kaggle/working so they land in the notebook's
    # Output file pane when the session ends (only needed if the user
    # overrode --output-dir to something outside /kaggle/working).
    def _inside_kaggle_working(path: Path) -> bool:
        try:
            path.relative_to(KAGGLE_WORKING)
        except ValueError:
            return False
        return True

    if KAGGLE_WORKING.exists() and not _inside_kaggle_working(output_dir):
        mirror = KAGGLE_WORKING / "outputs" / "baseline_eval"
        mirror.mkdir(parents=True, exist_ok=True)
        for name in (
            "baseline_components.csv",
            "baseline_system.csv",
            "baseline_per_image.csv",
            "baseline_summary.json",
            "test_splits.json",
            "baseline.kaggle.yaml",
            "paths.kaggle.yaml",
        ):
            src = output_dir / name
            if src.exists():
                shutil.copy2(src, mirror / name)
        print(f"Artefacts mirrored to: {mirror}")


if __name__ == "__main__":
    main()
