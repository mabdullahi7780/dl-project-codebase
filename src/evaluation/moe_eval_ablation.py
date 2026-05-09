"""MoE evaluation wrapper for ablation studies.

Wraps ``run_moe_evaluation`` and accepts a small set of overrides without
touching the production ``moe_eval.py``. Used by ``notebooks/ablation_studies.ipynb``.

Supported overrides
-------------------
- ``routing_temperature``    — softmax temperature applied at inference
- ``fusion_mode``            — "weighted_logit" | "weighted_prob"
- ``use_domain_ctx``         — enable/disable domain context input to the gate
                               (note: requires a gate retrained with the matching
                               flag — flipping at inference will silently produce
                               nonsense for the disabled variant)
- ``cavity_threshold``       — sigmoid cutoff for Expert 2 cavity binarisation
- ``otsu_floor``/``otsu_ceil`` — bounds for the adaptive Otsu lesion threshold;
                               set both equal to force a fixed threshold
- ``routing_override``       — "uniform", "top1", or "force_expert_<idx>" to
                               replace the gate's softmax weights at inference

All overrides default to None, in which case the base config / current
hardcoded value is used.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

import src.components.component8_timika as c8t
import src.utils.morphology as morphology
from src.evaluation.moe_eval import run_moe_evaluation


@dataclass(slots=True)
class AblationOverrides:
    """Bundle of overrides for a single ablation run."""

    name: str = "default"
    # Config-level (applied via patched config dict)
    routing_temperature: float | None = None
    fusion_mode: str | None = None
    use_domain_ctx: bool | None = None
    moe_checkpoint_path: str | None = None  # alternate gate checkpoint (for B)
    # Function-level (applied via monkey-patch)
    cavity_threshold: float | None = None
    otsu_floor: float | None = None
    otsu_ceil: float | None = None
    routing_override: str | None = None  # "uniform" | "top1" | "force_expert_<i>"

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "routing_temperature": self.routing_temperature,
            "fusion_mode": self.fusion_mode,
            "use_domain_ctx": self.use_domain_ctx,
            "moe_checkpoint_path": self.moe_checkpoint_path,
            "cavity_threshold": self.cavity_threshold,
            "otsu_floor": self.otsu_floor,
            "otsu_ceil": self.otsu_ceil,
            "routing_override": self.routing_override,
        }


def _patch_config(base_cfg: dict[str, Any], ov: AblationOverrides) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("moe", {})
    if ov.routing_temperature is not None:
        cfg["moe"]["routing_temperature"] = float(ov.routing_temperature)
    if ov.fusion_mode is not None:
        cfg["moe"]["fusion_mode"] = str(ov.fusion_mode)
    if ov.use_domain_ctx is not None:
        cfg["moe"]["use_domain_ctx"] = bool(ov.use_domain_ctx)
    if ov.moe_checkpoint_path is not None:
        cfg["moe"]["checkpoint_path"] = str(ov.moe_checkpoint_path)
    return cfg


@contextmanager
def _patched_runtime(ov: AblationOverrides):
    """Temporarily monkey-patch hardcoded knobs for the duration of one eval."""
    # 1) Adaptive Otsu floor/ceil
    orig_adaptive = morphology.adaptive_lesion_threshold
    if ov.otsu_floor is not None or ov.otsu_ceil is not None:
        floor = ov.otsu_floor if ov.otsu_floor is not None else 0.5
        ceil = ov.otsu_ceil if ov.otsu_ceil is not None else 0.8

        def _patched_adaptive(prob_map, lung_mask, **kw):
            kw.setdefault("floor", floor)
            kw.setdefault("ceil", ceil)
            return orig_adaptive(prob_map, lung_mask, **kw)

        morphology.adaptive_lesion_threshold = _patched_adaptive
        # moe_eval imports the symbol directly, patch it there too
        import src.evaluation.moe_eval as _mev
        _mev.adaptive_lesion_threshold = _patched_adaptive

    # 2) Cavity threshold passed through compute_moe_timika
    orig_moe_timika = c8t.compute_moe_timika
    if ov.cavity_threshold is not None:
        cav = float(ov.cavity_threshold)

        def _patched_moe_timika(*args, **kwargs):
            kwargs.setdefault("cavity_threshold", cav)
            return orig_moe_timika(*args, **kwargs)

        c8t.compute_moe_timika = _patched_moe_timika
        import src.evaluation.moe_eval as _mev
        _mev.compute_moe_timika = _patched_moe_timika

    # 3) Routing override — wrap the gate forward
    orig_gate_forward = None
    if ov.routing_override is not None:
        from src.components.component3_routing import Component3RoutingGate

        orig_gate_forward = Component3RoutingGate.forward
        mode = ov.routing_override

        def _override_forward(self, img_emb, domain_ctx=None):
            B = img_emb.shape[0]
            K = self.num_experts
            device = img_emb.device
            if mode == "uniform":
                w = torch.full((B, K), 1.0 / K, device=device)
            elif mode == "top1":
                w_full = orig_gate_forward(self, img_emb, domain_ctx)
                top_idx = w_full.argmax(dim=-1, keepdim=True)
                w = torch.zeros_like(w_full).scatter_(1, top_idx, 1.0)
            elif mode.startswith("force_expert_"):
                idx = int(mode.replace("force_expert_", ""))
                w = torch.zeros((B, K), device=device)
                w[:, idx] = 1.0
            else:
                raise ValueError(f"Unknown routing_override: {mode}")
            return w

        Component3RoutingGate.forward = _override_forward

    try:
        yield
    finally:
        # Restore everything
        morphology.adaptive_lesion_threshold = orig_adaptive
        c8t.compute_moe_timika = orig_moe_timika
        import src.evaluation.moe_eval as _mev
        _mev.adaptive_lesion_threshold = orig_adaptive
        _mev.compute_moe_timika = orig_moe_timika
        if orig_gate_forward is not None:
            from src.components.component3_routing import Component3RoutingGate
            Component3RoutingGate.forward = orig_gate_forward


def run_ablation(
    overrides: AblationOverrides,
    *,
    moe_config_path: Path,
    paths_config_path: Path,
    output_dir: Path,
    limit_per_domain: int = 200,
    holdout_frac: float = 0.2,
    seed: int = 42,
    tbx_list_name: str | None = "all_trainval.txt",
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run a single ablation variant and return its summary dict.

    Outputs are written under ``output_dir / overrides.name / ...`` so multiple
    variants do not overwrite each other.
    """
    output_dir = Path(output_dir)
    variant_dir = output_dir / overrides.name
    variant_dir.mkdir(parents=True, exist_ok=True)

    # Patch config file (write a temp copy under the variant directory)
    with Path(moe_config_path).open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f) or {}
    patched_cfg = _patch_config(base_cfg, overrides)
    patched_cfg_path = variant_dir / "_patched_moe.yaml"
    with patched_cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(patched_cfg, f, sort_keys=False)

    print(f"\n{'=' * 70}\nABLATION: {overrides.name}\n{'=' * 70}")
    print(f"  Overrides: {overrides.describe()}")

    with _patched_runtime(overrides):
        summary = run_moe_evaluation(
            moe_config_path=patched_cfg_path,
            paths_config_path=paths_config_path,
            output_dir=variant_dir,
            limit_per_domain=limit_per_domain,
            holdout_frac=holdout_frac,
            seed=seed,
            tbx_list_name=tbx_list_name,
            repo_root=repo_root,
        )

    summary["ablation"] = overrides.describe()
    return summary
