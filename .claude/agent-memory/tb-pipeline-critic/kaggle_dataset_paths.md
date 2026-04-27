---
name: Kaggle Dataset Mount Paths
description: Exact /kaggle/input candidate paths encoded in scripts/kaggle_moe_train.py and kaggle_component1_finetune.py
type: reference
---

Kaggle dataset mount candidates currently encoded in `scripts/kaggle_moe_train.py` (checked in order; first to exist wins; overridable via env vars KAGGLE_*_PATH):

- Montgomery: `/kaggle/input/datasets/iahmedhabib/montgomery/montgomery`, `/kaggle/input/montgomery/montgomery`, `/kaggle/input/montgomery`
- Shenzhen: `/kaggle/input/datasets/iahmedhabib/shehzhenn/shenzhen` (note: typo "shehzhenn" in the Kaggle dataset slug), `/kaggle/input/datasets/iahmedhabib/shenzhen/shenzhen`, `/kaggle/input/shenzhen/shenzhen`, `/kaggle/input/shenzhen`
- TBX11K: `/kaggle/input/datasets/usmanshams/tbx-11/TBX11K`, `/kaggle/input/tbx-11/TBX11K`, `/kaggle/input/TBX11K`
- NIH-CXR14: `/kaggle/input/datasets/organizations/nih-chest-xrays/data`, `/kaggle/input/nih-chest-xrays/data`, `/kaggle/input/nih-cxr14`
- MedSAM ViT-B: `/kaggle/input/datasets/iahmedhabib/medsam-vit-b/medsam_vit_b.pth`, `/kaggle/input/medsam-vit-b/medsam_vit_b.pth`, `/kaggle/input/medsam/medsam_vit_b.pth`
- Optional Component 1 adapters: `/kaggle/input/datasets/iahmedhabib/component1-artifacts/component1_adapters.safetensors` (+ other candidates)
- Optional Component 4 decoder: `/kaggle/input/datasets/iahmedhabib/component4-artifacts/component4_mask_decoder.pt` (+ other candidates)

**When to use:** When writing new Kaggle scripts or configs, point at these exact mounts (or add new candidates using the same `_resolve_path`/`_check_mount` pattern). When diagnosing a Kaggle run, FileNotFoundError usually means the user forgot to attach one of these datasets.
