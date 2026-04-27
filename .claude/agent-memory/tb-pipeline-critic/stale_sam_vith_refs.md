---
name: Stale SAM ViT-H References After MedSAM Pivot
description: .env.example and one mock class name still reference SAM ViT-H even though the backbone was swapped to MedSAM ViT-B
type: project
---

The repo pivoted from SAM ViT-H to MedSAM ViT-B (sam_model_registry["vit_b"] + MedSAM weights) to fit Kaggle T4 16 GB, but two stale references remain:

1. `.env.example` still defines `SAM_VIT_H_CKPT=checkpoints/sam/sam_vit_h_4b8939.pth`. No such file is present in `checkpoints/`; all real loading goes through `checkpoints/medsam/medsam_vit_b.pth`.
2. `src/components/component1_encoder.py` keeps the mock class name `MockSAMViTHImageEncoder`. The real loader path uses MedSAM ViT-B; only the mock's class name is stale. Output tensor contract ([B,256,64,64]) is correct regardless.

**Why:** Both are cosmetic / documentation-level. Renaming the mock class would require touching test fixtures that import it; low priority.

**How to apply:** When writing audit reports, flag these as stale-naming issues but not blockers. If the user asks to "finalize the MedSAM pivot," the fix is (a) replace the env var block in `.env.example` with `MEDSAM_VIT_B_CKPT` and (b) rename the mock class to `MockMedSAMViTBImageEncoder`, updating any test imports.
