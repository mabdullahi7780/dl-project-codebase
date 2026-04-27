---
name: Plan Deviation — MoE Path Implemented
description: Components 3, 5, 6 and upgraded C7/C8 are implemented despite plan.md marking them Priority C; this is accepted
type: project
---

Components 3 (routing gate), 5 (4-expert bank), 6 (fusion), the ResNet18 boundary critic (upgraded C7), and Expert-2 cavity-aware Timika scoring (upgraded C8) are all implemented and trained, even though plan.md §14 groups them as Priority C (after baseline). README.md explicitly calls this out.

**Why:** The user expanded scope past the baseline once MedSAM ViT-B freed GPU headroom; both the baseline path and MoE path coexist and share C0/C1/C2/C4. CLAUDE.md still says "Components 3, 5, 6 are intentionally not implemented in the baseline" — the baseline stand-in (`baseline_lesion_proposer.py`) is therefore still the intentional fallback and its interface must stay stable.

**How to apply:** Do not suggest *removing* C3/5/6 code to match plan.md. Do not suggest adding stubs either — they exist. When auditing or recommending work, treat both paths as first-class and select by the `moe.enabled` config flag / `pipeline_mode` bundle field. Report Priority C items that are *still* open (e.g. BioGPT report generator faithfulness hardening) separately from C3/5/6, which are no longer Priority C for this project.
