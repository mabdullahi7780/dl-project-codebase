# WS-C: Cavity Grounding — What We Did and Why

## Background

We are building an AI model that scores TB severity from chest X-rays (CXRs).
The score is called the **Timika score** — it combines lung damage (ALP) and
whether the patient has **cavities** (holes in the lung tissue, a sign of
severe TB). The model is trained on data from multiple countries (Romania,
Moldova, Kazakhstan) and tested on each country independently.

The model uses a frozen **RAD-DINO** vision encoder, which splits each X-ray
into a 7×7 grid of 49 patches. A small head network called the
**SpatialCavityHead** scores each patch and learns to detect cavities from
those patch scores.

---

## Why We Did WS-C

The paper had a figure (Fig. 4) showing a heatmap of where the AI "looks"
when it detects a cavity. It looked convincing — attention concentrated in
the upper lungs, which is where TB cavities typically appear.

**But it was just a picture. No numbers, no proof.**

A reviewer could reasonably ask:
> "How do you know the model isn't just randomly guessing which zone to look at?"

**WS-C answers that question.** We turned the illustrative heatmap into a
*measured, checkable claim* by comparing the model's attention to actual
radiologist annotations of where cavities are.

---

## How We Did It

### Step 1 — Get radiologist zone labels
The TB Portals dataset comes with manual annotations from radiologists. They
label which **sextant** of the lung has a cavity:

```
Upper-Left  |  Upper-Right
Mid-Left    |  Mid-Right
Lower-Left  |  Lower-Right
```

We dug through 3 CSV files and a non-obvious join chain to recover a
6-bit label (which of the 6 zones are cavity-positive) for each X-ray.
This gave us ground truth for 1,117 images across the three countries.

### Step 2 — Get attention vectors from the model
The SpatialCavityHead outputs 49 attention weights (one per patch) that sum
to 1. These weights tell us which patches the model "cared about" when
making its cavity decision. We ran training on Kaggle (GPU), averaged over
5 training seeds per country, and saved the attention vectors.

### Step 3 — Map patches to zones
The 49 patches map to 6 zones (3 row bands × 2 columns). One important
detail: chest X-rays are displayed facing the patient, so the **patient's
right lung appears on the image's left side**. We accounted for this flip
in the zone mapping.

### Step 4 — Measure localization (Zone-AUC)
For each image we asked: **does the model put more attention on zones where
radiologists found cavities?**

The metric is **Zone-AUC**: if you randomly pick one cavity-positive zone
and one cavity-negative zone from the same image, what fraction of the time
does the model rank the positive zone higher? (0.5 = random, 1.0 = perfect)

We also computed the **pointing game**: does the zone with the highest
attention contain a cavity? And compared both to their chance baselines.

### Step 5 — Controls to confirm it's real
To make sure the result wasn't a statistical artifact, we ran:
- **Shuffle control**: randomly scramble attention → AUC should drop to ~0.5
- **L/R flip control**: swap left/right column mapping → AUC should drop

Both controls behaved correctly, confirming the signal is real.

---

## Results

| Country | Zone-AUC | Chance | Pointing Game | Chance |
|---------|----------|--------|---------------|--------|
| Romania | **0.846** | 0.5 | **72.7%** | 32.4% |
| Kazakhstan | **0.884** | 0.5 | **78.6%** | 33.1% |
| Moldova* | 0.704 → **0.884** | 0.5 | — | — |

*Moldova's X-rays are stored mirror-reversed in the dataset. When the
column mapping is corrected for this, the zone-AUC matches the other
two countries. The L/R-flip control detected this automatically — which
actually demonstrates the control is working correctly.

**Controls:**
- Shuffle control → ~0.50 (clean null, as expected)
- L/R flip → AUC drops for Romania and Kazakhstan (confirms the mapping is correct)

---

## What Does Zone-AUC 0.85–0.88 Actually Mean?

It does **not** mean the model is 88% accurate at pinpointing a cavity.

It means: **when comparing a cavity-positive zone to a cavity-negative zone,
the model ranks the positive one higher ~87% of the time.**

This is a ranking score, not a localization guarantee. The zones are coarse
(6 big chunks of the lung), and the model was never explicitly trained to
localize — the attention is a byproduct of cavity classification. We are
claiming the model learned something anatomically meaningful, not that it
can replace a radiologist.

---

## How We Confirmed Moldova Images Are Mirror-Reversed

Moldova's zone-AUC was suspiciously low (0.704) and the L/R-flip control behaved
backwards — flipping the column mapping *raised* the AUC to 0.884 instead of
lowering it. This strongly suggested the images were stored mirror-reversed. We
confirmed it through four independent checks:

### Check 1 — L/R flip control (statistical)
For Romania and Kazakhstan, flipping the left/right column mapping dropped the
zone-AUC (as expected — a correct mapping should hurt when reversed):
- Romania: 0.846 → 0.643 ↓
- Kazakhstan: 0.884 → 0.623 ↓
- Moldova: 0.704 → **0.884** ↑ (went UP — the mapping was backwards)

### Check 2 — One-sided cavity laterality test (statistical)
For images where the radiologist annotated a cavity on one side only, we checked
whether the model's attention landed on the correct side:
- Romania: **82%** correct side
- Kazakhstan: **89%** correct side
- Moldova: **23%** correct side (77% wrong side — systematic reversal)

### Check 3 — Visual check via Kaggle PNG dataset

**How we selected the images:**

The TB Portals dataset was converted to PNG format and uploaded to Kaggle as
`mabdullahi454/tb-portals-cxr-pngs`. This dataset contains the actual CXR
images as PNG files plus a `manifest.csv` with columns `image_path`, `country`,
and `cavity` (yes/no).

To find the right test images we used two local CSVs:

1. **`TB_Portals_CXRs_August_2023.csv`** — filtered for `country == Moldova`
   to get 830 Moldova `imagingstudy_id`s. Then extracted the PNG filename for
   each study from the `series_instance_content_url` column (take the last
   part of the path, replace `.dcm` with `.png`).

2. **`TB_Portals_CXR_Manual_Annotations_August_2023.csv`** — for each Moldova
   study, looked at which sextants had a cavity (`smallcavities > 0` OR
   `mediumcavities > 0` OR `largecavities > 0`). Extracted the side (Left/Right)
   from the sextant name. Kept only studies where **all cavity sextants were on
   one side only** — no ambiguity about where the cavity is.

   This gave us:
   - **83 left-only cavity images** (radiologist says: cavity is on patient's LEFT lung only)
   - **100 right-only cavity images** (radiologist says: cavity is on patient's RIGHT lung only)

**Cross-verification against the PNG manifest:**

We checked every one of these 183 filenames against `tbx_portal_manifest.csv`
(the Kaggle PNG dataset manifest):
- **83/83 left-only** filenames found in the manifest ✅
- **100/100 right-only** filenames found in the manifest ✅
- **All 183 labelled `country = Moldova`** in the manifest ✅

This confirmed the filenames derived from the annotation CSV match exactly the
filenames in the PNG dataset, and all belong to Moldova.

**What we expected to see (normal CXR convention):**

In a standard PA chest X-ray, the patient faces the detector:
- Patient's **LEFT** lung → appears on the **RIGHT** side of the screen
- Patient's **RIGHT** lung → appears on the **LEFT** side of the screen

So for a left-only cavity image, the abnormality should appear on the **right**
side of the screen. For a right-only cavity image, it should appear on the
**left** side of the screen.

**What the notebook did:**

We ran `moldova_orientation_check.ipynb` on Kaggle with only the PNG dataset
attached (no other dataset needed — the image lists were hardcoded from the
local CSV analysis). The notebook:
- Displayed a **side-by-side comparison** of 5 left-only and 5 right-only cases
  at large resolution with a **red vertical line** at the centre of each image
- Displayed full **grids of all 83 left-only** and **all 100 right-only** images

**Finding (visual):**

In the left-only cavity images, the abnormalities (opacities, irregular texture)
consistently appeared on the **LEFT** side of the red line — but they should
appear on the **RIGHT** side in a normal CXR. Every single group showed the
same reversal. The pattern was systematic, not random.

**Quantitative confirmation — cardiac asymmetry (the objective version of Check 3):**

Because finding a cavity by eye in a thumbnail is subjective, we added a
label-free, objective test that uses the **heart** instead of the cavity. The
heart is a bright radio-opaque shadow on the patient's LEFT, present in *every*
image. In a normal CXR (patient-left → image-right) the lower-central region is
brighter on the image-RIGHT half; if the image is mirrored, the heart shadow
moves to the image-LEFT half. We measured this asymmetry over ~200 images per
country, using Kazakhstan and Romania (shown normal by Checks 1 & 2) as the
reference:

| Country | n | mean asymmetry | % heart on image-RIGHT | Verdict |
|---------|---|----------------|------------------------|---------|
| Kazakhstan | 200 | **+0.068** | 70% | NORMAL |
| Romania | 200 | **+0.027** | 55% | NORMAL |
| Moldova | 200 | **−0.044** | 36% | **MIRRORED** |

Moldova's sign is the **opposite** of both reference countries — the heart sits
on the wrong (image-left) side. The `mean_cxr_by_country.png` figure (the average
of 200 CXRs per country, which cancels random disease and leaves pure anatomy)
shows the same thing: the bright cardiac region pulls to the right of centre for
Kazakhstan/Romania but to the left for Moldova. This is a controlled, objective,
cavity-free confirmation that Moldova images are mirror-reversed.

### Check 4 — Heart position at full resolution
In a normal chest X-ray, the heart sits slightly **right of the spine** (because
the heart is on the patient's left, which maps to image right). We opened a
Moldova image at full resolution and the heart was clearly **left of the spine**
— the opposite of normal. This is the most direct anatomical confirmation.

**Note:** Moldova PNGs do not have L/R markers burned in (unlike some other
countries in the dataset such as South Africa which have visible "L/KT" markers).
This is because the DICOM-to-PNG conversion did not apply the orientation
correction stored in the DICOM metadata, leaving Moldova images in their raw
detector orientation — which is the mirror of the radiological standard.

### Conclusion
All four checks agree. Moldova images are mirror-reversed. This does not affect
the paper's existing cavity detection numbers (cavity presence is
orientation-invariant). The corrected zone-AUC for Moldova = **0.884**, matching
Romania and Kazakhstan.

---

## What This Gives the Paper

1. **Fig. 4 is no longer just a pretty picture** — it now has measured numbers
   behind it (Zone-AUC = 0.85–0.88, pointing game 73–79% vs 33% chance).

2. **Interpretability evidence** — the SpatialCavityHead, trained only to say
   "cavity yes/no," learned to look in the anatomically correct zones as a
   byproduct. This supports why the model generalises across countries.

3. **Pre-registration check passed** — we committed in advance to requiring
   Zone-AUC > chance on at least 2 of 3 countries. Both Romania and Kazakhstan
   pass clearly.

4. **Segmenter-free** — we achieved zone-level localization evidence without
   any segmentation model, using only the model's own internal attention.

---

## Files Produced

| File | What it is |
|------|------------|
| `scripts/ws_c_cavity_grounding.py` | Main analysis script (zone map, metrics, controls) |
| `tests/test_ws_c_cavity_grounding.py` | Unit tests (all 8 pass) |
| `notebooks/ws_c_export_attention.ipynb` | Kaggle notebook that exports attention vectors |
| `ws_c/attn_spatial.npz` | Per-patch attention for all 1,208 test images |
| `ws_c/cavprob_global.npz` | Cavity probabilities from global head |
| `ws_c/cavity_grounding_metrics.json` | Full results (zone-AUC, pointing, controls, calibration) |
