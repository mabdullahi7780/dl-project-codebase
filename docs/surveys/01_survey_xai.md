# Explainable AI for Medical Imaging — with an Edge-Deployment Lens

## 1. Introduction and Scope

Deep neural networks now match or exceed expert performance on many radiological classification tasks, but their adoption in clinical and—especially—low- and middle-income-country (LMIC) settings is gated less by accuracy than by *trust*: a clinician must be able to interrogate why a model produced an output, know when not to believe it, and operate it on the hardware actually available at the point of care. This survey organises the explainable-AI (XAI) literature for medical imaging around four complementary questions a deployable system must answer: **(i) where did the model look?** (post-hoc attribution and transformer-specific attribution), **(ii) can the model be interpretable by construction?** (prototype and concept-based ante-hoc models), **(iii) is the explanation actually faithful, and does it help a clinician?** (faithfulness, robustness, localization, and reader-study evaluation), and **(iv) should the output be believed at all?** (uncertainty quantification, abstention, and the clinical/regulatory frame).

Throughout, we keep a running thread of relevance to a concrete and increasingly common deployment archetype: a **frozen self-supervised ViT backbone (DINOv2-class, e.g. RAD-DINO [Pérez-García 2024]) with a small trainable head that produces a tuberculosis (TB) severity output, deployed on no-GPU edge hardware**. This archetype sharpens every trade-off in the literature—because the backbone forward pass dominates compute, because gradients into a frozen backbone are largely uninformative, and because the head's learned attention is a candidate "free" explanation whose faithfulness is contested. We use it as a stress-test, not as the subject of the survey.

A note on evidence quality: claims below are drawn only from citations that survived adversarial verification. Where verification refuted or could not confirm a specific number, we say so explicitly and fall back to the confirmed result.

---

## 2. A Taxonomy of Explainability for Medical Imaging

The field does not have a single accepted taxonomy, but four recent surveys converge on overlapping axes. Hou et al.'s *Self-eXplainable AI for Medical Image Analysis* (arXiv 2024, 200+ papers) partitions **self-explainable** (ante-hoc) methods into attention-, concept-, and prototype-based families, plus input- and output-level explainability. The MDPI *Electronics* X-CV review (2025) organises **post-hoc** XAI into attribution-, activation-, perturbation-, and transformer-based families. Synthesising these, we use a two-level taxonomy:

| Axis | Family | Representative methods | When the explanation is produced |
|---|---|---|---|
| **Post-hoc** | Gradient / backprop | Vanilla saliency, Guided Backprop, SmoothGrad, Integrated Gradients, DeepLIFT/DeepSHAP | After training, per prediction |
| | Activation / CAM | Grad-CAM, Grad-CAM++, HiRes-CAM, LayerCAM, XGrad-CAM, Eigen-CAM, Score-CAM, Ablation-CAM | After training, per prediction |
| | Perturbation / black-box | Occlusion, LIME, RISE, KernelSHAP | After training, per prediction |
| | Transformer-specific | Attention rollout/flow, Chefer relevance, AttnLRP, CDAM | After training, per prediction |
| **Ante-hoc (by design)** | Prototype / case-based | ProtoPNet, ProtoTree, Deformable ProtoPNet, PIP-Net, XProtoNet, Proto-BagNets | Intrinsic to architecture |
| | Concept-based | CBM, Post-hoc CBM, Label-free CBM, TCAV | Intrinsic / probe-based |
| | Alignment / attention | B-cos networks, attention-based MIL, attentive probing | Intrinsic to architecture |
| **Orthogonal** | Uncertainty as explanation | Deep ensembles, MC dropout, EDL, temperature scaling, conformal prediction, selective prediction | Wraps any model |

Two cross-cutting distinctions, formalised by Jacovi & Goldberg [Jacovi 2020], are load-bearing for everything below. **Plausibility** is whether an explanation is convincing to a human; **faithfulness** is whether it accurately reflects the model's actual computation. A heatmap that lands on a lesion is plausible but may be wholly unfaithful—produced even by a randomly weighted network [Adebayo 2018]. The two are measured by different metric families, and conflating them is the most common methodological error in medical-imaging XAI.

A second framing, from Rudin's influential position paper [Rudin 2019], argues that for high-stakes decisions one should *stop explaining black boxes* and instead build inherently interpretable models, precisely because post-hoc explanations can be unfaithful and harmful. Ghassemi, Oakden-Rayner & Beam [Ghassemi 2021] extend this to a healthcare-specific critique: current post-hoc saliency offers "false hope," and rigorous validation plus calibrated performance is a more defensible basis for trust than heatmaps. These two papers anchor the central tension of the survey.

---

## 3. Post-hoc Attribution: Gradient, Activation, and Perturbation Families

### 3.1 Gradient and backprop methods

The cheapest attributions are single- or few-backward-pass gradient methods. **SmoothGrad** [Smilkov 2017] averages gradient maps over *N* Gaussian-noised input copies to suppress speckle; it is composable on top of other gradient methods and costs *N×* forward+backward passes. **Integrated Gradients (IG)** [Sundararajan 2017] integrates gradients along a straight path from a baseline to the input. IG is frequently described—including in primary sources—as "the unique attribution satisfying *Sensitivity* and *Implementation Invariance*." This is an **overstatement that must be corrected**: *all* path methods satisfy those two axioms; IG is the unique *symmetry-preserving* path method (the Aumann–Shapley path). The substantive practical points stand: IG defeats **gradient saturation** (where local gradients flatten in a confident regime so vanilla gradients vanish despite the feature mattering), costs ~20–300 gradient steps, and is **baseline-sensitive**—a critical caveat in chest X-ray (CXR), where the conventional zero/black baseline encodes *air*, not *absence of finding*. **DeepLIFT** [Shrikumar 2017] and its Shapley extensions (DeepSHAP, GradientSHAP) approximate reference-difference attributions in one pass but violate implementation invariance.

A decisive negative result governs this family. Adebayo et al. [Adebayo 2018] showed via cascading model-parameter randomization and label randomization that Guided Backprop and Guided Grad-CAM are *invariant* to both the model's weights and the data labels—they behave like edge detectors and explain neither the model nor the task. These "sanity checks" (measured via Spearman rank correlation, SSIM, HOG similarity) are now a mandatory minimal bar; visually crisp maps that fail them are unfit for model debugging or clinical trust.

### 3.2 The CAM family

**Grad-CAM** [Selvaraju 2017] weights last-conv feature maps by globally-average-pooled class gradients to produce a coarse class-discriminative heatmap—architecture-agnostic, training-free, ~0.02 s for one backward pass. It is the de-facto CXR explanation and, as we discuss in §5, the best localiser among saliency methods in the strongest CXR benchmark. Its variants trade faithfulness against locality:

- **Grad-CAM++** [Chattopadhay 2018] uses higher-order gradients for better coverage of multiple object instances.
- **HiRes-CAM** [Draelos 2020] *element-wise* multiplies gradients and activations and is *provably* sensitive only to locations the model used—directly fixing Grad-CAM's gradient-averaging step, which can "smear" importance onto regions the model never used. It is a same-cost, strictly-more-faithful drop-in.
- **LayerCAM** fuses shallow+deep layers for finer maps; **XGrad-CAM** adds axiomatic weighting; **Eigen-CAM** takes the first principal component of activations (gradient-free, fast, but *class-insensitive*).
- **Score-CAM** [Wang 2020] and **Ablation-CAM** are gradient-free but require a forward pass *per channel*. Score-CAM costs ~3.4 s vs Grad-CAM's ~0.02 s—a ~150× penalty (benchmarked at 3.378 s vs 0.023 s), which **disqualifies it on heavy backbones at the edge**.

### 3.3 Perturbation / black-box methods

**Occlusion** [Zeiler 2014] slides a mask and records score drop; **LIME** [Ribeiro 2016] fits a sparse linear surrogate over superpixels; **RISE** [Petsiuk 2018] probes with thousands of random masks (4,000 for VGG16, 8,000 for ResNet50, confirmed) and weights them by output. RISE is model-agnostic and often *more faithful* than white-box Grad-CAM, but is the most edge-hostile method by orders of magnitude. **SHAP/KernelSHAP/DeepSHAP** [Lundberg 2017] unify LIME, DeepLIFT, and Shapley values; the framework unifies six prior methods and identifies a unique solution satisfying local accuracy, missingness, and consistency, with DeepSHAP/GradientSHAP as faster deep approximations.

### 3.4 Edge-cost ordering

The practical hierarchy for no-GPU deployment is clear. **Cheap (single backward/forward):** vanilla gradients, Grad-CAM, Grad-CAM++, HiRes-CAM, XGrad-CAM, LayerCAM, Eigen-CAM. **Moderate (N×):** SmoothGrad, IG. **Edge-hostile (per-channel or thousands of forwards):** Score-CAM, Ablation-CAM, RISE, KernelSHAP, occlusion. For a frozen-ViT TB head, the dominant cost is *always* the backbone forward pass; any method requiring repeated full forwards multiplies the single most expensive operation. This single fact rules out the entire perturbation family and the gradient-free CAMs for on-device explanation, while leaving them available for *offline* validation.

---

## 4. Attention-Based and Transformer-Specific Explainability

Because the project archetype is a ViT whose head is an attention pool, transformer-specific attribution is the most load-bearing literature. It splits into three lineages plus a prior question.

### 4.1 The prior question: is attention an explanation?

The foundational NLP debate transfers directly to vision attention. Jain & Wallace [Jain 2019] showed attention weights correlate weakly with gradient/leave-one-out importance and that *counterfactual* attention distributions can yield essentially unchanged predictions—so attention is not inherently faithful. Wiegreffe & Pinter [Wiegreffe 2019] rebutted that the conclusion is definition-dependent: with the right diagnostic baselines (uniform-weight, seed-variance, frozen-attention, and properly-trained adversarial baselines) attention retains explanatory value. The modern ViT consensus echoes Wiegreffe—raw attention *alone* is insufficient, but attention *plus* gradients/aggregation can be faithful. Liu et al. [Liu 2022] sharpened the warning with a **Faithfulness Violation Test (FVT)**: high-attention features can exert *suppressive* (negative) effects on the output, and raw attention was among the worst offenders for polarity consistency.

### 4.2 Raw attention and aggregation

**Raw last-layer [CLS]→patch attention** is zero-cost and built-in but unreliable: across layers, token information mixes, so a single layer's attention does not reflect input contribution; it is class-agnostic and ignores the sign of contribution. **Attention rollout** [Abnar 2020] recursively multiplies per-layer attention matrices, each blended with identity to model the residual stream (A = 0.5·W + 0.5·I), then renormalises; **attention flow** treats the stacked attention as a max-flow graph. Both correlate better with input gradients and ablation importance than raw attention, and rollout is the de-facto training-free ViT baseline—but both average heads and remain class-agnostic.

### 4.3 Attention + gradient / LRP attribution (the faithful SOTA family)

- **Chefer relevance** [Chefer 2021, CVPR] propagates class-specific relevance through attention layers and skip connections via a Deep-Taylor-Decomposition LRP formulation that conserves total relevance, weighting attention by the target-class gradient. It produces class-discriminative maps that beat raw attention and rollout on segmentation and perturbation benchmarks. The follow-up [Chefer 2021, ICCV Oral] generalises to bi-modal/co-attention and encoder-decoder transformers (DETR, VQA, CLIP), and an official PyTorch implementation exists.
- **AttnLRP** [Achtibat 2024] derives faithful LRP rules for the non-linear attention operations (softmax, matmul), attributing both inputs and latent neurons at roughly one-backward-pass cost, surpassing Chefer and gradient baselines on LLaMa/Mixtral/Flan-T5/ViT.
- **CDAM** [Brocki 2024] scales attention by gradients of a class *or latent-concept* similarity score with respect to final-block token activations, producing *signed, class-specific* maps that beat seven estimators (including rollout) on correctness, compactness, and class-sensitivity.
- **FViT** [Hu 2024] addresses fragility: ViT attention explanations are unstable under perturbation, and Denoised Diffusion Smoothing keeps top-k attention and the prediction distribution stable—at a cost that is *offline-only* for edge.

Critically, Wu et al. [Wu 2024] introduced the **Salience-guided Faithfulness Coefficient (SaCo)** and showed that many existing metrics cannot distinguish advanced ViT explanation methods from *Random Attribution*, and that gradient + multi-layer aggregation markedly improves faithfulness—confirming the field's direction and the inadequacy of raw attention.

### 4.4 Emergent semantic attention in self-supervised ViTs

Caron et al. [Caron 2021] showed that the [CLS]-token last-layer self-attention of a self-supervised ViT (DINO) segments objects with no segmentation supervision—an emergent property absent in supervised ViTs and convnets (78.3% kNN with ViT-S/8 and 80.1% linear with ViT-B/8 — two different architectures, not one model's paired result; this is DINO v1, a separate lineage from DINOv2/RAD-DINO). This is *why* a frozen DINOv2-class encoder yields semantically meaningful attention at all, and is the strongest justification for treating such attention as a built-in explanation substrate. RAD-DINO [Pérez-García 2024]—an image-only DINOv2 ViT-B continually pretrained on 838,336 CXRs (~838k; the paper's figure, which includes ~90k private data), whereas the public checkpoint is trained on 882,775 images (~883k) from five public datasets—inherits this property.

But Darcet et al. [Darcet 2024] found that DINOv2-class attention maps are corrupted by ~2% high-norm "artifact" tokens (~10× normal norm) in low-information background regions, which the model repurposes for global computation; adding learnable **register tokens** removes them, yielding smoother, more interpretable attention and SOTA dense prediction. This is a **direct caveat for any RAD-DINO-derived encoder**: the patch grid may contain artifact tokens that leak into a pooled attention map and produce spurious highlights unless a registers-equipped checkpoint or a training-free high-norm-token suppression is used.

### 4.5 Attention pooling as explanation

Learned attention-pooling heads—attentive probing [Psomas 2026] and attention-based MIL [Ilse 2018]—compute a softmax-weighted convex combination of token/instance embeddings; the pooling weights double as a localisation map. Attentive probing outperforms linear/CLS probing on frozen features and yields complementary interpretable localisation, but the weights are correlational, not causal, and inherit every faithfulness caveat above.

> **Relevance to the frozen-ViT TB model.** Because the backbone is frozen, gradient-into-backbone CAM methods are largely moot—gradients only flow into the trivial head. The head's learned softmax over its pooled patch tokens *is* an intrinsic, zero-cost attention map (an instance of attention-based MIL [Ilse 2018]), and the cheapest possible explanation on edge. Caron [Caron 2021] and RAD-DINO [Pérez-García 2024] justify why it is semantically meaningful. But the design should (a) upgrade from class-agnostic raw attention to a *class-specific, signed* map via CDAM/Chefer—one backward pass through the tiny head, trivially cheap on a small token grid even on CPU; (b) inspect patch-token norms for Darcet-style artifacts before pooling; and (c) reserve heavy faithful methods (FViT diffusion smoothing, IG noise averaging) for offline validation only.

---

## 5. Inherently Interpretable and Concept-Based Models

Ante-hoc interpretability is the direct response to Rudin [Rudin 2019]. The recurring empirical lesson across this section is that **interpretability-by-design need not cost accuracy**, but it can cost faithfulness, training complexity, and inference efficiency.

### 5.1 Prototype / case-based networks

**ProtoPNet** ("This Looks Like That") [Chen 2019] inserts a prototype layer that classifies by comparing image patches to a bank of learned class-specific prototypical parts, summing evidence linearly; it matches its non-interpretable counterpart on fine-grained tasks (CUB-200, Stanford Cars). The family then diversified:

- **ProtoTree** [Nauta 2021a] routes an image through a soft binary decision tree of prototypes—globally faithful, human-followable, ~8 prototypes per path after pruning.
- **Deformable ProtoPNet** [Donnelly 2022] uses adaptively-positioned prototypical parts to capture pose/context, improving accuracy and explanation richness—relevant where lesion position varies (e.g. upper-lobe cavities).
- **PIP-Net** [Nauta 2023] learns self-supervised prototypes with a *sparse* linear scoring sheet, narrowing the latent–pixel semantic gap and abstaining when no prototype matches; sparsity aids edge deployment.
- **ProtoPNeXt** [Willard 2024] is a unified hyperparameter-search framework producing stronger ProtoPNet variants.

In radiology, **XProtoNet** [Kim 2021] is the landmark CXR adaptation: rather than fixed-size patches, it learns a per-prototype *occurrence module* that predicts *where* a disease sign appears, then compares features in that region against disease prototypes, giving global + local explanations and SOTA on NIH ChestX-ray14 (112,120 images, 14 multi-label pathologies). **Proto-BagNets** [Djoumessi 2024] combine BagNets with prototypes so the receptive-field-limited evidence map is faithful by construction—the visualised evidence is (provably, by the BagNet additive design) the evidence used—validated on retinal OCT. A TB-relevant prototype method, **CXR-NeXus** [Chiang 2026], couples per-class prototype memory with counterfactual consistency on a ResNet-50 backbone for COVID/pneumonia/**tuberculosis**/normal, reporting better calibration under perturbation (~74% accuracy at the most severe corruption vs ≤67% for baselines). *Caveat: CXR-NeXus is published in Cureus, a low-bar pay-to-publish, single-author venue—cite it as a TB-relevant exemplar, not as strong evidence.*

The principal faithfulness caveat is Hoffmann et al. [Hoffmann 2021]: ProtoPNet latent-space similarity does *not* guarantee input-space similarity, and JPEG/adversarial perturbations can corrupt its "this looks like that" explanations. The 2025 part-prototype survey [Elhadri 2025] catalogues spatial misalignment, background bias, slower inference, complex multi-stage training, non-standardised evaluation, and the fact that most methods are untested on high-resolution/low-data medical regimes.

### 5.2 Concept-based models

**Concept Bottleneck Models (CBM)** [Koh 2020] force the network to predict human-specified concepts first, then predict the label *only* from those concepts—enabling *test-time intervention* (correcting a concept fixes the prediction), validated on knee-osteoarthritis X-ray grading with clinical concepts. Vanilla CBMs need dense concept annotations and often lose accuracy; two relaxations remove the annotation burden: **Post-hoc CBM (PCBM)** [Yuksekgonul 2023] converts *any* pretrained backbone into a CBM via concept activation vectors and borrowed/textual concepts, preserving accuracy and enabling concept-level editing without target-domain retraining; **Label-free CBM** [Oikarinen 2023] auto-generates the concept set via LLM prompting, grounds it with CLIP, and scales to ImageNet with no concept labels. The conceptual ancestor is **TCAV** [Kim 2018], which learns a linear concept activation vector in a hidden layer and uses directional derivatives to quantify a concept's influence on a class—operating on frozen representations. **Clinical-knowledge-guided CBMs** [Pang 2024] align concept importance with clinician priorities to improve out-of-domain robustness (validated on white-blood-cell and skin images, *not* CXR—so cross-country CXR relevance is an extrapolation). CBMs face concept incompleteness, leakage, and CLIP concept hallucination.

### 5.3 Alignment- and attention-based interpretability

**B-cos networks** [Böhle 2022] replace linear layers with a weight-input-aligned B-cos transform, making the entire model summarisable by a single input-dependent linear map—faithful by design at accuracy parity. **B-cosification** [Arya 2024] converts a *pretrained* model (including CLIP/ViT) into a B-cos model by fine-tuning, at 4.7–9× less compute than from-scratch (4.7× for DenseNet-121, 9.0× for ViT-S, confirmed). Crucially, B-cosification requires *modifying/fine-tuning the backbone*, which conflicts with a fully-frozen-encoder constraint—note it as a trade-off, not a drop-in.

**Attention-based deep MIL** [Ilse 2018] provides permutation-invariant learned-softmax (gated) attention pooling whose weights directly expose each instance's contribution—the dominant interpretable paradigm in weakly-supervised CXR/histopathology, and essentially the project archetype's head.

> **Relevance to the frozen-ViT TB model.** The project's head *is* attention-based MIL [Ilse 2018]; frame the localisation map as MIL instance-attribution and place it in Hou et al.'s [Hou 2024] "attention-based self-explainable" branch. XProtoNet [Kim 2021] and CXR-NeXus [Chiang 2026] are the closest interpretable-CXR related work; XProtoNet's occurrence module is conceptually similar to an attention map (predicting *where*) but heavier (full backbone + prototype bank). The edge advantage is real: Elhadri [Elhadri 2025] flags prototype models as slow, multi-stage, and untested at high resolution—exactly the costs a cheap attention/sparse-scoring head avoids. A concept upgrade path on the *frozen* encoder (PCBM [Yuksekgonul 2023], label-free CBM [Oikarinen 2023], TCAV [Kim 2018]) would make severity reasoning concept-grounded (cavity, consolidation, fibrosis, upper-lobe predominance) without dense annotations—a strong "future work." But faithfulness (Hoffmann [Hoffmann 2021]) is the main exposure: any interpretable head must be validated for input-space faithfulness, not just plausibility.

---

## 6. Evaluating Explanations: Faithfulness, Robustness, Localization, and Reader Studies

If attention or saliency is the explanation, the central contribution of a rigorous paper is no longer the heatmap but its *validation*. Jacovi & Goldberg [Jacovi 2020] supply the framing: plausibility and faithfulness must be measured separately.

### 6.1 Faithfulness / fidelity

The dominant paradigm is *perturbation-based*. RISE [Petsiuk 2018] introduced **Deletion AUC** (lower is better) and **Insertion AUC** (higher is better); the NLP analogue is ERASER's **comprehensiveness** (confidence drop when the rationale is removed) and **sufficiency** (confidence retained when only the rationale is kept), aggregated as AOPC [DeYoung 2020]. The naive version is confounded: masking creates out-of-distribution inputs, conflating explanation quality with distribution shift. **ROAR** [Hooker 2019] fixed this by retraining on perturbed data—at prohibitive cost. **ROAD** [Rong 2022] showed that perturbation curves leak class information through the *shape* of the removed mask; its noisy-linear-imputation removal makes MoRF and LeRF orderings consistent and is up to 99% cheaper than ROAR. ROAD-style imputation (not zeroing) is the preferred protocol.

### 6.2 Sanity checks and robustness

Adebayo et al.'s model-parameter-randomization test (MPRT) and data-randomization test [Adebayo 2018] are the minimal faithfulness bar. Robustness adds two findings: Ghorbani et al. [Ghorbani 2019] showed imperceptible, prediction-preserving perturbations can completely relocate saliency (gradients, DeepLIFT, IG), with a Hessian-geometry explanation for why fragility is generic; Heo et al. [Heo 2019] escalated to *adversarial model manipulation*—accuracy-preserving fine-tuning that radically changes Grad-CAM/LRP/SimpleGrad heatmaps, transferring across methods. Together these mandate repeatability across seeds/retrains and stability under small perturbations.

### 6.3 Localization against ground truth

The **Pointing Game** (max-saliency point inside the GT region = hit) is coarse; the **Energy-Based Pointing Game (EBPG)** [Wang 2020] computes the fraction of total saliency energy inside the GT region—threshold-free and more discriminative. **IoU/Dice** of a thresholded map vs a lesion mask rounds out the toolkit but is strongly threshold-dependent (report a sweep). **Quantus** [Hedström 2023] operationalises 35+ metrics across six axes (faithfulness, robustness, localization, complexity, randomization, axiomatic) as a single reproducible harness.

### 6.4 Medical-specific evidence: two sobering CXR benchmarks

Two landmark CXR studies are negative results for post-hoc saliency and define the bar.

**CheXlocalize** [Saporta 2022] benchmarked seven saliency methods (Grad-CAM, Grad-CAM++, Integrated Gradients, etc.) on CheXpert against board-certified expert segmentations. **Grad-CAM localised pathologies best of the saliency methods, but all methods significantly underperformed a human-radiologist benchmark** on mIoU and the looser hit rate; the gap was largest for small, complex, multi-instance findings, and localization correlated with model confidence (overconfidence risk). Notably, even expert-vs-expert mIoU was modest (highest ~0.72, for cardiomegaly), so overlap metrics conflate model error with annotation noise.

**Arun et al.** [Arun 2021] tested eight methods on SIIM-ACR pneumothorax (10,675 images) and RSNA pneumonia (14,863 images) against four criteria—localization utility, sensitivity to weight randomization, repeatability across retrains, reproducibility across architectures—and found *every* method failed at least one; none matched a dedicated U-Net/RetinaNet, and the explicit recommendation was to use a segmentation/detection model when localization is the goal. (The narrow sub-claim that "only XRAI passed utility" is reported by the paper but could not be word-for-word re-verified from the source PDF; the overall conclusion is robust.) The musculoskeletal follow-up (2024) reinforced the pattern outside the chest.

It is worth flagging a verification correction here: a 2024 *European Journal of Radiology* mammogram study is sometimes cited as showing "Grad-CAM and RISE performed well" on insertion/deletion/pointing-game metrics. **This is refuted.** That study evaluated only Grad-CAM, Grad-CAM++, and Eigen-CAM (RISE was not tested), used the Pointing Game as its primary metric, and reported *low* scores (Grad-CAM 0.41, Eigen-CAM 0.35, Grad-CAM++ 0.30), concluding the methods "frequently fall short." We therefore do *not* cite it as positive evidence for saliency faithfulness; it is, if anything, additional modality-adjacent evidence of saliency's weakness.

### 6.5 Attention faithfulness in CXR specifically

For ViT attention on CXR, the evidence is mixed and methodologically shallow. Wollek et al. [Wollek 2023] found radiologists rated ViT attention-based saliency useful in **47% of pneumothorax cases vs 39% for Grad-CAM**, with ViTs matching CNN AUC (0.84–0.95). Chung et al. [Chung 2024] found attention maps generally surpass Grad-CAM but lose to transformer-specific methods, with efficacy context-dependent. TB-specific transformer-XAI work overwhelmingly defaults to Grad-CAM heatmaps with mIoU against bounding boxes and rarely applies faithfulness tests (perturbation AUC, SaCo, FVT) or transformer-native attribution—a clear quality gap a rigorous TB paper can exceed.

### 6.6 Clinical reader / utility studies

The end goal is clinician benefit. Ahn et al. [Ahn 2022], a 6-reader randomized crossover on 497 CXRs with heatmap/contour overlays, showed AI assistance raised sensitivity (pneumothorax AUROC 0.885→0.969; nodules 0.724→0.752) without specificity loss and cut reporting time ~10% (40.8→36.9 s)—all figures verified. But cognitive-bias reader studies temper this: a 2026 *European Radiology* mammography simulation [Eur Radiol 2026] found saliency-based XAI **roughly halved but did not eliminate** automation bias (36.1%→~17–18%) and anchoring bias (33.9%→~17–18%), with less-experienced readers most vulnerable (junior automation bias 41.7% vs senior 30.0%, cut to 20.0% with XAI). *(A previously circulated figure that accuracy "stayed ~9 points below baseline under biased AI even with the explanation" could not be verified and appears to be fabricated precision; the confirmed result is that ~1 in 5 manipulated cases remained biased.)* The lesson: providing a heatmap is *necessary but not sufficient* to prevent over-reliance—especially for non-expert LMIC operators.

> **Relevance to the frozen-ViT TB model.** This literature converts a qualitative attention figure into a publishable, validated localizer. Concretely: (1) frame contributions with Jacovi–Goldberg [Jacovi 2020]—claim *faithfulness*, not just upper-lobe concentration; (2) for coarse zone/sextant cavity annotations, **EBPG** [Wang 2020] is the natural primary metric (fraction of attention energy inside the GT zone), with Pointing Game and an IoU/Dice threshold sweep as complements; (3) run **MPRT** [Adebayo 2018]—randomize the head (and last block) weights and show the map degrades, cheap because the heads are trivial; (4) adapt **deletion/insertion or comprehensiveness/sufficiency** [Petsiuk 2018; DeYoung 2020] to *tokens*—ablate top-k attended tokens with ROAD-style neighbour-mean imputation [Rong 2022] and measure logit drop, cheap because only the head re-evaluates while the expensive ViT forward is cached; (5) report **attention-map repeatability across ensemble members** as a near-free Wiegreffe–Pinter-style seed-variance test [Wiegreffe 2019], answering the Ghorbani/Heo/Arun fragility critiques; (6) run the whole battery in **Quantus** [Hedström 2023]; (7) benchmark against the human ceiling [Saporta 2022; Arun 2021] and frame the map as decision support, not a localizer of record, citing Ahn [Ahn 2022] and the bias studies for the deployment caveat.

---

## 7. Uncertainty Quantification as Explanation, and the Clinical/Regulatory Frame

A growing position—articulated by Ghassemi et al. [Ghassemi 2021]—is that calibrated uncertainty is a *more actionable* form of explanation than saliency: instead of telling a clinician *where* the model looked, it tells them *whether to believe the output and when to defer*.

### 7.1 Bayesian approximations, ensembles, and evidential UQ

**Deep ensembles** [Lakshminarayanan 2017]—M independently initialised networks—are embarrassingly parallel, better-calibrated, and better at flagging OOD inputs, and are the de-facto strong baseline. **MC dropout** [Gal 2016] casts test-time dropout as approximate variational inference in a deep Gaussian process, giving epistemic uncertainty from T stochastic forward passes at no extra training cost, though its calibration is often inferior to ensembles. **Evidential deep learning (EDL)** [Sensoy 2018] places a Dirichlet over class probabilities to output uncertainty from a *single* forward pass—attractive for edge compute, though its aleatoric/epistemic decomposition is a slightly generous gloss on a method foregrounding OOD/adversarial robustness. The JMIR Medical Informatics systematic review [Kurz 2022] (22 studies) found MC dropout and deep ensembles dominate the medical-imaging literature, that combining methods helps, and that "discarding uncertain predictions leads to improved accuracy on the remaining samples"—but warned that nearly all studies were run in "very artificial settings" with little real human–AI collaboration.

### 7.2 Calibration

Guo et al. [Guo 2017] documented that modern deep nets are systematically *overconfident*, popularised **Expected Calibration Error (ECE)**, and showed a single-parameter **temperature scaling** is a remarkably effective, accuracy-preserving post-hoc fix. Calibration is necessary but not sufficient: a calibrated probability is a marginal statement, not a per-prediction guarantee.

### 7.3 Conformal prediction (CP)

CP supplies the missing finite-sample, distribution-free guarantee, wrapping *any* pretrained model [Angelopoulos 2023]. **Split CP** returns a prediction set/interval with coverage ≥1−α under exchangeability. Variants matter:

- **Conformalized Quantile Regression (CQR)** [Romano 2019, NeurIPS] produces adaptive, heteroscedastic interval widths—directly relevant to a continuous severity target.
- **Jackknife+** [Barber 2021, AoS] reuses all data for fit and calibration with a worst-case 1−2α guarantee—useful when per-fold calibration data is scarce.
- **Mondrian / group-conditional CP** gives per-group coverage (per country, per severity stratum)—a fairness/heterogeneity audit.
- CP's exchangeability assumption is broken by distribution shift: **Weighted CP under covariate shift** [Tibshirani 2019, NeurIPS] restores coverage via likelihood-ratio weights, and **CP beyond exchangeability** [Barber 2023, AoS] handles general non-exchangeable data with bounded coverage gaps.

### 7.4 Selective prediction / abstention

**SelectiveNet** [Geifman 2019] jointly trains predictor + selector to a target coverage, advancing the risk–coverage frontier. Recent clinical work fuses CP with **cost-aware deferral**: a 2026 *Scientific Reports* framework [Sci Rep 2026] combining calibrated probabilities, importance-weighted split CP, and cost-aware deferral reported error reductions on retained cases of ~49.6% in-distribution and ~46.7% OOD on early-sepsis prediction. *(This and the catheter-CP preprint below are recent/limited-review, self-reported numbers—hedge if load-bearing.)*

### 7.5 Medical / CXR-specific UQ

For CXR specifically: risk-sensitive CP for catheter-placement detection [Long Hui 2025] reports 90.68% overall and 99.29% coverage on critical conditions with zero high-risk mispredictions (single-author arXiv preprint). The closest TB-CXR precedent is **Rajaraman et al.** [Rajaraman 2022]: a VGG16-based U-Net with MC dropout segments TB-consistent findings on frontal CXRs and uses an uncertainty threshold to refer ambiguous cases to experts—directly paralleling an uncertainty-driven TB severity/abstention loop. *(Author-list correction: the correct authors are Rajaraman, Zamzmi, Yang, Xue, Jaeger, Antani; "Les R. Folio" is not an author of this paper.)*

### 7.6 Regulatory frame

What regulators demand converges less on heatmaps and more on *calibrated confidence, known operating limits, graceful abstention, and monitoring*. The **FDA** frames AI/ML as Software as a Medical Device (SaMD) under a Total-Product-Lifecycle view. Two distinct documents are often conflated and must be kept separate: (i) the **joint FDA / Health Canada / MHRA "Predetermined Change Control Plans … Guiding Principles"** (Oct 2023), a high-level harmonisation document; and (ii) the **FDA-only final guidance "Marketing Submission Recommendations for a Predetermined Change Control Plan for AI-Enabled Device Software Functions"** (3 Dec 2024), which codifies the operative **PCCP**—a pre-authorised description of planned model changes, methodology, and impact assessment that allows updates (e.g. recalibration, new-country folds) without a new marketing submission. The **EU AI Act** (in force Aug 2024; high-risk obligations from Aug 2026) classes most AI medical devices as high-risk, layering transparency, human oversight, data governance, risk management, and post-market monitoring on top of the **EU MDR** (Article 61 clinical evaluation; MDR does not mandate "explainable AI" but auditors expect demonstrable understanding of model logic). **GDPR** Articles 13–15 ("meaningful information about the logic involved") and Article 22 (limits on solely-automated decisions) underpin the contested "right to explanation."

> **Relevance to the frozen-ViT TB model.** A frozen encoder with cheap heads is the ideal substrate for the entire UQ toolkit, because every method here is post-hoc or head-level. The single most important architectural point for no-GPU deployment: **ensemble only the heads, not the backbone.** Run the ViT forward once and ensemble the M cheap heads [Lakshminarayanan 2017]; do not re-run the backbone M times. For a continuous severity target, prefer CQR [Romano 2019] over constant-width CP; for a composite score mixing a regression term and a thresholded binary term, conformalise the regression component and handle the binary term via the calibrated/conformal classifier (an explicit gap). Because cross-country deployment is covariate (and label) shift, pair split CP with weighted/non-exchangeable CP [Tibshirani 2019; Barber 2023] and report per-country (Mondrian) coverage—with confidence bands given small per-country negative counts (echoing the known high-variance-AUROC warning on small validation sets). Apply temperature scaling and report ECE [Guo 2017] before thresholding. Add an abstain/defer mode [Geifman 2019; Sci Rep 2026], routing high-uncertainty CXRs to a human—exactly what the TB-CXR precedent [Rajaraman 2022] and the JMIR review [Kurz 2022] endorse, and the most credible LMIC deployment story. Frame uncertainty (per Ghassemi [Ghassemi 2021]) as a *more defensible* trust signal than saliency, while shipping the built-in attention map as a complementary, model-faithful explanation. Position the system under FDA SaMD with a PCCP (Dec 2024 final guidance) covering recalibration; map abstention onto EU AI Act human-oversight and GDPR Art. 22; and disclose that RAD-DINO's own model card states it is research-only, not for clinical use.

---

## 8. Synthesis: Consensus, Debates, and the Edge Trade-off

**Points of consensus across the four sub-literatures.**
1. *Plausibility ≠ faithfulness*, and the latter must be measured, not assumed [Jacovi 2020; Adebayo 2018; Saporta 2022; Arun 2021].
2. *Post-hoc saliency under-localises in CXR.* Even the best saliency method (Grad-CAM) trails radiologists, worst on small/complex/multi-instance lesions—precisely TB cavities and infiltrates [Saporta 2022; Arun 2021].
3. *Raw attention alone is insufficient*; gradient/aggregation/LRP augmentation markedly improves faithfulness [Liu 2022; Wu 2024; Abnar 2020; Chefer 2021].
4. *Interpretability-by-design need not cost accuracy* [Chen 2019; Kim 2021; Böhle 2022].
5. *Uncertainty + abstention is what clinicians and regulators actually want*, and is more defensible than heatmaps [Ghassemi 2021; Kurz 2022; FDA/EU/GDPR].

**Live debates.**
- *Is attention explanation?* [Jain 2019] vs [Wiegreffe 2019] remains unresolved; the modern synthesis is "not alone, but yes with the right tests/augmentation."
- *Post-hoc vs ante-hoc.* Rudin [Rudin 2019] and Ghassemi [Ghassemi 2021] argue for inherently interpretable models; the prototype/CBM literature shows this is feasible at accuracy parity but with faithfulness [Hoffmann 2021] and efficiency [Elhadri 2025] costs.
- *Overlap metrics vs causal tests.* With expert-vs-expert mIoU only ~0.72 [Saporta 2022], localization-overlap metrics conflate model error with annotation noise, pushing the field toward perturbation/causal faithfulness.

**The edge trade-off, summarised.** The faithfulness–latency Pareto frontier is the organising constraint for edge deployment. Faithful methods (Chefer, AttnLRP, CDAM, FViT, RISE, Score-CAM) need gradients, smoothing, or many forward passes; the cheapest built-in attention is the least faithful. The frozen-ViT-plus-cheap-head archetype resolves this elegantly: the backbone forward is the only unavoidable cost, and the built-in attention map, a one-backward-pass class-specific upgrade (CDAM/Chefer), head-only ensembling, temperature scaling, and conformal intervals all cost essentially nothing on top of it—while heavy faithful methods are confined to offline validation.

| Explanation / UQ approach | Edge inference cost (frozen ViT) | Faithfulness evidence | Class-specific? | Best CXR use |
|---|---|---|---|---|
| Raw [CLS]/pool attention | Free (built-in) | Weak alone [Liu 2022; Jain 2019] | No | Cheap primary map, must validate |
| Attention rollout | 1 forward | Better than raw [Abnar 2020] | No | Training-free ViT baseline |
| CDAM / Chefer relevance | +1 backward (head) | Strong, class-discriminative [Brocki 2024; Chefer 2021] | Yes | Cheap signed upgrade on frozen feats |
| Grad-CAM (on patch grid) | +1 backward | Best saliency in CheXlocalize but sub-human [Saporta 2022] | Yes | CNN baseline / corroboration |
| HiRes-CAM | +1 backward | Provably faithful drop-in [Draelos 2020] | Yes | Faithful CAM where needed |
| IG / SmoothGrad | N× backward | Axiomatic; baseline-sensitive [Sundararajan 2017] | Yes | Offline; CXR baseline ≠ black |
| Score-CAM / Ablation-CAM | per-channel forwards (~150×) | Faithful but edge-hostile [Wang 2020] | Yes | Offline only |
| RISE / KernelSHAP / occlusion | 4k–8k forwards | Often most faithful, most expensive [Petsiuk 2018] | Yes | Offline validation only |
| Prototype head (ProtoPNet/XProtoNet) | Backbone + prototype bank | Plausibility ≠ faithfulness [Hoffmann 2021] | Yes | Heavier ante-hoc alternative |
| Deep ensemble (heads only) | Free if backbone cached [Lakshminarayanan 2017] | Strong UQ/OOD | n/a | Epistemic uncertainty + repeatability |
| MC dropout | T× forwards | Calibration < ensembles [Gal 2016] | n/a | Single-model UQ |
| EDL | 1 forward | Single-pass UQ [Sensoy 2018] | n/a | Edge-friendly UQ |
| Conformal prediction (split/CQR) | Negligible post-hoc [Angelopoulos 2023; Romano 2019] | Distribution-free coverage | n/a | Defensible trust signal + defer |

---

## 9. Open Problems & Research Gaps

1. **Faithfulness of learned attention-pooling weights on frozen features.** The faithfulness literature (FVT [Liu 2022], SaCo [Wu 2024], perturbation) targets raw/rollout/LRP maps over the *backbone*. Almost nobody has rigorously tested whether the softmax weights of a *trained downstream attention-pool head* on *frozen* features are faithful—exactly the project archetype's setting.

2. **A faithful, cheap explainer for frozen self-supervised ViTs.** When the classifier is a tiny head on frozen DINOv2/RAD-DINO features, gradients into the backbone are uninformative; Chefer/rollout/AttnLRP/ReciproCAM disagree, and no consensus method exists for this specific regime.

3. **Class-agnostic vs class-specific medical attention.** DINO/DINOv2 attention segments the *salient object*, not the *target pathology*; a single cavity cannot be assumed most salient. Bridging emergent self-supervised attention to disease-specific localization without retraining is open.

4. **Artifact tokens in medical DINOv2 encoders.** Whether RAD-DINO-class encoders exhibit Darcet-style high-norm artifact tokens [Darcet 2024] on CXR, and whether training-free mitigations work on frozen medical encoders, is essentially unstudied.

5. **Explaining regression and composite clinical scores.** Almost all attribution and ante-hoc interpretability targets classification logits. Attributing a continuous severity output, or a composite score mixing a regression term and a thresholded binary term, is under-explored—including which target to backprop and how to compose heterogeneous explanations into one coherent overlay.

6. **Edge/on-device XAI cost.** Little work measures attribution/UQ latency and memory on Core ML / TFLite / ONNX-Runtime-Web / WASM. A systematic on-device faithfulness-vs-latency Pareto frontier for CXR does not exist, and single-pass UQ (EDL, last-layer Laplace) is under-validated on CXR foundation features.

7. **Baseline choice for IG/SHAP in radiology.** Black = air, not absence-of-finding; principled CXR baselines (lung-field mean, contralateral mirror) are unstudied.

8. **Conditional coverage under real (LOCO) shift.** Split CP only guarantees marginal coverage; patient-/group-conditional coverage under simultaneous covariate *and* label shift is unsolved, and weighted CP needs an estimable likelihood ratio that is hard to obtain in LMIC settings.

9. **Conformalising hybrid clinical targets.** Conformal inference for a composite score combining continuous regression and thresholded binary terms, while preserving clinically meaningful interval semantics, is largely untouched.

10. **Standardised localization for coarse (zone/sextant) annotations.** TB zone-level labels sit between bounding boxes and segmentation masks; localization metrics for this granularity are not standardized, and faithfulness vs localization are routinely conflated (a map can pass EBPG yet fail MPRT).

11. **Real human–AI collaboration evaluation.** The JMIR review's [Kurz 2022] central gap: almost no study measures whether uncertainty-driven deferral actually improves clinician decisions/outcomes in deployment—especially in low-resource clinics where deferred cases lack specialist backup. Reader studies overwhelmingly use Grad-CAM overlays; whether built-in attention or conformal intervals change reader trust/over-reliance differently is open, and saliency XAI reduces but does not eliminate automation/anchoring bias [Eur Radiol 2026; Ahn 2022].

12. **Calibration drift, monitoring, and regulatory mapping.** Temperature-scaling/CP calibration degrade silently under shift/time; lightweight on-device recalibration and drift detection that fit a PCCP are under-developed, and there is no standardised way to map conformal sets/abstention rates/ECE onto MDR/AI-Act/GDPR explanation obligations.

13. **Joint uncertainty + saliency as one explanation.** No established protocol exists for jointly presenting "where" (attention) and "how sure" (conformal) to clinicians, nor for whether overconfidence-correlated saliency [Saporta 2022] misleads when shown alongside intervals.

14. **Group-conditional fairness of abstention.** Selective prediction can systematically defer on minority subgroups/countries—an under-audited harm.

---

## References

1. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., Batra, D. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV 2017 / IJCV 2019. https://arxiv.org/abs/1610.02391
2. Chattopadhay, A., Sarkar, A., Howlader, P., Balasubramanian, V. N. *Grad-CAM++: Generalized Gradient-Based Visual Explanations for Deep Convolutional Networks.* WACV 2018. https://arxiv.org/abs/1710.11063
3. Draelos, R. L., Carin, L. *Use HiResCAM instead of Grad-CAM for faithful explanations of convolutional neural networks.* arXiv 2020/2021. https://arxiv.org/abs/2011.08891
4. Wang, H., Wang, Z., Du, M., Yang, F., Zhang, Z., Ding, S., Mardziel, P., Hu, X. *Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks.* CVPR Workshops 2020. https://arxiv.org/abs/1910.01279
5. Sundararajan, M., Taly, A., Yan, Q. *Axiomatic Attribution for Deep Networks (Integrated Gradients).* ICML 2017. https://arxiv.org/abs/1703.01365
6. Lundberg, S. M., Lee, S.-I. *A Unified Approach to Interpreting Model Predictions (SHAP).* NeurIPS 2017. https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html
7. Shrikumar, A., Greenside, P., Kundaje, A. *Learning Important Features Through Propagating Activation Differences (DeepLIFT).* ICML 2017. https://arxiv.org/abs/1704.02685
8. Petsiuk, V., Das, A., Saenko, K. *RISE: Randomized Input Sampling for Explanation of Black-box Models.* BMVC 2018. https://arxiv.org/abs/1806.07421
9. Smilkov, D., Thorat, N., Kim, B., Viégas, F., Wattenberg, M. *SmoothGrad: removing noise by adding noise.* arXiv 2017 (ICML Workshop on Visualization for Deep Learning). https://arxiv.org/abs/1706.03825
10. Ribeiro, M. T., Singh, S., Guestrin, C. *"Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME).* KDD 2016. https://arxiv.org/abs/1602.04938
11. Zeiler, M. D., Fergus, R. *Visualizing and Understanding Convolutional Networks (Occlusion).* ECCV 2014. https://arxiv.org/abs/1311.2901
12. Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., Kim, B. *Sanity Checks for Saliency Maps.* NeurIPS 2018. https://arxiv.org/abs/1810.03292
13. Chefer, H., Gur, S., Wolf, L. *Transformer Interpretability Beyond Attention Visualization.* CVPR 2021. https://arxiv.org/abs/2012.09838
14. Chefer, H., Gur, S., Wolf, L. *Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers.* ICCV 2021 (Oral). https://arxiv.org/abs/2103.15679
15. Abnar, S., Zuidema, W. *Quantifying Attention Flow in Transformers (Attention Rollout / Flow).* ACL 2020. https://aclanthology.org/2020.acl-main.385/
16. Jain, S., Wallace, B. C. *Attention is not Explanation.* NAACL-HLT 2019. https://arxiv.org/abs/1902.10186
17. Wiegreffe, S., Pinter, Y. *Attention is not not Explanation.* EMNLP-IJCNLP 2019. https://aclanthology.org/D19-1002/
18. Liu, Y., Li, H., Guo, Y., Kong, C., Li, J., Wang, S. *Rethinking Attention-Model Explainability through Faithfulness Violation Test.* ICML 2022. https://arxiv.org/abs/2201.12114
19. Wu, J., Kang, W., Tang, H., Hong, Y., Yan, Y. *On the Faithfulness of Vision Transformer Explanations (SaCo).* CVPR 2024. https://arxiv.org/abs/2404.01415
20. Achtibat, R., Hatefi, S. M. V., Dreyer, M., Jain, A., Wiegand, T., Lapuschkin, S., Samek, W. *AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers.* ICML 2024. https://arxiv.org/abs/2402.05602
21. Brocki, L., Binda, J., Chung, N. C. *Class-Discriminative Attention Maps for Vision Transformers (CDAM).* TMLR 2024 (also IJCAI 2024 XAI Workshop). https://arxiv.org/abs/2312.02364
22. Hu, L., Liu, Y., Liu, N., Huai, M., Sun, L., Wang, D. *Improving Interpretation Faithfulness for Vision Transformers (Faithful ViTs).* ICML 2024. https://arxiv.org/abs/2311.17983
23. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A. *Emerging Properties in Self-Supervised Vision Transformers (DINO).* ICCV 2021. https://arxiv.org/abs/2104.14294
24. Darcet, T., Oquab, M., Mairal, J., Bojanowski, P. *Vision Transformers Need Registers.* ICLR 2024 (Oral). https://arxiv.org/abs/2309.16588
25. Psomas, B., Christopoulos, D., Baltzi, E., Kakogeorgiou, I., Aravanis, T., Komodakis, N., Karantzalos, K., Avrithis, Y., Tolias, G. *Attention, Please! Revisiting Attentive Probing Through the Lens of Efficiency.* arXiv 2506.10178 (ICLR 2026, venue not independently confirmed). https://arxiv.org/abs/2506.10178
26. Ilse, M., Tomczak, J. M., Welling, M. *Attention-based Deep Multiple Instance Learning.* ICML 2018. https://proceedings.mlr.press/v80/ilse18a.html
27. Chen, C., Li, O., Tao, C., Barnett, A. J., Su, J., Rudin, C. *This Looks Like That: Deep Learning for Interpretable Image Recognition (ProtoPNet).* NeurIPS 2019. https://arxiv.org/abs/1806.10574
28. Kim, E., Kim, S., Seo, M., Yoon, S. *XProtoNet: Diagnosis in Chest Radiography with Global and Local Explanations.* CVPR 2021. https://openaccess.thecvf.com/content/CVPR2021/html/Kim_XProtoNet_Diagnosis_in_Chest_Radiography_With_Global_and_Local_Explanations_CVPR_2021_paper.html
29. Nauta, M., van Bree, R., Seifert, C. *Neural Prototype Trees for Interpretable Fine-grained Image Recognition (ProtoTree).* CVPR 2021. https://arxiv.org/abs/2012.02046
30. Donnelly, J., Barnett, A. J., Chen, C. *Deformable ProtoPNet: An Interpretable Image Classifier Using Deformable Prototypes.* CVPR 2022. https://openaccess.thecvf.com/content/CVPR2022/html/Donnelly_Deformable_ProtoPNet_An_Interpretable_Image_Classifier_Using_Deformable_Prototypes_CVPR_2022_paper.html
31. Nauta, M., Schlötterer, J., van Keulen, M., Seifert, C. *PIP-Net: Patch-Based Intuitive Prototypes for Interpretable Image Classification.* CVPR 2023. https://openaccess.thecvf.com/content/CVPR2023/html/Nauta_PIP-Net_Patch-Based_Intuitive_Prototypes_for_Interpretable_Image_Classification_CVPR_2023_paper.html
32. Djoumessi, K., Bah, B., Kühlewein, L., Berens, P., Koch, L. *This actually looks like that: Proto-BagNets for local and global interpretability-by-design.* MICCAI 2024. https://arxiv.org/abs/2406.15168
33. Chiang, L.-F. *An Interpretable Chest X-ray Classification Framework Using Prototype Memory and Counterfactual Consistency (CXR-NeXus).* Cureus, 2026 (low-bar venue). https://pmc.ncbi.nlm.nih.gov/articles/PMC12972620/
34. Koh, P. W., Nguyen, T., Tang, Y. S., Mussmann, S., Pierson, E., Kim, B., Liang, P. *Concept Bottleneck Models.* ICML 2020. https://proceedings.mlr.press/v119/koh20a.html
35. Yuksekgonul, M., Wang, M., Zou, J. *Post-hoc Concept Bottleneck Models.* ICLR 2023 (Spotlight). https://arxiv.org/abs/2205.15480
36. Oikarinen, T., Das, S., Nguyen, L. M., Weng, T.-W. (Lily Weng). *Label-Free Concept Bottleneck Models.* ICLR 2023. https://arxiv.org/abs/2304.06129
37. Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viégas, F., Sayres, R. *Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV).* ICML 2018. https://proceedings.mlr.press/v80/kim18d.html
38. Böhle, M., Fritz, M., Schiele, B. *B-cos Networks: Alignment is All We Need for Interpretability.* CVPR 2022. https://arxiv.org/abs/2205.10268
39. Arya, S., Rao, S., Böhle, M., Schiele, B. *B-cosification: Transforming Deep Neural Networks to be Inherently Interpretable.* NeurIPS 2024. https://arxiv.org/abs/2411.00715
40. Pang, W., Ke, X., Tsutsui, S., Wen, B. *Integrating Clinical Knowledge into Concept Bottleneck Models.* MICCAI 2024 (validated on WBC/skin, not CXR). https://papers.miccai.org/miccai-2024/415-Paper1786.html
41. Rudin, C. *Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead.* Nature Machine Intelligence, 2019. https://www.nature.com/articles/s42256-019-0048-x
42. Hoffmann, A., Fanconi, C., Rade, R., Köhler, J. *This Looks Like That... Does it? Shortcomings of Latent Space Prototype Interpretability in Deep Networks.* ICML 2021 XAI Workshop. https://arxiv.org/abs/2105.02968
43. Elhadri, K., et al. *This looks like what? Challenges and Future Research Directions for Part-Prototype Models.* arXiv 2025. https://arxiv.org/abs/2502.09340
44. Willard, F., et al. *ProtoPNeXt* (unified hyperparameter-search framework for ProtoPNet variants), 2024 (Duke). [in-text reference; see ProtoPNet ecosystem]
45. Hou, J., Liu, S., Bie, Y., Wang, H., Tan, A., Luo, L., Chen, H. *Self-eXplainable AI for Medical Image Analysis: A Survey and New Outlooks.* arXiv 2024. https://arxiv.org/abs/2410.02331
46. Jacovi, A., Goldberg, Y. *Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate Faithfulness?* ACL 2020. https://aclanthology.org/2020.acl-main.386/
47. DeYoung, J., Jain, S., Rajani, N. F., Lehman, E., Xiong, C., Socher, R., Wallace, B. C. *ERASER: A Benchmark to Evaluate Rationalized NLP Models (Comprehensiveness/Sufficiency).* ACL 2020. https://aclanthology.org/2020.acl-main.408/
48. Hooker, S., Erhan, D., Kindermans, P.-J., Kim, B. *A Benchmark for Interpretability Methods in Deep Neural Networks (ROAR).* NeurIPS 2019. https://arxiv.org/abs/1806.10758
49. Rong, Y., Leemann, T., Borisov, V., Kasneci, G., Kasneci, E. *A Consistent and Efficient Evaluation Strategy for Attribution Methods (ROAD).* ICML 2022. https://proceedings.mlr.press/v162/rong22a.html
50. Ghorbani, A., Abid, A., Zou, J. *Interpretation of Neural Networks Is Fragile.* AAAI 2019. https://arxiv.org/abs/1710.10547
51. Heo, J., Joo, S., Moon, T. *Fooling Neural Network Interpretations via Adversarial Model Manipulation.* NeurIPS 2019. https://arxiv.org/abs/1902.02041
52. Hedström, A., Weber, L., Krakowczyk, D., Bareeva, D., Motzkus, F., Samek, W., Lapuschkin, S., Höhne, M. M.-C. *Quantus: An Explainable AI Toolkit for Responsible Evaluation of Neural Network Explanations and Beyond.* JMLR 2023. https://www.jmlr.org/papers/volume24/22-0142/22-0142.pdf
53. Saporta, A., Gui, X., Agrawal, A., Pareek, A., Truong, S. Q. H., Nguyen, C. D. T., Ngo, V.-D., Seekins, J., Blankenberg, F. G., Ng, A. Y., Lungren, M. P., Rajpurkar, P. *Benchmarking saliency methods for chest X-ray interpretation (CheXlocalize).* Nature Machine Intelligence 2022. https://www.nature.com/articles/s42256-022-00536-x
54. Arun, N., Gaw, N., Singh, P., Chang, K., Aggarwal, M., Chen, B., et al., Kalpathy-Cramer, J. *Assessing the Trustworthiness of Saliency Maps for Localizing Abnormalities in Medical Imaging.* Radiology: Artificial Intelligence 2021;3(6):e200267. https://pubs.rsna.org/doi/10.1148/ryai.2021200267
55. Wollek, A., Graf, R., Čečatka, S., et al. *Attention-based Saliency Maps Improve Interpretability of Pneumothorax Classification.* Radiology: Artificial Intelligence 2023. https://arxiv.org/abs/2303.01871
56. Chung, M., Won, J. B., Kim, G., Kim, Y., Ozbulak, U. *Evaluating Visual Explanations of Attention Maps for Transformer-based Medical Imaging.* MICCAI 2024 Workshop (iMIMIC). https://arxiv.org/abs/2503.09535
57. Ahn, J. S., Ebrahimian, S., McDermott, S., et al. *Association of Artificial Intelligence-Aided Chest Radiograph Interpretation With Reader Performance and Efficiency.* JAMA Network Open 2022;5(8):e2229289. https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2795798
58. *Evaluating cognitive biases in AI-assisted mammography interpretation: a simulation reader study of explainable AI across radiologist experience levels.* European Radiology 2026 (the "~9 points below baseline" figure is unverified and omitted here). https://link.springer.com/article/10.1007/s00330-026-12666-6
59. Angelopoulos, A. N., Bates, S. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* arXiv 2021 / Foundations and Trends in Machine Learning 2023. https://arxiv.org/abs/2107.07511
60. Romano, Y., Patterson, E., Candès, E. J. *Conformalized Quantile Regression (CQR).* NeurIPS 2019. https://arxiv.org/abs/1905.03222
61. Barber, R. F., Candès, E. J., Ramdas, A., Tibshirani, R. J. *Predictive inference with the jackknife+.* Annals of Statistics 2021;49(1):486–507. https://arxiv.org/abs/1905.02928
62. Tibshirani, R. J., Barber, R. F., Candès, E. J., Ramdas, A. *Conformal Prediction Under Covariate Shift.* NeurIPS 2019. https://arxiv.org/abs/1904.06019
63. Barber, R. F., Candès, E. J., Ramdas, A., Tibshirani, R. J. *Conformal prediction beyond exchangeability.* Annals of Statistics 2023;51(2):816–845. https://arxiv.org/abs/2202.13415
64. Lakshminarayanan, B., Pritzel, A., Blundell, C. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS 2017. https://arxiv.org/abs/1612.01474
65. Gal, Y., Ghahramani, Z. *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.* ICML 2016. https://arxiv.org/abs/1506.02142
66. Sensoy, M., Kaplan, L., Kandemir, M. *Evidential Deep Learning to Quantify Classification Uncertainty.* NeurIPS 2018. https://arxiv.org/abs/1806.01768
67. Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. *On Calibration of Modern Neural Networks (Temperature Scaling, ECE).* ICML 2017. https://arxiv.org/abs/1706.04599
68. Geifman, Y., El-Yaniv, R. *SelectiveNet: A Deep Neural Network with an Integrated Reject Option.* ICML 2019. https://arxiv.org/abs/1901.09192
69. Kurz, A., Hauser, K., Mehrtens, H. A., et al. (Brinker, T. J.). *Uncertainty Estimation in Medical Image Classification: Systematic Review.* JMIR Medical Informatics 2022;10(8):e36427. https://medinform.jmir.org/2022/8/e36427
70. Rajaraman, S., Zamzmi, G., Yang, F., Xue, Z., Jaeger, S., Antani, S. *Uncertainty Quantification in Segmenting Tuberculosis-Consistent Findings in Frontal Chest X-rays.* Biomedicines 2022;10(6):1323. https://www.mdpi.com/2227-9059/10/6/1323
71. Long Hui. *Risk-Sensitive Conformal Prediction for Catheter Placement Detection in Chest X-rays.* arXiv 2505.22496, 2025 (single-author preprint, self-reported numbers). https://arxiv.org/abs/2505.22496
72. *Conformal selective prediction with cost-aware deferral for safe clinical triage under distribution shift.* Scientific Reports 2026; s41598-026-40637-w. https://www.nature.com/articles/s41598-026-40637-w
73. Ghassemi, M., Oakden-Rayner, L., Beam, A. L. *The false hope of current approaches to explainable artificial intelligence in health care.* The Lancet Digital Health 2021;3(11):e745–e750. https://www.thelancet.com/journals/landig/article/PIIS2589-7500(21)00208-9/fulltext
74. U.S. FDA. *Marketing Submission Recommendations for a Predetermined Change Control Plan for AI-Enabled Device Software Functions* (final guidance, 3 Dec 2024); and *Predetermined Change Control Plans for Machine Learning-Enabled Medical Devices: Guiding Principles* (joint FDA / Health Canada / MHRA, Oct 2023). https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
75. Pérez-García, F., Sharma, H., Bond-Taylor, S., Bouzid, K., et al., Oktay, O. *RAD-DINO: Exploring Scalable Medical Image Encoders Beyond Text Supervision.* arXiv 2401.10815, 2024 / Nature Machine Intelligence 2025. https://arxiv.org/abs/2401.10815
76. *Attribution-Based Explainability in Medical Imaging: A Critical Review on Explainable Computer Vision (X-CV) Techniques and Their Applications in Medical AI* (Alam, Zadeh, Sheikh-Akbari). Electronics (MDPI) 2025;14(15):3024 (taxonomy used; the specific "wrong anatomical structure" wording could not be verified and is not quoted as load-bearing). https://www.mdpi.com/2079-9292/14/15/3024
