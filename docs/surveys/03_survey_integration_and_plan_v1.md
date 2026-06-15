# XAI ↔ Edge: An Integration Survey and Concrete Expansion Plan for a Frozen-RAD-DINO TB Severity System

## Part C1 — The Interconnection: XAI ↔ Edge

The two source surveys treat explainability and edge deployment as parallel concerns. They are not. Every compression lever that makes a model deployable on a GPU-less LMIC device also mutates the very computations an explanation is supposed to faithfully report, and every uncertainty guarantee that makes a model trustworthy assumes a numeric precision and a data distribution that compression and cross-country deployment both violate. This part maps that interaction surface and flags where the literature is thin enough to publish into.

The unifying observation for *our* system is structural: because the backbone is a **frozen RAD-DINO ViT-B/14** and the explanation/uncertainty assets (49-token spatial cavity attention; split-conformal intervals) live in trivially cheap heads, the compression we apply to make the system deployable is applied *almost entirely to the encoder that produces the features the explanation is computed over*. The explanation is downstream of the thing being compressed. That is exactly the regime the literature has not characterised.

### C1.1 How compression degrades or preserves explanation faithfulness

**Quantization and saliency/attention drift.** The ViT-PTQ literature documents that the three transformer pathologies — post-LayerNorm inter-channel variance, the power-law post-Softmax attention distribution, and asymmetric post-GELU activations — are precisely the operations an attention explanation reads out [Yuan 2022; Lin 2022]. FQ-ViT's *Log-Int-Softmax* executes 4-bit log-domain attention with bit-shifts [Lin 2022]; this demonstrably alters attention *sharpness* because it requantizes the softmax distribution that the spatial cavity head pools over. RepQ-ViT pushes W4A4 to a usable accuracy level but does so by reparameterizing the LayerNorm and Softmax quantizers [Li 2023a] — i.e., by changing the numerics of the two ops that most directly shape an attention map. The critical gap: every one of these papers reports *task accuracy* (top-1) after quantization; **none reports whether the attention/saliency map is preserved**. Accuracy parity at W8A8 does not imply explanation parity, and for a system whose attention map *is* the clinical cavity explanation, that distinction is load-bearing. The mobile-ViT latency study adds a perverse twist: INT8 can *degrade* latency on some cores (a QNNPACK GELU pathology of 2.85×) [Z. Li 2025], so the cheap-and-faithful corner of the design space is not even monotone in bit-width.

**Distillation and attention survival.** Here the evidence is more hopeful and more directly actionable. Attention Transfer [Zagoruyko 2017] explicitly matches pooled spatial attention maps teacher→student, and the strongest recent result is that copying *only* a teacher ViT's attention maps recovers most of the benefit of pretraining [A. Li 2024] — strong evidence that attention patterns are a transferable, distillable object. This means a distilled student of RAD-DINO *can* in principle inherit the cavity-attention behaviour, but only if the KD objective includes an attention-matching term; naive logit or embedding KD carries no such guarantee. The SSL-distillation lessons (SEED's similarity-distribution matching [Fang 2021]; DisCo's embedding matching [Gao 2022]; the warning that naive feature-map MSE fails on low-rank ViT features [Tian 2025]) all concern *embedding* fidelity, not *attention-map* fidelity. The one medical study that checks explanation survival under KD — Explainable Medical KD [Mir 2025], which uses Score-CAM to confirm the student attends to the same pathological zones — does so **only qualitatively**. There is no standard quantitative certificate (attention-map IoU, rank correlation, pointing-game vs radiologist masks) that a distilled model's explanation survived.

**Pruning/sparsity and attribution stability.** Token reduction is the highest-leverage edge lever for our 1369-token bottleneck, and also the most explanation-hostile in a subtle way. Token Merging (ToMe) [Bolya 2023] merges the most-similar tokens via bipartite soft matching; Adaptive Token Merging (ATM) [Lee 2025] does so training-free with per-image thresholds. Both *mix the spatial provenance* of tokens — and our explanation is a spatial map over the 7×7 grid. ToMe's proportional-attention bookkeeping is the safest choice for keeping the map interpretable, but whether the resulting map still localises cavities to the upper lobes is unmeasured. On weight pruning: structured pruning (NViT [Yang 2023]) mutates backbone weights and therefore *breaks the frozen-encoder contract* outright; N:M 2:4 sparsity [Mishra 2021] accelerates nothing on ANE/NNAPI/QNN/WASM and is irrelevant to our targets. Head pruning [Michel 2019] removes 20–40% of heads — but if the discriminative cavity signal lives in a pruned head, the explanation simply vanishes. The general fragility results from Survey A reinforce the worry: saliency is already fragile under imperceptible input perturbation [Ghorbani 2019] and under accuracy-preserving weight manipulation [Heo 2019]; compression is precisely an accuracy-preserving weight/numeric manipulation, so the *a priori* expectation should be drift, and the burden is on the deployer to show the explanation survived (e.g., via MPRT [Adebayo 2018] run before *and* after compression).

### C1.2 The interpretability–efficiency trade-off and inherently-interpretable-yet-efficient designs

The two surveys jointly define a Pareto frontier with two axes that are usually treated separately: *faithfulness* (Survey A) and *on-device latency/memory* (Survey B). The cheapest explanation — raw attention pooling — is also the least faithful [Jain 2019; Liu 2022], while the faithful methods (Chefer relevance [Chefer 2021], AttnLRP [Achtibat 2024], CDAM [Brocki 2024], FViT diffusion smoothing [Hu 2024], RISE [Petsiuk 2018], Score-CAM [Wang 2020]) cost extra gradients, smoothing passes, or thousands of forwards. On a frozen ViT where the backbone forward is the only unavoidable cost, any method requiring repeated full forwards multiplies the single most expensive operation, which rules out the entire perturbation family and the gradient-free CAMs for *on-device* explanation while leaving them for offline validation.

The frozen-encoder-plus-cheap-head archetype resolves this trade-off elegantly *on the explanation side*: a one-backward-pass class-specific upgrade (CDAM/Chefer through the tiny head) costs essentially nothing on top of the single backbone forward. But the inherently-interpretable-yet-efficient designs from Survey A interact badly with the efficient-architecture designs from Survey B. Prototype models (ProtoPNet [Chen 2019], XProtoNet [Kim 2021]) are flagged as slow, multi-stage, and untested at high resolution [Elhadri 2025] — i.e., interpretable-by-design but *not* edge-efficient. B-cos networks [Böhle 2022] are faithful-by-design at accuracy parity, and B-cosification [Arya 2024] converts a pretrained ViT at 9× less compute, but it requires *fine-tuning the backbone*, which conflicts with the frozen-encoder constraint and with deployment as a fixed exportable graph. Concept Bottleneck Models on frozen features (PCBM [Yuksekgonul 2023], label-free CBM [Oikarinen 2023]) are the rare design that is both interpretable-by-construction *and* efficient (they are tiny probes on frozen features), making them the most promising ante-hoc upgrade compatible with edge constraints — at the cost of CLIP-concept hallucination and concept leakage.

The honest synthesis: **attention-based MIL pooling [Ilse 2018] — which is exactly our head — is the unique point that is simultaneously inherently-interpretable, zero-added-inference-cost, and frozen-encoder-compatible.** Its weakness is faithfulness, not efficiency, which is why the publishable move is to *validate and cheaply upgrade* it rather than replace it.

### C1.3 Uncertainty under compression

This is the thinnest and most important intersection. Survey A establishes the UQ toolkit; Survey B establishes the compression toolkit; **neither connects them, and the joint literature is essentially empty** (Survey B, Open Problem 4, states verbatim that "no work connects low-bit inference, token reduction, distillation, or DP noise to split-conformal coverage validity or deep-ensemble diversity").

**Does split-conformal coverage hold after INT8/INT4?** Conformal prediction wraps any model and guarantees marginal coverage *under exchangeability* [Angelopoulos 2023]. Quantization changes the model's nonconformity score function. If calibration is performed *after* quantization on a held-out INT8 calibration set, split-CP coverage should hold by construction (the guarantee is distribution-free in the scores, whatever model produced them) — but if the model is calibrated in FP and then quantized for deployment, the calibration scores no longer match the deployed scores and coverage can break. This FP-calibrate-then-quantize failure mode is not characterised anywhere in either survey and is a clean, cheap experiment for us. INT4 is worse: the larger the quantization error, the larger the potential shift between calibration-time and inference-time scores, and our system's *already-fragile* Moldova under-coverage (0.83 vs nominal 0.90) is exactly the kind of margin that low-bit noise could erode further.

**Cost of deep ensembles on edge and cheap alternatives.** Survey A's single most important architectural point transfers directly: **ensemble only the heads, not the backbone** [Lakshminarayanan 2017]. Run the frozen ViT forward once, then run M=10 cheap heads — our M=10 ensemble is therefore nearly free on edge, because the expensive operation is shared. This is a genuine structural advantage of the frozen-encoder design that the edge-UQ literature has not exploited. The cheap-UQ alternatives from Survey A all apply: MC-dropout costs T× forwards [Gal 2016] (expensive if dropout is in the backbone, cheap if only in the head); evidential deep learning gives single-pass UQ [Sensoy 2018]; last-layer Laplace and distilled ensembles are noted as under-validated on CXR foundation features (Survey A, Open Problem 6). Conformal-without-ensembling (split-CP on a single head) is the cheapest defensible option and is what makes the system deployable even when ensemble memory is tight on a browser tab.

**Calibration drift under compression and shift.** Temperature scaling [Guo 2017] and CP calibration degrade silently under shift and time (Survey A, Open Problem 12). Compression is an additional, *static* shift in the model itself; LOCO is a *data* shift; DP noise (if federated) is a third. Our known calibration-slope problem (0.55–0.68, severe-band under-prediction) means the system is *already* miscalibrated in FP, so any compression-induced calibration drift compounds a known weakness. Weighted CP [Tibshirani 2019] and CP-beyond-exchangeability [Barber 2023] are the shift-robust tools; their interaction with a *quantized* score function is unstudied.

### C1.4 Privacy/federated ↔ explainability interactions

Survey B establishes that federated PEFT on a frozen encoder exchanges only tiny head weights [Alkhunaizi 2024], and that FedBN-style local normalization [Li 2021] is the right analogue for cross-country (LOCO) feature shift. The explainability interaction is two-sided. On the protective side: keeping data on-device (ORT-Web's "data never leaves the tab") means the attention overlay is computed locally and never transmitted, sidestepping the privacy risk of shipping a saliency map that could leak patient anatomy. On the adversarial side: gradient-inversion attacks reconstruct training images from shared gradients [Liu 2024], and an *attention map* is itself a compressed spatial fingerprint of the input — a federated system that shares explanation artifacts for audit could leak more than it intends. Differential privacy (DP-SGD) is the defence, but DP noise degrades faster on already-compressed updates (Survey B, §9.4), and a 74-study review shows DP can *widen subgroup fairness gaps* [Mohammadi 2026] — meaning the very subgroups whose explanations matter most (HIV+, pediatric, prior-TB) are where DP-induced explanation degradation would concentrate. FedCTTA [Rajib 2025] shows test-time adaptation (our transductive CORAL analogue) can run on-device without feature exchange, but **whether transductive CORAL alignment at inference leaks patient information is explicitly uncharacterised** (Survey B, Open Problem 4) — and CORAL changes the features the attention is computed over, so it is simultaneously a privacy question and an explanation-stability question.

### C1.5 Evaluation: measuring that an explanation is BOTH faithful AND cheap on-device

Neither survey supplies a joint metric, and Survey B (Open Problem 6) states that "no standardised benchmark reports DINOv2/RAD-DINO-class encoder latency, energy, and peak memory across Core ML/ANE, TFLite-NNAPI/QNN, ORT-Web/WebGPU/WASM, and commodity CPU simultaneously," while Survey A (Open Problem 6) states "little work measures attribution/UQ latency and memory on Core ML / TFLite / ONNX-Runtime-Web / WASM" and "a systematic on-device faithfulness-vs-latency Pareto frontier for CXR does not exist." The composite metric a rigorous paper needs has three legs, all of which the constituent surveys supply tools for:

1. **Faithfulness leg** — perturbation-based with ROAD-style neighbour-mean imputation [Rong 2022] (not zeroing, to avoid OOD confound), adapted to *tokens*: ablate top-k attended tokens and measure logit drop, cheap because only the head re-evaluates while the ViT forward is cached. Plus MPRT [Adebayo 2018] (randomize head weights, show the map degrades) run before and after each compression step. Plus EBPG [Wang 2020] against TB Portals sextant annotations. All orchestrated in Quantus [Hedström 2023].
2. **Explanation-survival-under-compression leg** — the missing metric both surveys flag: attention-map IoU / Spearman rank correlation / EBPG *between the FP and the compressed model*, as a function of compression budget (bit-width, token-keep-rate, student size). This is the publishable white space.
3. **Cost leg** — measured latency, energy, and *peak memory* per backend under MLPerf-Mobile-style methodology [Janapa Reddi 2022], with the FLOPs≠latency caveat [Almeida 2021; Vasu 2023a] enforced and thermal/numeric-parity disclosed.

### C1.6 Where the literature is THIN — the publishable white space

Cross-referencing the two surveys' open-problem lists yields five gaps that are *intersection* gaps — neither survey alone owns them, and they are directly addressable on our compute:

- **WS1 — Explanation fidelity vs compression budget.** No standard metric certifies that a compressed model's attention/saliency survives (Survey B OP2; Survey A OP6). For a system whose attention map *is* the cavity explanation, this is acute and ownable.
- **WS2 — Conformal coverage under quantization.** Whether split-CP coverage holds after INT8/INT4, and whether FP-calibrate-then-quantize breaks it, is unstudied (Survey B OP4; Survey A OP6).
- **WS3 — Faithfulness of learned attention-pooling weights on *frozen* features.** The faithfulness literature targets backbone maps; almost nobody has tested whether a trained downstream attention-pool head on frozen features is faithful (Survey A OP1) — exactly our head.
- **WS4 — Sub-INT8 PTQ of self-distilled DINOv2/RAD-DINO encoders.** All ViT-PTQ numbers are on supervised ImageNet ViTs; whether RepQ/AdaLog fixes hold for a register-token-free self-distilled medical encoder is unmeasured (Survey B OP1).
- **WS5 — Explaining a composite regression+threshold clinical score on edge.** Almost all attribution targets classification logits; attributing Timika = ALP + 40·1[cavity] (a continuous term plus a thresholded binary term) and conformalising it is largely untouched (Survey A OP5/OP9; Survey B OP3).

---

## Part C2 — Mapping Techniques onto Our Pipeline

The pipeline has exactly two cost centres and two explanation/UQ assets. Cost: the **frozen RAD-DINO ViT-B encoder** (1369 tokens at 518²) dominates; the **ALP-MLP and 49-token cavity head plus the kNN/CORAL/ensemble/conformal ladder** are trivially cheap. Explanation/UQ assets: the **spatial cavity attention** and the **split-conformal intervals**. Every technique below is positioned relative to those four.

| # | Technique | Where in pipeline | Accuracy impact | Latency / size / memory impact | Risk | Targets |
|---|---|---|---|---|---|---|
| T1 | **INT8 PTQ (W8A8, per-channel weights, RepQ/SmoothQuant outlier handling, 32 in-domain CXR calib imgs)** [Li 2023a; Xiao 2023; Yuan 2022] | Frozen encoder only; heads stay FP | <1% expected on classification analogues; **Timika MAE impact unmeasured** (regression is more exposed) | ~4× size; INT8 accelerated on ANE A17/M4, Hexagon, WASM-SIMD; ~28% latency / ~70% memory on a Pixel-class device | Activation outliers in LayerNorm/Softmax/GELU; cavity-texture signal loss; **possible INT8 latency *regression* on some mobile cores** [Z. Li 2025] | iPad, Android, browser, CPU |
| T2 | **Sub-INT8 weight-only PTQ (AWQ/GPTQ INT4)** [Lin 2024; Frantar 2023] | Encoder weights only; activations FP | Lowest accuracy risk of the low-bit options; no matmul speedup | Memory only (~4× weights); no latency gain on commodity CPU | Buys memory not speed; pointless unless memory-bound (browser tab, low-RAM tablet) | browser, low-RAM Android |
| T3 | **Training-free token merging in the encoder (ToMe / ATM)** [Bolya 2023; Lee 2025] | *Inside* the ViT-B blocks (attacks the 1369-token O(N²) cost) | 0.2–0.5% on ImageNet ViTs; **Timika/cavity impact unmeasured** | ~1.5–2× throughput; reduces compute *and* memory; no training | **Mixes spatial provenance → may corrupt the 7×7 cavity map** (use proportional-attention bookkeeping) | all four |
| T4 | **Distill frozen RAD-DINO → small token-grid student (DINOv2 iBOT self-distill + SEED/DisCo embedding KD + attention-matching KD)** [Oquab 2023; Fang 2021; Gao 2022; Zagoruyko 2017; A. Li 2024] | Replaces the encoder with EfficientFormerV2-S0 / FastViT-SA12 / TinyViT-class student | Risk of MAE regression; LOCO gap must be preserved (per MEMORY: compression must not erode cross-country generalization) | 30–200× fewer params; MobileNet-speed; the single biggest latency win | Naive logit/feature KD fails for SSL ViTs [Tian 2025]; **attention map may not transfer without explicit AT term**; re-fit calibration ladder | all four |
| T5 | **ANE-friendly attention rewrite ((B,C,1,S) layout, Conv2d-1×1 for Linear, reshape-free attention)** [Apple ANE Transformers 2022; Apple ViT-ANE 2023] | One-time surgical export of the frozen encoder | None (numeric-parity rewrite) | Up to ~10× faster / ~14× lower peak memory at long seq; **without it the model lands on GPU/CPU not ANE** | Manual per-model effort; must verify 37×37→7×7 grid is byte-identical post-rewrite | iPad |
| T6 | **Head-only deep ensemble (M=10) + split-conformal** [Lakshminarayanan 2017; Angelopoulos 2023; Romano 2019] | Heads only; backbone forward shared | Improves calibration/UQ; no accuracy cost | **Nearly free on edge** (M heads on one shared backbone forward); tiny memory | Ensemble memory if heads grow; **CP coverage validity after encoder quantization is unverified** (WS2) | all four |
| T7 | **CDAM/Chefer class-specific signed attention upgrade** [Brocki 2024; Chefer 2021] | +1 backward through the tiny head | None (explanation-only) | Negligible (1 backward on 49 tokens, CPU-fine) | Upgrades class-agnostic raw attention to signed/class-specific; must re-validate on compressed model | all four |
| T8 | **CQR instead of constant-width split-CP** [Romano 2019] | Conformal layer (continuous ALP target) | Better adaptive interval width; addresses Moldova under-coverage | Negligible | Binary cavity term still needs separate conformal handling (WS5) | all four |
| T9 | **Static-shape ONNX export → ORT-Web (WebGPU, WASM-INT8 fallback)** [Microsoft ORT-Web 2024; Hugging Face 2024] | Whole graph; 518² fixed so shapes freeze | None (export fidelity) | WebGPU large speedup over WASM; WASM-INT8 ~2–4× over scalar FP32; zero-install, privacy-by-construction | DINOv2/RAD-DINO needs custom ONNX export; op-coverage for attention | browser |
| T10 | **ExecuTorch + XNNPACK (PT2E INT8) / ONNX+OpenVINO-NNCF** [PyTorch ExecuTorch 2024; OpenVINO/NNCF] | Frozen encoder on mini-PC | INT8 <1–3% on classification; regression unmeasured | ~50 KB runtime; INT8 on x86/ARM | OpenVINO warns ViT INT8 PTQ is fragile (post-GELU, batch>1) | CPU / mini-PC |
| T11 | **Federated PEFT (share head weights only) + FedBN-style local calibration + FedProx** [Alkhunaizi 2024; Li 2021; Li 2020] | Training-time; heads + per-country calibration | ~4% accuracy cost per 10× param-exchange reduction on non-IID | Tiny communication; no inference cost | DP noise widens subgroup gaps [Mohammadi 2026]; CORAL/CP under federation unstudied | training infra |
| T12 | **Keep heads in FP while backbone is INT8 (mixed precision)** [Survey B §3.5] | Encoder INT8, heads FP | Protects the continuous Timika output and conformal scores from quantization error | Minimal cost (heads are tiny) | None significant; recommended default | all four |

**Prose mapping by target.**

*iPad (Core ML / ANE).* The ANE is transformer-hostile by default and a stock RAD-DINO export silently lands on GPU/CPU [Apple ANE Transformers 2022], so **T5 (attention rewrite) is mandatory, not optional**, and is acceptable precisely because the encoder is frozen (one-time surgical cost). Stack T5 + T1 (W8A8, accelerated on A17/M4) + T12 (FP heads) + T6 (head ensemble) + T7 (signed attention). T3 (ToMe) is additive but must be validated against the cavity map. Expected: this is the highest-end target; target sub-50ms encoder forward on A17 ANE is plausible for a distilled student (T4), less so for full ViT-B even quantized.

*Android (TFLite/LiteRT + QNN/Hexagon).* XNNPACK CPU is the guaranteed fallback; the QNN/Hexagon delegate (INT4/8/16/FP16, MHA acceleration) is the accelerated path [Qualcomm 2024; Google AI Edge 2024]. **Benchmark T1 — do not assume it; INT8 can regress latency on some cores** [Z. Li 2025]. Stack T1 + T12 + T6, add T3, and T4 if ViT-B is too slow. T2 (INT4 weight-only) helps low-RAM tablets.

*Browser (ORT-Web / WebGPU / WASM).* Most deployment-friendly for LMIC clinics (zero install, data never leaves the tab). T9 (static-shape ONNX) + WebGPU primary, WASM-INT8 fallback. T2 (INT4 weight-only) is attractive here because browser memory is the binding constraint. WebNN is too immature [W3C 2026] — do not target it for clinical use. T4 student is most valuable here, since a 30–200× smaller encoder transforms feasibility on commodity browsers.

*CPU / mini-PC.* T10 (ExecuTorch+XNNPACK or ONNX+OpenVINO-NNCF) + T1 + T12. LeViT/EfficientFormerV2-class students (T4) are CPU-oriented and directly relevant to GPU-less clinics. This is the most forgiving target and the natural place to first validate the full XAI+UQ battery.

A cross-cutting rule from both surveys: **report measured latency, post-quantization size, peak memory, and per-subgroup (HIV+, pediatric, prior-TB) Timika error per backend** — aggregate metrics hide the failures that matter clinically [Worodria 2024; Alege 2025].

---

## Part C3 — Prioritised Expansion Roadmap (P0/P1/P2)

### P0 — Foundational, low-compute, high-certainty (do first; all fit free-tier T4)

**P0.1 — Quantitative localization validation of the cavity-attention map against TB Portals sextant annotations.**
*Idea:* Convert the qualitative R4b attention overlay into a validated localizer. Compute EBPG [Wang 2020] (fraction of attention energy inside the GT sextant) as primary metric, Pointing Game and an IoU/Dice threshold sweep as complements, and run MPRT [Adebayo 2018] (randomize the head weights, show the map degrades) as the faithfulness floor. Adapt token-level deletion/insertion with ROAD-style neighbour-mean imputation [Rong 2022; DeYoung 2020], ablating top-k attended tokens and measuring Timika logit drop. Orchestrate in Quantus [Hedström 2023].
*Why:* This is Known Limitation (i) and the single most defensible XAI contribution; without it the attention map is plausible but unvalidated [Jacovi 2020; Saporta 2022]. It directly addresses WS3 (faithfulness of attention-pool weights on frozen features).
*Dependencies:* TB Portals sextant cavity annotations; cached frozen features (already built).
*Compute:* Negligible — heads are trivial and features are cached. **Free-tier T4, hours.**
*Expected result:* A faithfulness-validated attention map with EBPG and an MPRT pass; framing shifts from "concentrates on upper lobes" to "faithful, ground-truth-localized cavity evidence."
*Null/risk:* The attention may fail MPRT (map unchanged under head randomization → it is reading backbone saliency, not the trained head) or under-localise small cavities exactly as CheXlocalize found for saliency [Saporta 2022]. Either outcome is *itself publishable* as an honest negative.

**P0.2 — Class-specific signed attention upgrade (CDAM/Chefer through the head).**
*Idea:* Replace the class-agnostic softmax pool readout with a CDAM/Chefer signed, class-specific map [Brocki 2024; Chefer 2021] — one backward pass through the tiny head.
*Why:* Raw attention alone is weak [Jain 2019; Liu 2022]; gradient-augmented attention is markedly more faithful [Wu 2024]. Cheap on the 49-token grid even on CPU.
*Dependencies:* P0.1 metric harness (to show the upgrade improves faithfulness).
*Compute:* **Free-tier T4, trivial.**
*Expected result:* Higher EBPG/deletion-AUC than raw attention at zero deployment cost.
*Null/risk:* On a *frozen* backbone, gradients into the encoder are uninformative (Survey A OP2), so CDAM may add little over raw attention — a clean negative that motivates P1.

**P0.3 — INT8 PTQ of the encoder with explanation- and coverage-survival measurement (WS1 + WS2 pilot).**
*Idea:* RepQ-ViT/SmoothQuant W8A8 [Li 2023a; Xiao 2023] on the frozen encoder with 32 in-domain CXR calibration images, heads kept FP (T12). Then measure the **two missing metrics**: (a) attention-map IoU/Spearman/EBPG between FP and INT8 models; (b) split-conformal coverage on the INT8 model, comparing FP-calibrate-then-quantize vs quantize-then-calibrate.
*Why:* Directly attacks WS1 (explanation fidelity vs compression) and WS2 (conformal coverage under quantization), the two emptiest cells in the joint literature.
*Dependencies:* P0.1 harness; cached features for FP baseline.
*Compute:* PTQ is calibration-only, minutes. **Free-tier T4.**
*Expected result:* A first data point on the explanation-fidelity-vs-bit-width curve and on whether CP coverage survives INT8.
*Null/risk:* INT8 may preserve both perfectly (less novel but still a useful negative), or the regression target may degrade more than classification papers suggest (Survey B OP3) — the latter is the more interesting result.

### P1 — Core deployment, moderate compute (T4 sufficient for most; one item may want A100)

**P1.1 — Distill frozen RAD-DINO into a TinyViT/EfficientFormerV2-S0 token-grid student via feature + attention KD, preserving the LOCO gap.**
*Idea:* Re-run the DINOv2 iBOT objective with frozen RAD-DINO as teacher [Oquab 2023] to train a ViT-S/14-or-smaller token-producing student; add SEED similarity-distribution / DisCo embedding matching [Fang 2021; Gao 2022], per-patch feature matching with feature-lifting to handle low-rank ViT features [Tian 2025], and an explicit **attention-matching KD term** [Zagoruyko 2017; A. Li 2024] to preserve the cavity map. Distill on TB Portals + a public CXR pool (MIMIC/CheXpert/PadChest-class). Export to Core ML INT8. **Validate cavity-attention EBPG vs TB Portals sextant annotations is preserved within a pre-registered tolerance, and that the LOCO MAE gap is not eroded** (per MEMORY: compression must not collapse cross-country generalization).
*Why:* The encoder is the only heavy component; distillation is the single biggest latency/size win (30–200×) and the only path to a genuinely small browser/CPU footprint.
*Dependencies:* P0.1 (the EBPG-preservation certificate); the public CXR pool; the calibration ladder must be re-fit on student features and re-validated for conformal coverage and CORAL alignment.
*Compute:* SSL distillation over a large CXR pool is the **one item that genuinely benefits from an A100**; TinyViT's logit-caching trick [Wu 2022] makes a T4 run *tractable* but slow. Realistic: prototype on T4 with cached teacher features, scale on A100 if obtainable.
*Expected result:* A student at MobileNet-speed (EfficientFormerV2-S0 ~0.9 ms on iPhone-class [Y. Li 2023]) with Timika MAE within a tolerance of the teacher and EBPG preserved within Y%; target <X ms encoder forward on A17 ANE.
*Null/risk:* The distilled attention may not localise cavities (attention KD insufficient), or the LOCO gap may collapse because the student lacks RAD-DINO's 880k-CXR generalization — both are flagged risks and both are publishable negatives. MedAlmighty's caution that frozen-backbone+linear-probe underperforms [Ren 2025] cuts the other way and supports distilling a specialised student.

**P1.2 — Training-free token merging (ToMe/ATM) inside the encoder with cavity-map preservation.**
*Idea:* Apply ToMe [Bolya 2023] / ATM [Lee 2025] inside the ViT-B blocks with proportional-attention bookkeeping; measure cavity-EBPG drift as keep-rate decreases.
*Why:* Attacks the actual 1369-token O(N²) cost (which post-encoder pooling to 49 does not); training-free, frozen-compatible.
*Dependencies:* P0.1 harness for the drift measurement.
*Compute:* **Free-tier T4** (training-free).
*Expected result:* 1.5–2× throughput at <0.5% MAE loss *if* the cavity map survives; a keep-rate-vs-EBPG curve is itself a WS1 contribution.
*Null/risk:* Merging corrupts the spatial map below some keep-rate — the curve tells us exactly where, which is the point.

**P1.3 — Four-target on-device benchmark harness (latency, energy, peak memory, numeric parity).**
*Idea:* Export the (compressed) encoder to all four targets per Part C2 and measure under MLPerf-Mobile methodology [Janapa Reddi 2022] with thermal/numeric disclosure, plus per-subgroup Timika error.
*Why:* Fills Survey B OP6 / Survey A OP6 — no such benchmark exists for the 518²/1369-token regime.
*Dependencies:* P1.1 student (for the small-model rows); P0.3 INT8 encoder.
*Compute:* Inference-only on the target devices; no training. **Device access, not GPU, is the constraint.**
*Expected result:* The first faithfulness-and-cost Pareto frontier for a RAD-DINO-class CXR encoder across ANE/QNN/WebGPU/CPU.
*Null/risk:* Some backends may not support the attention ops (op-coverage gaps), which is itself a reportable finding.

### P2 — Higher-risk, higher-novelty (compute varies; some need A100)

**P2.1 — Conformalising the composite Timika score and fixing the calibration slope (WS5).**
*Idea:* CQR [Romano 2019] on the continuous ALP term + a separate calibrated/conformal treatment of the thresholded cavity term, composed into a coherent interval on Timika = ALP + 40·1[cavity]. Pair split-CP with weighted/non-exchangeable CP [Tibshirani 2019; Barber 2023] for LOCO shift, report Mondrian per-country coverage, and address the slope<1 severe-band under-prediction.
*Why:* Conformalising a composite regression+threshold clinical target is largely untouched (Survey A OP9; Survey B OP3); fixes Known Limitations (ii) and the Moldova under-coverage.
*Dependencies:* P0.3 (coverage-under-quantization result).
*Compute:* **Free-tier T4** (post-hoc).
*Expected result:* Restored ≥0.90 coverage on Moldova and a defensible composite interval; a method other composite-score systems can reuse.
*Null/risk:* The binary cavity term may resist clean conformalisation, forcing a two-number output (interval + flag) rather than one interval.

**P2.2 — Concept-grounded severity head on frozen features (PCBM / label-free CBM).**
*Idea:* Add a tiny concept bottleneck on frozen RAD-DINO features [Yuksekgonul 2023; Oikarinen 2023] grounding Timika in clinical concepts (cavity, consolidation, fibrosis, upper-lobe predominance), enabling test-time intervention.
*Why:* Inherently-interpretable-yet-efficient (tiny probe on frozen features) — the rare design compatible with both edge and interpretability constraints (C1.2).
*Dependencies:* concept annotations or CLIP-grounded auto-concepts; P0.1 harness.
*Compute:* **Free-tier T4.**
*Expected result:* Concept-level explanations and intervention without dense annotations or backbone fine-tuning.
*Null/risk:* CLIP concept hallucination / leakage; concept set may not capture TB severity drivers.

**P2.3 — Federated PEFT + on-device CORAL with privacy/coverage characterisation.**
*Idea:* Federated PEFT exchanging only head weights [Alkhunaizi 2024] with FedBN-style local calibration [Li 2021], FedProx regularization [Li 2020], moderate-ε DP, and on-device CORAL via the FedCTTA pattern [Rajib 2025]. **Characterise whether ALP regression and conformal coverage degrade under DP+federation, and whether CORAL leaks** (both flagged unstudied).
*Why:* The privacy/UQ intersection white space (C1.4); a credible multi-country TB Portals deployment story.
*Dependencies:* multi-site infra (simulatable on one machine); P2.1 conformal layer.
*Compute:* **A100 helpful** for realistic multi-client simulation but T4-feasible at small scale.
*Expected result:* A privacy-preserving cross-country deployment with quantified UQ/coverage cost.
*Null/risk:* DP noise may widen HIV+/pediatric subgroup gaps [Mohammadi 2026] — an honest fairness finding.

---

## Part C4 — Candidate Paper Contributions

**Contribution 1 — "Does the explanation survive compression?" An explanation-fidelity-vs-compression-budget benchmark for a frozen-ViT CXR severity model.**
*Gap:* No standard metric certifies that a compressed model's attention/saliency survives (WS1; Survey B OP2, Survey A OP6). For a system whose attention map *is* the cavity explanation, this is acute.
*Experiment:* Sweep INT8/INT4 PTQ (RepQ-ViT [Li 2023a]), token-keep-rate (ToMe [Bolya 2023]), and distilled-student size (P1.1) on the frozen encoder; for each, report attention-map IoU/Spearman/EBPG between FP and compressed, alongside Timika MAE. **Baselines:** FP model; naive zero-baseline ablation vs ROAD imputation [Rong 2022]. **Metrics:** EBPG [Wang 2020], deletion/insertion AUC, MPRT [Adebayo 2018], all in Quantus [Hedström 2023], vs measured latency/memory [Janapa Reddi 2022].
*Feasibility:* High — features cached, PTQ/ToMe are training-free, all on free-tier T4 (P0.3 + P1.2).
*Venue:* an XAI or efficient-ML workshop (NeurIPS/ICCV) for the first cut; **MIDL or IEEE JBHI** for the full benchmark.

**Contribution 2 — Quantitative validation and cheap class-specific upgrade of attention-pooling explanations on frozen self-supervised CXR features.**
*Gap:* Almost nobody has tested whether trained downstream attention-pool weights on frozen features are faithful (WS3; Survey A OP1), and TB transformer-XAI work defaults to Grad-CAM without faithfulness tests.
*Experiment:* P0.1 + P0.2 — EBPG/Pointing-Game/IoU vs TB Portals sextant annotations, MPRT, token-deletion; then CDAM/Chefer signed upgrade [Brocki 2024; Chefer 2021] and show faithfulness gain at ~zero edge cost. **Baselines:** raw attention, attention rollout [Abnar 2020], Grad-CAM on the patch grid (the field default), random attribution. **Metrics:** EBPG, deletion-AUC, MPRT pass/fail, SaCo [Wu 2024].
*Feasibility:* High — trivial compute, needs sextant annotations.
*Venue:* **ISBI or MICCAI** (interpretable-CXR) / iMIMIC workshop [Chung 2024].

**Contribution 3 — Conformal coverage under quantization for a composite clinical severity score (WS2 + WS5).**
*Gap:* Whether split-CP coverage holds after INT8/INT4 is unstudied (Survey B OP4), and conformalising a regression+threshold composite (Timika) is largely untouched (Survey A OP9).
*Experiment:* P0.3 + P2.1 — CQR [Romano 2019] + weighted/non-exchangeable CP [Tibshirani 2019; Barber 2023] on the composite Timika, measuring marginal and Mondrian per-country coverage for FP vs INT8 vs INT4, and FP-calibrate-then-quantize vs quantize-then-calibrate. **Baselines:** constant-width split-CP; FP-only coverage; temperature scaling [Guo 2017]. **Metrics:** empirical coverage at α=0.10, interval width, per-country (LOCO) coverage, calibration slope.
*Feasibility:* High — post-hoc, free-tier T4. Fixes our known Moldova under-coverage (0.83) and slope<1 problems.
*Venue:* **ML4H or IEEE JBHI**; a strong fit for a UQ/safe-ML workshop.

**Contribution 4 — Distilling a frozen CXR foundation encoder to an edge student *without eroding cross-country generalization or the cavity explanation*.**
*Gap:* SSL-distillation preserves embeddings, not attention maps or the LOCO gap; whether a small student inherits RAD-DINO's cross-country robustness *and* its cavity localization is unmeasured (WS4-adjacent; MEMORY constraint).
*Experiment:* P1.1 — iBOT self-distill + SEED/DisCo + attention-KD [Oquab 2023; Fang 2021; Gao 2022; Zagoruyko 2017; A. Li 2024] into EfficientFormerV2-S0/TinyViT; export Core ML INT8; report LOCO MAE gap (vs teacher) and cavity-EBPG preservation. **Baselines:** from-scratch student; embedding-only KD (no attention term); frozen-teacher linear probe [Ren 2025]; published TB baselines (TXV+CheXzero, GroupDRO, IW-regression). **Metrics:** Timika MAE per country, LOCO gap, EBPG preservation %, on-device latency [Y. Li 2023].
*Feasibility:* Medium — the **one item that may need an A100**; T4-feasible at reduced scale via logit caching [Wu 2022].
*Venue:* **MICCAI or IEEE JBHI** (clinical impact) / efficient-ML workshop for the methods cut.

**Contribution 5 — A four-backend on-device Pareto frontier (faithfulness × latency × memory) for a RAD-DINO-class CXR encoder.**
*Gap:* No benchmark reports DINOv2/RAD-DINO-class encoder latency, energy, *and* peak memory across Core ML/ANE, TFLite-QNN, ORT-Web/WebGPU/WASM, and CPU at the 518²/1369-token regime (Survey B OP6; Survey A OP6).
*Experiment:* P1.3 — export the FP, INT8 (P0.3), token-merged (P1.2), and distilled (P1.1) encoders to all four targets; jointly plot faithfulness (EBPG drift) against measured latency/peak-memory under MLPerf-Mobile methodology [Janapa Reddi 2022] with FLOPs≠latency disclosure [Almeida 2021], plus per-subgroup Timika error.
*Feasibility:* Medium — inference-only; the constraint is *device access* (iPad/Snapdragon tablet/browser/mini-PC), not GPU compute.
*Venue:* **IEEE JBHI or a systems/efficient-ML venue (MLSys workshop)**; the systems-plus-clinical framing is distinctive.

A defensible submission strategy: bundle Contributions 2 + 3 (both free-tier, both fix known limitations) as the immediate ICONIP-followup / ISBI paper; Contributions 1 + 5 as the integration "white space" paper (MIDL/JBHI); and Contribution 4 as the flagship MICCAI-2027-aligned deployment paper once A100 access is confirmed.

---

## References

(Citations are drawn from the two source surveys; URLs reproduced as verified there.)

- [Abnar 2020] Abnar, S., Zuidema, W. *Quantifying Attention Flow in Transformers.* ACL 2020. https://aclanthology.org/2020.acl-main.385/
- [Achtibat 2024] Achtibat, R., et al. *AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers.* ICML 2024. https://arxiv.org/abs/2402.05602
- [Adebayo 2018] Adebayo, J., et al. *Sanity Checks for Saliency Maps.* NeurIPS 2018. https://arxiv.org/abs/1810.03292
- [Alege 2025] Alege, A., et al. *Impact of the ultra-portable digital x-ray with CAD4TB for active case finding for TB in Nigeria.* Frontiers in Digital Health 2025. https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2025.1559203/full
- [Alkhunaizi 2024] Alkhunaizi, N., et al. *Probing the Efficacy of Federated Parameter-Efficient Fine-Tuning of Vision Transformers for Medical Image Classification.* arXiv:2407.11573, 2024. https://arxiv.org/abs/2407.11573
- [Almeida 2021] Almeida, M., et al. *Smart at what cost? Characterising Mobile Deep Neural Networks in the wild.* ACM IMC 2021. https://arxiv.org/pdf/2109.13963
- [Angelopoulos 2023] Angelopoulos, A. N., Bates, S. *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification.* Foundations and Trends in ML 2023. https://arxiv.org/abs/2107.07511
- [Apple ANE Transformers 2022] Apple ML Research. *Deploying Transformers on the Apple Neural Engine.* 2022. https://machinelearning.apple.com/research/neural-engine-transformers
- [Apple ViT-ANE 2023] Apple ML Research. *Deploying Attention-Based Vision Transformers to Apple Neural Engine.* 2023. https://machinelearning.apple.com/research/vision-transformers
- [Arya 2024] Arya, S., Rao, S., Böhle, M., Schiele, B. *B-cosification: Transforming Deep Neural Networks to be Inherently Interpretable.* NeurIPS 2024. https://arxiv.org/abs/2411.00715
- [Barber 2023] Barber, R. F., Candès, E. J., Ramdas, A., Tibshirani, R. J. *Conformal prediction beyond exchangeability.* Annals of Statistics 2023;51(2):816–845. https://arxiv.org/abs/2202.13415
- [Böhle 2022] Böhle, M., Fritz, M., Schiele, B. *B-cos Networks: Alignment is All We Need for Interpretability.* CVPR 2022. https://arxiv.org/abs/2205.10268
- [Bolya 2023] Bolya, D., et al. *Token Merging: Your ViT But Faster.* ICLR 2023 (Oral). https://arxiv.org/abs/2210.09461
- [Brocki 2024] Brocki, L., Binda, J., Chung, N. C. *Class-Discriminative Attention Maps for Vision Transformers (CDAM).* TMLR 2024. https://arxiv.org/abs/2312.02364
- [Chefer 2021] Chefer, H., Gur, S., Wolf, L. *Transformer Interpretability Beyond Attention Visualization.* CVPR 2021. https://arxiv.org/abs/2012.09838
- [Chen 2019] Chen, C., et al. *This Looks Like That: Deep Learning for Interpretable Image Recognition (ProtoPNet).* NeurIPS 2019. https://arxiv.org/abs/1806.10574
- [Chung 2024] Chung, M., et al. *Evaluating Visual Explanations of Attention Maps for Transformer-based Medical Imaging.* MICCAI 2024 Workshop (iMIMIC). https://arxiv.org/abs/2503.09535
- [DeYoung 2020] DeYoung, J., et al. *ERASER: A Benchmark to Evaluate Rationalized NLP Models.* ACL 2020. https://aclanthology.org/2020.acl-main.408/
- [Elhadri 2025] Elhadri, K., et al. *This looks like what? Challenges and Future Research Directions for Part-Prototype Models.* arXiv 2025. https://arxiv.org/abs/2502.09340
- [Fang 2021] Fang, Z., et al. *SEED: Self-supervised Distillation For Visual Representation.* ICLR 2021. https://arxiv.org/abs/2101.04731
- [Frantar 2023] Frantar, E., et al. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023. https://arxiv.org/abs/2210.17323
- [Gal 2016] Gal, Y., Ghahramani, Z. *Dropout as a Bayesian Approximation.* ICML 2016. https://arxiv.org/abs/1506.02142
- [Gao 2022] Gao, Y., et al. *DisCo: Remedy Self-supervised Learning on Lightweight Models with Distilled Contrastive Learning.* ECCV 2022. https://arxiv.org/abs/2104.09124
- [Ghorbani 2019] Ghorbani, A., Abid, A., Zou, J. *Interpretation of Neural Networks Is Fragile.* AAAI 2019. https://arxiv.org/abs/1710.10547
- [Google AI Edge 2024] Google AI Edge. *Utilizing Qualcomm NPUs for Mobile AI Development with LiteRT.* 2024. https://ai.google.dev/edge/litert/android/npu/qualcomm
- [Guo 2017] Guo, C., et al. *On Calibration of Modern Neural Networks.* ICML 2017. https://arxiv.org/abs/1706.04599
- [Hedström 2023] Hedström, A., et al. *Quantus: An Explainable AI Toolkit for Responsible Evaluation of Neural Network Explanations.* JMLR 2023. https://www.jmlr.org/papers/volume24/22-0142/22-0142.pdf
- [Heo 2019] Heo, J., Joo, S., Moon, T. *Fooling Neural Network Interpretations via Adversarial Model Manipulation.* NeurIPS 2019. https://arxiv.org/abs/1902.02041
- [Hu 2024] Hu, L., et al. *Improving Interpretation Faithfulness for Vision Transformers.* ICML 2024. https://arxiv.org/abs/2311.17983
- [Hugging Face 2024] Lochner, J., et al. *Transformers.js v3: WebGPU Support, New Models & Tasks, and More.* HF Blog 2024. https://huggingface.co/blog/transformersjs-v3
- [Ilse 2018] Ilse, M., Tomczak, J. M., Welling, M. *Attention-based Deep Multiple Instance Learning.* ICML 2018. https://proceedings.mlr.press/v80/ilse18a.html
- [Jacovi 2020] Jacovi, A., Goldberg, Y. *Towards Faithfully Interpretable NLP Systems.* ACL 2020. https://aclanthology.org/2020.acl-main.386/
- [Jain 2019] Jain, S., Wallace, B. C. *Attention is not Explanation.* NAACL-HLT 2019. https://arxiv.org/abs/1902.10186
- [Janapa Reddi 2022] Janapa Reddi, V., et al. *MLPerf Mobile Inference Benchmark.* MLSys 2022. https://proceedings.mlsys.org/paper_files/paper/2022/hash/a2b2702ea7e682c5ea2c20e8f71efb0c-Abstract.html
- [Kim 2021] Kim, E., et al. *XProtoNet: Diagnosis in Chest Radiography with Global and Local Explanations.* CVPR 2021. https://openaccess.thecvf.com/content/CVPR2021/html/Kim_XProtoNet_Diagnosis_in_Chest_Radiography_With_Global_and_Local_Explanations_CVPR_2021_paper.html
- [Lakshminarayanan 2017] Lakshminarayanan, B., et al. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS 2017. https://arxiv.org/abs/1612.01474
- [Lee 2025] Lee, J., Choi, D.-W. *Lossless Token Merging Even Without Fine-Tuning in Vision Transformers (ATM).* arXiv:2505.15160, 2025. https://arxiv.org/abs/2505.15160
- [Li 2020] Li, T., et al. *Federated Optimization in Heterogeneous Networks (FedProx).* MLSys 2020. https://arxiv.org/abs/1812.06127
- [Li 2021] Li, X., et al. *FedBN: Federated Learning on Non-IID Features via Local Batch Normalization.* ICLR 2021. https://openreview.net/forum?id=6YEQUn0QICG
- [Li 2023a] Li, Z., Xiao, J., Yang, L., Gu, Q. *RepQ-ViT: Scale Reparameterization for Post-Training Quantization of Vision Transformers.* ICCV 2023. https://arxiv.org/abs/2212.08254
- [A. Li 2024] Li, A. C., et al. *On the Surprising Effectiveness of Attention Transfer for Vision Transformers.* NeurIPS 2024. https://arxiv.org/abs/2411.09702
- [Y. Li 2023] Li, Y., et al. *Rethinking Vision Transformers for MobileNet Size and Speed (EfficientFormerV2).* ICCV 2023. https://arxiv.org/abs/2212.08059
- [Z. Li 2025] Li, Z., Paolieri, M., Golubchik, L. *A Study on Inference Latency for Vision Transformers on Mobile Devices.* arXiv:2510.25166, 2025. https://arxiv.org/abs/2510.25166
- [Lin 2022] Lin, Y., et al. *FQ-ViT: Post-Training Quantization for Fully Quantized Vision Transformer.* IJCAI 2022. https://www.ijcai.org/proceedings/2022/0164.pdf
- [Lin 2024] Lin, J., et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys 2024 (Best Paper). https://arxiv.org/abs/2306.00978
- [Liu 2022] Liu, Y., et al. *Rethinking Attention-Model Explainability through Faithfulness Violation Test.* ICML 2022. https://arxiv.org/abs/2201.12114
- [Liu 2024] Liu, C., et al. *AFGI: Towards Accurate and Fast-convergent Gradient Inversion Attack in Federated Learning.* arXiv:2403.08383, 2024. https://arxiv.org/abs/2403.08383
- [Michel 2019] Michel, P., Levy, O., Neubig, G. *Are Sixteen Heads Really Better than One?* NeurIPS 2019. https://arxiv.org/abs/1905.10650
- [Microsoft ORT-Web 2024] Microsoft ONNX Runtime team. *ONNX Runtime Web unleashes generative AI in the browser using WebGPU.* 2024. https://opensource.microsoft.com/blog/2024/02/29/onnx-runtime-web-unleashes-generative-ai-in-the-browser-using-webgpu/
- [Mir 2025] Mir, A. N., Rizvi, D. R. *Explainable Knowledge Distillation for Efficient Medical Image Classification.* arXiv:2508.15251, 2025. https://arxiv.org/abs/2508.15251
- [Mishra 2021] Mishra, A., et al. *Accelerating Sparse Deep Neural Networks (2:4 / N:M structured sparsity).* arXiv:2104.08378, 2021. https://arxiv.org/abs/2104.08378
- [Mohammadi 2026] Mohammadi, M., et al. *Differential privacy for medical deep learning: methods, tradeoffs, and deployment implications.* npj Digital Medicine 2026. https://pmc.ncbi.nlm.nih.gov/articles/PMC12855931/
- [Oikarinen 2023] Oikarinen, T., et al. *Label-Free Concept Bottleneck Models.* ICLR 2023. https://arxiv.org/abs/2304.06129
- [Oquab 2023] Oquab, M., et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024 / arXiv 2023. https://arxiv.org/abs/2304.07193
- [Pérez-García 2024] Pérez-García, F., et al. *RAD-DINO: Exploring Scalable Medical Image Encoders Beyond Text Supervision.* arXiv:2401.10815, 2024 / Nature Machine Intelligence 2025. https://arxiv.org/abs/2401.10815
- [Petsiuk 2018] Petsiuk, V., Das, A., Saenko, K. *RISE: Randomized Input Sampling for Explanation of Black-box Models.* BMVC 2018. https://arxiv.org/abs/1806.07421
- [PyTorch ExecuTorch 2024] PyTorch / Meta. *ExecuTorch — XNNPACK Backend.* 2024–2025. https://docs.pytorch.org/executorch/stable/backends-xnnpack.html
- [Qualcomm 2024] Qualcomm Technologies. *Unlocking on-device generative AI with an NPU and heterogeneous computing.* 2024. https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Unlocking-on-device-generative-AI-with-an-NPU-and-heterogeneous-computing.pdf
- [Rajib 2025] Rajib, R. H., et al. *FedCTTA: A Collaborative Approach to Continual Test-Time Adaptation in Federated Learning.* IJCNN 2025. https://arxiv.org/abs/2505.13643
- [Ren 2025] Ren, Y., Gu, Z., Liu, W. *MedAlmighty: enhancing disease diagnosis with large vision model distillation.* Frontiers in Artificial Intelligence 2025. https://pmc.ncbi.nlm.nih.gov/articles/PMC12378157/
- [Romano 2019] Romano, Y., Patterson, E., Candès, E. J. *Conformalized Quantile Regression (CQR).* NeurIPS 2019. https://arxiv.org/abs/1905.03222
- [Rong 2022] Rong, Y., et al. *A Consistent and Efficient Evaluation Strategy for Attribution Methods (ROAD).* ICML 2022. https://proceedings.mlr.press/v162/rong22a.html
- [Saporta 2022] Saporta, A., et al. *Benchmarking saliency methods for chest X-ray interpretation (CheXlocalize).* Nature Machine Intelligence 2022. https://www.nature.com/articles/s42256-022-00536-x
- [Sensoy 2018] Sensoy, M., Kaplan, L., Kandemir, M. *Evidential Deep Learning to Quantify Classification Uncertainty.* NeurIPS 2018. https://arxiv.org/abs/1806.01768
- [Tian 2025] Tian, H., Xu, B., Li, S. *From Per-Image Low-Rank to Encoding Mismatch: Rethinking Feature Distillation in Vision Transformers.* ICML 2026 (arXiv:2511.15572, 2025). https://arxiv.org/abs/2511.15572
- [Tibshirani 2019] Tibshirani, R. J., et al. *Conformal Prediction Under Covariate Shift.* NeurIPS 2019. https://arxiv.org/abs/1904.06019
- [Vasu 2023a] Vasu, P. K. A., et al. *MobileOne: An Improved One Millisecond Mobile Backbone.* CVPR 2023. https://arxiv.org/abs/2206.04040
- [W3C 2026] W3C Web ML Working Group. *Web Neural Network API (WebNN).* W3C Candidate Recommendation, Jan 2026. https://www.w3.org/TR/webnn/
- [Wang 2020] Wang, H., et al. *Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks.* CVPR Workshops 2020 (Score-CAM); EBPG metric per the same evaluation line. https://arxiv.org/abs/1910.01279
- [Worodria 2024] Worodria, W., et al. (R2D2 TB Network). *An independent, multi-country head-to-head accuracy comparison of automated chest x-ray algorithms for the triage of pulmonary tuberculosis.* medRxiv 2024 (Annals ATS). https://pmc.ncbi.nlm.nih.gov/articles/PMC11213091/
- [Wu 2022] Wu, K., et al. *TinyViT: Fast Pretraining Distillation for Small Vision Transformers.* ECCV 2022. https://arxiv.org/abs/2207.10666
- [Wu 2024] Wu, J., et al. *On the Faithfulness of Vision Transformer Explanations (SaCo).* CVPR 2024. https://arxiv.org/abs/2404.01415
- [Xiao 2023] Xiao, G., et al. *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* ICML 2023. https://arxiv.org/abs/2211.10438
- [Yang 2023] Yang, H., et al. *Global Vision Transformer Pruning with Hessian-Aware Saliency (NViT).* CVPR 2023. https://arxiv.org/abs/2110.04869
- [Yuan 2022] Yuan, Z., et al. *PTQ4ViT: Post-Training Quantization for Vision Transformers with Twin Uniform Quantization.* ECCV 2022. https://arxiv.org/abs/2111.12293
- [Yuksekgonul 2023] Yuksekgonul, M., Wang, M., Zou, J. *Post-hoc Concept Bottleneck Models.* ICLR 2023 (Spotlight). https://arxiv.org/abs/2205.15480
- [Zagoruyko 2017] Zagoruyko, S., Komodakis, N. *Paying More Attention to Attention: Improving the Performance of CNNs via Attention Transfer.* ICLR 2017. https://arxiv.org/abs/1612.03928
