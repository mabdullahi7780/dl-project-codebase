"""Domain-Adversarial Mixture-of-Experts (DA-MoE) for Timika scoring.

This is the project's novelty over Kantipudi et al. (JIIM 2024). Kantipudi train
ONE network per task (ALP regressor / cavity classifier / direct-Timika
regressor) and run their three approaches (A1/A2/A3) independently. Their own
results expose a cross-country generalisation failure (the held-out *Moldova*
cohort is the sickest — ALP mean ~40 vs a ~26 training pool — and a single
regressor collapses toward the training mean). This module attacks that failure
with three additions, all on a SHARED DenseNet121 trunk (memory ~= one DenseNet):

  1. **Mixture of experts + gate.** K experts produce Timika "views"; a gating
     network routes per-image. A high-severity expert can own Moldova-like cases
     instead of the whole model regressing to the mean.

  2. **Domain-adversarial trunk (DANN).** A gradient-reversal country classifier
     pushes the trunk to be country-invariant, so the gate routes on *content /
     severity* rather than memorising the training countries. This is what makes
     the routing transfer to the unseen held-out country (domain generalisation;
     the test country is NEVER shown to the adversary).

  3. **Critic refinement.** A small head reads the trunk features, the per-view
     predictions and their disagreement, and emits a residual correction gated by
     predicted uncertainty — it nudges only high-disagreement / low-confidence
     cases (perception -> critique -> refine).

The SAME model covers all three Kantipudi approaches via :class:`MoEConfig.mode`:

    a2      ALP experts -> view = (100*ALP_k + 40*cavity) / 140   (cavity = frozen agent)
    a3      direct-Timika experts -> view = T_m                   (no cavity agent)
    a1      detection-ALP view = (det_alp + 40*cavity) / 140      (det_alp precomputed)
    fusion  all of the above (the flagship)

Everything is expressed as a Timika *fraction* in [0, 1]; multiply by 140 to get
the radiologist-scale Timika score.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.components.baseline_paper import FEATURE_DIM, _DenseNet121Features

TIMIKA_MAX = 140.0
CAVITY_BONUS = 40.0


# ── Gradient Reversal Layer (Ganin & Lempitsky, 2015) ─────────────────────────

class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):  # type: ignore[no-untyped-def]
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[no-untyped-def]
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return _GradReverse.apply(x, lambd)


# ── Config + output container ─────────────────────────────────────────────────

@dataclass
class MoEConfig:
    mode: str = "a2"                 # a2 | a3 | a1 | fusion
    n_alp_experts: int = 4           # A2 views (active in a2, fusion)
    n_timika_experts: int = 4        # A3 views (active in a3, fusion)
    use_a1_view: bool = False        # A1 detection view (needs det_alp); set by mode/CLI
    n_train_countries: int = 5       # DANN output dim (set from the split)
    gate_temp: float = 1.0           # softmax temperature; <1 sharpens routing
    use_dann: bool = True
    use_critic: bool = True
    critic_scale: float = 0.25       # max critic correction in Timika-fraction space
    pretrained: bool = True

    def view_kinds(self) -> list[str]:
        """Ordered list of view kinds the gate fuses, given the mode."""
        kinds: list[str] = []
        if self.mode in ("a2", "fusion"):
            kinds += ["a2"] * self.n_alp_experts
        if self.mode in ("a3", "fusion"):
            kinds += ["a3"] * self.n_timika_experts
        if self.use_a1_view or self.mode == "a1":
            kinds += ["a1"]
        if not kinds:
            raise ValueError(f"mode={self.mode!r} produced no views; check the config.")
        return kinds


@dataclass
class MoEOutput:
    timika_frac: torch.Tensor                       # [B] final Timika in [0,1]
    view_preds: torch.Tensor                        # [B, V] per-view Timika fraction
    gate_w: torch.Tensor                            # [B, V] gate weights (sum to 1)
    disagreement: torch.Tensor                      # [B] std across views (detached input to critic)
    alp_frac: torch.Tensor | None = None            # [B] gate-weighted ALP fraction (a2/fusion)
    country_logits: torch.Tensor | None = None      # [B, C] DANN head (train only)
    critic_delta: torch.Tensor | None = None        # [B] applied residual correction
    expert_alp: torch.Tensor | None = None          # [B, n_alp_experts] raw ALP-expert fractions
    expert_timika: torch.Tensor | None = None       # [B, n_timika_experts] raw Timika-expert fractions


# ── The model ─────────────────────────────────────────────────────────────────

class RegressionMoE(nn.Module):
    """Shared-trunk MoE over Timika views, with DANN + critic.

    The cavity classifier is a SEPARATE frozen agent (trained on the paper's
    balanced split, exactly like A2); its probability is passed into ``forward``
    via ``cav_prob`` to assemble the A2/A1 views. ``det_alp`` (0-100, from the A1
    YOLO+lung pipeline) is passed in for the A1 view. Both are optional depending
    on the mode.
    """

    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.kinds = cfg.view_kinds()
        self.n_views = len(self.kinds)

        self.trunk = _DenseNet121Features(cfg.pretrained)

        # Expert heads (linear over shared features). Distinct inits give the
        # gate something to specialise over; a load-balance term (in training)
        # keeps experts from collapsing to one.
        self.alp_experts = nn.ModuleList(
            nn.Linear(FEATURE_DIM, 1) for _ in range(cfg.n_alp_experts)
        ) if cfg.mode in ("a2", "fusion") else nn.ModuleList()
        self.timika_experts = nn.ModuleList(
            nn.Linear(FEATURE_DIM, 1) for _ in range(cfg.n_timika_experts)
        ) if cfg.mode in ("a3", "fusion") else nn.ModuleList()

        self.gate = nn.Linear(FEATURE_DIM, self.n_views)

        if cfg.use_dann:
            self.domain_head = nn.Sequential(
                nn.Linear(FEATURE_DIM, 256), nn.ReLU(inplace=True),
                nn.Dropout(0.5), nn.Linear(256, cfg.n_train_countries),
            )
        if cfg.use_critic:
            # reads [feat, view_preds, gate_w, disagreement] -> (delta, confidence)
            critic_in = FEATURE_DIM + 2 * self.n_views + 1
            self.critic = nn.Sequential(
                nn.Linear(critic_in, 256), nn.ReLU(inplace=True),
                nn.Dropout(0.3), nn.Linear(256, 2),
            )

    # -- expert forward helpers -------------------------------------------------

    def _alp_fracs(self, feat: torch.Tensor) -> torch.Tensor:
        """[B, n_alp_experts] ALP fractions in [0,1]."""
        if not self.alp_experts:
            return feat.new_zeros((feat.size(0), 0))
        return torch.cat([torch.sigmoid(h(feat)) for h in self.alp_experts], dim=1)

    def _timika_fracs(self, feat: torch.Tensor) -> torch.Tensor:
        if not self.timika_experts:
            return feat.new_zeros((feat.size(0), 0))
        return torch.cat([torch.sigmoid(h(feat)) for h in self.timika_experts], dim=1)

    def features(self, image: torch.Tensor) -> torch.Tensor:
        return self.trunk(image)

    # -- full forward -----------------------------------------------------------

    def forward(
        self,
        image: torch.Tensor,
        *,
        cav_prob: torch.Tensor | None = None,   # [B] P(cavity) from frozen agent
        det_alp: torch.Tensor | None = None,    # [B] detection ALP in [0,100] (A1)
        grl_lambda: float = 0.0,
        feat: torch.Tensor | None = None,       # reuse precomputed features if given
    ) -> MoEOutput:
        feat = self.trunk(image) if feat is None else feat
        B = feat.size(0)
        if cav_prob is None:
            cav_prob = feat.new_zeros(B)

        alp_e = self._alp_fracs(feat)        # [B, Ka]
        tim_e = self._timika_fracs(feat)     # [B, Km]

        views: list[torch.Tensor] = []
        if self.cfg.mode in ("a2", "fusion") and alp_e.size(1):
            # (100*ALP + 40*cav) / 140 per ALP expert
            a2_views = (100.0 * alp_e + CAVITY_BONUS * cav_prob.unsqueeze(1)) / TIMIKA_MAX
            views.append(a2_views)
        if self.cfg.mode in ("a3", "fusion") and tim_e.size(1):
            views.append(tim_e)
        if (self.cfg.use_a1_view or self.cfg.mode == "a1"):
            if det_alp is None:
                det_alp = feat.new_zeros(B)
            a1_view = ((det_alp + CAVITY_BONUS * cav_prob) / TIMIKA_MAX).unsqueeze(1)
            views.append(a1_view)
        view_preds = torch.cat(views, dim=1).clamp(0.0, 1.0)   # [B, V]

        gate_logits = self.gate(feat) / max(self.cfg.gate_temp, 1e-3)
        gate_w = F.softmax(gate_logits, dim=1)                 # [B, V]
        gated = (gate_w * view_preds).sum(dim=1)               # [B]

        if view_preds.size(1) > 1:
            disagreement = view_preds.std(dim=1, unbiased=False)   # [B]
        else:
            disagreement = view_preds.new_zeros(B)                 # single view -> no disagreement

        critic_delta = None
        timika_frac = gated
        if self.cfg.use_critic:
            crit_in = torch.cat(
                [feat, view_preds, gate_w, disagreement.unsqueeze(1)], dim=1
            )
            raw = self.critic(crit_in)
            delta = torch.tanh(raw[:, 0]) * self.cfg.critic_scale
            # apply mainly where the experts disagree (uncertainty-gated refine)
            confidence = torch.sigmoid(raw[:, 1])
            critic_delta = delta * (1.0 - confidence)
            timika_frac = (gated + critic_delta).clamp(0.0, 1.0)

        country_logits = None
        if self.cfg.use_dann and self.training:
            country_logits = self.domain_head(grad_reverse(feat, grl_lambda))

        # gate-weighted ALP (over A2 views only) for ALP-MAE reporting
        alp_frac = None
        if alp_e.size(1):
            a2_w = gate_w[:, : alp_e.size(1)]
            denom = a2_w.sum(dim=1, keepdim=True).clamp_min(1e-6)
            alp_frac = (a2_w / denom * alp_e).sum(dim=1)

        return MoEOutput(
            timika_frac=timika_frac, view_preds=view_preds, gate_w=gate_w,
            disagreement=disagreement.detach(), alp_frac=alp_frac,
            country_logits=country_logits, critic_delta=critic_delta,
            expert_alp=alp_e if alp_e.size(1) else None,
            expert_timika=tim_e if tim_e.size(1) else None,
        )


def load_balance_loss(gate_w: torch.Tensor) -> torch.Tensor:
    """Encourage all experts to be used across the batch (Shazeer-style).

    Penalises the squared coefficient of variation of mean per-expert usage, so
    the gate cannot collapse onto a single expert. Returns 0 for a single view.
    """
    if gate_w.size(1) < 2:
        return gate_w.new_zeros(())
    usage = gate_w.mean(dim=0)                       # [V]
    return (usage.std() / (usage.mean() + 1e-6)) ** 2
