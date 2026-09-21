import torch
import torch.nn as nn
import torch.nn.functional as F
from tcif_ablation_config import validate_ablation


TCIF_OUTPUT_MODES = ("local", "context", "posterior")


class TemporalContextInnovationFilter(nn.Module):
    """Fuse a masked temporal prior with the current-clip observation.

    The context branch never reads ``local_features`` while constructing the
    prior.  This makes the returned context prediction a genuine masked-center
    diagnostic.  The posterior feature is written back as a zero-initialized
    residual so enabling TCIF preserves the pre-TCIF prediction function at
    initialization.
    """

    def __init__(
        self,
        feature_dim: int,
        latent_dim: int = 128,
        context_temperature: float = 1.0,
        min_variance: float = 1e-3,
        max_log_variance: float = 6.0,
        enable_transition_gate: bool = False,
        transition_gate_hidden_dim: int = 64,
        transition_gate_init_bias: float = 2.0,
        fusion_rule: str = "precision",
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.latent_dim = int(latent_dim)
        self.context_temperature = float(context_temperature)
        self.min_variance = float(min_variance)
        self.max_log_variance = float(max_log_variance)
        self.enable_transition_gate = bool(enable_transition_gate)
        self.transition_gate_hidden_dim = int(transition_gate_hidden_dim)
        self.transition_gate_init_bias = float(transition_gate_init_bias)
        if fusion_rule not in ("precision", "equal"):
            raise ValueError(f"Unknown fusion rule: {fusion_rule}")
        self.fusion_rule = fusion_rule

        if self.feature_dim <= 0:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if self.context_temperature <= 0.0:
            raise ValueError(
                "context_temperature must be positive, "
                f"got {context_temperature}"
            )
        if self.min_variance <= 0.0:
            raise ValueError(f"min_variance must be positive, got {min_variance}")
        if self.transition_gate_hidden_dim <= 0:
            raise ValueError(
                "transition_gate_hidden_dim must be positive, "
                f"got {transition_gate_hidden_dim}"
            )

        self.local_stats = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 2 * self.latent_dim),
        )
        self.context_stats = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 2 * self.latent_dim),
        )
        self.relative_position_encoder = nn.Sequential(
            nn.Linear(2, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.context_score = nn.Sequential(
            nn.LayerNorm(self.latent_dim),
            nn.Linear(self.latent_dim, 1),
        )
        self.output_projection = nn.Linear(self.latent_dim, self.feature_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        if self.enable_transition_gate:
            self.transition_gate = nn.Sequential(
                nn.LayerNorm(4 * self.latent_dim),
                nn.Linear(4 * self.latent_dim, self.transition_gate_hidden_dim),
                nn.GELU(),
                nn.Linear(self.transition_gate_hidden_dim, 1),
            )
            nn.init.zeros_(self.transition_gate[-1].weight)
            nn.init.constant_(
                self.transition_gate[-1].bias,
                self.transition_gate_init_bias,
            )
        else:
            self.transition_gate = None

    def _split_stats(self, stats: torch.Tensor):
        mean, raw_log_variance = stats.chunk(2, dim=-1)
        log_variance = raw_log_variance.clamp(
            min=-self.max_log_variance,
            max=self.max_log_variance,
        )
        variance = F.softplus(log_variance) + self.min_variance
        return mean, variance, log_variance

    @staticmethod
    def _validate_inputs(
        local_features: torch.Tensor,
        context_features: torch.Tensor,
        context_valid_mask: torch.Tensor,
        context_relative_pos: torch.Tensor,
    ):
        if local_features.ndim != 2:
            raise ValueError(
                "local_features must have shape [B,F], "
                f"got {tuple(local_features.shape)}"
            )
        if context_features.ndim != 3:
            raise ValueError(
                "context_features must have shape [B,C,F], "
                f"got {tuple(context_features.shape)}"
            )
        batch_size, context_count, feature_dim = context_features.shape
        if local_features.shape != (batch_size, feature_dim):
            raise ValueError(
                "local/context feature shapes disagree: "
                f"local={tuple(local_features.shape)} "
                f"context={tuple(context_features.shape)}"
            )
        expected_context_shape = (batch_size, context_count)
        if context_valid_mask.shape != expected_context_shape:
            raise ValueError(
                "context_valid_mask must match [B,C]="
                f"{expected_context_shape}, got {tuple(context_valid_mask.shape)}"
            )
        if context_relative_pos.shape != expected_context_shape:
            raise ValueError(
                "context_relative_pos must match [B,C]="
                f"{expected_context_shape}, got {tuple(context_relative_pos.shape)}"
            )

    def forward(
        self,
        local_features: torch.Tensor,
        context_features: torch.Tensor,
        context_valid_mask: torch.Tensor,
        context_relative_pos: torch.Tensor,
        output_mode: str = "posterior",
    ):
        self._validate_inputs(
            local_features,
            context_features,
            context_valid_mask,
            context_relative_pos,
        )
        if output_mode not in TCIF_OUTPUT_MODES:
            raise ValueError(
                f"Unsupported TCIF output_mode={output_mode!r}; "
                f"expected one of {TCIF_OUTPUT_MODES}"
            )

        context_valid_mask = context_valid_mask.to(
            device=local_features.device,
            dtype=torch.bool,
        )
        context_relative_pos = context_relative_pos.to(
            device=local_features.device,
            dtype=local_features.dtype,
        )
        context_features = context_features.to(
            device=local_features.device,
            dtype=local_features.dtype,
        )

        local_mean, local_variance, local_log_variance = self._split_stats(
            self.local_stats(local_features)
        )
        context_mean, context_variance, context_log_variance = self._split_stats(
            self.context_stats(context_features)
        )

        relative_position_input = torch.stack(
            [
                context_relative_pos,
                torch.sign(context_relative_pos)
                * torch.log1p(context_relative_pos.abs()),
            ],
            dim=-1,
        )
        context_keys = context_mean + self.relative_position_encoder(
            relative_position_input
        )
        context_scores = self.context_score(context_keys).squeeze(-1)
        context_scores = context_scores - (
            context_relative_pos.abs() / self.context_temperature
        )
        context_scores = context_scores - 0.5 * context_log_variance.mean(dim=-1)

        masked_scores = context_scores.masked_fill(~context_valid_mask, -1e4)
        context_attention = torch.softmax(masked_scores, dim=1)
        context_attention = context_attention * context_valid_mask.to(
            dtype=context_attention.dtype
        )
        context_attention = context_attention / context_attention.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-12)

        prior_mean = torch.sum(
            context_attention.unsqueeze(-1) * context_mean,
            dim=1,
        )
        centered_context = context_mean - prior_mean.unsqueeze(1)
        prior_variance = torch.sum(
            context_attention.unsqueeze(-1)
            * (context_variance + centered_context.square()),
            dim=1,
        ).clamp_min(self.min_variance)

        has_context = context_valid_mask.any(dim=1)
        has_context_float = has_context.to(dtype=local_features.dtype).unsqueeze(-1)
        prior_mean = torch.where(has_context.unsqueeze(-1), prior_mean, local_mean)
        prior_variance = torch.where(
            has_context.unsqueeze(-1),
            prior_variance,
            local_variance,
        )

        local_precision = local_variance.reciprocal()
        prior_precision = prior_variance.reciprocal()
        posterior_variance = (local_precision + prior_precision).reciprocal()
        posterior_mean = posterior_variance * (
            local_precision * local_mean + prior_precision * prior_mean
        )
        if self.fusion_rule == "equal":
            # Only change the feature update; retain learned statistics in
            # neighbor scoring and the continuation gate.
            posterior_mean = 0.5 * (local_mean + prior_mean)
            posterior_variance = 0.25 * (local_variance + prior_variance)

        if self.transition_gate is not None:
            transition_gate_input = torch.cat(
                [
                    (local_mean - prior_mean).abs(),
                    torch.log(local_variance.clamp_min(self.min_variance)),
                    torch.log(prior_variance.clamp_min(self.min_variance)),
                    prior_mean.abs(),
                ],
                dim=-1,
            )
            continuation_gate_logits = self.transition_gate(
                transition_gate_input
            ).squeeze(-1)
            continuation_gate = torch.sigmoid(continuation_gate_logits)
        else:
            continuation_gate_logits = local_mean.new_full(
                (local_mean.shape[0],),
                20.0,
            )
            continuation_gate = local_mean.new_ones((local_mean.shape[0],))
        continuation_gate = continuation_gate * has_context.to(
            dtype=continuation_gate.dtype
        )

        context_delta = (prior_mean - local_mean) * has_context_float
        posterior_delta = (posterior_mean - local_mean) * has_context_float
        gate_scale = continuation_gate.unsqueeze(-1)
        context_feature = local_features + gate_scale * self.output_projection(
            context_delta
        )
        posterior_feature = local_features + gate_scale * self.output_projection(
            posterior_delta
        )

        if output_mode == "local":
            output_feature = local_features
        elif output_mode == "context":
            output_feature = context_feature
        else:
            output_feature = posterior_feature

        return {
            "feature": output_feature,
            "local_feature": local_features,
            "context_feature": context_feature,
            "posterior_feature": posterior_feature,
            "local_mean": local_mean,
            "local_variance": local_variance,
            "local_log_variance": local_log_variance,
            "prior_mean": prior_mean,
            "prior_variance": prior_variance,
            "posterior_mean": posterior_mean,
            "posterior_variance": posterior_variance,
            "innovation": local_mean - prior_mean,
            "context_attention": context_attention,
            "has_context": has_context,
            "continuation_gate": continuation_gate,
            "continuation_gate_logits": continuation_gate_logits,
        }


class StandardContextFusion(nn.Module):
    """Masked neighbor mean + concatenation MLP, with the same context auxiliary head.

    Variance/innovation diagnostics are NaN because this baseline has no
    probabilistic filter. Prediction and auxiliary features remain finite.
    """

    def __init__(self, feature_dim, latent_dim, target_parameters):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        # Active parameters only: context projection and residual network.
        fixed = 7 * feature_dim + feature_dim * latent_dim + latent_dim
        per_hidden = 3 * feature_dim + 1
        self.hidden_dim = max(1, round((target_parameters - fixed) / per_hidden))
        self.context_projection = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, latent_dim)
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(2 * feature_dim),
            nn.Linear(2 * feature_dim, self.hidden_dim), nn.GELU(),
            nn.Linear(self.hidden_dim, feature_dim),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(self, local_features, context_features, context_valid_mask,
                context_relative_pos, output_mode="posterior"):
        TemporalContextInnovationFilter._validate_inputs(
            local_features, context_features, context_valid_mask, context_relative_pos
        )
        if output_mode not in ("posterior", "local"):
            raise ValueError("Standard context fusion supports posterior/local diagnostics only")
        mask = context_valid_mask.to(device=local_features.device, dtype=torch.bool)
        context_features = context_features.to(local_features)
        weights = mask.to(local_features.dtype)
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1)
        pooled = (context_features.masked_fill(~mask.unsqueeze(-1), 0)
                  * weights.unsqueeze(-1)).sum(1)
        valid = mask.any(1)
        delta = self.fusion(torch.cat((local_features, pooled), dim=-1))
        posterior = local_features + delta * valid.unsqueeze(-1).to(delta.dtype)
        prior = self.context_projection(pooled)
        unavailable = prior.new_full(prior.shape, float("nan"))
        return {
            "feature": local_features if output_mode == "local" else posterior,
            "prior_mean": prior, "has_context": valid,
            "context_attention": weights,
            "innovation": unavailable,
            "prior_variance": unavailable, "posterior_variance": unavailable,
            "continuation_gate": valid.to(local_features.dtype),
            "continuation_gate_logits": local_features.new_full((len(valid),), 20.),
        }


def build_task_filters(ablation="full", **kwargs):
    """Preserve original initialization and parameter names for legacy checkpoints.

    Draw both original filters before changing structure so variant creation
    does not shift initialization of subsequent context heads or the RNG state.
    The ordinary baseline's own random draws are isolated from that stream.
    """
    validate_ablation(ablation)
    original_kwargs = dict(kwargs)
    if ablation != "full":
        original_kwargs["enable_transition_gate"] = True
    regression = TemporalContextInnovationFilter(**original_kwargs)
    ordinal = TemporalContextInnovationFilter(**original_kwargs)
    if ablation == "equal_weight":
        regression.fusion_rule = ordinal.fusion_rule = "equal"
    elif ablation == "no_gate":
        regression.transition_gate = ordinal.transition_gate = None
        regression.enable_transition_gate = ordinal.enable_transition_gate = False
    elif ablation == "shared_filter":
        ordinal = regression
    elif ablation == "standard_context":
        target = sum(p.numel() for p in regression.parameters())
        with torch.random.fork_rng(devices=[]):
            regression = StandardContextFusion(kwargs["feature_dim"], kwargs["latent_dim"], target)
            ordinal = StandardContextFusion(kwargs["feature_dim"], kwargs["latent_dim"], target)
    return regression, ordinal
