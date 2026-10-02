import torch
import torch.nn as nn
from models.tcif import TemporalContextInnovationFilter

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
        fixed = 5 * feature_dim  # MOSI context_aux=0: match active posterior MLP only
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



def prepare(model,variant):
    if variant=='tcif':return
    if variant!='standard':raise ValueError(variant)
    original=model.tcif_regression
    target=sum(p.numel() for p in original.parameters())
    device=next(model.parameters()).device
    with torch.random.fork_rng(devices=[]):
        model.tcif_regression=StandardContextFusion(model.head_input_dim,model.tcif_latent_dim,target).to(device)
        model.tcif_ordinal=StandardContextFusion(model.head_input_dim,model.tcif_latent_dim,target).to(device)
    model.tcif_enable_transition_gate=False
    actual=sum(p.numel() for p in model.tcif_regression.fusion.parameters())
    total=sum(p.numel() for p in model.tcif_regression.parameters())
    assert abs(actual-target)/target<.005
    model.context_variant_record=dict(variant=variant,tcif_reference_params_per_branch=target,
        standard_active_posterior_params_per_branch=actual,standard_total_params_per_branch=total,
        relative_active_difference=(actual-target)/target,context_projection_usage="diagnostic prior only; context_aux=0")
