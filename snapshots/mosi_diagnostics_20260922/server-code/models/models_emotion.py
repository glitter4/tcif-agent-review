from functools import partial
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers.utils
import transformers.utils.import_utils as _import_utils

def _disable_sklearn():
    def _false():
        return False
    transformers.utils.is_sklearn_available = _false
    _import_utils.is_sklearn_available = _false

_disable_sklearn()

from transformers import AutoModel, BertConfig, BertModel, ViTConfig, ViTModel, HubertConfig, HubertModel, ResNetConfig, ResNetModel
from head_utils import (
    cls7_expected_value,
    cls7_expected_value_from_probs,
    compute_final_prediction,
    cumulative_logits_to_cls7_probs,
    get_cls7_centers,
    hier_sign_mag_to_cls7_probs,
    probs_to_logits,
)
from models.tcif import TCIF_OUTPUT_MODES, TemporalContextInnovationFilter

def softmax(x: torch.Tensor, dim) -> torch.Tensor:
    max_vals = torch.amax(x, dim=dim, keepdim=True)
    e_x = torch.exp(x - max_vals)
    sum_exp = e_x.sum(dim=dim, keepdim=True)
    return e_x / sum_exp


def _temporal_feature_mix(
    feat_task1: torch.Tensor,
    feat_task2: torch.Tensor,
    detach_task2: bool = False,
) -> torch.Tensor:
    """Keep the temporal forward value fixed while optionally blocking task-2 gradients."""
    task2_source = feat_task2.detach() if detach_task2 else feat_task2
    return 0.5 * (feat_task1 + task2_source)


class ExpertLoadTracker:
    def __init__(self, enabled: bool = True):
        self.enabled = bool(enabled)
        self.reset()

    def set_enabled(self, enabled: bool):
        self.enabled = bool(enabled)
        if not self.enabled:
            self.reset()

    def reset(self):
        self._agg = {
            "msoe": {"vision": {}, "text": {}, "audio": {}},
            "mtoe_blocks": [],
            "fusion": {"task1": {}, "task2": {}},
        }

    @staticmethod
    def _to_cpu_float_list(x):
        if x is None:
            return []
        if isinstance(x, list):
            return [float(v) for v in x]
        return [float(v) for v in x.detach().cpu().reshape(-1).tolist()]

    @staticmethod
    def _accumulate_target(target: dict, stats: dict):
        if "combine_expert_mass_sum" not in target:
            target["combine_expert_mass_sum"] = [0.0 for _ in ExpertLoadTracker._to_cpu_float_list(stats.get("combine_expert_mass"))]
            target["combine_slot_mass_sum"] = [0.0 for _ in ExpertLoadTracker._to_cpu_float_list(stats.get("combine_slot_mass"))]
            target["dispatch_expert_mass_sum"] = [0.0 for _ in ExpertLoadTracker._to_cpu_float_list(stats.get("dispatch_expert_mass"))]
            target["dispatch_slot_mass_sum"] = [0.0 for _ in ExpertLoadTracker._to_cpu_float_list(stats.get("dispatch_slot_mass"))]
            target["combine_total_mass_sum"] = 0.0
            target["dispatch_total_mass_sum"] = 0.0
            target["num_batches"] = 0
        ce = ExpertLoadTracker._to_cpu_float_list(stats.get("combine_expert_mass"))
        cs = ExpertLoadTracker._to_cpu_float_list(stats.get("combine_slot_mass"))
        de = ExpertLoadTracker._to_cpu_float_list(stats.get("dispatch_expert_mass"))
        ds = ExpertLoadTracker._to_cpu_float_list(stats.get("dispatch_slot_mass"))
        for i, v in enumerate(ce):
            target["combine_expert_mass_sum"][i] += float(v)
        for i, v in enumerate(cs):
            target["combine_slot_mass_sum"][i] += float(v)
        for i, v in enumerate(de):
            target["dispatch_expert_mass_sum"][i] += float(v)
        for i, v in enumerate(ds):
            target["dispatch_slot_mass_sum"][i] += float(v)
        target["combine_total_mass_sum"] += float(stats.get("combine_total_mass", 0.0))
        target["dispatch_total_mass_sum"] += float(stats.get("dispatch_total_mass", 0.0))
        target["num_batches"] += 1

    def update(self, batch_router_stats: dict):
        if not self.enabled:
            return
        msoe = batch_router_stats.get("msoe", {}) if isinstance(batch_router_stats, dict) else {}
        mtoe_blocks = batch_router_stats.get("mtoe_blocks", []) if isinstance(batch_router_stats, dict) else []
        fusion = batch_router_stats.get("fusion", {}) if isinstance(batch_router_stats, dict) else {}
        for mod_name in ["vision", "text", "audio"]:
            if mod_name in msoe:
                self._accumulate_target(self._agg["msoe"][mod_name], msoe[mod_name])
        for idx, block_stats in enumerate(mtoe_blocks):
            while len(self._agg["mtoe_blocks"]) <= idx:
                self._agg["mtoe_blocks"].append({})
            self._accumulate_target(self._agg["mtoe_blocks"][idx], block_stats)
        for task_name in ["task1", "task2"]:
            if task_name in fusion:
                self._accumulate_target(self._agg["fusion"][task_name], fusion[task_name])

    @staticmethod
    def _finalize_target(target: dict):
        combine_total = float(target.get("combine_total_mass_sum", 0.0))
        dispatch_total = float(target.get("dispatch_total_mass_sum", 0.0))
        combine_expert = [float(v) for v in target.get("combine_expert_mass_sum", [])]
        combine_slot = [float(v) for v in target.get("combine_slot_mass_sum", [])]
        dispatch_expert = [float(v) for v in target.get("dispatch_expert_mass_sum", [])]
        dispatch_slot = [float(v) for v in target.get("dispatch_slot_mass_sum", [])]
        return {
            "num_batches": int(target.get("num_batches", 0)),
            "num_experts": len(combine_expert),
            "num_slots": len(combine_slot),
            "combine_total_mass": combine_total,
            "dispatch_total_mass": dispatch_total,
            "combine_expert_mass": combine_expert,
            "combine_expert_share": [v / combine_total for v in combine_expert] if combine_total > 0 else [0.0 for _ in combine_expert],
            "combine_slot_mass": combine_slot,
            "combine_slot_share": [v / combine_total for v in combine_slot] if combine_total > 0 else [0.0 for _ in combine_slot],
            "dispatch_expert_mass": dispatch_expert,
            "dispatch_expert_share": [v / dispatch_total for v in dispatch_expert] if dispatch_total > 0 else [0.0 for _ in dispatch_expert],
            "dispatch_slot_mass": dispatch_slot,
            "dispatch_slot_share": [v / dispatch_total for v in dispatch_slot] if dispatch_total > 0 else [0.0 for _ in dispatch_slot],
        }

    def summary(self, reset: bool = False):
        if not self.enabled:
            return {}
        out = {"msoe": {}, "mtoe_blocks": [], "fusion": {}}
        for k in ["vision", "text", "audio"]:
            out["msoe"][k] = self._finalize_target(self._agg["msoe"][k]) if self._agg["msoe"][k] else {}
        for block in self._agg["mtoe_blocks"]:
            out["mtoe_blocks"].append(self._finalize_target(block) if block else {})
        for task_name in ["task1", "task2"]:
            out["fusion"][task_name] = self._finalize_target(self._agg["fusion"][task_name]) if self._agg["fusion"][task_name] else {}
        if reset:
            self.reset()
        return out

    @staticmethod
    def msoe_group_masses(summary: dict, mass_key: str = "dispatch_expert_mass"):
        layout = (summary or {}).get("expert_group_layout", {}) or {}
        if not isinstance(layout, dict):
            return {}
        group_names = [
            name
            for name in ["text_specific", "audio_specific", "vision_specific", "shared", "temporal_contrast"]
            if isinstance(layout.get(name), dict)
        ]
        if not group_names:
            return {}
        out = {}
        msoe = (summary or {}).get("msoe", {}) or {}
        for modality_name in ["vision", "text", "audio"]:
            modality_stats = msoe.get(modality_name, {}) if isinstance(msoe, dict) else {}
            masses = modality_stats.get(mass_key, []) if isinstance(modality_stats, dict) else []
            if not masses:
                continue
            masses = [float(v) for v in masses]
            total_mass = sum(masses)
            groups = {}
            for group_name in group_names:
                group = layout.get(group_name, {}) or {}
                indices = group.get("indices", []) if isinstance(group, dict) else []
                valid_indices = [int(idx) for idx in indices if 0 <= int(idx) < len(masses)]
                mass = sum(masses[idx] for idx in valid_indices)
                groups[group_name] = {
                    "mass": mass,
                    "share": mass / total_mass if total_mass > 0.0 else 0.0,
                    "indices": valid_indices,
                }
            out[modality_name] = {
                "total_mass": total_mass,
                "groups": groups,
            }
        return out

    @staticmethod
    def msoe_group_ratios(summary: dict, group_name: str, share_key: str = "dispatch_expert_share"):
        layout = (summary or {}).get("expert_group_layout", {}) or {}
        group = layout.get(group_name, {}) if isinstance(layout, dict) else {}
        indices = group.get("indices", []) if isinstance(group, dict) else []
        if not indices:
            return {}
        ratios = {}
        msoe = (summary or {}).get("msoe", {}) or {}
        for modality_name in ["vision", "text", "audio"]:
            modality_stats = msoe.get(modality_name, {}) if isinstance(msoe, dict) else {}
            shares = modality_stats.get(share_key, []) if isinstance(modality_stats, dict) else []
            if not shares:
                continue
            total_share = sum(float(v) for v in shares)
            if total_share <= 0.0:
                ratios[modality_name] = 0.0
                continue
            group_share = sum(float(shares[idx]) for idx in indices if 0 <= int(idx) < len(shares))
            ratios[modality_name] = group_share / total_share
        return ratios

    @staticmethod
    def brief(summary: dict, topk: int = 3):
        def _topk_share_text(shares):
            if not shares:
                return "none"
            pairs = sorted([(i, float(v)) for i, v in enumerate(shares)], key=lambda x: x[1], reverse=True)[:topk]
            return ",".join([f"e{idx}:{val:.4f}" for idx, val in pairs])
        parts = []
        for name in ["vision", "text", "audio"]:
            share = (((summary or {}).get("msoe", {}) or {}).get(name, {}) or {}).get("combine_expert_share", [])
            parts.append(f"msoe_{name}_top={_topk_share_text(share)}")
        mtoe_blocks = (summary or {}).get("mtoe_blocks", [])
        for bi, block in enumerate(mtoe_blocks):
            share = (block or {}).get("combine_expert_share", [])
            parts.append(f"mtoe_b{bi}_top={_topk_share_text(share)}")
        for task_name in ["task1", "task2"]:
            share = (((summary or {}).get("fusion", {}) or {}).get(task_name, {}) or {}).get("combine_expert_share", [])
            parts.append(f"fusion_{task_name}_top={_topk_share_text(share)}")
        return " ".join(parts)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class TemporalTransformer(nn.Module):
    def __init__(self, dim, depth=2, num_heads=8, dropout=0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x, attention_mask=None):
        if attention_mask is None:
            return self.encoder(x)
        key_padding = attention_mask == 0
        return self.encoder(x, src_key_padding_mask=key_padding)

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_layer=None, norm_layer=nn.LayerNorm, act_layer=nn.GELU, attention_dropout: float = 0.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer() if mlp_layer is not None else Mlp(dim)

    def forward(self, x, modality_token_ids: torch.Tensor = None, return_router_stats: bool = False):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x_norm = self.norm2(x)
        if return_router_stats:
            y, router_stats = self.mlp(x_norm, return_router_stats=True)
            x = x + y
            return x, router_stats
        x = x + self.mlp(x_norm)
        return x

class SoftMoELayerWrapper(nn.Module):
    def __init__(self, dim, num_experts, slots_per_expert, layer, normalize=True, router_temperature: float = 0.1, **layer_kwargs):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.normalize = normalize
        self.router_temperature = float(router_temperature)

        self.phi = nn.Parameter(torch.zeros(dim, num_experts, slots_per_expert))
        if self.normalize:
            self.scale = nn.Parameter(torch.ones(1))

        nn.init.normal_(self.phi, mean=0, std=1 / dim**0.5)
        self.experts = nn.ModuleList([layer(**layer_kwargs) for _ in range(num_experts)])

    def forward(
        self,
        x,
        modality_token_ids: torch.Tensor = None,
        return_router_stats: bool = False,
        router_logits_bias: torch.Tensor = None,
    ):
        if self.normalize:
            x = F.normalize(x, dim=2)
            phi = self.scale * F.normalize(self.phi, dim=0)
        else:
            phi = self.phi

        logits = torch.einsum("bmd,dnp->bmnp", x, phi)
        if router_logits_bias is not None:
            logits = logits + router_logits_bias
        logits = logits / self.router_temperature
        d = softmax(logits, dim=2)
        c = softmax(logits, dim=1)

        xs = torch.einsum("bmd,bmnp->bnpd", x, d)

        ys = torch.stack(
            [f_i(xs[:, i, :, :]) for i, f_i in enumerate(self.experts)],
            dim=1
        )

        y = torch.einsum("bnpd,bmnp->bmd", ys, c)
        if not return_router_stats:
            return y
        router_stats = {
            "combine_expert_mass": c.sum(dim=(0, 1, 3)).detach(),
            "combine_slot_mass": c.sum(dim=(0, 1, 2)).detach(),
            "combine_total_mass": float(c.sum().detach().item()),
            "dispatch_expert_mass": d.sum(dim=(0, 1, 3)).detach(),
            "dispatch_slot_mass": d.sum(dim=(0, 1, 2)).detach(),
            "dispatch_total_mass": float(d.sum().detach().item()),
        }
        return y, router_stats

class ConditionalMutualInformationLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, dispatch: torch.Tensor, modality_token_ids: torch.Tensor):
        if dispatch.ndim != 4:
            raise RuntimeError(f"dispatch must be 4D [B,L,N,P], got shape={tuple(dispatch.shape)}")
        bsz, seq_len, _, _ = dispatch.shape
        if modality_token_ids is None:
            return dispatch.new_tensor(0.0)
        if modality_token_ids.ndim != 1 or modality_token_ids.shape[0] != seq_len:
            raise RuntimeError(
                f"modality_token_ids shape mismatch: expected [{seq_len}], got {tuple(modality_token_ids.shape)}"
            )
        num_modalities = int(modality_token_ids.max().item()) + 1
        mass = dispatch.new_zeros((num_modalities, dispatch.shape[2], dispatch.shape[3]))
        for m in range(num_modalities):
            mask = (modality_token_ids == m)
            if mask.any():
                mass[m] = dispatch[:, mask, :, :].sum(dim=(0, 1))
        total = mass.sum()
        if total <= 0:
            return dispatch.new_tensor(0.0)
        p_met = mass / (total + self.epsilon)
        p_mt = p_met.sum(dim=1, keepdim=True)
        p_et = p_met.sum(dim=0, keepdim=True)
        p_t = p_met.sum(dim=(0, 1), keepdim=True)
        log_term = torch.log(p_met + self.epsilon) + torch.log(p_t + self.epsilon) - torch.log(p_mt + self.epsilon) - torch.log(p_et + self.epsilon)
        cond_mi = (p_met * log_term).sum()
        return -cond_mi

class SoftMoELayerWrapperMET(nn.Module):
    def __init__(self, dim, num_experts, slots_per_expert, layer, normalize=True, router_temperature: float = 0.1, **layer_kwargs):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.normalize = normalize
        self.router_temperature = float(router_temperature)

        self.task_embedding1 = nn.Parameter(torch.zeros(dim))
        self.task_embedding2 = nn.Parameter(torch.zeros(dim))

        self.phi = nn.Parameter(torch.zeros(dim, num_experts, slots_per_expert))
        if self.normalize:
            self.scale = nn.Parameter(torch.ones(1))

        nn.init.normal_(self.phi, mean=0, std=1 / dim**0.5)
        self.criterion = ConditionalMutualInformationLoss()

        self.experts = nn.ModuleList([layer(**layer_kwargs) for _ in range(num_experts)])

    def forward(self, x, modality_token_ids: torch.Tensor = None, return_router_stats: bool = False):
        if self.normalize:
            x = F.normalize(x, dim=2)
            phi = self.scale * F.normalize(self.phi, dim=0)
        else:
            phi = self.phi

        logits = torch.einsum("bmd,dnp->bmnp", x, phi)
        logits = logits / self.router_temperature
        d = softmax(logits, dim=2)
        c = softmax(logits, dim=1)
        cmi_loss = self.criterion(d, modality_token_ids)

        xs = torch.einsum("bmd,bmnp->bnpd", x, d)

        task_emb1 = self.task_embedding1.view(1, 1, 1, -1)
        task_emb2 = self.task_embedding2.view(1, 1, 1, -1)

        if self.slots_per_expert >= 1:
            xs[:, :, 0, :] = xs[:, :, 0, :] + task_emb1
        if self.slots_per_expert >= 2:
            xs[:, :, 1, :] = xs[:, :, 1, :] + task_emb2

        ys = torch.stack(
            [f_i(xs[:, i, :, :]) for i, f_i in enumerate(self.experts)],
            dim=1
        )

        y = torch.einsum("bnpd,bmnp->bmpd", ys, c)

        y_task1 = y[:, :, 0, :] if self.slots_per_expert >= 1 else y
        y_task2 = y[:, :, 1, :] if self.slots_per_expert >= 2 else None
        if not return_router_stats:
            return y_task1, y_task2, cmi_loss
        router_stats = {
            "combine_expert_mass": c.sum(dim=(0, 1, 3)).detach(),
            "combine_slot_mass": c.sum(dim=(0, 1, 2)).detach(),
            "combine_total_mass": float(c.sum().detach().item()),
            "dispatch_expert_mass": d.sum(dim=(0, 1, 3)).detach(),
            "dispatch_slot_mass": d.sum(dim=(0, 1, 2)).detach(),
            "dispatch_total_mass": float(d.sum().detach().item()),
        }
        return y_task1, y_task2, cmi_loss, router_stats


class TaskQueryPool(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        embed_dim = int(embed_dim)
        num_heads = int(num_heads)
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}")
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError(f"num_heads must evenly divide embed_dim, got embed_dim={embed_dim}, num_heads={num_heads}")
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)

    def forward(
        self,
        h: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        if h.ndim != 3:
            raise ValueError(f"TaskQueryPool expects [B, L, E], got {tuple(h.shape)}")
        if h.shape[1] <= 0:
            raise ValueError("TaskQueryPool received an empty token sequence")
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.to(device=h.device, dtype=torch.bool)
            if key_padding_mask.shape != h.shape[:2]:
                raise ValueError(
                    f"TaskQueryPool key_padding_mask must match [B, L]={tuple(h.shape[:2])}, "
                    f"got {tuple(key_padding_mask.shape)}"
                )
            all_masked = key_padding_mask.all(dim=1)
            if bool(all_masked.any().item()):
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False
        q = self.query.expand(h.shape[0], -1, -1)
        if return_attention:
            out, attn = self.attn(
                q,
                h,
                h,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
            return out, attn
        out, _ = self.attn(q, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        return out


class SharedSpecificMoELayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_text_specific_experts: int,
        num_audio_specific_experts: int,
        num_vision_specific_experts: int,
        num_shared_experts: int,
        num_temporal_contrast_experts: int = 0,
        enable_text_shared_experts: bool = True,
        enable_temporal_contrast_experts: bool = False,
        normalize: bool = True,
        router_temperature: float = 0.1,
        drop: float = 0.0,
        attention_dropout: float = 0.0,
        expert_mlp_ratio: float = 1.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_text_specific_experts = int(num_text_specific_experts)
        self.num_audio_specific_experts = int(num_audio_specific_experts)
        self.num_vision_specific_experts = int(num_vision_specific_experts)
        self.num_shared_experts = int(num_shared_experts)
        self.num_temporal_contrast_experts = int(num_temporal_contrast_experts)
        self.enable_text_shared_experts = bool(enable_text_shared_experts)
        self.enable_temporal_contrast_experts = bool(enable_temporal_contrast_experts)
        self.num_experts = (
            self.num_text_specific_experts
            + self.num_audio_specific_experts
            + self.num_vision_specific_experts
            + self.num_shared_experts
            + self.num_temporal_contrast_experts
        )
        if self.num_experts <= 0:
            raise ValueError("SharedSpecificMoELayer requires at least one expert")
        self.slots_per_expert = 1
        self.normalize = bool(normalize)
        self.router_temperature = float(router_temperature)
        if self.router_temperature <= 0:
            raise ValueError(f"router_temperature must be positive, got {self.router_temperature}")
        self.expert_mlp_ratio = float(expert_mlp_ratio)
        if self.expert_mlp_ratio <= 0:
            raise ValueError(
                f"expert_mlp_ratio must be positive, got {self.expert_mlp_ratio}"
            )

        self.attn_norm = nn.ModuleDict({
            "vision": nn.LayerNorm(self.dim),
            "text": nn.LayerNorm(self.dim),
            "audio": nn.LayerNorm(self.dim),
        })
        self.self_attn = nn.ModuleDict({
            "vision": nn.MultiheadAttention(self.dim, num_heads=num_heads, dropout=attention_dropout, batch_first=True),
            "text": nn.MultiheadAttention(self.dim, num_heads=num_heads, dropout=attention_dropout, batch_first=True),
            "audio": nn.MultiheadAttention(self.dim, num_heads=num_heads, dropout=attention_dropout, batch_first=True),
        })
        self.moe_norm = nn.ModuleDict({
            "vision": nn.LayerNorm(self.dim),
            "text": nn.LayerNorm(self.dim),
            "audio": nn.LayerNorm(self.dim),
        })

        self.phi = nn.Parameter(torch.zeros(self.dim, self.num_experts, self.slots_per_expert))
        if self.normalize:
            self.scale = nn.Parameter(torch.ones(1))
        nn.init.normal_(self.phi, mean=0, std=1 / self.dim ** 0.5)
        self.experts = nn.ModuleList([
            Mlp(
                in_features=self.dim,
                hidden_features=max(1, int(round(self.dim * self.expert_mlp_ratio))),
                out_features=self.dim,
                drop=drop,
            )
            for _ in range(self.num_experts)
        ])

        self.expert_group_layout = self._build_expert_group_layout()
        self.register_buffer("text_allowed_mask", self._build_allowed_mask("text"), persistent=False)
        self.register_buffer("audio_allowed_mask", self._build_allowed_mask("audio"), persistent=False)
        self.register_buffer("vision_allowed_mask", self._build_allowed_mask("vision"), persistent=False)
        self._latest_router_stats = {}

    def _build_expert_group_layout(self):
        start = 0
        layout = {}
        groups = [
            ("text_specific", self.num_text_specific_experts),
            ("audio_specific", self.num_audio_specific_experts),
            ("vision_specific", self.num_vision_specific_experts),
            ("shared", self.num_shared_experts),
            ("temporal_contrast", self.num_temporal_contrast_experts),
        ]
        for name, count in groups:
            end = start + count
            layout[name] = {
                "start": int(start),
                "end": int(end),
                "count": int(count),
                "indices": list(range(start, end)),
            }
            start = end
        layout["total_experts"] = int(self.num_experts)
        return layout

    def _build_allowed_mask(self, modality: str) -> torch.Tensor:
        mask = torch.zeros(self.num_experts, dtype=torch.bool)
        if modality == "text":
            text_groups = ["text_specific"]
            if self.enable_text_shared_experts:
                text_groups.append("shared")
            if self.enable_temporal_contrast_experts:
                text_groups.append("temporal_contrast")
            for key in text_groups:
                for idx in self.expert_group_layout[key]["indices"]:
                    mask[idx] = True
        elif modality == "audio":
            groups = ["audio_specific", "shared"]
            if self.enable_temporal_contrast_experts:
                groups.append("temporal_contrast")
            for key in groups:
                for idx in self.expert_group_layout[key]["indices"]:
                    mask[idx] = True
        elif modality == "vision":
            groups = ["vision_specific", "shared"]
            if self.enable_temporal_contrast_experts:
                groups.append("temporal_contrast")
            for key in groups:
                for idx in self.expert_group_layout[key]["indices"]:
                    mask[idx] = True
        else:
            raise ValueError(f"Unknown modality: {modality}")
        if not bool(mask.any()):
            raise ValueError(f"No experts are accessible for modality '{modality}'")
        return mask

    @staticmethod
    def _coerce_token_mask(token_mask: torch.Tensor, x: torch.Tensor, name: str = "token_mask"):
        if token_mask is None:
            return None
        token_mask = token_mask.to(device=x.device, dtype=torch.bool)
        if token_mask.shape != x.shape[:2]:
            raise ValueError(
                f"{name} must match token shape [B, L]={tuple(x.shape[:2])}, got {tuple(token_mask.shape)}"
            )
        return token_mask

    def _masked_moe(
        self,
        x: torch.Tensor,
        allowed_mask: torch.Tensor,
        token_mask: torch.Tensor = None,
        return_router_stats: bool = False,
    ):
        if x.ndim != 3:
            raise ValueError(f"Expected [B, L, E] tokens, got {tuple(x.shape)}")
        if x.shape[1] <= 0:
            raise ValueError("MoE received an empty token sequence")
        if x.shape[2] != self.dim:
            raise ValueError(f"Expected embedding dim {self.dim}, got {x.shape[2]}")
        token_mask = self._coerce_token_mask(token_mask, x)

        if self.normalize:
            x = F.normalize(x, dim=2)
            phi = self.scale * F.normalize(self.phi, dim=0)
        else:
            phi = self.phi

        logits = torch.einsum("bmd,dnp->bmnp", x, phi)
        logits = logits / self.router_temperature
        mask = allowed_mask.to(device=logits.device, dtype=torch.bool).view(1, 1, self.num_experts, 1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        mask_f = mask.to(dtype=logits.dtype)

        d = softmax(logits, dim=2)
        if token_mask is not None:
            token_mask_f = token_mask.to(dtype=logits.dtype).view(x.shape[0], x.shape[1], 1, 1)
            d = d * token_mask_f
            combine_logits = logits.masked_fill(
                ~token_mask.view(x.shape[0], x.shape[1], 1, 1),
                torch.finfo(logits.dtype).min,
            )
            c = softmax(combine_logits, dim=1) * token_mask_f * mask_f
        else:
            c = softmax(logits, dim=1) * mask_f
        xs = torch.einsum("bmd,bmnp->bnpd", x, d)
        ys = torch.stack([expert(xs[:, idx, :, :]) for idx, expert in enumerate(self.experts)], dim=1)
        y = torch.einsum("bnpd,bmnp->bmd", ys, c)

        if not return_router_stats:
            return y
        router_stats = {
            "combine_expert_mass": c.sum(dim=(0, 1, 3)).detach(),
            "combine_expert_mass_per_sample": c.sum(dim=(1, 3)).detach(),
            "combine_slot_mass": c.sum(dim=(0, 1, 2)).detach(),
            "combine_total_mass": float(c.sum().detach().item()),
            "dispatch_expert_mass": d.sum(dim=(0, 1, 3)).detach(),
            "dispatch_expert_mass_per_sample": d.sum(dim=(1, 3)).detach(),
            "dispatch_slot_mass": d.sum(dim=(0, 1, 2)).detach(),
            "dispatch_total_mass": float(d.sum().detach().item()),
            "allowed_expert_mask": allowed_mask.detach().cpu().tolist(),
            "expert_group_layout": self.expert_group_layout,
        }
        return y, router_stats

    def _forward_modality(
        self,
        modality: str,
        x: torch.Tensor,
        token_mask: torch.Tensor = None,
        return_router_stats: bool = False,
    ):
        token_mask = self._coerce_token_mask(token_mask, x, name=f"{modality}_token_mask")
        key_padding_mask = None
        if token_mask is not None:
            key_padding_mask = ~token_mask
            all_masked = key_padding_mask.all(dim=1)
            if bool(all_masked.any().item()):
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked, 0] = False
        x_norm = self.attn_norm[modality](x)
        attn_out, _ = self.self_attn[modality](
            x_norm,
            x_norm,
            x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x_norm = self.moe_norm[modality](x)
        allowed_mask = getattr(self, f"{modality}_allowed_mask")
        if return_router_stats:
            y, router_stats = self._masked_moe(x_norm, allowed_mask, token_mask=token_mask, return_router_stats=True)
            return x + y, router_stats
        y = self._masked_moe(x_norm, allowed_mask, token_mask=token_mask, return_router_stats=False)
        return x + y

    def forward(
        self,
        vision_tokens,
        text_tokens,
        audio_tokens,
        vision_mask: torch.Tensor = None,
        text_mask: torch.Tensor = None,
        audio_mask: torch.Tensor = None,
        return_router_stats: bool = False,
    ):
        if return_router_stats:
            z_v, vision_stats = self._forward_modality("vision", vision_tokens, token_mask=vision_mask, return_router_stats=True)
            z_t, text_stats = self._forward_modality("text", text_tokens, token_mask=text_mask, return_router_stats=True)
            z_a, audio_stats = self._forward_modality("audio", audio_tokens, token_mask=audio_mask, return_router_stats=True)
            self._latest_router_stats = {
                "vision": vision_stats,
                "text": text_stats,
                "audio": audio_stats,
            }
            return z_v, z_t, z_a, self._latest_router_stats
        z_v = self._forward_modality("vision", vision_tokens, token_mask=vision_mask, return_router_stats=False)
        z_t = self._forward_modality("text", text_tokens, token_mask=text_mask, return_router_stats=False)
        z_a = self._forward_modality("audio", audio_tokens, token_mask=audio_mask, return_router_stats=False)
        self._latest_router_stats = {}
        return z_v, z_t, z_a

class EmotionM4OE(nn.Module):
    def __init__(self, 
                 num_classes=7, 
                 num_aux_classes=2,
                 embed_dim=768, 
                 depth_msoe=2, 
                 depth_mtoe=2,
                 num_experts_msoe=16,
                 num_experts_mtoe=32,
                 num_shared_experts=2,
                 num_text_specific_experts=4,
                 num_audio_specific_experts=4,
                 num_vision_specific_experts=4,
                 num_temporal_contrast_experts=0,
                 enable_text_shared_experts: bool = True,
                 enable_temporal_contrast_experts: bool = False,
                 enable_temporal_contrast_loss: bool = False,
                 temporal_embedding_dim: int = 128,
                 vision_backbone_type="vit",
                 vision_backbone_path="/path/to/user/models/resnet-18",
                 vit_model_path="/path/to/user/models/vit-base-patch16-224-in21k",
                 bert_model_path="/path/to/user/models/google-bert_bert-base-uncased",
                 hubert_model_path="/path/to/user/models/hubert-base-ls960",
                 local_files_only=True,
                 temporal_depth=2,
                 temporal_heads=8,
                 dropout=0.1,
                 attention_dropout: float = 0.0,
                 expert_mlp_ratio: float = 1.0,
                 alignment_num_heads: int = 4,
                 alignment_mlp_ratio: float = 2.0,
                 router_temperature: float = 0.1,
                 enable_expert_load_output: bool = True,
                 enable_regression: bool = False,
                 vit_context_ratio: float = 0.0,
                 vit_context_lambda_mode: str = "static",
                 vit_context_lambda_init=None,
                 vit_context_attention_dim: int = 64,
                 output_head_mode: str = "legacy",
                 final_pred_eta: float = 0.0,
                 signed_class_count: int = 7,
                 label_min: float = -3.0,
                 label_max: float = 3.0,
                 enable_sign_head: bool = False,
                 final_pred_sign_beta: float = 0.0,
                 neutral_positive_gate_threshold: float = None,
                 cls7_head_type: str = "flat",
                 cumulative_p7_mix: float = 0.3,
                 temporal_detach_task2: bool = False,
                 enable_tcif: bool = False,
                 tcif_latent_dim: int = 128,
                 tcif_output_mode: str = "posterior",
                 tcif_context_temperature: float = 1.0,
                 tcif_enable_transition_gate: bool = False,
                 tcif_transition_gate_hidden_dim: int = 64,
                 tcif_transition_gate_init_bias: float = 2.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.enable_regression = bool(enable_regression)
        self.output_head_mode = str(output_head_mode)
        if self.output_head_mode not in {"legacy", "signed_reg_cls7"}:
            raise ValueError(f"Unsupported output_head_mode: {self.output_head_mode}")
        self.final_pred_eta = float(final_pred_eta)
        self.signed_class_count = int(signed_class_count)
        self.label_min = float(label_min)
        self.label_max = float(label_max)
        if self.signed_class_count <= 1:
            raise ValueError(
                f"signed_class_count must be > 1, got {self.signed_class_count}"
            )
        if self.label_max <= self.label_min:
            raise ValueError(
                f"label_max must be > label_min, got {self.label_min}..{self.label_max}"
            )
        self.enable_sign_head = bool(enable_sign_head)
        self.final_pred_sign_beta = float(final_pred_sign_beta)
        self.neutral_positive_gate_threshold = (
            None
            if neutral_positive_gate_threshold is None
            else float(neutral_positive_gate_threshold)
        )
        if self.neutral_positive_gate_threshold is not None and not (
            -1.0 <= self.neutral_positive_gate_threshold <= 1.0
        ):
            raise ValueError(
                "neutral_positive_gate_threshold must be in [-1, 1], "
                f"got {neutral_positive_gate_threshold}"
            )
        if self.neutral_positive_gate_threshold is not None and not self.enable_sign_head:
            raise ValueError("neutral positive gate requires enable_sign_head=True")
        if (
            self.neutral_positive_gate_threshold is not None
            and self.output_head_mode != "signed_reg_cls7"
        ):
            raise ValueError(
                "neutral positive gate requires output_head_mode='signed_reg_cls7'"
            )
        self.cls7_head_type = str(cls7_head_type)
        if self.cls7_head_type not in {"flat", "hier_sign_mag", "cumulative", "hybrid_cumulative"}:
            raise ValueError(f"Unsupported cls7_head_type: {self.cls7_head_type}")
        if self.signed_class_count != 7 and self.cls7_head_type != "flat":
            raise ValueError(
                "non-7-bin signed heads currently require cls7_head_type='flat'"
            )
        self.cumulative_p7_mix = float(cumulative_p7_mix)
        if not (0.0 <= self.cumulative_p7_mix <= 1.0):
            raise ValueError(f"cumulative_p7_mix must be in [0, 1], got {self.cumulative_p7_mix}")
        self.register_buffer(
            "cls7_centers",
            get_cls7_centers(
                num_classes=self.signed_class_count,
                label_min=self.label_min,
                label_max=self.label_max,
            ),
            persistent=False,
        )
        self.enable_expert_load_output = bool(enable_expert_load_output)
        self.vit_context_ratio = float(vit_context_ratio)
        self.vit_context_lambda_mode = str(vit_context_lambda_mode).lower()
        if self.vit_context_lambda_mode not in {"static", "adaptive", "attention"}:
            raise ValueError(f"Unsupported vit_context_lambda_mode: {self.vit_context_lambda_mode}")
        self.vit_context_attention_dim = int(vit_context_attention_dim)
        if self.vit_context_attention_dim <= 0:
            raise ValueError(f"vit_context_attention_dim must be positive, got {self.vit_context_attention_dim}")
        self.expert_load_tracker = ExpertLoadTracker(enabled=self.enable_expert_load_output)
        self.attention_dropout = float(attention_dropout)
        self.expert_mlp_ratio = float(expert_mlp_ratio)
        if self.expert_mlp_ratio <= 0:
            raise ValueError(
                f"expert_mlp_ratio must be positive, got {self.expert_mlp_ratio}"
            )
        self.num_experts_msoe = int(num_experts_msoe)
        self.num_experts_mtoe = int(num_experts_mtoe)
        self.num_shared_experts = int(num_shared_experts)
        self.num_text_specific_experts = int(num_text_specific_experts)
        self.num_audio_specific_experts = int(num_audio_specific_experts)
        self.num_vision_specific_experts = int(num_vision_specific_experts)
        self.num_temporal_contrast_experts = int(num_temporal_contrast_experts)
        self.enable_text_shared_experts = bool(enable_text_shared_experts)
        self.enable_temporal_contrast_experts = bool(enable_temporal_contrast_experts)
        self.enable_temporal_contrast_loss = bool(enable_temporal_contrast_loss)
        self.temporal_embedding_dim = int(temporal_embedding_dim)
        self.temporal_detach_task2 = bool(temporal_detach_task2)
        self.enable_tcif = bool(enable_tcif)
        self.tcif_latent_dim = int(tcif_latent_dim)
        self.tcif_output_mode = str(tcif_output_mode).lower()
        self.tcif_context_temperature = float(tcif_context_temperature)
        self.tcif_enable_transition_gate = bool(tcif_enable_transition_gate)
        self.tcif_transition_gate_hidden_dim = int(tcif_transition_gate_hidden_dim)
        self.tcif_transition_gate_init_bias = float(tcif_transition_gate_init_bias)
        if self.tcif_output_mode not in TCIF_OUTPUT_MODES:
            raise ValueError(
                f"Unsupported tcif_output_mode={tcif_output_mode!r}; "
                f"expected one of {TCIF_OUTPUT_MODES}"
            )
        if self.enable_tcif and self.output_head_mode != "signed_reg_cls7":
            raise ValueError("TCIF requires output_head_mode='signed_reg_cls7'")
        if self.enable_tcif and self.cls7_head_type != "flat":
            raise ValueError("TCIF stage-1 currently requires cls7_head_type='flat'")

        self.visual_backbone_type = vision_backbone_type
        self._validate_model_files(hubert_model_path)
        self.bert = self._load_backbone(bert_model_path, local_files_only, BertModel(BertConfig()))
        self.hubert = self._load_backbone(hubert_model_path, local_files_only, HubertModel(HubertConfig()))

        self._vit_vanilla_dim = int(ViTConfig().hidden_size)
        self._bert_vanilla_dim = int(BertConfig().hidden_size)
        self._hubert_vanilla_dim = int(HubertConfig().hidden_size)
        self._resnet_vanilla_dim = int(ResNetConfig().hidden_sizes[-1])

        vit_hidden_size = None
        bert_hidden_size = getattr(getattr(self.bert, "config", None), "hidden_size", None)
        hubert_hidden_size = getattr(getattr(self.hubert, "config", None), "hidden_size", None)
        bert_hidden_size = int(bert_hidden_size) if bert_hidden_size is not None else self._bert_vanilla_dim
        hubert_hidden_size = int(hubert_hidden_size) if hubert_hidden_size is not None else self._hubert_vanilla_dim

        self.vit = None
        self.resnet = None
        self.temporal_transformer = None
        if self.visual_backbone_type == "vit":
            self._validate_model_files(vit_model_path)
            self.vit = self._load_backbone(vit_model_path, local_files_only, ViTModel(ViTConfig()))
            vit_hidden_size = getattr(getattr(self.vit, "config", None), "hidden_size", None)
            vit_hidden_size = int(vit_hidden_size) if vit_hidden_size is not None else self._vit_vanilla_dim
            self.vit_projector = self._make_projector(vit_hidden_size, self._vit_vanilla_dim)
            self.visual_proj = self._make_projector(self._vit_vanilla_dim, embed_dim)
            self.vit_context_lambdas = None
            self.vit_context_lambda_gate_norm = None
            self.vit_context_lambda_gate = None
            self.vit_context_lambda_attn_norm = None
            self.vit_context_lambda_attn_q = None
            self.vit_context_lambda_attn_k = None
            self._vit_context_lambda_stats = None
            if self.vit_context_ratio > 0.0:
                init_lambdas = self._parse_vit_context_lambda_init(vit_context_lambda_init)
                self.vit_context_lambdas = nn.Parameter(
                    torch.log(init_lambdas)
                )
                if self.vit_context_lambda_mode == "adaptive":
                    self.vit_context_lambda_gate_norm = nn.LayerNorm(3 * embed_dim)
                    self.vit_context_lambda_gate = nn.Linear(3 * embed_dim, 3)
                    nn.init.zeros_(self.vit_context_lambda_gate.weight)
                    nn.init.zeros_(self.vit_context_lambda_gate.bias)
                elif self.vit_context_lambda_mode == "attention":
                    self.vit_context_lambda_attn_norm = nn.LayerNorm(embed_dim)
                    self.vit_context_lambda_attn_q = nn.Linear(embed_dim, self.vit_context_attention_dim, bias=False)
                    self.vit_context_lambda_attn_k = nn.Linear(embed_dim, self.vit_context_attention_dim, bias=False)
                    nn.init.zeros_(self.vit_context_lambda_attn_q.weight)
                    nn.init.xavier_uniform_(self.vit_context_lambda_attn_k.weight)
        else:
            self._validate_model_files(vision_backbone_path)
            self.resnet = self._load_backbone(vision_backbone_path, local_files_only, ResNetModel(ResNetConfig()))
            resnet_hidden_size = self._get_resnet_hidden_size(self.resnet)
            self.resnet_projector = self._make_projector(resnet_hidden_size, embed_dim)
            self.temporal_transformer = TemporalTransformer(embed_dim, depth=temporal_depth, num_heads=temporal_heads, dropout=dropout)
            self.vit_context_lambdas = None
            self.vit_context_lambda_gate_norm = None
            self.vit_context_lambda_gate = None
            self.vit_context_lambda_attn_norm = None
            self.vit_context_lambda_attn_q = None
            self.vit_context_lambda_attn_k = None
            self._vit_context_lambda_stats = None
            for p in self.resnet.parameters():
                p.requires_grad = False

        self.bert_projector = self._make_projector(bert_hidden_size, self._bert_vanilla_dim)
        self.hubert_projector = self._make_projector(hubert_hidden_size, self._hubert_vanilla_dim)

        self.text_proj = nn.Linear(self._bert_vanilla_dim, embed_dim)
        self.audio_proj = self._make_projector(self._hubert_vanilla_dim, embed_dim)
        self.shared_specific_layers = nn.ModuleList([
            SharedSpecificMoELayer(
                dim=embed_dim,
                num_heads=12,
                num_text_specific_experts=self.num_text_specific_experts,
                num_audio_specific_experts=self.num_audio_specific_experts,
                num_vision_specific_experts=self.num_vision_specific_experts,
                num_shared_experts=self.num_shared_experts,
                num_temporal_contrast_experts=self.num_temporal_contrast_experts,
                enable_text_shared_experts=self.enable_text_shared_experts,
                enable_temporal_contrast_experts=self.enable_temporal_contrast_experts,
                normalize=True,
                router_temperature=router_temperature,
                drop=dropout,
                attention_dropout=self.attention_dropout,
                expert_mlp_ratio=self.expert_mlp_ratio,
            )
            for _ in range(depth_msoe)
        ])
        self.expert_group_layout = (
            self.shared_specific_layers[0].expert_group_layout
            if len(self.shared_specific_layers) > 0
            else {
                "text_specific": {"start": 0, "end": self.num_text_specific_experts, "count": self.num_text_specific_experts, "indices": list(range(self.num_text_specific_experts))},
                "audio_specific": {
                    "start": self.num_text_specific_experts,
                    "end": self.num_text_specific_experts + self.num_audio_specific_experts,
                    "count": self.num_audio_specific_experts,
                    "indices": list(range(self.num_text_specific_experts, self.num_text_specific_experts + self.num_audio_specific_experts)),
                },
                "vision_specific": {
                    "start": self.num_text_specific_experts + self.num_audio_specific_experts,
                    "end": self.num_text_specific_experts + self.num_audio_specific_experts + self.num_vision_specific_experts,
                    "count": self.num_vision_specific_experts,
                    "indices": list(range(
                        self.num_text_specific_experts + self.num_audio_specific_experts,
                        self.num_text_specific_experts + self.num_audio_specific_experts + self.num_vision_specific_experts,
                    )),
                },
                "shared": {
                    "start": self.num_text_specific_experts + self.num_audio_specific_experts + self.num_vision_specific_experts,
                    "end": self.num_text_specific_experts + self.num_audio_specific_experts + self.num_vision_specific_experts + self.num_shared_experts,
                    "count": self.num_shared_experts,
                    "indices": list(range(
                        self.num_text_specific_experts + self.num_audio_specific_experts + self.num_vision_specific_experts,
                        self.num_text_specific_experts + self.num_audio_specific_experts + self.num_vision_specific_experts + self.num_shared_experts,
                    )),
                },
                "temporal_contrast": {
                    "start": self.num_text_specific_experts + self.num_audio_specific_experts + self.num_vision_specific_experts + self.num_shared_experts,
                    "end": self.num_text_specific_experts + self.num_audio_specific_experts + self.num_vision_specific_experts + self.num_shared_experts + self.num_temporal_contrast_experts,
                    "count": self.num_temporal_contrast_experts,
                    "indices": list(range(
                        self.num_text_specific_experts + self.num_audio_specific_experts + self.num_vision_specific_experts + self.num_shared_experts,
                        self.num_text_specific_experts + self.num_audio_specific_experts + self.num_vision_specific_experts + self.num_shared_experts + self.num_temporal_contrast_experts,
                    )),
                },
                "total_experts": self.num_text_specific_experts + self.num_audio_specific_experts + self.num_vision_specific_experts + self.num_shared_experts + self.num_temporal_contrast_experts,
            }
        )
        self.task_pool_intensity = TaskQueryPool(embed_dim=embed_dim, num_heads=alignment_num_heads)
        self.task_pool_polarity = TaskQueryPool(embed_dim=embed_dim, num_heads=alignment_num_heads)

        # Compatibility placeholders for old inspection code; active forward no longer uses them.
        self.msoe_v = nn.Sequential()
        self.msoe_t = nn.Sequential()
        self.msoe_a = nn.Sequential()
        self.mtoe_blocks = nn.ModuleList()
        self.fusion_task1 = nn.Identity()
        self.fusion_task2 = nn.Identity()
        self.fusion_task2_text_router = nn.Identity()

        self.head_input_dim = embed_dim * 4
        self.norm_task1 = nn.LayerNorm(self.head_input_dim)
        self.norm_task2 = nn.LayerNorm(self.head_input_dim)
        self.post_moe_dropout = nn.Dropout(dropout)

        if self.output_head_mode == "legacy":
            self.head_intensity = nn.Linear(self.head_input_dim, 1)
            self.head_polarity = nn.Linear(self.head_input_dim, num_classes)
        else:
            self.head_signed_reg = nn.Linear(self.head_input_dim, 1)
            self.head_cls7 = (
                nn.Linear(self.head_input_dim, self.signed_class_count)
                if self.cls7_head_type in {"flat", "hybrid_cumulative"}
                else None
            )
            self.head_hier_sign = nn.Linear(self.head_input_dim, 3) if self.cls7_head_type == "hier_sign_mag" else None
            self.head_hier_mag_neg = nn.Linear(self.head_input_dim, 3) if self.cls7_head_type == "hier_sign_mag" else None
            self.head_hier_mag_pos = nn.Linear(self.head_input_dim, 3) if self.cls7_head_type == "hier_sign_mag" else None
            self.head_cumulative = nn.Linear(self.head_input_dim, 6) if self.cls7_head_type in {"cumulative", "hybrid_cumulative"} else None
            self.head_sign = nn.Linear(self.head_input_dim, 1) if self.enable_sign_head else None
        if self.enable_temporal_contrast_experts or self.enable_temporal_contrast_loss:
            if self.temporal_embedding_dim <= 0:
                raise ValueError(f"temporal_embedding_dim must be positive, got {self.temporal_embedding_dim}")
            self.temporal_embedding_head = nn.Sequential(
                nn.LayerNorm(self.head_input_dim),
                nn.Linear(self.head_input_dim, self.head_input_dim),
                nn.GELU(),
                nn.Linear(self.head_input_dim, self.temporal_embedding_dim),
            )
        else:
            self.temporal_embedding_head = None
        if self.enable_tcif:
            self.tcif_regression = TemporalContextInnovationFilter(
                feature_dim=self.head_input_dim,
                latent_dim=self.tcif_latent_dim,
                context_temperature=self.tcif_context_temperature,
                enable_transition_gate=self.tcif_enable_transition_gate,
                transition_gate_hidden_dim=self.tcif_transition_gate_hidden_dim,
                transition_gate_init_bias=self.tcif_transition_gate_init_bias,
            )
            self.tcif_ordinal = TemporalContextInnovationFilter(
                feature_dim=self.head_input_dim,
                latent_dim=self.tcif_latent_dim,
                context_temperature=self.tcif_context_temperature,
                enable_transition_gate=self.tcif_enable_transition_gate,
                transition_gate_hidden_dim=self.tcif_transition_gate_hidden_dim,
                transition_gate_init_bias=self.tcif_transition_gate_init_bias,
            )
            self.tcif_context_reg_head = nn.Linear(self.tcif_latent_dim, 1)
            self.tcif_context_cls7_head = nn.Linear(
                self.tcif_latent_dim,
                self.signed_class_count,
            )
        else:
            self.tcif_regression = None
            self.tcif_ordinal = None
            self.tcif_context_reg_head = None
            self.tcif_context_cls7_head = None

    @staticmethod
    def _make_projector(in_dim: int, out_dim: int) -> nn.Module:
        if int(in_dim) == int(out_dim):
            return nn.Identity()
        proj = nn.Linear(int(in_dim), int(out_dim))
        nn.init.xavier_uniform_(proj.weight)
        if proj.bias is not None:
            nn.init.zeros_(proj.bias)
        return proj

    @staticmethod
    def _get_resnet_hidden_size(model: nn.Module) -> int:
        cfg = getattr(model, "config", None)
        hidden_sizes = getattr(cfg, "hidden_sizes", None)
        if isinstance(hidden_sizes, (list, tuple)) and hidden_sizes:
            return int(hidden_sizes[-1])
        embedding_size = getattr(cfg, "embedding_size", None)
        if embedding_size is not None:
            return int(embedding_size)
        return int(ResNetConfig().hidden_sizes[-1])

    @staticmethod
    def _validate_model_files(model_path: str) -> None:
        if not model_path:
            return
        if not os.path.isdir(model_path):
            raise RuntimeError(f"Model path not found: {model_path}")
        has_config = os.path.exists(os.path.join(model_path, "config.json"))
        has_weights = any(
            os.path.exists(os.path.join(model_path, fname))
            for fname in ["model.safetensors", "pytorch_model.bin"]
        )
        if not (has_config and has_weights):
            raise RuntimeError(f"Model files incomplete in: {model_path}")

    @staticmethod
    def _maybe_convert_bin_to_safetensors(model_dir: str) -> bool:
        safetensors_path = os.path.join(model_dir, "model.safetensors")
        bin_path = os.path.join(model_dir, "pytorch_model.bin")
        if os.path.exists(safetensors_path):
            return False
        if not os.path.exists(bin_path):
            return False
        try:
            import safetensors.torch
        except Exception:
            return False
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        safetensors.torch.save_file(state_dict, safetensors_path)
        return True

    @classmethod
    def _load_backbone(cls, model_path: str, local_files_only: bool, fallback_model: nn.Module) -> nn.Module:
        if not model_path:
            return fallback_model
        try:
            return AutoModel.from_pretrained(model_path, local_files_only=local_files_only, use_safetensors=True)
        except Exception:
            pass
        try:
            return AutoModel.from_pretrained(
                model_path,
                local_files_only=local_files_only,
                use_safetensors=False,
            )
        except Exception as e:
            if (
                isinstance(e, AttributeError)
                and "'NoneType' object has no attribute 'get'" in str(e)
                and os.path.isfile(os.path.join(model_path, "pytorch_model.bin"))
            ):
                return AutoModel.from_pretrained(
                    model_path,
                    local_files_only=local_files_only,
                    use_safetensors=False,
                )
            if "serious vulnerability issue in `torch.load`" in str(e):
                if cls._maybe_convert_bin_to_safetensors(model_path):
                    return AutoModel.from_pretrained(model_path, local_files_only=local_files_only, use_safetensors=True)
            raise

    @staticmethod
    def _extract_last_hidden_state(outputs):
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        if isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            return outputs[0]
        raise RuntimeError("Model outputs do not contain last_hidden_state")

    @staticmethod
    def _extract_resnet_features(outputs):
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            h = outputs.pooler_output
            if h.dim() == 4:
                return h.mean(dim=(-2, -1))
            if h.dim() == 3:
                return h.mean(dim=1)
            return h
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            h = outputs.last_hidden_state
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            h = outputs[0]
        else:
            raise RuntimeError("ResNet outputs do not contain usable features")
        if h.dim() == 4:
            return h.mean(dim=(-2, -1))
        if h.dim() == 3:
            return h.mean(dim=1)
        return h

    @staticmethod
    def _parse_vit_context_lambda_init(vit_context_lambda_init):
        if vit_context_lambda_init is None or vit_context_lambda_init == "":
            values = [1.0 - 2e-6, 1e-6, 1e-6]
        elif isinstance(vit_context_lambda_init, str):
            values = [float(part.strip()) for part in vit_context_lambda_init.split(",") if part.strip()]
        else:
            values = [float(x) for x in vit_context_lambda_init]
        if len(values) != 3:
            raise ValueError("vit_context_lambda_init must contain exactly three values: main,prev,next")
        init_lambdas = torch.tensor(values, dtype=torch.float32)
        if torch.any(init_lambdas <= 0):
            raise ValueError("vit_context_lambda_init values must all be positive")
        init_lambdas = init_lambdas / init_lambdas.sum().clamp_min(1e-12)
        return init_lambdas.clamp_min(1e-8)

    def _encode_vit_tokens(self, vit_images: torch.Tensor) -> torch.Tensor:
        bsz, num_frames, channels, height, width = vit_images.shape
        vit_in = vit_images.reshape(bsz * num_frames, channels, height, width)
        vit_out = self._extract_last_hidden_state(self.vit(vit_in))
        vit_out = self.vit_projector(vit_out)
        vit_out = self.visual_proj(vit_out)
        return vit_out.reshape(bsz, num_frames, vit_out.shape[1], vit_out.shape[2])

    def reset_vit_context_lambda_stats(self):
        self._vit_context_lambda_stats = None

    def _update_vit_context_lambda_stats(
        self,
        lambdas,
        attention_center_prob=None,
        attention_matrix=None,
        attention_center_entropy=None,
    ):
        if lambdas is None:
            return
        with torch.no_grad():
            vals = lambdas.detach().reshape(-1, 3).float().cpu()
            if vals.numel() == 0:
                return
            batch_count = int(vals.shape[0])
            batch_sum = vals.sum(dim=0)
            batch_sumsq = (vals * vals).sum(dim=0)
            batch_min = vals.min(dim=0).values
            batch_max = vals.max(dim=0).values
            stats = self._vit_context_lambda_stats
            if stats is None:
                self._vit_context_lambda_stats = {
                    "count": batch_count,
                    "sum": batch_sum,
                    "sumsq": batch_sumsq,
                    "min": batch_min,
                    "max": batch_max,
                }
                stats = self._vit_context_lambda_stats
            else:
                stats["count"] += batch_count
                stats["sum"] += batch_sum
                stats["sumsq"] += batch_sumsq
                stats["min"] = torch.minimum(stats["min"], batch_min)
                stats["max"] = torch.maximum(stats["max"], batch_max)

            if attention_center_prob is not None:
                attn_vals = attention_center_prob.detach().reshape(-1, 3).float().cpu()
                if attn_vals.numel() > 0:
                    attn_count = int(attn_vals.shape[0])
                    attn_sum = attn_vals.sum(dim=0)
                    attn_sumsq = (attn_vals * attn_vals).sum(dim=0)
                    attn_min = attn_vals.min(dim=0).values
                    attn_max = attn_vals.max(dim=0).values
                    if "attention_center_prob_count" not in stats:
                        stats["attention_center_prob_count"] = attn_count
                        stats["attention_center_prob_sum"] = attn_sum
                        stats["attention_center_prob_sumsq"] = attn_sumsq
                        stats["attention_center_prob_min"] = attn_min
                        stats["attention_center_prob_max"] = attn_max
                    else:
                        stats["attention_center_prob_count"] += attn_count
                        stats["attention_center_prob_sum"] += attn_sum
                        stats["attention_center_prob_sumsq"] += attn_sumsq
                        stats["attention_center_prob_min"] = torch.minimum(stats["attention_center_prob_min"], attn_min)
                        stats["attention_center_prob_max"] = torch.maximum(stats["attention_center_prob_max"], attn_max)

            if attention_matrix is not None:
                matrix_vals = attention_matrix.detach().reshape(-1, 3, 3).float().cpu()
                if matrix_vals.numel() > 0:
                    matrix_count = int(matrix_vals.shape[0])
                    matrix_sum = matrix_vals.sum(dim=0)
                    if "attention_matrix_count" not in stats:
                        stats["attention_matrix_count"] = matrix_count
                        stats["attention_matrix_sum"] = matrix_sum
                    else:
                        stats["attention_matrix_count"] += matrix_count
                        stats["attention_matrix_sum"] += matrix_sum

            if attention_center_entropy is not None:
                entropy_vals = attention_center_entropy.detach().reshape(-1).float().cpu()
                if entropy_vals.numel() > 0:
                    entropy_count = int(entropy_vals.shape[0])
                    entropy_sum = entropy_vals.sum()
                    if "attention_center_entropy_count" not in stats:
                        stats["attention_center_entropy_count"] = entropy_count
                        stats["attention_center_entropy_sum"] = entropy_sum
                    else:
                        stats["attention_center_entropy_count"] += entropy_count
                        stats["attention_center_entropy_sum"] += entropy_sum

    def get_vit_context_lambda_stats(self, reset: bool = False):
        stats = self._vit_context_lambda_stats
        if stats is None or int(stats.get("count", 0)) <= 0:
            return None
        count = float(stats["count"])
        mean = stats["sum"] / count
        var = (stats["sumsq"] / count) - mean * mean
        payload = {
            "count": int(stats["count"]),
            "mean": [float(x) for x in mean.tolist()],
            "std": [float(x) for x in torch.sqrt(var.clamp_min(0.0)).tolist()],
            "min": [float(x) for x in stats["min"].tolist()],
            "max": [float(x) for x in stats["max"].tolist()],
        }
        attn_count = int(stats.get("attention_center_prob_count", 0))
        if attn_count > 0:
            attn_mean = stats["attention_center_prob_sum"] / float(attn_count)
            attn_var = (stats["attention_center_prob_sumsq"] / float(attn_count)) - attn_mean * attn_mean
            payload.update(
                {
                    "attention_center_prob_mean": [float(x) for x in attn_mean.tolist()],
                    "attention_center_prob_std": [float(x) for x in torch.sqrt(attn_var.clamp_min(0.0)).tolist()],
                    "attention_center_prob_min": [float(x) for x in stats["attention_center_prob_min"].tolist()],
                    "attention_center_prob_max": [float(x) for x in stats["attention_center_prob_max"].tolist()],
                }
            )
        matrix_count = int(stats.get("attention_matrix_count", 0))
        if matrix_count > 0:
            matrix_mean = stats["attention_matrix_sum"] / float(matrix_count)
            payload["attention_matrix_mean"] = [
                [float(x) for x in row]
                for row in matrix_mean.tolist()
            ]
        entropy_count = int(stats.get("attention_center_entropy_count", 0))
        if entropy_count > 0:
            payload["attention_center_entropy_mean"] = float(
                (stats["attention_center_entropy_sum"] / float(entropy_count)).item()
            )
        if reset:
            self.reset_vit_context_lambda_stats()
        return payload

    def _fuse_vit_context_triplet(self, vit_groups):
        if vit_groups.dim() != 5:
            raise ValueError(f"vit_groups must have shape [B,T,K,L,D], got {tuple(vit_groups.shape)}")
        b, t, k, token_count, dim = vit_groups.shape
        center_idx = 1 if k > 1 else 0
        center_tokens = vit_groups[:, :, center_idx, :, :]
        if self.vit_context_lambdas is None or k < 3:
            return center_tokens

        prev_tokens = vit_groups[:, :, 0, :, :]
        next_tokens = vit_groups[:, :, 2, :, :]
        base_logits = self.vit_context_lambdas.to(device=center_tokens.device, dtype=center_tokens.dtype)

        center_cls = center_tokens[:, :, 0, :]
        prev_cls = prev_tokens[:, :, 0, :]
        next_cls = next_tokens[:, :, 0, :]

        if self.vit_context_lambda_mode == "adaptive":
            gate_input = torch.cat([center_cls, prev_cls, next_cls], dim=-1)
            gate_norm_weight = getattr(self.vit_context_lambda_gate_norm, "weight", None)
            gate_dtype = getattr(gate_norm_weight, "dtype", center_tokens.dtype)
            gate_input = gate_input.to(dtype=gate_dtype)
            dynamic_logits = self.vit_context_lambda_gate(
                self.vit_context_lambda_gate_norm(gate_input)
            ).to(dtype=center_tokens.dtype)
            logits = base_logits.view(1, 1, 3) + dynamic_logits
            lambdas = softmax(logits, dim=-1)
            self._update_vit_context_lambda_stats(lambdas)
            lambda_main = lambdas[..., 0].unsqueeze(-1).unsqueeze(-1)
            lambda_prev = lambdas[..., 1].unsqueeze(-1).unsqueeze(-1)
            lambda_next = lambdas[..., 2].unsqueeze(-1).unsqueeze(-1)
        elif self.vit_context_lambda_mode == "attention":
            frame_cls = torch.stack([center_cls, prev_cls, next_cls], dim=2)
            attn_norm_weight = getattr(self.vit_context_lambda_attn_norm, "weight", None)
            attn_dtype = getattr(attn_norm_weight, "dtype", center_tokens.dtype)
            frame_cls = self.vit_context_lambda_attn_norm(frame_cls.to(dtype=attn_dtype))
            q = self.vit_context_lambda_attn_q(frame_cls)
            k_proj = self.vit_context_lambda_attn_k(frame_cls)
            attn_logits = torch.matmul(q, k_proj.transpose(-1, -2)) / math.sqrt(float(self.vit_context_attention_dim))
            attn_logits = attn_logits.to(dtype=center_tokens.dtype)
            center_row_logits = attn_logits[:, :, 0, :]
            logits = base_logits.view(1, 1, 3) + center_row_logits
            lambdas = softmax(logits, dim=-1)
            attention_matrix = softmax(attn_logits, dim=-1)
            attention_center_prob = attention_matrix[:, :, 0, :]
            attention_center_entropy = -(
                attention_center_prob
                * torch.log(attention_center_prob.clamp_min(1e-12))
            ).sum(dim=-1)
            self._update_vit_context_lambda_stats(
                lambdas,
                attention_center_prob=attention_center_prob,
                attention_matrix=attention_matrix,
                attention_center_entropy=attention_center_entropy,
            )
            lambda_main = lambdas[..., 0].unsqueeze(-1).unsqueeze(-1)
            lambda_prev = lambdas[..., 1].unsqueeze(-1).unsqueeze(-1)
            lambda_next = lambdas[..., 2].unsqueeze(-1).unsqueeze(-1)
        else:
            lambda_main, lambda_prev, lambda_next = softmax(base_logits, dim=0).unbind()

        return (
            lambda_main * center_tokens
            + lambda_prev * prev_tokens
            + lambda_next * next_tokens
        )

    def reset_expert_load_stats(self):
        self.expert_load_tracker.reset()

    def get_expert_load_summary(self, reset: bool = False):
        summary = self.expert_load_tracker.summary(reset=reset)
        if summary:
            summary["expert_group_layout"] = self.expert_group_layout
            summary["msoe_group_dispatch"] = ExpertLoadTracker.msoe_group_masses(summary, mass_key="dispatch_expert_mass")
            summary["msoe_group_combine"] = ExpertLoadTracker.msoe_group_masses(summary, mass_key="combine_expert_mass")
        return summary

    @staticmethod
    def _merge_router_stats(stats_list):
        merged = {}
        for stats in stats_list:
            if not isinstance(stats, dict):
                continue
            for key in [
                "combine_expert_mass",
                "combine_expert_mass_per_sample",
                "combine_slot_mass",
                "dispatch_expert_mass",
                "dispatch_expert_mass_per_sample",
                "dispatch_slot_mass",
            ]:
                if key in stats and stats[key] is not None:
                    val = stats[key]
                    merged[key] = val.detach() if key not in merged else merged[key] + val.detach()
            merged["combine_total_mass"] = float(merged.get("combine_total_mass", 0.0)) + float(stats.get("combine_total_mass", 0.0))
            merged["dispatch_total_mass"] = float(merged.get("dispatch_total_mass", 0.0)) + float(stats.get("dispatch_total_mass", 0.0))
        return merged

    @staticmethod
    def _task_attention_by_modality(attn_weights: torch.Tensor, modality_masks):
        if attn_weights is None:
            return None
        weights = attn_weights
        if weights.ndim == 4:
            weights = weights.mean(dim=1).squeeze(1)
        elif weights.ndim == 3:
            weights = weights.squeeze(1)
        else:
            raise ValueError(f"Unexpected task attention shape: {tuple(weights.shape)}")
        if weights.ndim != 2:
            raise ValueError(f"Task attention must reduce to [B, L], got {tuple(weights.shape)}")
        pieces = []
        start = 0
        for mask in modality_masks:
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            end = start + mask.shape[1]
            mass = (weights[:, start:end] * mask).sum(dim=1)
            pieces.append(mass)
            start = end
        out = torch.stack(pieces, dim=1)
        denom = out.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return out / denom

    def _run_msoe_stack(self, stack, x, return_router_stats: bool = False):
        if not return_router_stats:
            for block in stack:
                x = block(x)
            return x
        stats_list = []
        for block in stack:
            x, block_stats = block(x, return_router_stats=True)
            stats_list.append(block_stats)
        return x, self._merge_router_stats(stats_list)

    @staticmethod
    def _all_valid_mask(tokens: torch.Tensor):
        return torch.ones(tokens.shape[:2], device=tokens.device, dtype=torch.bool)

    @staticmethod
    def _coerce_sequence_mask(mask: torch.Tensor, tokens: torch.Tensor, name: str):
        if mask is None:
            return EmotionM4OE._all_valid_mask(tokens)
        mask = mask.to(device=tokens.device, dtype=torch.bool)
        if mask.shape != tokens.shape[:2]:
            raise ValueError(f"{name} must match token shape [B, L]={tuple(tokens.shape[:2])}, got {tuple(mask.shape)}")
        return mask

    def _build_audio_token_mask(self, audio_attention_mask: torch.Tensor, audio_embed: torch.Tensor):
        get_feature_mask = getattr(self.hubert, "_get_feature_vector_attention_mask", None)
        if callable(get_feature_mask):
            try:
                feature_mask = get_feature_mask(audio_embed.shape[1], audio_attention_mask)
                return self._coerce_sequence_mask(feature_mask, audio_embed, "audio_token_mask")
            except Exception:
                pass
        return self._all_valid_mask(audio_embed)

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, token_mask: torch.Tensor = None):
        if token_mask is None:
            return tokens.mean(dim=1)
        token_mask = token_mask.to(device=tokens.device, dtype=torch.bool)
        weights = token_mask.to(dtype=tokens.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (tokens * weights).sum(dim=1) / denom

    @staticmethod
    def _build_task_head_feature(task_tokens, modality_tokens, modality_masks=None):
        pooled = [task_tokens.mean(dim=1)]
        if modality_masks is None:
            modality_masks = [None for _ in modality_tokens]
        pooled.extend([
            EmotionM4OE._masked_mean(tokens, mask)
            for tokens, mask in zip(modality_tokens, modality_masks)
        ])
        return torch.cat(pooled, dim=-1)

    @staticmethod
    def _maybe_cache_tensor(tensor):
        if tensor is None:
            return None
        if isinstance(tensor, torch.Tensor) and tensor.numel() == 0:
            return None
        return tensor

    @staticmethod
    def _module_param_dtype(module, fallback=torch.float32):
        for param in module.parameters(recurse=True):
            return param.dtype
        return fallback

    @staticmethod
    def _flatten_tcif_context_tensor(
        tensor,
        batch_size: int,
        context_count: int,
        name: str,
    ):
        if tensor is None:
            return None
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a tensor, got {type(tensor).__name__}")
        if tensor.ndim < 2 or tuple(tensor.shape[:2]) != (
            batch_size,
            context_count,
        ):
            raise ValueError(
                f"{name} must start with [B,C]={(batch_size, context_count)}, "
                f"got {tuple(tensor.shape)}"
            )
        return tensor.reshape(batch_size * context_count, *tensor.shape[2:])

    def _apply_tcif_to_local_output(
        self,
        local_output,
        context_output,
        context_valid_mask,
        context_relative_pos,
        requested_return_analysis: bool,
    ):
        local_extras = local_output.get("extras", {})
        context_extras = context_output.get("extras", {})
        local_reg_feature = local_extras.get("feat_task1")
        local_cls_feature = local_extras.get("feat_task2")
        context_reg_feature = context_extras.get("feat_task1")
        context_cls_feature = context_extras.get("feat_task2")
        if any(
            feature is None
            for feature in (
                local_reg_feature,
                local_cls_feature,
                context_reg_feature,
                context_cls_feature,
            )
        ):
            raise RuntimeError("TCIF requires task features from local and context forwards")

        batch_size, context_count = context_valid_mask.shape
        context_reg_feature = context_reg_feature.reshape(
            batch_size,
            context_count,
            -1,
        )
        context_cls_feature = context_cls_feature.reshape(
            batch_size,
            context_count,
            -1,
        )
        regression_result = self.tcif_regression(
            local_reg_feature,
            context_reg_feature,
            context_valid_mask,
            context_relative_pos,
            output_mode=self.tcif_output_mode,
        )
        ordinal_result = self.tcif_ordinal(
            local_cls_feature,
            context_cls_feature,
            context_valid_mask,
            context_relative_pos,
            output_mode=self.tcif_output_mode,
        )

        tcif_context_y_reg = self.tcif_context_reg_head(
            regression_result["prior_mean"]
        ).squeeze(-1)
        tcif_context_cls7_logits = self.tcif_context_cls7_head(
            ordinal_result["prior_mean"]
        )
        tcif_payload = {
            "output_mode": self.tcif_output_mode,
            "has_context": regression_result["has_context"],
            "context_y_reg": tcif_context_y_reg,
            "context_cls7_logits": tcif_context_cls7_logits,
            "local_y_reg": local_output["y_reg"],
            "local_cls7_logits": local_output["cls7_logits"],
            "regression_attention": regression_result["context_attention"],
            "ordinal_attention": ordinal_result["context_attention"],
            "regression_innovation_norm": regression_result["innovation"].norm(
                dim=-1
            ),
            "ordinal_innovation_norm": ordinal_result["innovation"].norm(dim=-1),
            "regression_prior_variance": regression_result["prior_variance"].mean(
                dim=-1
            ),
            "ordinal_prior_variance": ordinal_result["prior_variance"].mean(
                dim=-1
            ),
            "regression_posterior_variance": regression_result[
                "posterior_variance"
            ].mean(dim=-1),
            "ordinal_posterior_variance": ordinal_result[
                "posterior_variance"
            ].mean(dim=-1),
            "regression_continuation_gate": regression_result[
                "continuation_gate"
            ],
            "ordinal_continuation_gate": ordinal_result[
                "continuation_gate"
            ],
            "regression_continuation_gate_logits": regression_result[
                "continuation_gate_logits"
            ],
            "ordinal_continuation_gate_logits": ordinal_result[
                "continuation_gate_logits"
            ],
        }
        local_extras["tcif"] = tcif_payload

        if self.tcif_output_mode != "local":
            feat_task1 = regression_result["feature"]
            feat_task2 = ordinal_result["feature"]
            y_reg = self.head_signed_reg(feat_task1).squeeze(-1)
            cls7_logits = self.head_cls7(feat_task2)
            cls7_probs = torch.softmax(cls7_logits, dim=-1)
            sign_logits = (
                self.head_sign(feat_task2).squeeze(-1)
                if self.head_sign is not None
                else None
            )
            temporal_embedding = None
            if self.temporal_embedding_head is not None:
                temporal_base = _temporal_feature_mix(
                    feat_task1,
                    feat_task2,
                    detach_task2=self.temporal_detach_task2,
                )
                temporal_embedding = F.normalize(
                    self.temporal_embedding_head(temporal_base),
                    dim=-1,
                )
            centers = self.cls7_centers.to(
                device=cls7_logits.device,
                dtype=cls7_logits.dtype,
            )
            y_cls_expected = cls7_expected_value_from_probs(
                cls7_probs,
                centers=centers,
            )
            y_final = compute_final_prediction(
                y_reg,
                cls7_logits,
                eta=self.final_pred_eta,
                centers=centers,
                sign_logits=sign_logits,
                sign_beta=self.final_pred_sign_beta,
                neutral_positive_gate_threshold=self.neutral_positive_gate_threshold,
            )
            local_output["y_reg"] = y_reg
            local_output["cls7_logits"] = cls7_logits
            local_output["sign_logits"] = sign_logits
            local_extras.update(
                {
                    "y_cls_expected": y_cls_expected,
                    "y_final": y_final,
                    "cls7_probs": cls7_probs,
                    "feat_task2": feat_task2,
                    "temporal_embedding": temporal_embedding,
                }
            )
            if requested_return_analysis:
                local_extras["feat_task1"] = feat_task1
                local_extras["affective_repr"] = F.normalize(
                    0.5 * (feat_task1 + feat_task2),
                    dim=-1,
                )

        if not requested_return_analysis:
            local_extras.pop("feat_task1", None)
            local_extras.pop("affective_repr", None)
            local_extras.pop("task_query_attention", None)
        local_output["extras"] = local_extras
        return local_output

    def forward(
        self,
        images,
        input_ids,
        attention_mask,
        audio_values=None,
        audio_attention_mask=None,
        return_router_stats: bool = False,
        return_analysis: bool = False,
        cached_vision=None,
        cached_text=None,
        cached_audio=None,
        cached_text_mask=None,
        cached_audio_mask=None,
        tcif_context_images=None,
        tcif_context_input_ids=None,
        tcif_context_attention_mask=None,
        tcif_context_audio_values=None,
        tcif_context_audio_attention_mask=None,
        tcif_context_cached_vision=None,
        tcif_context_cached_text=None,
        tcif_context_cached_audio=None,
        tcif_context_cached_text_mask=None,
        tcif_context_cached_audio_mask=None,
        tcif_context_valid_mask=None,
        tcif_context_relative_pos=None,
        _tcif_skip: bool = False,
    ):
        if self.enable_tcif and not _tcif_skip:
            if tcif_context_valid_mask is None or tcif_context_relative_pos is None:
                raise ValueError(
                    "TCIF is enabled but context_valid_mask/context_relative_pos "
                    "were not provided"
                )
            if tcif_context_valid_mask.ndim != 2:
                raise ValueError(
                    "tcif_context_valid_mask must have shape [B,C], "
                    f"got {tuple(tcif_context_valid_mask.shape)}"
                )
            batch_size, context_count = tcif_context_valid_mask.shape
            if context_count <= 0:
                raise ValueError("TCIF requires at least one context slot")

            local_output = self.forward(
                images,
                input_ids,
                attention_mask,
                audio_values,
                audio_attention_mask,
                return_router_stats=return_router_stats,
                return_analysis=return_analysis,
                cached_vision=cached_vision,
                cached_text=cached_text,
                cached_audio=cached_audio,
                cached_text_mask=cached_text_mask,
                cached_audio_mask=cached_audio_mask,
                _tcif_skip=True,
            )
            context_output = self.forward(
                self._flatten_tcif_context_tensor(
                    tcif_context_images,
                    batch_size,
                    context_count,
                    "tcif_context_images",
                ),
                self._flatten_tcif_context_tensor(
                    tcif_context_input_ids,
                    batch_size,
                    context_count,
                    "tcif_context_input_ids",
                ),
                self._flatten_tcif_context_tensor(
                    tcif_context_attention_mask,
                    batch_size,
                    context_count,
                    "tcif_context_attention_mask",
                ),
                self._flatten_tcif_context_tensor(
                    tcif_context_audio_values,
                    batch_size,
                    context_count,
                    "tcif_context_audio_values",
                ),
                self._flatten_tcif_context_tensor(
                    tcif_context_audio_attention_mask,
                    batch_size,
                    context_count,
                    "tcif_context_audio_attention_mask",
                ),
                return_router_stats=False,
                return_analysis=False,
                cached_vision=self._flatten_tcif_context_tensor(
                    tcif_context_cached_vision,
                    batch_size,
                    context_count,
                    "tcif_context_cached_vision",
                ),
                cached_text=self._flatten_tcif_context_tensor(
                    tcif_context_cached_text,
                    batch_size,
                    context_count,
                    "tcif_context_cached_text",
                ),
                cached_audio=self._flatten_tcif_context_tensor(
                    tcif_context_cached_audio,
                    batch_size,
                    context_count,
                    "tcif_context_cached_audio",
                ),
                cached_text_mask=self._flatten_tcif_context_tensor(
                    tcif_context_cached_text_mask,
                    batch_size,
                    context_count,
                    "tcif_context_cached_text_mask",
                ),
                cached_audio_mask=self._flatten_tcif_context_tensor(
                    tcif_context_cached_audio_mask,
                    batch_size,
                    context_count,
                    "tcif_context_cached_audio_mask",
                ),
                _tcif_skip=True,
            )
            return self._apply_tcif_to_local_output(
                local_output,
                context_output,
                tcif_context_valid_mask,
                tcif_context_relative_pos,
                requested_return_analysis=return_analysis,
            )

        cached_vision = self._maybe_cache_tensor(cached_vision)
        cached_text = self._maybe_cache_tensor(cached_text)
        cached_audio = self._maybe_cache_tensor(cached_audio)
        cached_text_mask = self._maybe_cache_tensor(cached_text_mask)
        cached_audio_mask = self._maybe_cache_tensor(cached_audio_mask)

        if cached_vision is None and images is None:
            raise ValueError("images or cached_vision must be provided")
        if cached_text is None and (input_ids is None or attention_mask is None):
            raise ValueError("input_ids/attention_mask or cached_text must be provided")
        if cached_text is not None and cached_text_mask is None and attention_mask is None:
            raise ValueError("attention_mask or cached_text_mask must be provided with cached_text")
        if cached_audio is None and audio_values is None:
            raise ValueError("audio_values or cached_audio must be provided")
        if cached_audio is None and audio_attention_mask is None:
            raise ValueError("audio_attention_mask must be provided with raw audio")

        device = None
        for tensor in (cached_vision, cached_text, cached_audio, images, input_ids, audio_values):
            if isinstance(tensor, torch.Tensor):
                device = tensor.device
                break
        if device is None:
            raise ValueError("Could not infer input device")

        need_router_stats = bool(return_router_stats or self.enable_expert_load_output)
        if cached_vision is not None:
            if self.visual_backbone_type != "vit":
                raise ValueError("cached_vision is only supported with the ViT visual backbone")
            vision_dtype = self._module_param_dtype(
                self.vit_projector,
                self._module_param_dtype(self.visual_proj, torch.float32),
            )
            vit_groups = self.visual_proj(self.vit_projector(cached_vision.to(device=device, dtype=vision_dtype)))
            if vit_groups.dim() == 5:
                b, t, k, token_count, dim = vit_groups.shape
                center_tokens = self._fuse_vit_context_triplet(vit_groups)
                vision_embed = center_tokens.reshape(b, t * token_count, dim)
            elif vit_groups.dim() == 4:
                b, t, token_count, dim = vit_groups.shape
                vision_embed = vit_groups.reshape(b, t * token_count, dim)
            elif vit_groups.dim() == 3:
                vision_embed = vit_groups
            else:
                raise ValueError(f"cached_vision must have shape [B,L,D], [B,T,L,D], or [B,T,K,L,D], got {tuple(vit_groups.shape)}")
        else:
            images = images.to(device=device)
            if images.dim() == 4:
                images = images.unsqueeze(1)
            if images.shape[1] <= 0:
                raise ValueError(f"images must contain at least one frame, got shape={tuple(images.shape)}")
            if self.visual_backbone_type == "vit":
                if images.dim() == 6:
                    b, t, k, c, h, w = images.shape
                    vit_groups = self._encode_vit_tokens(images.reshape(b, t * k, c, h, w))
                    vit_groups = vit_groups.reshape(b, t, k, vit_groups.shape[2], vit_groups.shape[3])
                    center_tokens = self._fuse_vit_context_triplet(vit_groups)
                    vision_embed = center_tokens.reshape(b, t * center_tokens.shape[2], center_tokens.shape[3])
                else:
                    vit_out = self._encode_vit_tokens(images)
                    vision_embed = vit_out.reshape(vit_out.shape[0], vit_out.shape[1] * vit_out.shape[2], vit_out.shape[3])
            else:
                if images.dim() == 6:
                    center_idx = 1 if images.shape[2] > 1 else 0
                    images = images[:, :, center_idx, :, :, :]
                b, t, c, h, w = images.shape
                resnet_in = images.reshape(b * t, c, h, w)
                resnet_out = self.resnet(pixel_values=resnet_in)
                feats = self._extract_resnet_features(resnet_out)
                feats = self.resnet_projector(feats)
                feats = feats.reshape(b, t, feats.shape[-1])
                feats = self.temporal_transformer(feats)
                vision_embed = feats

        if cached_text is not None:
            text_dtype = self._module_param_dtype(
                self.bert_projector,
                self._module_param_dtype(self.text_proj, torch.float32),
            )
            text_embed = self.bert_projector(cached_text.to(device=device, dtype=text_dtype))
        else:
            input_ids = input_ids.to(device=device)
            attention_mask = attention_mask.to(device=device)
            text_embed = self._extract_last_hidden_state(self.bert(input_ids=input_ids, attention_mask=attention_mask))
            text_embed = self.bert_projector(text_embed)
        text_embed = self.text_proj(text_embed)

        if cached_audio is not None:
            audio_dtype = self._module_param_dtype(
                self.hubert_projector,
                self._module_param_dtype(self.audio_proj, torch.float32),
            )
            audio_embed = self.hubert_projector(cached_audio.to(device=device, dtype=audio_dtype))
        else:
            audio_dtype = images.dtype if isinstance(images, torch.Tensor) else self._module_param_dtype(self.hubert_projector, torch.float32)
            audio_values = audio_values.to(device=device, dtype=audio_dtype)
            audio_attention_mask = audio_attention_mask.to(device=device, dtype=torch.long)
            if audio_values.ndim != 2:
                raise ValueError(f"audio_values must have shape [B, T], got {tuple(audio_values.shape)}")
            if audio_attention_mask.shape != audio_values.shape:
                raise ValueError(
                    f"audio_attention_mask must match audio_values shape, got mask={tuple(audio_attention_mask.shape)} values={tuple(audio_values.shape)}"
                )
            if audio_values.shape[1] <= 0:
                raise ValueError("audio_values must contain at least one timestep")
            audio_embed = self._extract_last_hidden_state(self.hubert(input_values=audio_values, attention_mask=audio_attention_mask))
            audio_embed = self.hubert_projector(audio_embed)
        audio_embed = self.audio_proj(audio_embed)
        if vision_embed.shape[1] <= 0 or text_embed.shape[1] <= 0 or audio_embed.shape[1] <= 0:
            raise ValueError(
                f"Encoded modality tokens must all be non-empty, got vision={tuple(vision_embed.shape)} text={tuple(text_embed.shape)} audio={tuple(audio_embed.shape)}"
            )

        vision_mask = self._all_valid_mask(vision_embed)
        if cached_text is not None and cached_text_mask is not None:
            text_mask = self._coerce_sequence_mask(cached_text_mask.to(device=device), text_embed, "cached_text_mask")
        else:
            text_mask = self._coerce_sequence_mask(attention_mask.to(device=device), text_embed, "attention_mask")
        if cached_audio is not None:
            if cached_audio_mask is not None:
                audio_mask = self._coerce_sequence_mask(cached_audio_mask.to(device=device), audio_embed, "cached_audio_mask")
            else:
                audio_mask = self._all_valid_mask(audio_embed)
        else:
            audio_mask = self._build_audio_token_mask(audio_attention_mask, audio_embed)

        z_v, z_t, z_a = vision_embed, text_embed, audio_embed
        msoe_stats = {"vision": {}, "text": {}, "audio": {}}
        if need_router_stats:
            layer_stats = {"vision": [], "text": [], "audio": []}
            for layer in self.shared_specific_layers:
                z_v, z_t, z_a, block_stats = layer(
                    z_v,
                    z_t,
                    z_a,
                    vision_mask=vision_mask,
                    text_mask=text_mask,
                    audio_mask=audio_mask,
                    return_router_stats=True,
                )
                for modality_name in ("vision", "text", "audio"):
                    layer_stats[modality_name].append(block_stats.get(modality_name, {}))
            msoe_stats = {
                modality_name: self._merge_router_stats(stats_list)
                for modality_name, stats_list in layer_stats.items()
            }
        else:
            for layer in self.shared_specific_layers:
                z_v, z_t, z_a = layer(
                    z_v,
                    z_t,
                    z_a,
                    vision_mask=vision_mask,
                    text_mask=text_mask,
                    audio_mask=audio_mask,
                    return_router_stats=False,
                )

        h = torch.cat([z_v, z_t, z_a], dim=1)
        h_mask = torch.cat([vision_mask, text_mask, audio_mask], dim=1)
        task_key_padding_mask = ~h_mask
        total_reg_loss = h.new_zeros(())
        if return_analysis:
            o_task1, task1_attn = self.task_pool_intensity(
                h,
                key_padding_mask=task_key_padding_mask,
                return_attention=True,
            )
            o_task2, task2_attn = self.task_pool_polarity(
                h,
                key_padding_mask=task_key_padding_mask,
                return_attention=True,
            )
        else:
            o_task1 = self.task_pool_intensity(h, key_padding_mask=task_key_padding_mask)
            o_task2 = self.task_pool_polarity(h, key_padding_mask=task_key_padding_mask)
            task1_attn = None
            task2_attn = None

        modality_masks = (vision_mask, text_mask, audio_mask)
        feat_task1 = self._build_task_head_feature(o_task1, (z_v, z_t, z_a), modality_masks)
        feat_task2 = self._build_task_head_feature(o_task2, (z_v, z_t, z_a), modality_masks)

        feat_task1 = self.norm_task1(feat_task1)
        feat_task2 = self.norm_task2(feat_task2)
        feat_task1 = self.post_moe_dropout(feat_task1)
        feat_task2 = self.post_moe_dropout(feat_task2)

        if self.output_head_mode == "legacy":
            intensity = self.head_intensity(feat_task1).squeeze(-1)
            polarity_logits = self.head_polarity(feat_task2)
            model_output = (intensity, polarity_logits, total_reg_loss)
        else:
            y_reg = self.head_signed_reg(feat_task1).squeeze(-1)
            cls7_aux = {}
            if self.cls7_head_type == "flat":
                cls7_logits = self.head_cls7(feat_task2)
                cls7_probs = torch.softmax(cls7_logits, dim=-1)
            elif self.cls7_head_type == "hier_sign_mag":
                hier_sign_logits = self.head_hier_sign(feat_task2)
                hier_mag_neg_logits = self.head_hier_mag_neg(feat_task2)
                hier_mag_pos_logits = self.head_hier_mag_pos(feat_task2)
                cls7_probs = hier_sign_mag_to_cls7_probs(hier_sign_logits, hier_mag_neg_logits, hier_mag_pos_logits)
                cls7_logits = probs_to_logits(cls7_probs)
                cls7_aux.update(
                    {
                        "hier_sign_logits": hier_sign_logits,
                        "hier_mag_neg_logits": hier_mag_neg_logits,
                        "hier_mag_pos_logits": hier_mag_pos_logits,
                    }
                )
            elif self.cls7_head_type == "cumulative":
                cumulative_logits = self.head_cumulative(feat_task2)
                cls7_probs = cumulative_logits_to_cls7_probs(cumulative_logits)
                cls7_logits = probs_to_logits(cls7_probs)
                cls7_aux["cumulative_logits"] = cumulative_logits
            elif self.cls7_head_type == "hybrid_cumulative":
                flat_logits = self.head_cls7(feat_task2)
                cumulative_logits = self.head_cumulative(feat_task2)
                flat_probs = torch.softmax(flat_logits, dim=-1)
                cumulative_probs = cumulative_logits_to_cls7_probs(cumulative_logits)
                mix = self.cumulative_p7_mix
                cls7_probs = (1.0 - mix) * flat_probs + mix * cumulative_probs
                cls7_logits = probs_to_logits(cls7_probs)
                cls7_aux.update(
                    {
                        "flat_cls7_logits": flat_logits,
                        "cumulative_logits": cumulative_logits,
                        "cumulative_probs": cumulative_probs,
                    }
                )
            else:
                raise AssertionError(f"Unhandled cls7_head_type: {self.cls7_head_type}")
            sign_logits = self.head_sign(feat_task2).squeeze(-1) if self.head_sign is not None else None
            temporal_embedding = None
            if self.temporal_embedding_head is not None:
                temporal_base = _temporal_feature_mix(
                    feat_task1,
                    feat_task2,
                    detach_task2=self.temporal_detach_task2,
                )
                temporal_embedding = F.normalize(self.temporal_embedding_head(temporal_base), dim=-1)
            centers = self.cls7_centers.to(device=cls7_logits.device, dtype=cls7_logits.dtype)
            y_cls_expected = cls7_expected_value_from_probs(cls7_probs, centers=centers)
            y_final = compute_final_prediction(
                y_reg,
                cls7_logits,
                eta=self.final_pred_eta,
                centers=centers,
                sign_logits=sign_logits,
                sign_beta=self.final_pred_sign_beta,
                neutral_positive_gate_threshold=self.neutral_positive_gate_threshold,
            )
            model_output = {
                "y_reg": y_reg,
                "cls7_logits": cls7_logits,
                "sign_logits": sign_logits,
                "tc_loss": total_reg_loss,
                "extras": {
                    "y_cls_expected": y_cls_expected,
                    "y_final": y_final,
                    "cls7_probs": cls7_probs,
                    "cls7_head_type": self.cls7_head_type,
                    "feat_task2": feat_task2,
                    "temporal_embedding": temporal_embedding,
                },
            }
            if self.enable_tcif and _tcif_skip:
                model_output["extras"]["feat_task1"] = feat_task1
            model_output["extras"].update(cls7_aux)
            if return_analysis:
                modality_attention = {
                    "regression": self._task_attention_by_modality(task1_attn, modality_masks),
                    "ordinal": self._task_attention_by_modality(task2_attn, modality_masks),
                }
                affective_repr = F.normalize(0.5 * (feat_task1 + feat_task2), dim=-1)
                model_output["extras"].update({
                    "feat_task1": feat_task1,
                    "affective_repr": affective_repr,
                    "task_query_attention": modality_attention,
                })

        if not need_router_stats:
            return model_output
        router_stats = {
            "msoe": msoe_stats,
            "mtoe_blocks": [],
            "fusion": {
                "task1": {},
                "task2": {},
            },
            "expert_group_layout": self.expert_group_layout,
        }
        if self.enable_expert_load_output:
            self.expert_load_tracker.update(router_stats)
        if self.output_head_mode == "signed_reg_cls7":
            if return_router_stats:
                model_output["router_stats"] = router_stats
            return model_output
        if not return_router_stats:
            return model_output
        intensity, polarity_logits, total_reg_loss = model_output
        return intensity, polarity_logits, total_reg_loss, router_stats

class MToEBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_layer, attention_dropout: float = 0.0, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = mlp_layer()

    def forward(self, x, modality_token_ids: torch.Tensor = None, return_router_stats: bool = False):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        x_norm = self.norm2(x)
        if return_router_stats:
            y1, y2, loss, router_stats = self.mlp(
                x_norm,
                modality_token_ids=modality_token_ids,
                return_router_stats=True,
            )
        else:
            y1, y2, loss = self.mlp(x_norm, modality_token_ids=modality_token_ids)

        y1 = y1 + x
        if y2 is not None:
            y2 = y2 + x

        if return_router_stats:
            return y1, y2, loss, router_stats
        return y1, y2, loss
