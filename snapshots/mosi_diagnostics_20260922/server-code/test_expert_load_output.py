import json
import math
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from models.models_emotion import (
    ConditionalMutualInformationLoss,
    EmotionM4OE,
    ExpertLoadTracker,
    SharedSpecificMoELayer,
    TaskQueryPool,
)


def _fake_layer_stats():
    return {
        "combine_expert_mass": torch.tensor([8.0, 2.0], dtype=torch.float32),
        "combine_slot_mass": torch.tensor([10.0], dtype=torch.float32),
        "combine_total_mass": 10.0,
        "dispatch_expert_mass": torch.tensor([7.0, 3.0], dtype=torch.float32),
        "dispatch_slot_mass": torch.tensor([10.0], dtype=torch.float32),
        "dispatch_total_mass": 10.0,
    }


class TinyBertConfig:
    def __init__(self, hidden_size=12):
        self.hidden_size = hidden_size


class TinyHubertConfig:
    def __init__(self, hidden_size=12):
        self.hidden_size = hidden_size


class TinyViTConfig:
    def __init__(self, hidden_size=12):
        self.hidden_size = hidden_size


class TinyResNetConfig:
    def __init__(self, hidden_sizes=None):
        self.hidden_sizes = hidden_sizes or [12]


class TinyBertModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or TinyBertConfig()

    def forward(self, input_ids=None, attention_mask=None):
        x = input_ids.float().unsqueeze(-1).repeat(1, 1, self.config.hidden_size)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()
        return SimpleNamespace(last_hidden_state=x)


class TinyHubertModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or TinyHubertConfig()

    def forward(self, input_values=None, attention_mask=None):
        x = input_values.float().unsqueeze(-1).repeat(1, 1, self.config.hidden_size)
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1).float()
        return SimpleNamespace(last_hidden_state=x)


class TinyViTModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or TinyViTConfig()

    def forward(self, pixel_values=None):
        base = pixel_values.float().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)
        cls = base.repeat(1, self.config.hidden_size)
        patch1 = (base + 1.0).repeat(1, self.config.hidden_size)
        patch2 = (base + 2.0).repeat(1, self.config.hidden_size)
        return SimpleNamespace(last_hidden_state=torch.stack([cls, patch1, patch2], dim=1))


class TinyResNetModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or TinyResNetConfig()

    def forward(self, pixel_values=None):
        x = pixel_values.float().mean(dim=(1, 2, 3), keepdim=False)
        x = x.unsqueeze(-1).repeat(1, self.config.hidden_sizes[-1])
        return SimpleNamespace(pooler_output=x)


class RecordingSharedSpecificLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen_vision = None
        self.expert_group_layout = {
            "text_specific": {"start": 0, "end": 1, "count": 1, "indices": [0]},
            "audio_specific": {"start": 1, "end": 2, "count": 1, "indices": [1]},
            "vision_specific": {"start": 2, "end": 3, "count": 1, "indices": [2]},
            "shared": {"start": 3, "end": 4, "count": 1, "indices": [3]},
            "temporal_contrast": {"start": 4, "end": 4, "count": 0, "indices": []},
            "total_experts": 4,
        }

    def forward(
        self,
        vision_tokens,
        text_tokens,
        audio_tokens,
        vision_mask=None,
        text_mask=None,
        audio_mask=None,
        return_router_stats=False,
    ):
        self.seen_vision = vision_tokens.detach().clone()
        if return_router_stats:
            return vision_tokens, text_tokens, audio_tokens, {
                "vision": _fake_layer_stats(),
                "text": _fake_layer_stats(),
                "audio": _fake_layer_stats(),
            }
        return vision_tokens, text_tokens, audio_tokens


class ExpertLoadOutputTests(unittest.TestCase):
    def test_enabled_collects_and_formats(self):
        tracker = ExpertLoadTracker(enabled=True)
        batch_stats = {
            "msoe": {
                "vision": _fake_layer_stats(),
                "text": _fake_layer_stats(),
                "audio": _fake_layer_stats(),
            },
            "mtoe_blocks": [],
            "fusion": {"task1": {}, "task2": {}},
        }
        tracker.update(batch_stats)
        tracker.update(batch_stats)
        summary = tracker.summary(reset=False)
        self.assertIn("msoe", summary)
        self.assertIn("vision", summary["msoe"])
        vision = summary["msoe"]["vision"]
        self.assertEqual(vision["num_experts"], 2)
        self.assertAlmostEqual(sum(vision["combine_expert_share"]), 1.0, places=6)
        self.assertAlmostEqual(sum(vision["dispatch_expert_share"]), 1.0, places=6)
        brief = ExpertLoadTracker.brief(summary)
        self.assertIn("msoe_vision_top=", brief)

    def test_msoe_group_ratios_extract_shared_usage(self):
        summary = {
            "msoe": {
                "vision": {"dispatch_expert_mass": [0.0, 0.0, 0.0, 70.0, 30.0], "dispatch_expert_share": [0.0, 0.0, 0.0, 0.70, 0.30]},
                "text": {"dispatch_expert_mass": [25.0, 45.0, 0.0, 0.0, 30.0], "dispatch_expert_share": [0.25, 0.45, 0.0, 0.0, 0.30]},
                "audio": {"dispatch_expert_mass": [0.0, 0.0, 45.0, 0.0, 55.0], "dispatch_expert_share": [0.0, 0.0, 0.45, 0.0, 0.55]},
            },
            "expert_group_layout": {
                "text_specific": {"indices": [0, 1]},
                "audio_specific": {"indices": [2]},
                "vision_specific": {"indices": [3]},
                "shared": {"indices": [4]},
            },
        }
        ratios = ExpertLoadTracker.msoe_group_ratios(summary, group_name="shared")
        self.assertAlmostEqual(ratios["vision"], 0.30, places=6)
        self.assertAlmostEqual(ratios["text"], 0.30, places=6)
        self.assertAlmostEqual(ratios["audio"], 0.55, places=6)
        group_dispatch = ExpertLoadTracker.msoe_group_masses(summary, mass_key="dispatch_expert_mass")
        self.assertAlmostEqual(group_dispatch["vision"]["groups"]["vision_specific"]["mass"], 70.0, places=6)
        self.assertAlmostEqual(group_dispatch["vision"]["groups"]["shared"]["share"], 0.30, places=6)
        self.assertAlmostEqual(group_dispatch["text"]["groups"]["text_specific"]["mass"], 70.0, places=6)
        self.assertAlmostEqual(group_dispatch["text"]["groups"]["shared"]["share"], 0.30, places=6)

    def test_cmi_loss_is_finite(self):
        loss_fn = ConditionalMutualInformationLoss()
        dispatch = torch.softmax(torch.randn(2, 6, 4, 2), dim=2)
        modality_token_ids = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
        loss = loss_fn(dispatch, modality_token_ids)
        self.assertTrue(torch.isfinite(loss).item())

    def test_task_query_pool_shapes_and_grad(self):
        pool = TaskQueryPool(embed_dim=8, num_heads=2)
        h = torch.randn(2, 5, 8, requires_grad=True)
        out = pool(h)
        self.assertEqual(tuple(out.shape), (2, 1, 8))
        out.sum().backward()
        self.assertIsNotNone(pool.query.grad)
        self.assertIsNotNone(h.grad)

    def test_shared_specific_layer_masks_include_shared_for_text(self):
        layer = SharedSpecificMoELayer(
            dim=8,
            num_heads=2,
            num_text_specific_experts=2,
            num_audio_specific_experts=3,
            num_vision_specific_experts=1,
            num_shared_experts=2,
            normalize=False,
            router_temperature=0.1,
            drop=0.0,
            attention_dropout=0.0,
        )
        layout = layer.expert_group_layout
        self.assertEqual(layout["shared"]["indices"], [6, 7])
        self.assertEqual(layer.text_allowed_mask.tolist(), [True, True, False, False, False, False, True, True])
        self.assertEqual(layer.audio_allowed_mask.tolist(), [False, False, True, True, True, False, True, True])
        self.assertEqual(layer.vision_allowed_mask.tolist(), [False, False, False, False, False, True, True, True])

        x = torch.randn(2, 4, 8)
        z_v, z_t, z_a, stats = layer(x, x, x, return_router_stats=True)
        self.assertEqual(tuple(z_v.shape), (2, 4, 8))
        self.assertEqual(tuple(z_t.shape), (2, 4, 8))
        self.assertEqual(tuple(z_a.shape), (2, 4, 8))
        self.assertEqual(stats["text"]["allowed_expert_mask"], layer.text_allowed_mask.cpu().tolist())
        self.assertEqual(stats["audio"]["allowed_expert_mask"], layer.audio_allowed_mask.cpu().tolist())
        self.assertEqual(stats["vision"]["allowed_expert_mask"], layer.vision_allowed_mask.cpu().tolist())

    def test_shared_specific_layer_can_disable_shared_for_text(self):
        layer = SharedSpecificMoELayer(
            dim=8,
            num_heads=2,
            num_text_specific_experts=2,
            num_audio_specific_experts=3,
            num_vision_specific_experts=1,
            num_shared_experts=2,
            enable_text_shared_experts=False,
            normalize=False,
            router_temperature=0.1,
            drop=0.0,
            attention_dropout=0.0,
        )
        self.assertEqual(layer.text_allowed_mask.tolist(), [True, True, False, False, False, False, False, False])
        self.assertEqual(layer.audio_allowed_mask.tolist(), [False, False, True, True, True, False, True, True])
        self.assertEqual(layer.vision_allowed_mask.tolist(), [False, False, False, False, False, True, True, True])

        x = torch.randn(2, 4, 8)
        _, _, _, stats = layer(x, x, x, return_router_stats=True)
        self.assertEqual(stats["text"]["allowed_expert_mask"], layer.text_allowed_mask.cpu().tolist())
        tracker = ExpertLoadTracker(enabled=True)
        tracker.update({"msoe": stats, "mtoe_blocks": [], "fusion": {"task1": {}, "task2": {}}})
        summary = tracker.summary(reset=False)
        summary["expert_group_layout"] = layer.expert_group_layout
        group_dispatch = ExpertLoadTracker.msoe_group_masses(summary, mass_key="dispatch_expert_mass")
        self.assertAlmostEqual(group_dispatch["text"]["groups"]["shared"]["mass"], 0.0, places=6)
        self.assertAlmostEqual(group_dispatch["text"]["groups"]["shared"]["share"], 0.0, places=6)

    def _build_tiny_emotion_model(self, enable_expert_load_output=True, vision_backbone_type="resnet18_temporal", **kwargs):
        with patch("models.models_emotion.BertConfig", TinyBertConfig), \
             patch("models.models_emotion.ViTConfig", TinyViTConfig), \
             patch("models.models_emotion.HubertConfig", TinyHubertConfig), \
             patch("models.models_emotion.ResNetConfig", TinyResNetConfig), \
             patch("models.models_emotion.BertModel", TinyBertModel), \
             patch("models.models_emotion.ViTModel", TinyViTModel), \
             patch("models.models_emotion.HubertModel", TinyHubertModel), \
             patch("models.models_emotion.ResNetModel", TinyResNetModel):
            model = EmotionM4OE(
                num_classes=3,
                embed_dim=12,
                depth_msoe=1,
                depth_mtoe=1,
                num_experts_msoe=2,
                num_experts_mtoe=2,
                num_shared_experts=kwargs.pop("num_shared_experts", 2),
                num_text_specific_experts=kwargs.pop("num_text_specific_experts", 4),
                num_audio_specific_experts=kwargs.pop("num_audio_specific_experts", 4),
                num_vision_specific_experts=kwargs.pop("num_vision_specific_experts", 4),
                enable_text_shared_experts=kwargs.pop("enable_text_shared_experts", True),
                vision_backbone_type=vision_backbone_type,
                vision_backbone_path="",
                vit_model_path="",
                bert_model_path="",
                hubert_model_path="",
                local_files_only=True,
                temporal_depth=1,
                temporal_heads=3,
                dropout=0.0,
                attention_dropout=0.0,
                alignment_num_heads=3,
                enable_expert_load_output=enable_expert_load_output,
                vit_context_ratio=kwargs.pop("vit_context_ratio", 0.0),
                **kwargs,
            )
        model.temporal_transformer = nn.Identity()
        return model

    def _prepare_vit_context_recording_model(self, **kwargs):
        model = self._build_tiny_emotion_model(
            enable_expert_load_output=False,
            vision_backbone_type="vit",
            vit_context_ratio=kwargs.pop("vit_context_ratio", 0.2),
            **kwargs,
        )
        model.vit_projector = nn.Identity()
        model.visual_proj = nn.Identity()
        model.bert_projector = nn.Identity()
        model.text_proj = nn.Identity()
        model.hubert_projector = nn.Identity()
        model.audio_proj = nn.Identity()
        model.norm_task1 = nn.Identity()
        model.norm_task2 = nn.Identity()
        model.post_moe_dropout = nn.Identity()
        recorder = RecordingSharedSpecificLayer()
        model.shared_specific_layers = nn.ModuleList([recorder])
        model.expert_group_layout = recorder.expert_group_layout
        return model, recorder

    def test_emotion_m4oe_end_to_end_forward(self):
        model = self._build_tiny_emotion_model(enable_expert_load_output=True)
        images = torch.randn(2, 2, 3, 4, 4)
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)
        audio_values = torch.randn(2, 4)
        audio_attention_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.long)

        intensity, polarity_logits, reg_loss, router_stats = model(
            images,
            input_ids,
            attention_mask,
            audio_values,
            audio_attention_mask,
            return_router_stats=True,
        )

        self.assertEqual(tuple(intensity.shape), (2,))
        self.assertEqual(tuple(polarity_logits.shape), (2, 3))
        self.assertEqual(tuple(reg_loss.shape), ())
        self.assertEqual(float(reg_loss), 0.0)
        self.assertEqual(router_stats["mtoe_blocks"], [])
        self.assertEqual(router_stats["fusion"]["task1"], {})
        self.assertIn("expert_group_layout", router_stats)
        self.assertGreater(router_stats["msoe"]["vision"]["combine_total_mass"], 0.0)
        summary = model.get_expert_load_summary(reset=True)
        self.assertIn("expert_group_layout", summary)
        self.assertIn("msoe_group_dispatch", summary)
        self.assertIn("vision", summary["msoe"])

    def test_emotion_m4oe_supports_non_default_expert_counts(self):
        model = self._build_tiny_emotion_model(
            enable_expert_load_output=False,
            num_shared_experts=1,
            num_text_specific_experts=2,
            num_audio_specific_experts=3,
            num_vision_specific_experts=5,
        )
        images = torch.randn(1, 2, 3, 4, 4)
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        audio_values = torch.randn(1, 4)
        audio_attention_mask = torch.ones_like(audio_values, dtype=torch.long)
        intensity, polarity_logits, reg_loss = model(
            images,
            input_ids,
            attention_mask,
            audio_values,
            audio_attention_mask,
        )
        self.assertEqual(tuple(intensity.shape), (1,))
        self.assertEqual(tuple(polarity_logits.shape), (1, 3))
        self.assertEqual(float(reg_loss), 0.0)
        layout = model.expert_group_layout
        self.assertEqual(layout["shared"]["count"], 1)
        self.assertEqual(layout["text_specific"]["count"], 2)
        self.assertEqual(layout["audio_specific"]["count"], 3)
        self.assertEqual(layout["vision_specific"]["count"], 5)

    def test_emotion_m4oe_can_disable_text_shared_experts(self):
        model = self._build_tiny_emotion_model(
            enable_expert_load_output=False,
            num_shared_experts=2,
            num_text_specific_experts=2,
            num_audio_specific_experts=2,
            num_vision_specific_experts=2,
            enable_text_shared_experts=False,
        )
        self.assertFalse(model.enable_text_shared_experts)
        self.assertEqual(
            model.shared_specific_layers[0].text_allowed_mask.tolist(),
            [True, True, False, False, False, False, False, False],
        )

    def test_emotion_m4oe_fail_fast_on_missing_audio_inputs(self):
        model = self._build_tiny_emotion_model(enable_expert_load_output=False)
        images = torch.randn(1, 2, 3, 4, 4)
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        audio_values = torch.randn(1, 4)
        audio_attention_mask = torch.ones_like(audio_values, dtype=torch.long)

        with self.assertRaisesRegex(ValueError, "audio_values or cached_audio must be provided"):
            model(images, input_ids, attention_mask, None, audio_attention_mask)
        with self.assertRaisesRegex(ValueError, "audio_attention_mask must be provided"):
            model(images, input_ids, attention_mask, audio_values, None)

    def test_vit_context_weighted_sum_is_injected_before_shared_specific_stack(self):
        model, recorder = self._prepare_vit_context_recording_model()
        with torch.no_grad():
            model.vit_context_lambdas.copy_(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32))

        images = torch.tensor(
            [
                [
                    [
                        [[[1.0, 1.0], [1.0, 1.0]]] * 3,
                        [[[2.0, 2.0], [2.0, 2.0]]] * 3,
                        [[[5.0, 5.0], [5.0, 5.0]]] * 3,
                    ],
                    [
                        [[[10.0, 10.0], [10.0, 10.0]]] * 3,
                        [[[13.0, 13.0], [13.0, 13.0]]] * 3,
                        [[[17.0, 17.0], [17.0, 17.0]]] * 3,
                    ],
                ]
            ],
            dtype=torch.float32,
        )
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        audio_values = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        audio_attention_mask = torch.ones_like(audio_values, dtype=torch.long)

        intensity, polarity_logits, reg_loss = model(
            images,
            input_ids,
            attention_mask,
            audio_values,
            audio_attention_mask,
        )

        self.assertEqual(tuple(intensity.shape), (1,))
        self.assertEqual(tuple(polarity_logits.shape), (1, 3))
        self.assertEqual(float(reg_loss), 0.0)
        self.assertIsNotNone(recorder.seen_vision)

        vision_tokens = recorder.seen_vision
        weights = torch.softmax(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32), dim=0)
        first_triplet = torch.tensor([1.0, 2.0, 5.0], dtype=torch.float32)
        second_triplet = torch.tensor([10.0, 13.0, 17.0], dtype=torch.float32)
        first_base = float(torch.dot(weights[[1, 0, 2]], first_triplet))
        second_base = float(torch.dot(weights[[1, 0, 2]], second_triplet))
        expected = torch.tensor(
            [
                [first_base] * 12,
                [first_base + 1.0] * 12,
                [first_base + 2.0] * 12,
                [second_base] * 12,
                [second_base + 1.0] * 12,
                [second_base + 2.0] * 12,
            ],
            dtype=torch.float32,
        ).unsqueeze(0)
        self.assertTrue(torch.allclose(vision_tokens, expected))

    def test_adaptive_vit_context_zero_gate_matches_static(self):
        static_model, static_recorder = self._prepare_vit_context_recording_model(
            vit_context_lambda_mode="static",
            vit_context_lambda_init="0.8,0.1,0.1",
        )
        adaptive_model, adaptive_recorder = self._prepare_vit_context_recording_model(
            vit_context_lambda_mode="adaptive",
            vit_context_lambda_init="0.8,0.1,0.1",
        )

        images = torch.randn(2, 2, 3, 3, 2, 2)
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)
        audio_values = torch.randn(2, 4)
        audio_attention_mask = torch.ones_like(audio_values, dtype=torch.long)

        static_model(images, input_ids, attention_mask, audio_values, audio_attention_mask)
        adaptive_model(images, input_ids, attention_mask, audio_values, audio_attention_mask)

        self.assertTrue(torch.allclose(static_recorder.seen_vision, adaptive_recorder.seen_vision))
        stats = adaptive_model.get_vit_context_lambda_stats()
        self.assertIsNotNone(stats)
        self.assertEqual(stats["count"], 4)
        for got, expected in zip(stats["mean"], [0.8, 0.1, 0.1]):
            self.assertAlmostEqual(got, expected, places=6)

    def test_attention_vit_context_zero_q_matches_static(self):
        static_model, static_recorder = self._prepare_vit_context_recording_model(
            vit_context_lambda_mode="static",
            vit_context_lambda_init="0.8,0.1,0.1",
        )
        attention_model, attention_recorder = self._prepare_vit_context_recording_model(
            vit_context_lambda_mode="attention",
            vit_context_lambda_init="0.8,0.1,0.1",
            vit_context_attention_dim=4,
        )

        images = torch.randn(2, 2, 3, 3, 2, 2)
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)
        audio_values = torch.randn(2, 4)
        audio_attention_mask = torch.ones_like(audio_values, dtype=torch.long)

        static_model(images, input_ids, attention_mask, audio_values, audio_attention_mask)
        attention_model(images, input_ids, attention_mask, audio_values, audio_attention_mask)

        self.assertTrue(torch.allclose(static_recorder.seen_vision, attention_recorder.seen_vision))
        stats = attention_model.get_vit_context_lambda_stats()
        self.assertIsNotNone(stats)
        self.assertEqual(stats["count"], 4)
        for got, expected in zip(stats["mean"], [0.8, 0.1, 0.1]):
            self.assertAlmostEqual(got, expected, places=6)
        for got in stats["attention_center_prob_mean"]:
            self.assertAlmostEqual(got, 1.0 / 3.0, places=6)
        self.assertAlmostEqual(stats["attention_center_entropy_mean"], math.log(3.0), places=6)

    def test_attention_vit_context_gate_varies_by_sample_and_fuses_tokens(self):
        model, recorder = self._prepare_vit_context_recording_model(
            vit_context_lambda_mode="attention",
            vit_context_lambda_init="1.0,1.0,1.0",
            vit_context_attention_dim=1,
        )
        model.vit_context_lambda_attn_norm = nn.Identity()
        with torch.no_grad():
            model.vit_context_lambda_attn_q.weight.zero_()
            model.vit_context_lambda_attn_k.weight.zero_()
            model.vit_context_lambda_attn_q.weight[0, 0] = 1.0
            model.vit_context_lambda_attn_k.weight[0, 0] = 1.0

        prev0 = torch.full((2, model.embed_dim), 1.0)
        center0 = torch.full((2, model.embed_dim), 2.0)
        next0 = torch.full((2, model.embed_dim), 5.0)
        prev1 = torch.full((2, model.embed_dim), 5.0)
        center1 = torch.full((2, model.embed_dim), 2.0)
        next1 = torch.full((2, model.embed_dim), 1.0)
        cached_vision = torch.stack(
            [
                torch.stack([prev0, center0, next0], dim=0),
                torch.stack([prev1, center1, next1], dim=0),
            ],
            dim=0,
        ).unsqueeze(1)
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        audio_values = torch.randn(2, 4)
        audio_attention_mask = torch.ones_like(audio_values, dtype=torch.long)

        model(
            None,
            input_ids,
            attention_mask,
            audio_values,
            audio_attention_mask,
            cached_vision=cached_vision,
        )

        weights0 = torch.softmax(torch.tensor([4.0, 2.0, 10.0]), dim=0)
        weights1 = torch.softmax(torch.tensor([4.0, 10.0, 2.0]), dim=0)
        expected0 = weights0[0] * center0 + weights0[1] * prev0 + weights0[2] * next0
        expected1 = weights1[0] * center1 + weights1[1] * prev1 + weights1[2] * next1
        expected = torch.stack([expected0, expected1], dim=0)
        self.assertTrue(torch.allclose(recorder.seen_vision, expected, atol=1e-6))

        stats = model.get_vit_context_lambda_stats()
        self.assertIsNotNone(stats)
        self.assertGreater(stats["std"][1], 0.0)
        self.assertGreater(stats["std"][2], 0.0)
        self.assertIn("attention_matrix_mean", stats)
        self.assertEqual(len(stats["attention_matrix_mean"]), 3)
        self.assertEqual(len(stats["attention_matrix_mean"][0]), 3)

    def test_attention_lambda_history_stats_fields(self):
        from train_emotion import (
            LAMBDA_HISTORY_NAME,
            _append_lambda_history,
            _get_vit_context_lambda_state,
            _write_final_lambda_distribution,
        )

        model, _ = self._prepare_vit_context_recording_model(
            vit_context_lambda_mode="attention",
            vit_context_lambda_init="0.8,0.1,0.1",
        )
        cached_vision = torch.randn(2, 1, 3, 2, model.embed_dim)
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        audio_values = torch.randn(2, 4)
        audio_attention_mask = torch.ones_like(audio_values, dtype=torch.long)
        model(
            None,
            input_ids,
            attention_mask,
            audio_values,
            audio_attention_mask,
            cached_vision=cached_vision,
        )

        lambda_state = _get_vit_context_lambda_state(model)
        self.assertEqual(lambda_state["mode"], "attention")
        for key in [
            "base_lambda",
            "effective_lambda_mean",
            "effective_lambda_std",
            "effective_lambda_min",
            "effective_lambda_max",
            "attention_center_prob_mean",
            "attention_center_prob_std",
            "attention_center_prob_min",
            "attention_center_prob_max",
            "attention_matrix_mean",
            "attention_center_entropy_mean",
        ]:
            self.assertIn(key, lambda_state)
        self.assertEqual(len(lambda_state["attention_matrix_mean"]), 3)
        self.assertEqual(len(lambda_state["attention_matrix_mean"][0]), 3)
        for value in lambda_state["attention_center_prob_mean"]:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            _append_lambda_history(f"{tmpdir}/run", 1, lambda_state)
            _write_final_lambda_distribution(f"{tmpdir}/run", 1, lambda_state)
            with open(f"{tmpdir}/{LAMBDA_HISTORY_NAME}", "r", encoding="utf-8") as f:
                payload = json.loads(f.readline())
            with open(f"{tmpdir}/final_lambda_distribution.json", "r", encoding="utf-8") as f:
                final_payload = json.load(f)
        self.assertIn("attention_matrix_mean", payload)
        self.assertIn("attention_center_entropy_mean", payload)
        self.assertIn("attention_matrix_mean", final_payload)
        self.assertIn("effective_lambda_mean", final_payload)

    def test_attention_checkpoint_strict_load_and_detection(self):
        from test_emotion import (
            _infer_vit_context_attention_dim,
            _state_has_attention_vit_context_gate,
        )

        model, _ = self._prepare_vit_context_recording_model(
            vit_context_lambda_mode="attention",
            vit_context_lambda_init="0.8,0.1,0.1",
            vit_context_attention_dim=5,
        )
        state_dict = model.state_dict()
        self.assertTrue(_state_has_attention_vit_context_gate(state_dict))
        self.assertEqual(_infer_vit_context_attention_dim(state_dict, 64), 5)

        reloaded, _ = self._prepare_vit_context_recording_model(
            vit_context_lambda_mode="attention",
            vit_context_lambda_init="0.8,0.1,0.1",
            vit_context_attention_dim=5,
        )
        reloaded.load_state_dict(state_dict, strict=True)

    def test_eval_all_final_pred_eta_cli_override(self):
        from eval_all_mosei_maefixed import resolve_final_pred_eta

        hparams = {"final_pred_eta": 0.4}
        self.assertEqual(resolve_final_pred_eta(hparams, None), (0.4, "hparams"))
        self.assertEqual(resolve_final_pred_eta(hparams, 0.0), (0.0, "cli"))
        self.assertEqual(resolve_final_pred_eta({}, None), (0.0, "default"))

    def test_eval_all_neutral_positive_gate_cli_override(self):
        from eval_all_mosei_maefixed import resolve_neutral_positive_gate_threshold

        hparams = {"neutral_positive_gate_threshold": 0.2}
        self.assertEqual(
            resolve_neutral_positive_gate_threshold(hparams, None),
            (0.2, "hparams"),
        )
        self.assertEqual(
            resolve_neutral_positive_gate_threshold(hparams, 0.3),
            (0.3, "cli"),
        )
        self.assertEqual(
            resolve_neutral_positive_gate_threshold(
                hparams,
                None,
                cli_disabled=True,
            ),
            (None, "cli_disabled"),
        )
        self.assertEqual(
            resolve_neutral_positive_gate_threshold({}, None),
            (None, "disabled"),
        )
        with self.assertRaises(ValueError):
            resolve_neutral_positive_gate_threshold({}, 1.1)


if __name__ == "__main__":
    unittest.main()
