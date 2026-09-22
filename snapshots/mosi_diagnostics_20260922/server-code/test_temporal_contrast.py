import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from datasets.emotion_dataset import build_temporal_metadata, parse_temporal_sample_id
from models.models_emotion import EmotionM4OE, SharedSpecificMoELayer, _temporal_feature_mix
from test_expert_load_output import (
    TinyBertConfig,
    TinyBertModel,
    TinyHubertConfig,
    TinyHubertModel,
    TinyResNetConfig,
    TinyResNetModel,
    TinyViTConfig,
    TinyViTModel,
)
from train_emotion import _compute_temporal_contrast_loss, compute_soft_temporal_contrastive_loss


class TemporalMetadataTests(unittest.TestCase):
    def test_parse_sample_id_uses_last_underscore_as_clip(self):
        base_id, clip_idx, pos = parse_temporal_sample_id("video_part_A_003")
        self.assertEqual(base_id, "video_part_A")
        self.assertEqual(clip_idx, "003")
        self.assertEqual(pos, 3.0)

    def test_label_start_time_overrides_clip_position(self):
        meta = build_temporal_metadata("speaker_007", {"start": "12.5"}, group_id=4)
        self.assertEqual(meta["base_id"], "speaker")
        self.assertEqual(meta["clip_idx"], "007")
        self.assertEqual(meta["temporal_pos"], 12.5)
        self.assertEqual(meta["temporal_group_id"], 4)


class TemporalContrastLossTests(unittest.TestCase):
    def test_detach_task2_preserves_forward_and_blocks_only_task2_gradient(self):
        task1 = torch.randn(3, 5, requires_grad=True)
        task2 = torch.randn(3, 5, requires_grad=True)
        expected = 0.5 * (task1 + task2)

        mixed = _temporal_feature_mix(task1, task2, detach_task2=True)
        self.assertTrue(torch.equal(mixed, expected))
        mixed.sum().backward()

        self.assertTrue(torch.equal(task1.grad, torch.full_like(task1, 0.5)))
        self.assertIsNone(task2.grad)

    def test_default_temporal_mix_keeps_task2_gradient(self):
        task1 = torch.randn(2, 4, requires_grad=True)
        task2 = torch.randn(2, 4, requires_grad=True)
        mixed = _temporal_feature_mix(task1, task2)
        mixed.sum().backward()
        self.assertTrue(torch.equal(task1.grad, torch.full_like(task1, 0.5)))
        self.assertTrue(torch.equal(task2.grad, torch.full_like(task2, 0.5)))

    def test_no_same_group_positive_returns_zero(self):
        embeddings = torch.randn(3, 5, requires_grad=True)
        group_id = torch.tensor([0, 1, 2], dtype=torch.long)
        temporal_pos = torch.tensor([0.0, 0.0, 0.0])
        loss, stats = compute_soft_temporal_contrastive_loss(embeddings, group_id, temporal_pos)
        self.assertEqual(float(loss.item()), 0.0)
        self.assertEqual(stats["valid_anchor_count"], 0)
        self.assertEqual(stats["skipped_anchor_count"], 3)
        self.assertEqual(stats["positive_pair_count"], 0)

    def test_same_group_neighbors_create_soft_positive_pairs(self):
        embeddings = torch.randn(4, 6, requires_grad=True)
        group_id = torch.tensor([0, 0, 0, 1], dtype=torch.long)
        temporal_pos = torch.tensor([1.0, 2.0, 5.0, 2.0])
        loss, stats = compute_soft_temporal_contrastive_loss(
            embeddings,
            group_id,
            temporal_pos,
            temperature=0.2,
            decay_tau=1.0,
            positive_radius=1.0,
            weak_positive_radius=4.0,
            min_positive_weight=0.2,
        )
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(float(loss.item()), 0.0)
        self.assertEqual(stats["valid_anchor_count"], 3)
        self.assertEqual(stats["skipped_anchor_count"], 1)
        self.assertEqual(stats["positive_pair_count"], 6)
        loss.backward()
        self.assertIsNotNone(embeddings.grad)

    def test_training_loss_path_reads_batch_metadata(self):
        args = SimpleNamespace(
            output_head_mode="signed_reg_cls7",
            enable_temporal_contrast_experts=True,
            temporal_contrast_weight=0.03,
            temporal_contrast_temperature=0.07,
            temporal_decay_tau=1.0,
            temporal_positive_radius=1.0,
            temporal_weak_positive_radius=4.0,
            temporal_min_positive_weight=0.2,
        )
        model_out = {
            "y_reg": torch.zeros(3),
            "extras": {
                "temporal_embedding": torch.randn(3, 4, requires_grad=True),
            },
        }
        batch = {
            "temporal_group_id": torch.tensor([0, 0, 1], dtype=torch.long),
            "temporal_pos": torch.tensor([1.0, 2.0, 1.0]),
        }
        loss, stats = _compute_temporal_contrast_loss(model_out, batch, args)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertGreater(stats["valid_anchor_count"], 0)


class TemporalExpertLayoutTests(unittest.TestCase):
    def test_temporal_experts_are_compatible_when_disabled(self):
        layer = SharedSpecificMoELayer(
            dim=8,
            num_heads=2,
            num_text_specific_experts=2,
            num_audio_specific_experts=1,
            num_vision_specific_experts=1,
            num_shared_experts=1,
            normalize=False,
            drop=0.0,
        )
        self.assertEqual(layer.expert_group_layout["temporal_contrast"]["count"], 0)
        self.assertEqual(layer.expert_group_layout["total_experts"], 5)
        self.assertEqual(layer.text_allowed_mask.tolist(), [True, True, False, False, True])

    def test_temporal_experts_are_routable_for_all_modalities(self):
        layer = SharedSpecificMoELayer(
            dim=8,
            num_heads=2,
            num_text_specific_experts=2,
            num_audio_specific_experts=1,
            num_vision_specific_experts=1,
            num_shared_experts=1,
            num_temporal_contrast_experts=4,
            enable_temporal_contrast_experts=True,
            normalize=False,
            drop=0.0,
        )
        self.assertEqual(layer.expert_group_layout["temporal_contrast"]["indices"], [5, 6, 7, 8])
        self.assertEqual(layer.expert_group_layout["total_experts"], 9)
        self.assertTrue(all(layer.text_allowed_mask[idx].item() for idx in [5, 6, 7, 8]))
        self.assertTrue(all(layer.audio_allowed_mask[idx].item() for idx in [5, 6, 7, 8]))
        self.assertTrue(all(layer.vision_allowed_mask[idx].item() for idx in [5, 6, 7, 8]))
        x = torch.randn(2, 4, 8)
        _, _, _, stats = layer(x, x, x, return_router_stats=True)
        self.assertEqual(stats["text"]["allowed_expert_mask"], layer.text_allowed_mask.cpu().tolist())

    def test_tiny_model_returns_temporal_embedding(self):
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
                num_shared_experts=1,
                num_text_specific_experts=1,
                num_audio_specific_experts=1,
                num_vision_specific_experts=1,
                num_temporal_contrast_experts=2,
                enable_temporal_contrast_experts=True,
                temporal_embedding_dim=5,
                temporal_detach_task2=True,
                vision_backbone_type="resnet18_temporal",
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
                enable_expert_load_output=False,
                output_head_mode="signed_reg_cls7",
            )
        model.temporal_transformer = nn.Identity()
        out = model(
            torch.randn(2, 2, 3, 4, 4),
            torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long),
            torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long),
            torch.randn(2, 4),
            torch.ones(2, 4, dtype=torch.long),
            return_router_stats=True,
            return_analysis=True,
        )
        temporal_embedding = out["extras"]["temporal_embedding"]
        self.assertEqual(tuple(temporal_embedding.shape), (2, 5))
        norms = torch.linalg.vector_norm(temporal_embedding, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))
        feat_task2 = out["extras"]["feat_task2"]
        feat_task2.retain_grad()
        temporal_embedding.sum().backward()
        self.assertIsNone(feat_task2.grad)
        affective_repr = out["extras"]["affective_repr"]
        self.assertEqual(tuple(affective_repr.shape), (2, 48))
        attention = out["extras"]["task_query_attention"]
        self.assertEqual(tuple(attention["regression"].shape), (2, 3))
        self.assertEqual(tuple(attention["ordinal"].shape), (2, 3))
        self.assertTrue(torch.allclose(attention["regression"].sum(dim=1), torch.ones(2), atol=1e-5))
        self.assertTrue(torch.allclose(attention["ordinal"].sum(dim=1), torch.ones(2), atol=1e-5))
        router_stats = out["router_stats"]
        self.assertEqual(tuple(router_stats["msoe"]["vision"]["dispatch_expert_mass_per_sample"].shape), (2, 6))


if __name__ == "__main__":
    unittest.main()
