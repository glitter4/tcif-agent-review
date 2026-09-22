import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

from head_utils import (
    cls7_expected_value,
    compute_final_prediction,
    continuous_to_cls7_hard,
    continuous_to_cls7_soft,
    soft_cross_entropy,
)
from models.models_emotion import EmotionM4OE
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
from train_emotion import (
    _is_better_acc7,
    _resolve_use_text_cache,
    _signed_prediction_tensors,
    _write_checkpoint_selection_summary,
    score_metrics_from_lists,
)


def build_tiny_model(output_head_mode="legacy", **model_kwargs):
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
            num_shared_experts=2,
            num_text_specific_experts=2,
            num_audio_specific_experts=2,
            num_vision_specific_experts=2,
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
            output_head_mode=output_head_mode,
            **model_kwargs,
        )
    model.temporal_transformer = nn.Identity()
    return model


def tiny_batch(batch_size=2):
    return {
        "image": torch.randn(batch_size, 2, 3, 4, 4),
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]], dtype=torch.long)[:batch_size],
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.long)[:batch_size],
        "audio_values": torch.randn(batch_size, 4),
        "audio_attention_mask": torch.ones(batch_size, 4, dtype=torch.long),
        "raw_valence": torch.tensor([-2.2, 0.7], dtype=torch.float32)[:batch_size],
        "intensity": torch.tensor([2.2, 0.7], dtype=torch.float32)[:batch_size],
        "polarity": torch.tensor([0, 1], dtype=torch.long)[:batch_size],
    }


class SignedRegCls7HeadTests(unittest.TestCase):
    def test_validation_checkpoint_order_is_independent_of_test_metrics(self):
        current = {
            "epoch": 2,
            "val_acc7": 0.55,
            "val_mae": 0.50,
            "test_acc7": 0.99,
            "test_mae": 0.01,
        }
        better_val_worse_test = {
            "epoch": 3,
            "val_acc7": 0.56,
            "val_mae": 0.51,
            "test_acc7": 0.01,
            "test_mae": 3.0,
        }
        worse_val_better_test = {
            "epoch": 4,
            "val_acc7": 0.54,
            "val_mae": 0.40,
            "test_acc7": 1.0,
            "test_mae": 0.0,
        }
        self.assertTrue(_is_better_acc7(better_val_worse_test, current, "val"))
        self.assertFalse(_is_better_acc7(worse_val_better_test, current, "val"))

    def test_last2_rejects_text_cache(self):
        self.assertFalse(_resolve_use_text_cache(SimpleNamespace(unfreeze_bert_last_n_layers=2, use_text_cache=None)))
        self.assertFalse(_resolve_use_text_cache(SimpleNamespace(unfreeze_bert_last_n_layers=2, use_text_cache=False)))
        with self.assertRaises(ValueError):
            _resolve_use_text_cache(SimpleNamespace(unfreeze_bert_last_n_layers=2, use_text_cache=True))

    def test_continuous_to_cls7_hard_matches_v6_rounding(self):
        y = torch.tensor([-3.4, -2.51, -0.49, 0.0, 0.49, 2.51, 3.4])
        labels = continuous_to_cls7_hard(y)
        self.assertEqual(labels.tolist(), [0, 0, 3, 3, 3, 6, 6])

    def test_continuous_to_cls7_soft_distribution(self):
        y = torch.tensor([-2.8, 0.1, 2.7])
        q = continuous_to_cls7_soft(y, tau=0.5)
        self.assertEqual(tuple(q.shape), (3, 7))
        self.assertTrue(torch.allclose(q.sum(dim=1), torch.ones(3), atol=1e-6))
        self.assertEqual(int(q[0].argmax().item()), 0)
        self.assertEqual(int(q[1].argmax().item()), 3)
        self.assertEqual(int(q[2].argmax().item()), 6)
        if torch.cuda.is_available():
            q_cuda = continuous_to_cls7_soft(y.cuda(), tau=0.5)
            self.assertEqual(q_cuda.device.type, "cuda")

    def test_soft_cross_entropy_backward(self):
        logits = torch.randn(4, 7, requires_grad=True)
        target = continuous_to_cls7_soft(torch.tensor([-3.0, -0.2, 1.4, 2.9]), tau=0.5)
        loss = soft_cross_entropy(logits, target)
        self.assertEqual(tuple(loss.shape), ())
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_legacy_forward_shapes_unchanged(self):
        model = build_tiny_model("legacy")
        batch = tiny_batch()
        intensity, polarity_logits, tc_loss = model(
            batch["image"],
            batch["input_ids"],
            batch["attention_mask"],
            batch["audio_values"],
            batch["audio_attention_mask"],
        )
        self.assertEqual(tuple(intensity.shape), (2,))
        self.assertEqual(tuple(polarity_logits.shape), (2, 3))
        self.assertEqual(tuple(tc_loss.shape), ())

    def test_signed_reg_cls7_forward_shapes(self):
        model = build_tiny_model("signed_reg_cls7")
        batch = tiny_batch()
        out = model(
            batch["image"],
            batch["input_ids"],
            batch["attention_mask"],
            batch["audio_values"],
            batch["audio_attention_mask"],
        )
        self.assertEqual(tuple(out["y_reg"].shape), (2,))
        self.assertEqual(tuple(out["cls7_logits"].shape), (2, 7))
        self.assertEqual(tuple(out["tc_loss"].shape), ())
        self.assertIn("y_cls_expected", out["extras"])
        self.assertTrue(torch.isfinite(out["y_reg"]).all().item())
        self.assertTrue(torch.isfinite(out["cls7_logits"]).all().item())

    def test_prediction_metrics_for_reg_expected_and_final(self):
        logits = torch.randn(3, 7)
        y_reg = torch.tensor([-2.5, 0.1, 2.8])
        expected = cls7_expected_value(logits)
        final = compute_final_prediction(y_reg, logits, eta=0.2)
        raw = [-3.0, 0.0, 3.0]
        for scores in [y_reg.tolist(), expected.tolist(), final.tolist()]:
            metrics = score_metrics_from_lists(scores, raw)
            self.assertIn("mae", metrics)
            self.assertIn("acc7", metrics)
            self.assertIn("acc2", metrics)

    def test_neutral_positive_gate_disabled_matches_legacy_decoder_bit_exactly(self):
        logits = torch.randn(5, 7)
        y_reg = torch.tensor([-0.49, -0.2, 0.0, 0.2, 0.49])
        sign_logits = torch.linspace(-2.0, 2.0, steps=5)
        eta = 0.4
        base = (1.0 - eta) * y_reg + eta * cls7_expected_value(logits)
        sign_score = (
            2.0 * torch.sigmoid(sign_logits.reshape_as(base)) - 1.0
        ) * base.abs()

        for sign_beta in (0.0, 0.2):
            legacy = (
                base
                if sign_beta <= 0.0
                else (1.0 - sign_beta) * base + sign_beta * sign_score
            )
            decoded = compute_final_prediction(
                y_reg,
                logits,
                eta=eta,
                sign_logits=sign_logits,
                sign_beta=sign_beta,
                neutral_positive_gate_threshold=None,
            )
            self.assertTrue(torch.equal(decoded, legacy))

        decoded_without_sign_head = compute_final_prediction(
            y_reg,
            logits,
            eta=eta,
            sign_logits=None,
            sign_beta=0.2,
            neutral_positive_gate_threshold=None,
        )
        self.assertTrue(torch.equal(decoded_without_sign_head, base))

    def test_neutral_positive_gate_only_mirrors_high_confidence_negative_neutral_scores(self):
        logits = torch.zeros(5, 7)
        y_reg = torch.tensor([-0.4999, -0.5, -0.1, -0.1, 0.1])
        sign_logits = torch.tensor([10.0, 10.0, 10.0, 0.0, 10.0])
        gated = compute_final_prediction(
            y_reg,
            logits,
            eta=0.0,
            sign_logits=sign_logits,
            sign_beta=0.0,
            neutral_positive_gate_threshold=0.3,
        )
        expected = torch.tensor([0.4999, -0.5, 0.1, -0.1, 0.1])
        self.assertTrue(torch.allclose(gated, expected, atol=1e-7))
        self.assertTrue(
            torch.equal(
                continuous_to_cls7_hard(y_reg),
                continuous_to_cls7_hard(gated),
            )
        )

    def test_neutral_positive_gate_rejects_invalid_contracts(self):
        logits = torch.zeros(1, 7)
        y_reg = torch.tensor([-0.1])
        with self.assertRaises(ValueError):
            compute_final_prediction(
                y_reg,
                logits,
                eta=0.0,
                neutral_positive_gate_threshold=0.3,
            )
        with self.assertRaises(ValueError):
            compute_final_prediction(
                y_reg,
                logits,
                eta=0.0,
                sign_logits=torch.ones(1),
                neutral_positive_gate_threshold=1.1,
            )

        with self.assertRaisesRegex(ValueError, "output_head_mode"):
            build_tiny_model(
                "legacy",
                enable_sign_head=True,
                neutral_positive_gate_threshold=0.3,
            )

    def test_training_prediction_helper_forwards_neutral_positive_gate(self):
        outputs = {
            "y_reg": torch.tensor([-0.1, -0.1]),
            "cls7_logits": torch.zeros(2, 7),
            "sign_logits": torch.tensor([10.0, 0.0]),
        }
        args = SimpleNamespace(
            clamp_regression_eval=True,
            final_pred_eta=0.0,
            final_pred_sign_beta=0.0,
            neutral_positive_gate_threshold=0.3,
        )
        _, _, y_final = _signed_prediction_tensors(outputs, args)
        self.assertTrue(
            torch.allclose(y_final, torch.tensor([0.1, -0.1]), atol=1e-7)
        )

    def test_model_forward_uses_the_production_neutral_positive_gate(self):
        model = build_tiny_model(
            "signed_reg_cls7",
            enable_sign_head=True,
            final_pred_eta=0.4,
            final_pred_sign_beta=0.2,
            neutral_positive_gate_threshold=0.3,
        )
        batch = tiny_batch()
        out = model(
            batch["image"],
            batch["input_ids"],
            batch["attention_mask"],
            batch["audio_values"],
            batch["audio_attention_mask"],
        )
        expected = compute_final_prediction(
            out["y_reg"],
            out["cls7_logits"],
            eta=0.4,
            sign_logits=out["sign_logits"],
            sign_beta=0.2,
            neutral_positive_gate_threshold=0.3,
        )
        self.assertTrue(torch.equal(out["extras"]["y_final"], expected))

    def test_tiny_training_step_both_modes_and_metadata(self):
        for mode in ["legacy", "signed_reg_cls7"]:
            model = build_tiny_model(mode)
            batch = tiny_batch()
            optim = torch.optim.SGD(model.parameters(), lr=1e-3)
            out = model(
                batch["image"],
                batch["input_ids"],
                batch["attention_mask"],
                batch["audio_values"],
                batch["audio_attention_mask"],
            )
            if mode == "legacy":
                intensity, polarity_logits, tc_loss = out
                loss = F.smooth_l1_loss(intensity, batch["intensity"]) + F.cross_entropy(polarity_logits, batch["polarity"]) + tc_loss
            else:
                cls7_target = continuous_to_cls7_hard(batch["raw_valence"])
                loss = F.smooth_l1_loss(out["y_reg"], batch["raw_valence"]) + F.cross_entropy(out["cls7_logits"], cls7_target) + out["tc_loss"]
            optim.zero_grad()
            loss.backward()
            optim.step()
            self.assertTrue(torch.isfinite(loss).item())

        args = SimpleNamespace(
            output_head_mode="signed_reg_cls7",
            reg_loss_type="smooth_l1",
            cls7_loss_type="soft_ce",
            cls7_loss_weight=0.5,
            cls7_soft_tau=0.5,
            zero_sign_margin_weight=0.02,
            zero_sign_margin=0.02,
            zero_sign_margin_temperature=0.02,
            final_pred_eta=0.2,
            neutral_positive_gate_threshold=0.3,
            sign_aux_weight=0.0,
            clamp_regression_eval=True,
        )
        records = {
            "best_test_acc7": {"epoch": 1, "val_acc7": 0.1, "val_mae": 1.0, "val_acc2": 0.2, "test_acc7": 0.1, "test_mae": 1.0, "test_acc2": 0.2, "path": "x"},
            "best_test_mae": None,
        }
        with tempfile.TemporaryDirectory() as tmp:
            summary = _write_checkpoint_selection_summary(tmp, records, args=args)
            self.assertEqual(summary["head_config"]["output_head_mode"], "signed_reg_cls7")
            self.assertEqual(summary["head_config"]["zero_sign_margin_weight"], 0.02)
            self.assertEqual(summary["head_config"]["zero_sign_margin"], 0.02)
            self.assertEqual(summary["head_config"]["zero_sign_margin_temperature"], 0.02)
            self.assertEqual(
                summary["head_config"]["neutral_positive_gate_threshold"], 0.3
            )


if __name__ == "__main__":
    unittest.main()
