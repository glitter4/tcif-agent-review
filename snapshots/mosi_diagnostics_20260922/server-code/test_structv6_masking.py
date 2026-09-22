import unittest

import torch

from models.models_emotion import SharedSpecificMoELayer, TaskQueryPool


class StructV6MaskingTests(unittest.TestCase):
    def test_forbidden_experts_have_zero_dispatch_and_combine_mass(self):
        torch.manual_seed(7)
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
        vision = torch.randn(2, 4, 8)
        text = torch.randn(2, 5, 8)
        audio = torch.randn(2, 3, 8)
        vision_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
        text_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
        audio_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)

        _, _, _, stats = layer(
            vision,
            text,
            audio,
            vision_mask=vision_mask,
            text_mask=text_mask,
            audio_mask=audio_mask,
            return_router_stats=True,
        )

        for modality in ("vision", "text", "audio"):
            allowed = getattr(layer, f"{modality}_allowed_mask")
            forbidden = ~allowed
            combine = stats[modality]["combine_expert_mass"]
            dispatch = stats[modality]["dispatch_expert_mass"]
            self.assertTrue(torch.allclose(combine[forbidden], torch.zeros_like(combine[forbidden]), atol=1e-6))
            self.assertTrue(torch.allclose(dispatch[forbidden], torch.zeros_like(dispatch[forbidden]), atol=1e-6))
            self.assertGreater(float(combine[allowed].sum().item()), 0.0)
            self.assertGreater(float(dispatch[allowed].sum().item()), 0.0)

    def test_masked_moe_zeroes_padded_token_outputs(self):
        torch.manual_seed(11)
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
        tokens = torch.randn(2, 5, 8)
        token_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)

        out, stats = layer._masked_moe(
            tokens,
            layer.text_allowed_mask,
            token_mask=token_mask,
            return_router_stats=True,
        )

        self.assertTrue(torch.allclose(out[~token_mask], torch.zeros_like(out[~token_mask]), atol=1e-6))
        self.assertTrue(torch.allclose(
            stats["combine_expert_mass"][~layer.text_allowed_mask],
            torch.zeros_like(stats["combine_expert_mass"][~layer.text_allowed_mask]),
            atol=1e-6,
        ))

    def test_task_query_pool_accepts_padding_mask(self):
        torch.manual_seed(13)
        pool = TaskQueryPool(embed_dim=8, num_heads=2)
        tokens = torch.randn(2, 4, 8, requires_grad=True)
        key_padding_mask = torch.tensor(
            [[False, False, True, True], [True, True, True, True]],
            dtype=torch.bool,
        )

        out = pool(tokens, key_padding_mask=key_padding_mask)
        self.assertEqual(tuple(out.shape), (2, 1, 8))
        self.assertTrue(torch.isfinite(out).all())
        out.sum().backward()
        self.assertIsNotNone(tokens.grad)


if __name__ == "__main__":
    unittest.main()
