import copy
import tempfile
import unittest
from pathlib import Path

import torch

from magnitude_adapter import (
    MAGNITUDE_KNOTS,
    TwoSidedSignPreservingMagnitudeAdapter,
    adapter_state_sha256,
    assert_magnitude_adapter_contract,
    load_magnitude_adapter_checkpoint,
    magnitude_adapter_checkpoint_payload,
    monotonicity_audit,
    save_magnitude_adapter_checkpoint,
    sign_invariance_audit,
    validate_magnitude_adapter_payload,
)


class MagnitudeAdapterTest(unittest.TestCase):
    def test_identity_initialization_is_exact(self):
        adapter = TwoSidedSignPreservingMagnitudeAdapter().double()
        scores = torch.linspace(-3.0, 3.0, 1201, dtype=torch.float64)
        output = adapter(scores)
        self.assertTrue(torch.equal(output, scores))
        self.assertEqual(adapter.trainable_parameter_count, 12)

    def test_endpoints_and_strict_monotonicity_hold_for_extreme_parameters(self):
        adapter = TwoSidedSignPreservingMagnitudeAdapter().double()
        with torch.no_grad():
            adapter.theta_negative.copy_(
                torch.tensor([1e6, -1e6, 100.0, -100.0, 3.0, -3.0], dtype=torch.float64)
            )
            adapter.theta_positive.copy_(
                torch.tensor([-1e6, 1e6, -100.0, 100.0, -3.0, 3.0], dtype=torch.float64)
            )
        grid = torch.linspace(0.0, 3.0, 1201, dtype=torch.float64)
        for side in ("negative", "positive"):
            mapped = adapter.map_magnitude(grid, side)
            self.assertEqual(mapped[0].item(), 0.0)
            self.assertEqual(mapped[-1].item(), 3.0)
            self.assertTrue(torch.all(mapped[1:] > mapped[:-1]))
            self.assertTrue(torch.all(adapter.branch_slopes(side) > 0.0))
        self.assertTrue(assert_magnitude_adapter_contract(adapter)["valid"])

    def test_positive_and_negative_parameters_are_independent(self):
        adapter = TwoSidedSignPreservingMagnitudeAdapter().double()
        scores = torch.tensor(
            [-2.25, -1.25, -0.25, 0.25, 1.25, 2.25], dtype=torch.float64
        )
        with torch.no_grad():
            adapter.theta_positive.copy_(
                torch.tensor([2.0, -2.0, 1.5, -1.5, 1.0, -1.0], dtype=torch.float64)
            )
        output = adapter(scores)
        self.assertTrue(torch.equal(output[:3], scores[:3]))
        self.assertFalse(torch.equal(output[3:], scores[3:]))

        adapter = TwoSidedSignPreservingMagnitudeAdapter().double()
        with torch.no_grad():
            adapter.theta_negative.copy_(
                torch.tensor([2.0, -2.0, 1.5, -1.5, 1.0, -1.0], dtype=torch.float64)
            )
        output = adapter(scores)
        self.assertFalse(torch.equal(output[:3], scores[:3]))
        self.assertTrue(torch.equal(output[3:], scores[3:]))

    def test_zero_threshold_sign_is_invariant_including_negative_zero(self):
        adapter = TwoSidedSignPreservingMagnitudeAdapter().double()
        with torch.no_grad():
            adapter.theta_negative.copy_(
                torch.linspace(-8.0, 8.0, 6, dtype=torch.float64)
            )
            adapter.theta_positive.copy_(
                torch.linspace(8.0, -8.0, 6, dtype=torch.float64)
            )
        scores = torch.tensor(
            [-3.0, -2.0, -0.5, -1e-15, -0.0, 0.0, 1e-15, 0.5, 2.0, 3.0],
            dtype=torch.float64,
        )
        output = adapter(scores)
        self.assertTrue(torch.equal(output >= 0.0, scores >= 0.0))
        self.assertEqual(output[4].item(), 0.0)
        self.assertEqual(output[5].item(), 0.0)
        audit = sign_invariance_audit(scores, output)
        self.assertTrue(audit["binary_prediction_equal"])
        self.assertEqual(audit["binary_mismatch_count"], 0)

    def test_state_round_trip_and_source_hash_validation(self):
        adapter = TwoSidedSignPreservingMagnitudeAdapter().double()
        with torch.no_grad():
            adapter.theta_negative.copy_(torch.arange(6, dtype=torch.float64) / 7.0)
            adapter.theta_positive.copy_(-torch.arange(6, dtype=torch.float64) / 9.0)
        scores = torch.linspace(-3.0, 3.0, 101, dtype=torch.float64)
        expected = adapter(scores)
        source_hash = "a" * 64
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "adapter.pth"
            payload = save_magnitude_adapter_checkpoint(
                path,
                adapter,
                metadata={
                    "source_checkpoint_sha256": source_hash,
                    "selection_split": "test",
                },
            )
            loaded, loaded_payload = load_magnitude_adapter_checkpoint(
                path,
                expected_source_checkpoint_sha256=source_hash,
            )
        loaded = loaded.double()
        self.assertTrue(torch.equal(loaded(scores), expected))
        self.assertEqual(payload["state_sha256"], adapter_state_sha256(adapter))
        self.assertEqual(loaded_payload["metadata"]["selection_split"], "test")
        self.assertEqual(adapter_state_sha256(loaded), adapter_state_sha256(adapter))

    def test_loader_validation_rejects_bad_protocol_and_state(self):
        adapter = TwoSidedSignPreservingMagnitudeAdapter()
        payload = magnitude_adapter_checkpoint_payload(
            adapter,
            metadata={"source_checkpoint_sha256": "b" * 64},
        )

        bad_knots = copy.deepcopy(payload)
        bad_knots["protocol_config"]["knots"][-1] = 4.0
        with self.assertRaisesRegex(ValueError, "knots"):
            validate_magnitude_adapter_payload(bad_knots)

        bad_shape = copy.deepcopy(payload)
        bad_shape["state_dict"]["theta_positive"] = torch.zeros(5)
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_magnitude_adapter_payload(bad_shape)

        bad_finite = copy.deepcopy(payload)
        bad_finite["state_dict"]["theta_negative"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_magnitude_adapter_payload(bad_finite)

        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_magnitude_adapter_payload(
                payload,
                expected_source_checkpoint_sha256="c" * 64,
            )

    def test_extreme_parameters_have_no_nan_or_inf(self):
        adapter = TwoSidedSignPreservingMagnitudeAdapter().double()
        with torch.no_grad():
            adapter.theta_negative.copy_(
                torch.tensor(
                    [1e30, -1e30, 1e20, -1e20, 1e10, -1e10], dtype=torch.float64
                )
            )
            adapter.theta_positive.copy_(-adapter.theta_negative)
        scores = torch.linspace(-3.0, 3.0, 10001, dtype=torch.float64)
        output = adapter(scores)
        self.assertTrue(torch.isfinite(output).all())
        audit = monotonicity_audit(adapter)
        self.assertTrue(audit["valid"])
        self.assertTrue(audit["branches"]["negative"]["finite"])
        self.assertTrue(audit["branches"]["positive"]["finite"])

    def test_both_parameter_vectors_receive_finite_nonzero_gradients(self):
        adapter = TwoSidedSignPreservingMagnitudeAdapter().double()
        scores = torch.tensor([-2.7, -1.8, -0.7, 0.2, 1.1, 2.4], dtype=torch.float64)
        targets = torch.tensor([-2.2, -1.1, -0.4, 0.4, 1.6, 2.8], dtype=torch.float64)
        weights = torch.tensor([1.0, 0.7, 1.3, 0.9, 1.5, 0.6], dtype=torch.float64)
        loss = (weights * (adapter(scores) - targets).square()).mean()
        loss.backward()
        for parameter in (adapter.theta_negative, adapter.theta_positive):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum().item()), 0.0)

    def test_fixed_knots_match_protocol(self):
        self.assertEqual(MAGNITUDE_KNOTS, (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0))


if __name__ == "__main__":
    unittest.main()
