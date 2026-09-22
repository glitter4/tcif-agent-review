"""Small sign-preserving monotone calibration adapters for valence scores.

The adapter is intentionally independent of :class:`EmotionM4OE`.  It is
applied *after* the production decoder (including the neutral-positive rescue
gate) to a frozen score in ``[-3, 3]``.  Positive and negative scores use
separate monotone magnitude maps, but neither branch can cross zero.

Each branch has six trainable scalars and the fixed knots
``[0, .5, 1, 1.5, 2, 2.5, 3]``.  If ``theta`` is a branch parameter, its output
increments are

``delta = 3 * softmax(log(2) * tanh(theta))``.

Consequently every interval has strictly positive slope and the endpoint is
fixed at three.  Zero initialization gives six equal increments of ``0.5``
and therefore the exact identity map.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn


MAGNITUDE_KNOTS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
NUM_INTERVALS = len(MAGNITUDE_KNOTS) - 1
MAX_MAGNITUDE = MAGNITUDE_KNOTS[-1]
LOG_INCREMENT_BOUND = math.log(2.0)
CHECKPOINT_FORMAT_VERSION = 1
ADAPTER_TYPE = "two_sided_sign_preserving_monotone_magnitude"
EXPECTED_STATE_KEYS = frozenset({"theta_negative", "theta_positive"})


def _as_floating_tensor(values: Any, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values)
    if not tensor.is_floating_point():
        tensor = tensor.to(dtype=torch.float32)
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


class TwoSidedSignPreservingMagnitudeAdapter(nn.Module):
    """A frozen-score calibrator with independent negative/positive branches.

    The full signed mapping is globally strictly increasing on ``[-3, 3]``:
    negative inputs map to ``-g_negative(abs(x))`` and nonnegative inputs map
    to ``g_positive(x)``.  Exact zero remains exact zero, matching MOSEI's
    ``score >= 0`` positive-class convention.
    """

    def __init__(self) -> None:
        super().__init__()
        self.theta_negative = nn.Parameter(torch.zeros(NUM_INTERVALS))
        self.theta_positive = nn.Parameter(torch.zeros(NUM_INTERVALS))
        # Knots are a fixed protocol constant rather than checkpoint state.
        self.register_buffer(
            "_knots",
            torch.tensor(MAGNITUDE_KNOTS, dtype=torch.float32),
            persistent=False,
        )

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def extra_repr(self) -> str:
        return (
            f"knots={list(MAGNITUDE_KNOTS)}, "
            f"parameters={self.trainable_parameter_count}"
        )

    @staticmethod
    def _validate_side(side: str) -> str:
        normalized = str(side).strip().lower()
        aliases = {
            "negative": "negative",
            "neg": "negative",
            "-": "negative",
            "positive": "positive",
            "pos": "positive",
            "+": "positive",
        }
        if normalized not in aliases:
            raise ValueError(f"side must be 'negative' or 'positive', got {side!r}")
        return aliases[normalized]

    def _theta_for_side(self, side: str) -> torch.Tensor:
        return (
            self.theta_negative
            if self._validate_side(side) == "negative"
            else self.theta_positive
        )

    def branch_increments(self, side: str) -> torch.Tensor:
        """Return the six strictly positive output-height increments."""

        theta = self._theta_for_side(side)
        bounded_logits = LOG_INCREMENT_BOUND * torch.tanh(theta)
        return MAX_MAGNITUDE * torch.softmax(bounded_logits, dim=0)

    def branch_heights(self, side: str) -> torch.Tensor:
        """Return seven output knot heights, with exact endpoints 0 and 3."""

        increments = self.branch_increments(side)
        zero = increments.new_zeros(1)
        interior = torch.cumsum(increments, dim=0)[:-1]
        # Construct the final endpoint explicitly to remove accumulated
        # floating-point error while retaining gradients for every interior
        # height and every increment through the softmax normalization.
        endpoint = increments.new_full((1,), MAX_MAGNITUDE)
        return torch.cat((zero, interior, endpoint), dim=0)

    def branch_slopes(self, side: str) -> torch.Tensor:
        """Return the six positive piecewise-linear slopes."""

        heights = self.branch_heights(side)
        knots = self._knots.to(device=heights.device, dtype=heights.dtype)
        return (heights[1:] - heights[:-1]) / (knots[1:] - knots[:-1])

    def map_magnitude(self, magnitudes: torch.Tensor, side: str) -> torch.Tensor:
        """Map nonnegative magnitudes for one branch.

        Inputs outside ``[0, 3]`` are rejected rather than silently clipped;
        the production decoder is responsible for its documented score range.
        """

        magnitudes = _as_floating_tensor(magnitudes, name="magnitudes")
        if bool((magnitudes < 0.0).any().item()):
            raise ValueError("magnitudes must be nonnegative")
        if bool((magnitudes > MAX_MAGNITUDE).any().item()):
            maximum = float(magnitudes.detach().max().item())
            raise ValueError(
                f"magnitudes must be <= {MAX_MAGNITUDE}, got maximum {maximum}"
            )

        theta = self._theta_for_side(side)
        # Keep the computation on the score tensor's dtype/device without
        # detaching the parameter cast from autograd.
        heights = self.branch_heights(side).to(
            device=magnitudes.device, dtype=magnitudes.dtype
        )
        knots = self._knots.to(device=magnitudes.device, dtype=magnitudes.dtype)
        if theta.device != magnitudes.device:
            raise ValueError(
                "adapter parameters and scores must be on the same device; "
                f"got {theta.device} and {magnitudes.device}"
            )

        flat = magnitudes.reshape(-1)
        # bucketize against interior knots gives interval ids 0..5.  The
        # rightmost endpoint is clamped to the last interval.
        interval = torch.bucketize(flat, knots[1:-1], right=False)
        interval = interval.clamp(max=NUM_INTERVALS - 1)
        x0 = knots[interval]
        x1 = knots[interval + 1]

        # Interpolate displacement from identity, not the absolute height.
        # At theta=0 every displacement is exactly zero, so forward() returns
        # the input score bit-for-bit while parameter gradients remain live.
        offsets = heights - knots
        offset0 = offsets[interval]
        offset1 = offsets[interval + 1]
        fraction = (flat - x0) / (x1 - x0)
        mapped = flat + offset0 + fraction * (offset1 - offset0)
        return mapped.reshape_as(magnitudes)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        scores = _as_floating_tensor(scores, name="scores")
        if bool((scores.abs() > MAX_MAGNITUDE).any().item()):
            maximum = float(scores.detach().abs().max().item())
            raise ValueError(
                f"scores must be within [-{MAX_MAGNITUDE}, {MAX_MAGNITUDE}], "
                f"got max abs {maximum}"
            )

        magnitude = scores.abs()
        negative_magnitude = self.map_magnitude(magnitude, "negative")
        positive_magnitude = self.map_magnitude(magnitude, "positive")
        return torch.where(scores < 0.0, -negative_magnitude, positive_magnitude)

    def protocol_config(self) -> dict[str, Any]:
        return {
            "adapter_type": ADAPTER_TYPE,
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "knots": list(MAGNITUDE_KNOTS),
            "negative_parameter_count": NUM_INTERVALS,
            "positive_parameter_count": NUM_INTERVALS,
            "trainable_parameter_count": self.trainable_parameter_count,
            "parameterization": "delta=3*softmax(log(2)*tanh(theta))",
            "decoder_position": "after_production_decoder_and_neutral_positive_gate",
        }


def adapter_state_sha256(adapter: TwoSidedSignPreservingMagnitudeAdapter) -> str:
    """Return a deterministic hash of protocol configuration and parameters."""

    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            adapter.protocol_config(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    for name, value in sorted(adapter.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        # Viewing raw bytes also supports dtypes such as bfloat16 that NumPy
        # cannot represent directly.
        digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def magnitude_adapter_checkpoint_payload(
    adapter: TwoSidedSignPreservingMagnitudeAdapter,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    state_dict = {
        name: value.detach().cpu().clone()
        for name, value in adapter.state_dict().items()
    }
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "adapter_type": ADAPTER_TYPE,
        "protocol_config": adapter.protocol_config(),
        "state_dict": state_dict,
        "state_sha256": adapter_state_sha256(adapter),
        "metadata": dict(metadata or {}),
    }
    validate_magnitude_adapter_payload(payload)
    return payload


def validate_magnitude_adapter_payload(
    payload: Mapping[str, Any],
    *,
    expected_source_checkpoint_sha256: str | None = None,
) -> None:
    """Validate an adapter checkpoint without trusting its tensor contents."""

    if not isinstance(payload, Mapping):
        raise TypeError("adapter checkpoint must be a mapping")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "unsupported adapter checkpoint format_version: "
            f"{payload.get('format_version')!r}"
        )
    if payload.get("adapter_type") != ADAPTER_TYPE:
        raise ValueError(f"unexpected adapter_type: {payload.get('adapter_type')!r}")

    config = payload.get("protocol_config")
    if not isinstance(config, Mapping):
        raise ValueError("protocol_config must be a mapping")
    knots = tuple(float(value) for value in config.get("knots", ()))
    if knots != MAGNITUDE_KNOTS:
        raise ValueError(f"checkpoint knots do not match protocol: {knots}")
    if int(config.get("trainable_parameter_count", -1)) != 2 * NUM_INTERVALS:
        raise ValueError("checkpoint must declare exactly 12 trainable parameters")

    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("state_dict must be a mapping")
    actual_keys = frozenset(str(key) for key in state_dict)
    if actual_keys != EXPECTED_STATE_KEYS:
        raise ValueError(
            f"unexpected state_dict keys: {sorted(actual_keys)}; "
            f"expected {sorted(EXPECTED_STATE_KEYS)}"
        )
    for name in sorted(EXPECTED_STATE_KEYS):
        value = state_dict[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state_dict[{name!r}] must be a tensor")
        if tuple(value.shape) != (NUM_INTERVALS,):
            raise ValueError(
                f"state_dict[{name!r}] must have shape {(NUM_INTERVALS,)}, "
                f"got {tuple(value.shape)}"
            )
        if not value.is_floating_point() or not bool(
            torch.isfinite(value).all().item()
        ):
            raise ValueError(f"state_dict[{name!r}] must be finite floating point")

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    if expected_source_checkpoint_sha256 is not None:
        actual_hash = metadata.get("source_checkpoint_sha256")
        if actual_hash != expected_source_checkpoint_sha256:
            raise ValueError(
                "source checkpoint SHA-256 mismatch: "
                f"expected {expected_source_checkpoint_sha256!r}, got {actual_hash!r}"
            )


def save_magnitude_adapter_checkpoint(
    path: str | Path,
    adapter: TwoSidedSignPreservingMagnitudeAdapter,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = magnitude_adapter_checkpoint_payload(adapter, metadata=metadata)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return payload


def load_magnitude_adapter_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    expected_source_checkpoint_sha256: str | None = None,
) -> tuple[TwoSidedSignPreservingMagnitudeAdapter, dict[str, Any]]:
    path = Path(path)
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility.
        payload = torch.load(path, map_location=map_location)
    validate_magnitude_adapter_payload(
        payload,
        expected_source_checkpoint_sha256=expected_source_checkpoint_sha256,
    )
    state_dtype = payload["state_dict"]["theta_negative"].dtype
    adapter = TwoSidedSignPreservingMagnitudeAdapter().to(
        device=torch.device(map_location), dtype=state_dtype
    )
    adapter.load_state_dict(payload["state_dict"], strict=True)
    actual_hash = adapter_state_sha256(adapter)
    recorded_hash = payload.get("state_sha256")
    if recorded_hash is not None and recorded_hash != actual_hash:
        raise ValueError(
            f"adapter state SHA-256 mismatch: recorded={recorded_hash}, actual={actual_hash}"
        )
    return adapter, dict(payload)


def sign_invariance_audit(
    source_scores: torch.Tensor,
    adapted_scores: torch.Tensor,
) -> dict[str, Any]:
    """Audit the exact MOSEI zero-threshold prediction invariant."""

    source = _as_floating_tensor(source_scores, name="source_scores")
    adapted = _as_floating_tensor(adapted_scores, name="adapted_scores")
    if source.shape != adapted.shape:
        raise ValueError(
            f"score shape mismatch: source={tuple(source.shape)}, "
            f"adapted={tuple(adapted.shape)}"
        )
    source_binary = source >= 0.0
    adapted_binary = adapted >= 0.0
    mismatch = source_binary != adapted_binary
    return {
        "sample_count": int(source.numel()),
        "binary_prediction_equal": bool(not mismatch.any().item()),
        "binary_mismatch_count": int(mismatch.sum().item()),
        "source_zero_count": int((source == 0.0).sum().item()),
        "adapted_zero_count": int((adapted == 0.0).sum().item()),
        "source_finite": bool(torch.isfinite(source).all().item()),
        "adapted_finite": bool(torch.isfinite(adapted).all().item()),
        "max_abs_score_delta": float((adapted - source).abs().max().item())
        if source.numel()
        else 0.0,
    }


def monotonicity_audit(
    adapter: TwoSidedSignPreservingMagnitudeAdapter,
) -> dict[str, Any]:
    """Return JSON-serializable endpoint/slope diagnostics for both branches."""

    branches: dict[str, Any] = {}
    for side in ("negative", "positive"):
        heights = adapter.branch_heights(side).detach().cpu()
        slopes = adapter.branch_slopes(side).detach().cpu()
        branches[side] = {
            "heights": [float(value) for value in heights.tolist()],
            "slopes": [float(value) for value in slopes.tolist()],
            "start_is_zero": bool(heights[0].item() == 0.0),
            "end_is_three": bool(heights[-1].item() == MAX_MAGNITUDE),
            "strictly_monotone": bool((slopes > 0.0).all().item()),
            "finite": bool(
                torch.isfinite(heights).all().item()
                and torch.isfinite(slopes).all().item()
            ),
            "minimum_slope": float(slopes.min().item()),
            "maximum_slope": float(slopes.max().item()),
        }
    return {
        "adapter_type": ADAPTER_TYPE,
        "trainable_parameter_count": adapter.trainable_parameter_count,
        "valid": all(
            branch["start_is_zero"]
            and branch["end_is_three"]
            and branch["strictly_monotone"]
            and branch["finite"]
            for branch in branches.values()
        ),
        "branches": branches,
        "state_sha256": adapter_state_sha256(adapter),
    }


def assert_magnitude_adapter_contract(
    adapter: TwoSidedSignPreservingMagnitudeAdapter,
) -> dict[str, Any]:
    audit = monotonicity_audit(adapter)
    if not audit["valid"]:
        raise ValueError(f"magnitude adapter contract failed: {audit}")
    return audit
