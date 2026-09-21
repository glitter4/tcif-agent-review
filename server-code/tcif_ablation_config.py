"""Serializable, dependency-free definitions for the five paper ablations."""
import json
from pathlib import Path

TCIF_ABLATIONS = ("full", "standard_context", "equal_weight", "no_gate", "shared_filter")


def validate_ablation(name):
    if name not in TCIF_ABLATIONS:
        raise ValueError(f"Unknown TCIF ablation {name!r}; choose from {TCIF_ABLATIONS}")
    return name


def validate_training_ablation(args):
    name = validate_ablation(args.tcif_ablation)
    if name == "full":
        return
    if not args.enable_tcif or args.tcif_output_mode != "posterior":
        raise ValueError("Paper ablations require TCIF and posterior output mode")
    if name in ("standard_context", "no_gate"):
        if args.tcif_enable_transition_gate or args.tcif_transition_gate_loss_weight != 0:
            raise ValueError(f"{name} requires gate disabled and gate loss weight zero")
    elif not args.tcif_enable_transition_gate:
        raise ValueError(f"{name} retains the full model's continuation gate")


def load_training_defaults(parser, path):
    """Load explicit training values, rejecting unknown options and invalid types."""
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Training configuration must be a JSON object")
    actions = {a.dest: a for a in parser._actions}
    unknown = set(values) - set(actions)
    if unknown or "training_config" in values:
        raise ValueError(f"Unsupported training configuration keys: {sorted(unknown)}")
    for key, value in values.items():
        action = actions[key]
        if value is None:
            if action.default is not None:
                raise ValueError(f"{key} cannot be null")
            continue
        if isinstance(action.default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be a JSON boolean")
        elif action.type is not None:
            if isinstance(value, bool) or not isinstance(value, action.type):
                # JSON integers are also valid for floating-point arguments.
                if action.type is float and type(value) is int:
                    values[key] = float(value)
                else:
                    raise ValueError(f"Invalid type for {key}: {type(value).__name__}")
        if action.choices is not None and value not in action.choices:
            raise ValueError(f"Invalid value for {key}: {value!r}")
    parser.set_defaults(**values)
