import os


def default_structv5_root() -> str:
    override = os.environ.get("M4OE_STRUCTV5_ROOT", "").strip()
    return os.path.abspath(override or os.path.join(os.path.dirname(__file__), "..", "outputs"))


def structv5_path(*parts: str) -> str:
    return os.path.join(default_structv5_root(), *parts)


def ensure_structv5_root() -> str:
    root = default_structv5_root()
    os.makedirs(root, exist_ok=True)
    return root
