import os


REMOTE_WORKTREE_ROOT = "/path/to/workspaces/m4oe-structv6"
REMOTE_STRUCTV5_ROOT = "/path/to/m40e/StructV6.0"
LOCAL_STRUCTV5_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "StructV6.0")
)


def default_structv5_root() -> str:
    override = os.environ.get("M4OE_STRUCTV5_ROOT", "").strip()
    if override:
        return os.path.abspath(override)
    script_dir = os.path.abspath(os.path.dirname(__file__))
    if os.name != "nt" and script_dir.startswith(REMOTE_WORKTREE_ROOT):
        return REMOTE_STRUCTV5_ROOT
    return LOCAL_STRUCTV5_ROOT


def structv5_path(*parts: str) -> str:
    return os.path.join(default_structv5_root(), *parts)


def ensure_structv5_root() -> str:
    root = default_structv5_root()
    os.makedirs(root, exist_ok=True)
    return root
