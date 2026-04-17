from __future__ import annotations

from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CURRENT_DIR.parent
MODEL_EXTENSIONS = {".pth", ".pt", ".ckpt", ".tar"}


def discover_models(search_root=None):
    root = Path(search_root) if search_root else WORKSPACE_ROOT
    models = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if ".venv" in path.parts:
            continue
        suffixes = {suffix.lower() for suffix in path.suffixes}
        if not suffixes.intersection(MODEL_EXTENSIONS):
            continue
        file_name = path.name.lower()
        if not (file_name.endswith(".pth") or file_name.endswith(".pth.tar")):
            continue
        relative_path = path.relative_to(root)
        models.append(
            {
                "label": _make_label(relative_path),
                "path": str(path),
                "relative_path": str(relative_path),
            }
        )
    models.sort(key=lambda item: item["relative_path"].lower())
    return models


def _make_label(relative_path):
    parts = list(relative_path.parts)
    if len(parts) <= 2:
        return str(relative_path)
    return f"{parts[0]}/{parts[-2]}/{parts[-1]}"