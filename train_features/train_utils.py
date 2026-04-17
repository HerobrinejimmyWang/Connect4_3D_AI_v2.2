from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

try:
    from .feature_extractor import FeatureShape
    from .tiny_policy_model import TinyCandidatePolicyNet, count_parameters
except ImportError:
    from feature_extractor import FeatureShape
    from tiny_policy_model import TinyCandidatePolicyNet, count_parameters


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(requested: str | None = None) -> str:
    text = (requested or "").strip().lower()
    has_cuda = bool(torch.cuda.is_available())
    if not text:
        return "cuda" if has_cuda else "cpu"
    if text.startswith("cuda") and not has_cuda:
        return "cpu"
    return text


def build_or_resume_tiny_model(
    feature_shape: FeatureShape,
    resume_path: str | None = None,
) -> Tuple[TinyCandidatePolicyNet, Dict[str, Any] | None]:
    if not resume_path:
        model = TinyCandidatePolicyNet(
            global_dim=int(feature_shape.global_dim),
            candidate_dim=int(feature_shape.candidate_dim),
        )
        return model, None

    checkpoint_path = Path(resume_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"Invalid tiny checkpoint format: {checkpoint_path}")

    model_cfg = dict(payload.get("model_config") or {})
    global_dim = int(model_cfg.get("global_dim", feature_shape.global_dim))
    candidate_dim = int(model_cfg.get("candidate_dim", feature_shape.candidate_dim))
    candidate_count = int(model_cfg.get("candidate_count", feature_shape.candidate_count))

    if global_dim != int(feature_shape.global_dim):
        raise ValueError(
            f"Resume checkpoint global_dim={global_dim} does not match current extractor global_dim={feature_shape.global_dim}."
        )
    if candidate_dim != int(feature_shape.candidate_dim):
        raise ValueError(
            f"Resume checkpoint candidate_dim={candidate_dim} does not match current extractor candidate_dim={feature_shape.candidate_dim}."
        )
    if candidate_count != int(feature_shape.candidate_count):
        raise ValueError(
            f"Resume checkpoint candidate_count={candidate_count} does not match current extractor candidate_count={feature_shape.candidate_count}."
        )

    model = TinyCandidatePolicyNet(
        global_dim=global_dim,
        candidate_dim=candidate_dim,
        global_hidden=int(model_cfg.get("global_hidden", 24)),
        candidate_hidden=int(model_cfg.get("candidate_hidden", 24)),
        fusion_hidden=int(model_cfg.get("fusion_hidden", 16)),
        dropout=float(model_cfg.get("dropout", 0.05)),
        value_hidden=int(model_cfg.get("value_hidden", 12)),
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model, payload


def save_tiny_checkpoint(
    output_path: str | Path,
    model: TinyCandidatePolicyNet,
    train_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    metrics: list,
    resume_from: str | None = None,
) -> Tuple[Path, Path]:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "architecture": "tiny-candidate-policy-v1",
            "global_dim": int(model.global_dim),
            "candidate_dim": int(model.candidate_dim),
            "candidate_count": 25,
            "global_hidden": int(model.global_hidden),
            "candidate_hidden": int(model.candidate_hidden),
            "fusion_hidden": int(model.fusion_hidden),
            "value_hidden": int(model.value_hidden),
            "dropout": float(model.dropout_rate),
        },
        "train_config": dict(train_config),
        "dataset_config": dict(dataset_config),
        "metrics": list(metrics),
        "parameter_count": int(count_parameters(model)),
        "resume_from": str(resume_from) if resume_from else None,
    }
    torch.save(payload, out_path)

    metrics_path = out_path.with_suffix(out_path.suffix + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path, metrics_path
