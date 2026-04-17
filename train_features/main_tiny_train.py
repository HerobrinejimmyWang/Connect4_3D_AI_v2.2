from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, random_split

from feature_extractor import CandidateFeatureExtractor
from history_dataset import HistoryDatasetConfig, HumanHistoryDataset
from teacher_distill import (
    TeacherSelfPlayConfig,
    TeacherSelfPlayDataset,
    generate_teacher_self_play_samples,
    load_teacher_cache_samples,
)
from tiny_policy_model import count_parameters
from tiny_trainer import TinyPolicyTrainer, TinyTrainConfig
from train_utils import build_or_resume_tiny_model, resolve_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tiny 25-candidate policy model from human history")
    parser.add_argument(
        "--history",
        nargs="+",
        default=["test/history/*.json"],
        help="History file paths, directory paths, or glob patterns.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="Tiny training device, auto if omitted")
    parser.add_argument("--resume", type=str, default=None, help="Resume from existing tiny checkpoint")
    parser.add_argument("--winner-weight", type=float, default=1.0)
    parser.add_argument("--loser-weight", type=float, default=0.35)
    parser.add_argument("--draw-weight", type=float, default=0.7)
    parser.add_argument("--teacher-kl-weight", type=float, default=0.25)
    parser.add_argument("--value-loss-weight", type=float, default=0.2)
    parser.add_argument("--teacher-data-source", type=str, choices=["auto", "cache", "self-play"], default="auto", help="Teacher data source")
    parser.add_argument("--teacher-cache-path", type=str, default="distillation/cache/teacher_examples.pth.tar", help="Cached teacher examples path")
    parser.add_argument("--teacher-cache-max-samples", type=int, default=0, help="Use at most N cached samples, 0 means all")
    parser.add_argument("--teacher-model", type=str, default=None, help="Teacher model path for KL distillation self-play.")
    parser.add_argument("--teacher-games", type=int, default=500, help="Number of teacher self-play games to generate.")
    parser.add_argument("--teacher-max-steps", type=int, default=160, help="Max steps per teacher self-play game.")
    parser.add_argument("--teacher-temp", type=float, default=0.5, help="Sampling temperature for teacher self-play.")
    parser.add_argument("--teacher-opening-random-moves", type=int, default=6, help="Random opening moves per teacher game.")
    parser.add_argument("--teacher-device", type=str, default=None, help="Teacher inference device.")
    parser.add_argument("--teacher-sample-weight", type=float, default=1.0, help="Per-sample weight for teacher self-play samples.")
    parser.add_argument("--teacher-mcts-sims", type=int, default=256, help="Teacher MCTS simulations per move")
    parser.add_argument("--teacher-mcts-threads", type=int, default=4, help="Teacher MCTS worker threads")
    parser.add_argument("--teacher-cpuct", type=float, default=1.0, help="Teacher MCTS cpuct")
    parser.add_argument("--teacher-virtual-loss", type=float, default=1.0, help="Teacher MCTS virtual loss")
    parser.add_argument("--teacher-infer-batch-size", type=int, default=32, help="Teacher inference batch size")
    parser.add_argument("--teacher-infer-timeout", type=float, default=0.003, help="Teacher inference batch timeout")
    parser.add_argument("--teacher-dirichlet-alpha", type=float, default=0.3, help="Teacher root dirichlet alpha")
    parser.add_argument("--teacher-dirichlet-epsilon", type=float, default=0.10, help="Teacher root dirichlet epsilon")
    parser.add_argument("--teacher-parallel-workers", type=int, default=0, help="Teacher self-play workers, 0 means auto")
    parser.add_argument("--teacher-cpu-worker-ratio", type=float, default=0.5, help="CPU worker budget ratio for teacher self-play")
    parser.add_argument("--output", type=str, default="train_features/checkpoints/tiny_policy_v1.pth")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    extractor = CandidateFeatureExtractor()
    history_dataset_config = HistoryDatasetConfig(
        winner_weight=float(args.winner_weight),
        loser_weight=float(args.loser_weight),
        draw_weight=float(args.draw_weight),
    )

    datasets = []
    dataset_stats = {
        "history_samples": 0,
        "teacher_samples": 0,
    }
    teacher_stats = None

    history_dataset = HumanHistoryDataset(args.history, feature_extractor=extractor, config=history_dataset_config)
    if len(history_dataset) > 0:
        datasets.append(history_dataset)
        dataset_stats["history_samples"] = len(history_dataset)

    teacher_dataset = None
    teacher_source = str(args.teacher_data_source).strip().lower()
    teacher_samples = []
    if teacher_source in ("cache", "auto"):
        cache_path = str(args.teacher_cache_path)
        try:
            teacher_samples, cache_stats = load_teacher_cache_samples(
                feature_extractor=extractor,
                cache_path=cache_path,
                sample_weight=float(args.teacher_sample_weight),
                max_samples=max(0, int(args.teacher_cache_max_samples)),
            )
            teacher_stats = {
                "source": "cache",
                **cache_stats,
            }
        except Exception:
            if teacher_source == "cache":
                raise

    if not teacher_samples and teacher_source in ("self-play", "auto") and args.teacher_model and int(args.teacher_games) > 0:
        teacher_config = TeacherSelfPlayConfig(
            model_path=str(args.teacher_model),
            num_games=int(args.teacher_games),
            max_steps_per_game=int(args.teacher_max_steps),
            temperature=float(args.teacher_temp),
            opening_random_moves=int(args.teacher_opening_random_moves),
            sample_weight=float(args.teacher_sample_weight),
            device=resolve_device(args.teacher_device),
            mcts_sims=max(1, int(args.teacher_mcts_sims)),
            mcts_threads=max(1, int(args.teacher_mcts_threads)),
            cpuct=float(args.teacher_cpuct),
            virtual_loss=float(args.teacher_virtual_loss),
            inference_batch_size=max(1, int(args.teacher_infer_batch_size)),
            inference_timeout_s=max(0.001, float(args.teacher_infer_timeout)),
            dirichlet_alpha=float(args.teacher_dirichlet_alpha),
            dirichlet_epsilon=float(args.teacher_dirichlet_epsilon),
            parallel_workers=max(0, int(args.teacher_parallel_workers)),
            cpu_worker_ratio=max(0.1, min(1.0, float(args.teacher_cpu_worker_ratio))),
        )
        teacher_samples, self_play_stats = generate_teacher_self_play_samples(extractor, teacher_config)
        teacher_stats = {
            "source": "self-play",
            **self_play_stats,
        }

    if teacher_samples:
        teacher_dataset = TeacherSelfPlayDataset(teacher_samples, feature_extractor=extractor)
        datasets.append(teacher_dataset)
        dataset_stats["teacher_samples"] = len(teacher_dataset)

    if not datasets:
        raise RuntimeError("No valid training samples were found in history or teacher self-play sources.")

    dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    if len(dataset) == 0:
        raise RuntimeError("No valid training samples were found in the provided data sources.")

    val_ratio = max(0.0, min(0.5, float(args.val_ratio)))
    val_size = int(round(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size

    if val_size > 0:
        train_set, val_set = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(int(args.seed)),
        )
    else:
        train_set = dataset
        val_set = None

    train_loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=int(args.batch_size), shuffle=False) if val_set else None

    shape = extractor.feature_shape
    model, _ = build_or_resume_tiny_model(shape, resume_path=args.resume)

    trainer = TinyPolicyTrainer(
        model=model,
        config=TinyTrainConfig(
            epochs=int(args.epochs),
            learning_rate=float(args.lr),
            weight_decay=float(args.weight_decay),
            teacher_kl_weight=float(args.teacher_kl_weight),
            value_loss_weight=float(args.value_loss_weight),
            device=resolve_device(args.device),
        ),
    )

    history = trainer.fit(train_loader, val_loader)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "architecture": "tiny-candidate-policy-v1",
            "global_dim": shape.global_dim,
            "candidate_dim": shape.candidate_dim,
            "candidate_count": shape.candidate_count,
            "global_hidden": int(model.global_hidden),
            "candidate_hidden": int(model.candidate_hidden),
            "fusion_hidden": int(model.fusion_hidden),
            "value_hidden": int(model.value_hidden),
            "dropout": float(model.dropout_rate),
        },
        "train_config": trainer.export_config(),
        "dataset_config": {
            "history": history_dataset_config.__dict__,
            "stats": dataset_stats,
            "teacher_self_play": teacher_stats,
        },
        "metrics": history,
        "parameter_count": count_parameters(model),
        "resume_from": str(args.resume) if args.resume else None,
    }
    torch.save(payload, out_path)

    metrics_path = out_path.with_suffix(out_path.suffix + ".metrics.json")
    metrics_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"[tiny-train] samples={len(dataset)} train={train_size} val={val_size} "
        f"history={dataset_stats['history_samples']} teacher={dataset_stats['teacher_samples']}"
    )
    print(f"[tiny-train] params={count_parameters(model)}")
    print(f"[tiny-train] model_saved={out_path}")
    print(f"[tiny-train] metrics_saved={metrics_path}")


if __name__ == "__main__":
    main()
