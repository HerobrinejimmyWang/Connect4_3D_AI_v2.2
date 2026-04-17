from __future__ import annotations

import argparse

from torch.utils.data import DataLoader, random_split

from feature_extractor import CandidateFeatureExtractor
from teacher_distill import (
    TeacherSelfPlayConfig,
    TeacherSelfPlayDataset,
    generate_teacher_self_play_samples,
    load_teacher_cache_samples,
)
from tiny_trainer import TinyPolicyTrainer, TinyTrainConfig
from train_utils import build_or_resume_tiny_model, resolve_device, save_tiny_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tiny model from AlphaZero teacher self-play with MCTS")
    parser.add_argument("--teacher-data-source", type=str, choices=["cache", "self-play", "auto"], default="cache", help="Teacher data source")
    parser.add_argument("--teacher-cache-path", type=str, default="distillation/cache/teacher_examples.pth.tar", help="Cached teacher examples path")
    parser.add_argument("--teacher-cache-max-samples", type=int, default=200, help="Use at most N cached samples, 0 means all")
    parser.add_argument("--teacher-model", type=str, required=False, default="D:/四字棋3D/AI_v2.2/save_model/v2.2_fast/model.pth", help="Path to AlphaZero teacher model (used by self-play)")
    parser.add_argument("--teacher-games", type=int, default=200)
    parser.add_argument("--teacher-max-steps", type=int, default=160)
    parser.add_argument("--teacher-temp", type=float, default=0.5, help="Move sampling temperature")
    parser.add_argument("--teacher-opening-random-moves", type=int, default=6)
    parser.add_argument("--teacher-sample-weight", type=float, default=1.0)
    parser.add_argument("--teacher-device", type=str, default=None, help="Teacher inference device, auto if omitted")
    parser.add_argument("--teacher-mcts-sims", type=int, default=256)
    parser.add_argument("--teacher-mcts-threads", type=int, default=4)
    parser.add_argument("--teacher-cpuct", type=float, default=1.0)
    parser.add_argument("--teacher-virtual-loss", type=float, default=1.0)
    parser.add_argument("--teacher-infer-batch-size", type=int, default=32)
    parser.add_argument("--teacher-infer-timeout", type=float, default=0.003)
    parser.add_argument("--teacher-dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--teacher-dirichlet-epsilon", type=float, default=0.10)
    parser.add_argument("--teacher-parallel-workers", type=int, default=0, help="Teacher self-play workers, 0 means auto")
    parser.add_argument("--teacher-cpu-worker-ratio", type=float, default=0.5, help="CPU worker budget ratio for teacher self-play")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="Tiny training device, auto if omitted")
    parser.add_argument("--resume", type=str, default=None, help="Resume from existing tiny checkpoint")
    parser.add_argument("--teacher-kl-weight", type=float, default=0.35)
    parser.add_argument("--value-loss-weight", type=float, default=0.2)
    parser.add_argument("--output", type=str, default="train_features/checkpoints/tiny_policy_teacher_v1.pth")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    extractor = CandidateFeatureExtractor()

    source = str(args.teacher_data_source).strip().lower()
    teacher_samples = []
    teacher_stats = None

    if source in ("cache", "auto"):
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
            if source == "cache":
                raise
            teacher_samples = []
            teacher_stats = {"source": "cache", "cache_loaded": 0.0, "cache_path": cache_path}

    if not teacher_samples:
        if source == "cache":
            raise RuntimeError("Teacher cache mode selected, but no usable cache samples were loaded.")
        if not args.teacher_model:
            raise ValueError("--teacher-model is required when teacher data source falls back to self-play.")

        teacher_cfg = TeacherSelfPlayConfig(
            model_path=str(args.teacher_model),
            num_games=int(args.teacher_games),
            max_steps_per_game=int(args.teacher_max_steps),
            temperature=float(args.teacher_temp),
            opening_random_moves=int(args.teacher_opening_random_moves),
            sample_weight=float(args.teacher_sample_weight),
            device=resolve_device(args.teacher_device),
            mcts_sims=int(args.teacher_mcts_sims),
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

        teacher_samples, self_play_stats = generate_teacher_self_play_samples(extractor, teacher_cfg)
        teacher_stats = {
            "source": "self-play",
            **self_play_stats,
        }

    if len(teacher_samples) == 0:
        raise RuntimeError("Teacher data source produced no samples.")

    dataset = TeacherSelfPlayDataset(teacher_samples, feature_extractor=extractor)

    val_ratio = max(0.0, min(0.5, float(args.val_ratio)))
    val_size = int(round(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size

    if val_size > 0:
        train_set, val_set = random_split(dataset, [train_size, val_size])
    else:
        train_set = dataset
        val_set = None

    train_loader = DataLoader(train_set, batch_size=int(args.batch_size), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=int(args.batch_size), shuffle=False) if val_set else None

    model, _ = build_or_resume_tiny_model(extractor.feature_shape, resume_path=args.resume)

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

    metrics = trainer.fit(train_loader, val_loader)
    out_path, metrics_path = save_tiny_checkpoint(
        output_path=args.output,
        model=model,
        train_config=trainer.export_config(),
        dataset_config={
            "entry": "teacher",
            "teacher_self_play": teacher_stats,
            "teacher_config": {
                "data_source": source,
                "cache_path": str(args.teacher_cache_path),
                "cache_max_samples": int(args.teacher_cache_max_samples),
                "model_path": str(args.teacher_model) if args.teacher_model else None,
                "mcts_sims": int(args.teacher_mcts_sims),
                "mcts_threads": int(args.teacher_mcts_threads),
            },
            "stats": {"teacher_samples": len(dataset)},
        },
        metrics=metrics,
        resume_from=args.resume,
    )

    print(f"[tiny-teacher] samples={len(dataset)} train={train_size} val={val_size}")
    print(f"[tiny-teacher] source={teacher_stats.get('source', source)}")
    print(f"[tiny-teacher] mcts_sims={int(args.teacher_mcts_sims)}")
    print(f"[tiny-teacher] workers={int(teacher_stats.get('parallel_workers', 1))} cpu_ratio={float(teacher_stats.get('cpu_worker_ratio', 0.5)):.2f}")
    print(f"[tiny-teacher] model_saved={out_path}")
    print(f"[tiny-teacher] metrics_saved={metrics_path}")


if __name__ == "__main__":
    main()
