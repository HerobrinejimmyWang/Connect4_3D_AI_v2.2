from __future__ import annotations

import argparse

from torch.utils.data import DataLoader, random_split

from feature_extractor import CandidateFeatureExtractor
from history_dataset import HistoryDatasetConfig, HumanHistoryDataset
from tiny_trainer import TinyPolicyTrainer, TinyTrainConfig
from train_utils import build_or_resume_tiny_model, resolve_device, save_tiny_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tiny model from human-vs-model history samples")
    parser.add_argument("--history", nargs="+", default=["test/history/*.json"], help="History paths or glob patterns")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="Training device, auto if omitted")
    parser.add_argument("--resume", type=str, default=None, help="Resume from existing tiny checkpoint")
    parser.add_argument("--winner-weight", type=float, default=1.0)
    parser.add_argument("--loser-weight", type=float, default=0.35)
    parser.add_argument("--draw-weight", type=float, default=0.7)
    parser.add_argument("--teacher-kl-weight", type=float, default=0.0, help="Optional KL if teacher_probs exists in dataset")
    parser.add_argument("--value-loss-weight", type=float, default=0.0)
    parser.add_argument("--output", type=str, default="train_features/checkpoints/tiny_policy_human_v1.pth")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    extractor = CandidateFeatureExtractor()
    dataset_config = HistoryDatasetConfig(
        winner_weight=float(args.winner_weight),
        loser_weight=float(args.loser_weight),
        draw_weight=float(args.draw_weight),
    )
    dataset = HumanHistoryDataset(args.history, feature_extractor=extractor, config=dataset_config)
    if len(dataset) == 0:
        raise RuntimeError("No valid human-history training samples were found.")

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
            "entry": "human",
            "history": dataset_config.__dict__,
            "stats": {"history_samples": len(dataset)},
        },
        metrics=metrics,
        resume_from=args.resume,
    )

    print(f"[tiny-human] samples={len(dataset)} train={train_size} val={val_size}")
    print(f"[tiny-human] model_saved={out_path}")
    print(f"[tiny-human] metrics_saved={metrics_path}")


if __name__ == "__main__":
    main()
