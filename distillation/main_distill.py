import argparse
import json
import multiprocessing
import os
import random
import sys

import numpy as np
import torch
import torch.multiprocessing as mp

from distill_trainer import DistillationArgs, DistillationTrainer


def parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3D Connect4 distillation entrypoint")
    parser.add_argument("--config", type=str, default="distill_config.py", help="Path to distillation config (.json/.py)")
    parser.add_argument("--run-name", type=str, default=None, help="Run name suffix")
    parser.add_argument(
        "--active-model-preset",
        type=str,
        choices=["balanced", "fast"],
        default=None,
        help="Select which student preset to train. Both presets can be configured in config file, but only one runs per process.",
    )
    parser.add_argument("--teacher-data-generate-mode", action="store_true", help="Generate teacher cache and exit")
    parser.add_argument("--teacher-data-generation-enabled", action="store_true", help="Enable teacher cache generation if missing")
    parser.add_argument("--teacher-data-regenerate", action="store_true", help="Force regenerate teacher cache")

    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint under run dir")
    parser.add_argument("--resume-path", type=str, default=None, help="Resume or rollback from specific checkpoint path")
    parser.add_argument("--resume-weights-only", action="store_true", help="Only restore model weights")
    parser.add_argument("--rollback-iteration", type=int, default=None, help="Rollback to checkpoint_<iteration>")
    parser.add_argument("--continue-from-iteration", type=int, default=None, help="Force start iteration")
    parser.add_argument("--force-overwrite", action="store_true", help="Overwrite run directory when not resuming")

    parser.add_argument("--num-iterations", type=int, default=None, help="Override total distillation iterations")
    parser.add_argument("--num-self-play-games", type=int, default=None, help="Override self-play games per iteration")
    parser.add_argument("--checkpoint-interval", type=int, default=None, help="Override checkpoint save interval")
    parser.add_argument("--eval-interval", type=int, default=None, help="Override evaluation interval")

    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--print-config", action="store_true", help="Print resolved config and exit")
    return parser.parse_args()


def resolve_args(cli_args: argparse.Namespace) -> DistillationArgs:
    if cli_args.config:
        args = DistillationArgs.from_config_file(cli_args.config)
    else:
        args = DistillationArgs()

    args.seed = int(cli_args.seed)

    if cli_args.run_name:
        args.run_name = str(cli_args.run_name)
    if cli_args.active_model_preset:
        args.active_model_preset = str(cli_args.active_model_preset)

    if cli_args.teacher_data_generate_mode:
        args.teacher_data_generate_mode = True
    if cli_args.teacher_data_generation_enabled:
        args.teacher_data_generation_enabled = True
    if cli_args.teacher_data_regenerate:
        args.teacher_data_regenerate = True

    if cli_args.resume:
        args.resume = True
    if cli_args.resume_path:
        args.resume_path = str(cli_args.resume_path)
    if cli_args.resume_weights_only:
        args.resume_weights_only = True
    if cli_args.rollback_iteration is not None:
        args.rollback_iteration = int(cli_args.rollback_iteration)
    if cli_args.continue_from_iteration is not None:
        args.continue_from_iteration = int(cli_args.continue_from_iteration)
    if cli_args.force_overwrite:
        args.force_overwrite = True

    if cli_args.num_iterations is not None:
        args.num_iterations = int(cli_args.num_iterations)
    if cli_args.num_self_play_games is not None:
        args.num_self_play_games = int(cli_args.num_self_play_games)
    if cli_args.checkpoint_interval is not None:
        args.checkpoint_interval = int(cli_args.checkpoint_interval)
    if cli_args.eval_interval is not None:
        args.eval_interval = int(cli_args.eval_interval)

    if not torch.cuda.is_available():
        args.train_device = "cpu"
        args.shared_inference_device = "cpu"

    return args


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    cli_args = parse_cli_args()
    args = resolve_args(cli_args)

    multiprocessing.freeze_support()
    mp.set_start_method("spawn", force=True)

    set_global_seed(args.seed)

    if cli_args.print_config:
        print(json.dumps(args.to_dict(), ensure_ascii=False, indent=2))
        return

    print("Resolved distillation presets:")
    print(json.dumps(
        {
            "active_model_preset": args.active_model_preset,
            "balanced_model_preset": args.balanced_model_preset,
            "fast_model_preset": args.fast_model_preset,
        },
        ensure_ascii=False,
        indent=2,
    ))

    trainer = DistillationTrainer(args)

    if args.teacher_data_generate_mode:
        print("Teacher cache generation enabled; training will continue after refresh.")

    trainer.train()


if __name__ == "__main__":
    main()
