import sys
import os
import torch
import numpy as np
import multiprocessing

from trainer import Trainer, TrainerArgs

if __name__ == "__main__":
    # Use 'spawn' to avoid CUDA context issues in child processes
    multiprocessing.set_start_method('spawn', force=True)
    multiprocessing.freeze_support()

    args = TrainerArgs()

    # --- Training hyper-parameters ---
    args.num_iterations = 10
    args.num_self_play_games = 100
    args.checkpoint_interval = 2
    args.epochs = 5
    args.max_checkpoints = 3
    args.eval_games = 10
    args.update_threshold = 0.55
    args.cooldown_minutes = 8

    # --- Resume configuration ---
    RESUME_TRAINING = True
    CUSTOM_RESUME_PATH = None

    resume_checkpoint = None
    if RESUME_TRAINING:
        latest_path = os.path.join(args.checkpoint_dir, 'latest.pth.tar')
        if CUSTOM_RESUME_PATH and os.path.exists(CUSTOM_RESUME_PATH):
            resume_checkpoint = CUSTOM_RESUME_PATH
        elif os.path.exists(latest_path):
            resume_checkpoint = latest_path

        if resume_checkpoint:
            print(f"Resuming from checkpoint: {resume_checkpoint}")
        else:
            print("No checkpoint found – starting from scratch.")

    # --- Adjust num_iterations if checkpoint already past it ---
    if resume_checkpoint:
        try:
            ckpt = torch.load(resume_checkpoint, map_location='cpu',
                              weights_only=False)
        except Exception:
            ckpt = {}
        if isinstance(ckpt, dict) and 'iteration' in ckpt:
            last_iter = ckpt['iteration']
            if last_iter >= args.num_iterations:
                print(f"Checkpoint at iter {last_iter} >= "
                      f"num_iterations {args.num_iterations}. "
                      f"Extending to {last_iter + 10}.")
                args.num_iterations = last_iter + 10
        del ckpt

    print("Initializing Trainer …")
    trainer = Trainer(args, resume_path=resume_checkpoint)

    print("Starting Training Loop …")
    trainer.train()