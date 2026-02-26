import sys
import torch
import numpy as np
import random
import multiprocessing
from trainer import Trainer, TrainerArgs

# Protect entry point for multiprocessing
if __name__ == "__main__":
    # Ensure standard python multiprocessing behavior
    multiprocessing.freeze_support()

    # --- Global random seed for reproducibility ---
    GLOBAL_SEED = 42
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)
    torch.manual_seed(GLOBAL_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(GLOBAL_SEED)
    
    # Define Args
    args = TrainerArgs()
    
    # --- Custom Configuration ---
    args.num_iterations = 10      # Adjust based on how long you want to run
    args.num_self_play_games = 100 # Increase this if using parallel to get better data
    args.checkpoint_interval = 2
    args.epochs = 5
    args.max_checkpoints = 3      # Keep only 3 latest checkpoints to save space
    args.eval_games = 10          # Games to play during evaluation
    args.update_threshold = 0.55  # Required win rate to become the new 'best' model
    args.cooldown_minutes = 8     # Hardware protection: stop for N minutes after each iteration

    
    # --- Resume Training Configuration ---
    # Set this to True to resume from the latest checkpoint automatically
    RESUME_TRAINING = True 
    
    # Path to a specific checkpoint if you want to load a specific one (e.g., './checkpoints/checkpoint_15/checkpoint.pth.tar')
    # If None and RESUME_TRAINING is True, it will look for './checkpoints/latest.pth.tar'
    CUSTOM_RESUME_PATH = None
    # CUSTOM_RESUME_PATH = "D:\\四字棋3D\\connect4_3d_ai_train\\AI_v2.1.2\\checkpoints\\best.pth.tar"
    
    resume_checkpoint = None
    if RESUME_TRAINING:
        import os
        latest_path = os.path.join(args.checkpoint_dir, 'latest.pth.tar')
        if CUSTOM_RESUME_PATH and os.path.exists(CUSTOM_RESUME_PATH):
            resume_checkpoint = CUSTOM_RESUME_PATH
        elif os.path.exists(latest_path):
            resume_checkpoint = latest_path
        
        if resume_checkpoint:
            print(f"Resume Training requested. Using checkpoint: {resume_checkpoint}")
        else:
            print("Resume Training requested but no checkpoint found. Starting from scratch.")
    
    # --- Check Iterations ---
    if RESUME_TRAINING and resume_checkpoint:
        # Simple check: if start_iter (from checkpoint) >= num_iterations, warn user
        import numpy as np
        try:
            with torch.serialization.safe_globals([np._core.multiarray._reconstruct]):
                checkpoint_data = torch.load(resume_checkpoint, map_location='cpu')
        except Exception:
            # 备用路径（不同 numpy 版本）
            try:
                with torch.serialization.safe_globals([np.core.multiarray._reconstruct]):
                    checkpoint_data = torch.load(resume_checkpoint, map_location='cpu')
            except Exception as e:
                print("警告：safe_globals 两次尝试均失败；仅在信任 checkpoint 时使用 weights_only=False")
                checkpoint_data = torch.load(resume_checkpoint, map_location='cpu', weights_only=False) 
        
        # checkpoint_data = torch.load(resume_checkpoint, map_location='cpu')

        if isinstance(checkpoint_data, dict) and 'iteration' in checkpoint_data:
            last_iter = checkpoint_data['iteration']
            if last_iter >= args.num_iterations:
                print(f"WARNING: Checkpoint is at iteration {last_iter}, but num_iterations is set to {args.num_iterations}.")
                print(f"Increasing num_iterations to {last_iter + 10} to continue training.")
                args.num_iterations = last_iter + 10
        del checkpoint_data # Free memory

    print("Initializing Training...")
    trainer = Trainer(args, resume_path=resume_checkpoint)
    
    print("Starting Training Loop...")
    trainer.train()