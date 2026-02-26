import os
import sys
import time
import copy
import shutil
import logging
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import multiprocessing
import gc
from tqdm import tqdm
from collections import deque
from random import shuffle

from game_rules import GameRules
from model import Connect4Net, board_to_channels, NUM_INPUT_CHANNELS
from mcts import MCTS

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Connect4Dataset(torch.utils.data.Dataset):
    """Dataset wrapping training examples for use with DataLoader."""
    def __init__(self, examples):
        boards, pis, vs = zip(*examples)
        self.boards = [np.array(b) for b in boards]
        self.pis = np.array(pis, dtype=np.float32)
        self.vs = np.array(vs, dtype=np.float32)

    def __len__(self):
        return len(self.vs)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(board_to_channels(self.boards[idx])),
            torch.FloatTensor(self.pis[idx]),
            torch.FloatTensor([self.vs[idx]]),
        )


class TrainerArgs:
    def __init__(self):
        self.num_iterations = 200     # Total training iterations
        self.num_self_play_games = 100 # Games per iteration (Parallelized)
        self.num_mcts_sims = 64       # MCTS simulations per move
        self.cpuct = 1.0              # PUCT exploration constant
        self.batch_size = 64          # Training batch size
        self.epochs = 10              # Training epochs per iteration
        self.checkpoint_dir = './checkpoints'
        self.learning_rate = 0.001
        self.weight_decay = 1e-4      # L2 Regularization
        self.temp_threshold = 15      # Temperature threshold for exploration
        self.history_len = 20             # Number of iterations to keep history
        self.min_game_steps = 10          # Filter games shorter than this
        self.latest_data_weight = 2.0     # Weight for the latest iteration's data (2x oversampling)
        self.checkpoint_interval = 5      # Checkpoint every X iterations
        self.max_checkpoints = 3          # Number of old checkpoints to keep (excluding best/latest)
        self.update_threshold = 0.55      # Win rate required to replace the 'best' model
        self.eval_games = 10              # Number of games to play during evaluation
        self.cooldown_minutes = 5         # Cooldown time after each iteration (minutes)
        self.train_device = 'cpu'         # Device used for training (e.g. 'cpu', 'cuda:0')
        self.infer_device = 'cpu'         # Device used for inference / self-play workers

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('__')}

# --- Worker Function for Parallel Processing ---
# Must be top-level for pickling in multiprocessing
def self_play_worker(args_tuple):
    """
    Worker function to play ONE game independently.
    Args: (game_rules, model_state_dict, trainer_args, seed)
    """
    game_rules, model_state, args, seed = args_tuple
    
    # Set seed for this worker
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Reconstruct environment
    game = GameRules()
    net = Connect4Net()
    net.load_state_dict(model_state)
    net.to(args.infer_device)
    net.eval() # Set to eval mode for inference
    
    mcts = MCTS(game, net, args)
    
    train_examples = []
    board = game.get_init_board()
    cur_player = 1
    episode_step = 0
    
    while True:
        episode_step += 1
        canonical_board = game.get_canonical_form(board, cur_player)
        
        # Temperature function: Higher exploration in early steps
        temp = 1 if episode_step < args.temp_threshold else 0
        
        pi = mcts.get_action_prob(canonical_board, temp=temp)
        train_examples.append([canonical_board, cur_player, pi, None])
        
        action = np.random.choice(len(pi), p=pi)
        board, cur_player = game.get_next_state(board, cur_player, action)
        
        r = game.get_game_ended(board, cur_player)
        
        if r != 0:
            # Game ended. 
            # Data filtering: filter out games that ended too early
            if episode_step < args.min_game_steps:
                return []
                
            return_data = []
            for x in train_examples:
                # Reward for the player who was in 'canonical_board' perspective
                reward = r * (1 if x[1] == cur_player else -1)
                
                # Data Augmentation: Add all 8 symmetries
                syms = game.get_symmetries(x[0], x[2])
                for b, p in syms:
                    # Memory optimization: Use int8 for board and float32 for policy
                    return_data.append((b.astype(np.int8), p.astype(np.float32), float(reward)))
            return return_data

class Trainer:
    def __init__(self, args, resume_path=None):
        self.args = args
        self.game = GameRules()
        self.nnet = Connect4Net().to(args.train_device)
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        # Learning rate scheduler: reduce LR by 0.5 every 50 iterations
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.5)
        
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
            
        self.train_examples_history = []  # history of examples
        self.start_iter = 1
        self.eval_history = []  # list of dicts: {'iteration', 'wins','losses','draws','games'}
        self.best_win_rate = -1.0 # Track best performance
        
        # Best model for pitting
        self.best_nnet = Connect4Net().to(args.train_device)
        self.best_nnet.load_state_dict(self.nnet.state_dict()) # Initial best is current
        
        # Resume functionality
        if resume_path:
            self.load_checkpoint(resume_path)
            # After loading resume, try to load the best model as opponent
            best_path = os.path.join(args.checkpoint_dir, 'best.pth.tar')
            if os.path.exists(best_path):
                logging.info(f"Loading previous best model for evaluation: {best_path}")
                try:
                    try:
                        with torch.serialization.safe_globals([np._core.multiarray._reconstruct]):
                            best_checkpoint = torch.load(best_path, map_location='cpu')
                    except Exception:
                        try:
                            with torch.serialization.safe_globals([np.core.multiarray._reconstruct]):
                                best_checkpoint = torch.load(best_path, map_location='cpu')
                        except Exception:
                            best_checkpoint = torch.load(best_path, map_location='cpu', weights_only=False)
                    self.best_nnet.load_state_dict(best_checkpoint['state_dict'])
                except Exception as e:
                    logging.warning(f"Could not load best model for pitting: {e}")
        else:
            logging.info("Starting training from scratch.")

    def validate_model(self):
        """
        Performs a dummy forward pass to ensure model and weights are compatible.
        """
        try:
            self.nnet.eval()
            device = next(self.nnet.parameters()).device
            dummy_input = torch.randn(1, NUM_INPUT_CHANNELS, 8, 5, 5).to(device)
            with torch.no_grad():
                pi, v = self.nnet(dummy_input)
            return True
        except Exception as e:
            logging.error(f"Model validation failed: {e}")
            return False

    def load_checkpoint(self, resume_path):
        """
        Loads the training state from a checkpoint file.
        Supports both simple model weights and full training state.
        Now includes model validation and best win rate restoration.
        """
        if not os.path.isfile(resume_path):
            logging.warning(f"Checkpoint file not found: {resume_path}")
            return

        logging.info(f"Loading checkpoint: {resume_path}")
        try:
            try:
                with torch.serialization.safe_globals([np._core.multiarray._reconstruct]):
                    checkpoint = torch.load(resume_path, map_location='cpu')
            except Exception:
                try:
                    with torch.serialization.safe_globals([np.core.multiarray._reconstruct]):
                        checkpoint = torch.load(resume_path, map_location='cpu')
                except Exception:
                    logging.warning("safe_globals failed, falling back to weights_only=False")
                    checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
            
            # Case 1: Full training state (dictionary)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                # 1. Load weights with strict=True to catch architecture mismatches
                try:
                    self.nnet.load_state_dict(checkpoint['state_dict'], strict=True)
                except RuntimeError as e:
                    logging.error(f"Architecture mismatch during checkpoint load: {e}")
                    raise ValueError("Cannot resume: model architecture in checkpoint doesn't match current code.")

                # 2. Basic forward pass validation
                if not self.validate_model():
                    raise ValueError("Model validation failed after loading state_dict.")

                # 3. Restore optimizer
                if 'optimizer' in checkpoint:
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer'])
                    except Exception as e:
                        logging.warning(f"Could not load optimizer state: {e}. Starting with fresh optimizer.")
                
                # 4. Restore other states
                self.start_iter = checkpoint.get('iteration', 0) + 1
                self.train_examples_history = checkpoint.get('train_examples_history', [])
                self.eval_history = checkpoint.get('eval_history', [])
                self.best_win_rate = checkpoint.get('best_win_rate', -1.0)
                
                # Optional: Restore/Compare parameters
                saved_args = checkpoint.get('args', {})
                if saved_args:
                    # Check for significant mismatches if needed, but here we just log
                    logging.info(f"Loaded hyperparameters from checkpoint (e.g. lr={saved_args.get('learning_rate')})")

                logging.info(f"Resuming from iteration {self.start_iter}. Previous best win rate: {self.best_win_rate:.3f}")
                if len(self.train_examples_history) > 0:
                    logging.info(f"Restored {len(self.train_examples_history)} iterations of training data.")
            
            # Case 2: Just model weights (OrderedDict from state_dict)
            else:
                self.nnet.load_state_dict(checkpoint)
                if self.validate_model():
                    logging.info("Loaded model weights only. Starting from iteration 1.")
                else:
                    raise ValueError("Model weights loaded but validation failed.")
                
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
            logging.info("Critical error during resume. Ensure your model architecture has not changed.")
            sys.exit(1) # Exit to allow user to investigate architecture issues

    def execute_episode_parallel(self):
        """
        Runs self-play using multiprocessing pool.
        """
        cpu_count = multiprocessing.cpu_count()
        # Use 60% of CPU cores, ensuring at least 1
        num_workers = max(1, int(cpu_count * 0.6))
        logging.info(f"Spawning {num_workers} workers for self-play...")
        
        # Send CPU state dict to workers (handles GPU->CPU transfer if needed)
        model_state = {k: v.cpu() for k, v in self.nnet.state_dict().items()}
        
        # Prepare arguments for each game
        # We need unique seeds for randomness
        tasks = []
        for i in range(self.args.num_self_play_games):
            seed = int(time.time()) + i
            tasks.append((self.game, model_state, self.args, seed))
        
        iteration_examples = []
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            # tqdm for progress bar
            results = list(tqdm(pool.imap(self_play_worker, tasks), total=self.args.num_self_play_games, desc="Self-Play"))
            
            for game_data in results:
                iteration_examples.extend(game_data)
                
        return iteration_examples

    def train(self):
        for i in range(self.start_iter, self.args.num_iterations + 1):
            logging.info(f'Starting Iteration {i}/{self.args.num_iterations}')
            
            # 1. Self-Play
            iter_examples = self.execute_episode_parallel()
            self.train_examples_history.append(iter_examples)
            
            # Keep history limited
            if len(self.train_examples_history) > self.args.history_len:
                logging.info(f"Removing oldest history (keep last {self.args.history_len})")
                self.train_examples_history.pop(0)
            
            # Flatten list with weighting for the latest iteration
            train_data = []
            for idx, e in enumerate(self.train_examples_history):
                if idx == len(self.train_examples_history) - 1:
                    # Latest iteration: apply weight
                    train_data.extend(e)
                    if self.args.latest_data_weight > 1:
                        extra_count = int(len(e) * (self.args.latest_data_weight - 1))
                        if extra_count > 0:
                            # Sample extra data from the latest iteration
                            extra_indices = np.random.choice(len(e), extra_count, replace=extra_count > len(e))
                            train_data.extend([e[i] for i in extra_indices])
                        logging.info(f"Weighted latest iteration: {len(e)} original + {extra_count} extra samples")
                else:
                    train_data.extend(e)
            
            shuffle(train_data)
            
            # 2. Train Neural Net
            self.train_network(train_data)
            
            # Free memory from train_data
            del train_data
            gc.collect()
            
            self.scheduler.step()
            
            # 3. Save Checkpoint & Evaluate
            if i % self.args.checkpoint_interval == 0:
                self.save_checkpoint(i)
                self.evaluate_model(i)

            # 4. Hardware Protection: Cooldown period
            if self.args.cooldown_minutes > 0 and i < self.args.num_iterations:
                logging.info(f"Hardware Protection: Cooling down for {self.args.cooldown_minutes} minutes...")
                for minute in range(self.args.cooldown_minutes, 0, -1):
                    logging.info(f"Cooling down... {minute} minutes remaining.")
                    time.sleep(60)
                logging.info("Cooldown finished. Resuming training.")

        # 训练循环结束后：确保最终模型被保存一次
        logging.info("Saving final checkpoint...")
        self.save_checkpoint(self.args.num_iterations)
        self.final_report()

    def train_network(self, examples):
        """
        Train the network on examples using DataLoader for proper epoch-based sampling.
        examples: list of (board, policy, value)
        """
        self.nnet.train()
        device = next(self.nnet.parameters()).device
        dataset = Connect4Dataset(examples)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.args.batch_size, shuffle=True
        )

        for epoch in range(self.args.epochs):
            total_loss = 0.0
            pbar = tqdm(loader, desc=f"Training Epoch {epoch + 1}/{self.args.epochs}")
            for batch_idx, (boards, target_pis, target_vs) in enumerate(pbar):
                boards = boards.to(device)
                target_pis = target_pis.to(device)
                target_vs = target_vs.to(device)

                out_pi, out_v = self.nnet(boards)

                # Value loss: mean squared error
                l_v = F.mse_loss(out_v.view(-1), target_vs.view(-1))
                # Policy loss: cross-entropy (out_pi is log_softmax output)
                l_pi = -(target_pis * out_pi).sum(dim=1).mean()

                total_l = l_v + l_pi

                self.optimizer.zero_grad()
                total_l.backward()
                torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), max_norm=5.0)
                self.optimizer.step()

                total_loss += total_l.item()
                pbar.set_postfix(loss=total_loss / (batch_idx + 1))

    def save_checkpoint(self, iteration):
        """
        Saves the training state and model weights.
        """
        folder = os.path.join(self.args.checkpoint_dir, f'checkpoint_{iteration}')
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            
        filepath = os.path.join(folder, 'checkpoint.pth.tar')
        model_filepath = os.path.join(folder, 'model.pth')
        latest_filepath = os.path.join(self.args.checkpoint_dir, 'latest.pth.tar')

        # Memory Optimization: Prune history strictly before saving
        while len(self.train_examples_history) > self.args.history_len:
            self.train_examples_history.pop(0)

        state = {
            'iteration': iteration,
            'state_dict': self.nnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_examples_history': self.train_examples_history,
        }

        # Save full state
        torch.save(state, filepath)
        # Also save as latest for easy resume
        torch.save(state, latest_filepath)
        
        # Legacy/Simple model save (weights only)
        torch.save(self.nnet.state_dict(), model_filepath)
        
        logging.info(f"Checkpoint saved: {filepath}")
        
        # Cleanup old checkpoints
        self.cleanup_checkpoints()

    def save_best_model(self):
        """
        Separately saves the best performing model.
        """
        best_filepath = os.path.join(self.args.checkpoint_dir, 'best.pth.tar')
        state = {
            'state_dict': self.best_nnet.state_dict(),
            'best_win_rate': self.best_win_rate,
            'args': self.args.to_dict()
        }
        torch.save(state, best_filepath)
        logging.info(f"Updated BEST model saved: {best_filepath} (WinRate: {self.best_win_rate:.3f})")

    def cleanup_checkpoints(self):
        """
        Keeps only the most recent N checkpoints to save disk space.
        """
        checkpoint_folders = [f for f in os.listdir(self.args.checkpoint_dir) 
                             if f.startswith("checkpoint_") and os.path.isdir(os.path.join(self.args.checkpoint_dir, f))]
        
        # Extract iteration numbers and sort
        iterations = []
        for folder in checkpoint_folders:
            try:
                iter_num = int(folder.split('_')[1])
                iterations.append((iter_num, folder))
            except ValueError:
                continue
        
        iterations.sort(key=lambda x: x[0])
        
        # If we have more than max_checkpoints, delete the oldest ones
        if len(iterations) > self.args.max_checkpoints:
            to_delete = iterations[:-self.args.max_checkpoints]
            for it_num, folder_name in to_delete:
                folder_path = os.path.join(self.args.checkpoint_dir, folder_name)
                logging.info(f"Cleaning up old checkpoint: {folder_name}")
                try:
                    shutil.rmtree(folder_path)
                except Exception as e:
                    logging.warning(f"Failed to delete {folder_path}: {e}")

    def evaluate_model(self, iteration):
        """
        AlphaZero-style evaluation: Pit the current model against the best model so far.
        If the current model wins by a margin (update_threshold), it becomes the new best.
        """
        logging.info(f"--- Evaluating model vs Best version at iteration {iteration} ---")
        
        num_games = self.args.eval_games
        wins = 0
        losses = 0
        draws = 0
        
        # MCTS for both sides
        # Current nnet is P1 half the time, P2 half the time
        mcts_new = MCTS(self.game, self.nnet, self.args)
        mcts_best = MCTS(self.game, self.best_nnet, self.args)
        
        for i in range(num_games):
            board = self.game.get_init_board()
            cur_player = 1
            game_over = False
            
            # Alternate who starts
            # If i is even: NewModel=P1 (Player 1), BestModel=P2 (Player -1)
            # If i is odd: BestModel=P1 (Player 1), NewModel=P2 (Player -1)
            new_model_player = 1 if i % 2 == 0 else -1
            
            while not game_over:
                # Decide which MCTS to use
                active_mcts = mcts_new if cur_player == new_model_player else mcts_best
                
                canonical_board = self.game.get_canonical_form(board, cur_player)
                pi = active_mcts.get_action_prob(canonical_board, temp=0) # Deterministic
                action = np.argmax(pi)
                
                board, cur_player = self.game.get_next_state(board, cur_player, action)
                r = self.game.get_game_ended(board, 1) # Check from P1 perspective
                
                if r != 0:
                    game_over = True
                    # Determine result for the NEW model
                    # If r=1, P1 won. If new_model was P1, it won.
                    # If r=-1, P2 won. If new_model was P2, it won.
                    if (r == 1 and new_model_player == 1) or (r == -1 and new_model_player == -1):
                        wins += 1
                    elif r == 1e-4: # Draw
                        draws += 1
                    else:
                        losses += 1
                    
        win_rate = (wins + 0.5 * draws) / num_games
        logging.info(f"Pitting Result - Wins: {wins}, Losses: {losses}, Draws: {draws} | WinRate: {win_rate:.3f}")
        
        # Save to log
        with open(os.path.join(self.args.checkpoint_dir, "eval_log.txt"), "a") as f:
            f.write(f"Iter {iteration} vs Best: Wins {wins}/{num_games}, Losses {losses}, Draws {draws}, WinRate {win_rate:.3f}\n")

        # Update best model if win_rate exceeds threshold
        if win_rate >= self.args.update_threshold:
            logging.info(f"SUCCESS: New model is better than best (Threshold: {self.args.update_threshold}). Updating best.")
            self.best_win_rate = win_rate
            self.best_nnet.load_state_dict(self.nnet.state_dict())
            self.save_best_model()
        else:
            logging.info(f"REJECTED: New model rejected. Best model remains unchanged.")

        self.eval_history.append({
            'iteration': iteration,
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'games': num_games,
            'win_rate': win_rate
        })

    def final_report(self):
        report_path = os.path.join(self.args.checkpoint_dir, "FINAL_REPORT.txt")
        with open(report_path, "w") as f:
            f.write("=== Training Final Report ===\n")
            f.write(f"Total Iterations: {self.args.num_iterations}\n")
            f.write(f"Self-Play Games per Iter: {self.args.num_self_play_games}\n")
            f.write("Model trained successfully.\n")
            f.write("Check 'eval_log.txt' for progress history.\n")

            # 新增：如果有评估历史，输出评估指标摘要
            if len(self.eval_history) > 0:
                f.write("\n=== Evaluation Summary (vs Best) ===\n")
                total_evals = len(self.eval_history)
                f.write(f"Number of Evaluations: {total_evals}\n")
                # 计算平均/最佳/最后胜率
                win_rates = [e['win_rate'] for e in self.eval_history]
                avg_win = sum(win_rates)/len(win_rates)
                last_entry = self.eval_history[-1]
                f.write(f"Average Win Rate vs Previous Best: {avg_win:.3f}\n")
                f.write(f"Final Best Win Rate Recorded: {self.best_win_rate:.3f}\n")
                f.write(f"Last Eval (Iter {last_entry['iteration']}): Wins {last_entry['wins']}, Losses {last_entry['losses']}, Draws {last_entry['draws']}\n")
            
            # 最终模型路径信息
            latest_model_path = os.path.join(self.args.checkpoint_dir, 'latest.pth.tar')
            best_model_path = os.path.join(self.args.checkpoint_dir, 'best.pth.tar')
            
            if os.path.exists(latest_model_path):
                f.write(f"\nLatest training state (Resume point): {latest_model_path}\n")
            if os.path.exists(best_model_path):
                f.write(f"Best model tracked (Win rate {self.best_win_rate:.3f}): {best_model_path}\n")
            
            last_checkpoint_model = os.path.join(self.args.checkpoint_dir, f'checkpoint_{self.args.num_iterations}', 'model.pth')
            if os.path.exists(last_checkpoint_model):
                f.write(f"Final model weights (weights only): {last_checkpoint_model}\n")
            else:
                f.write("\nCheckpoints available in the checkpoint directory.\n")
        logging.info(f"Training Complete. Report saved to {report_path}")
