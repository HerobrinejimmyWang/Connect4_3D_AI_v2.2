import os
import sys
import time
import shutil
import logging
import gc
from random import shuffle

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing
from tqdm import tqdm

from game_rules import GameRules
from model import Connect4Net
from mcts import MCTS

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class TrainerArgs:
    def __init__(self):
        self.num_iterations = 200
        self.num_self_play_games = 100
        self.num_mcts_sims = 64
        self.cpuct = 1.0
        self.batch_size = 64
        self.epochs = 10
        self.checkpoint_dir = './checkpoints'
        self.learning_rate = 0.001
        self.weight_decay = 1e-4
        self.temp_threshold = 15
        self.history_len = 3
        self.min_game_steps = 10
        self.latest_data_weight = 1.3
        self.checkpoint_interval = 5
        self.max_checkpoints = 3
        self.update_threshold = 0.55
        self.eval_games = 10
        self.cooldown_minutes = 5

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}


# ------------------------------------------------------------------ #
# Top-level worker – must be picklable for multiprocessing
# ------------------------------------------------------------------ #
def self_play_worker(args_tuple):
    """
    Play ONE self-play game on **CPU** and return training samples.
    Receives: (game_rules_instance, model_state_dict, trainer_args, seed)
    """
    game_rules, model_state, args, seed = args_tuple

    np.random.seed(seed)
    torch.manual_seed(seed)

    game = GameRules()
    net = Connect4Net()
    net.load_state_dict(model_state)
    net.eval()

    mcts_engine = MCTS(game, net, args, device='cpu')

    examples = []
    board = game.get_init_board()
    cur_player = 1
    step = 0

    while True:
        step += 1
        canonical = game.get_canonical_form(board, cur_player)
        temp = 1 if step < args.temp_threshold else 0
        pi = mcts_engine.get_action_prob(canonical, temp=temp)
        examples.append([canonical, cur_player, pi, None])

        action = np.random.choice(len(pi), p=pi)
        board, cur_player = game.get_next_state(board, cur_player, action)
        r = game.get_game_ended(board, cur_player)

        if r != 0:
            if step < args.min_game_steps:
                return []
            data = []
            for x in examples:
                reward = r * (1 if x[1] == cur_player else -1)
                syms = game.get_symmetries(x[0], x[2])
                for b, p in syms:
                    data.append((b.astype(np.int8),
                                 p.astype(np.float32),
                                 np.float32(reward)))
            return data


# ------------------------------------------------------------------ #
class Trainer:
    """
    GPU-accelerated AlphaZero trainer for 3-D Connect Four.
    • Self-play runs on CPU workers (avoids CUDA fork issues).
    • Neural-network training uses GPU with mixed precision (AMP).
    • Model evaluation (pitting) uses GPU inference.
    """

    def __init__(self, args, resume_path=None):
        self.args = args

        # ---- Device ------------------------------------------------ #
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
            logging.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            logging.info("No GPU found – falling back to CPU.")

        self.game = GameRules()

        # Models on GPU
        self.nnet = Connect4Net().to(self.device)
        self.best_nnet = Connect4Net().to(self.device)
        self.best_nnet.load_state_dict(self.nnet.state_dict())

        self.optimizer = optim.Adam(self.nnet.parameters(),
                                    lr=args.learning_rate,
                                    weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.5)

        # Mixed-precision scaler (no-op on CPU)
        self.use_amp = (self.device.type == 'cuda')
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)

        os.makedirs(args.checkpoint_dir, exist_ok=True)

        self.train_examples_history = []
        self.start_iter = 1
        self.eval_history = []
        self.best_win_rate = -1.0

        if resume_path:
            self._load_checkpoint(resume_path)
            best_path = os.path.join(args.checkpoint_dir, 'best.pth.tar')
            if os.path.exists(best_path):
                self._load_best_model(best_path)
        else:
            logging.info("Starting training from scratch.")

    # ---- checkpoint helpers ---------------------------------------- #
    @staticmethod
    def _safe_load(path):
        """Load a torch checkpoint with numpy-compat fallbacks."""
        try:
            return torch.load(path, map_location='cpu', weights_only=False)
        except Exception:
            pass
        # Fallback with safe_globals for newer PyTorch
        for reconstruct in [getattr(np, '_core', getattr(np, 'core', None))]:
            if reconstruct is not None:
                try:
                    with torch.serialization.safe_globals(
                            [reconstruct.multiarray._reconstruct]):
                        return torch.load(path, map_location='cpu')
                except Exception:
                    continue
        raise RuntimeError(f"Cannot load checkpoint: {path}")

    def _validate_model(self):
        try:
            self.nnet.eval()
            dummy = torch.randn(1, 8, 5, 5, device=self.device)
            with torch.no_grad():
                self.nnet(dummy)
            return True
        except Exception as e:
            logging.error(f"Model validation failed: {e}")
            return False

    def _load_checkpoint(self, path):
        if not os.path.isfile(path):
            logging.warning(f"Checkpoint not found: {path}")
            return
        logging.info(f"Loading checkpoint: {path}")
        try:
            ckpt = self._safe_load(path)
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                self.nnet.load_state_dict(ckpt['state_dict'], strict=True)
                self.nnet.to(self.device)
                if not self._validate_model():
                    raise ValueError("Model validation failed after loading.")
                if 'optimizer' in ckpt:
                    try:
                        self.optimizer.load_state_dict(ckpt['optimizer'])
                    except Exception as e:
                        logging.warning(f"Optimizer state skipped: {e}")
                self.start_iter = ckpt.get('iteration', 0) + 1
                self.train_examples_history = ckpt.get(
                    'train_examples_history', [])
                self.eval_history = ckpt.get('eval_history', [])
                self.best_win_rate = ckpt.get('best_win_rate', -1.0)
                logging.info(
                    f"Resuming from iteration {self.start_iter}. "
                    f"Best win rate: {self.best_win_rate:.3f}")
            else:
                self.nnet.load_state_dict(ckpt)
                self.nnet.to(self.device)
                logging.info("Loaded weights only. Starting iter 1.")
        except Exception as e:
            logging.error(f"Checkpoint load error: {e}")
            sys.exit(1)

    def _load_best_model(self, path):
        try:
            ckpt = self._safe_load(path)
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                self.best_nnet.load_state_dict(ckpt['state_dict'])
            else:
                self.best_nnet.load_state_dict(ckpt)
            self.best_nnet.to(self.device)
            logging.info(f"Loaded best model from {path}")
        except Exception as e:
            logging.warning(f"Could not load best model: {e}")

    # ---- Self-play ------------------------------------------------- #
    def _execute_self_play(self):
        """Parallel self-play on CPU workers."""
        cpu_count = multiprocessing.cpu_count()
        num_workers = max(1, int(cpu_count * 0.6))
        logging.info(
            f"Self-play: {self.args.num_self_play_games} games "
            f"across {num_workers} workers")

        # Copy model weights to CPU for workers
        model_state = {k: v.cpu() for k, v in self.nnet.state_dict().items()}

        tasks = [(self.game, model_state, self.args,
                  int(time.time() * 1000) % (2 ** 31) + i)
                 for i in range(self.args.num_self_play_games)]

        results = []
        with multiprocessing.Pool(num_workers) as pool:
            for data in tqdm(pool.imap_unordered(self_play_worker, tasks),
                             total=self.args.num_self_play_games,
                             desc="Self-Play"):
                results.extend(data)
        return results

    # ---- Network training ------------------------------------------ #
    def _train_network(self, examples):
        """Train on GPU with mixed precision."""
        self.nnet.train()
        n = len(examples)
        batch_count = max(1, n // self.args.batch_size)
        total_loss = 0.0

        pbar = tqdm(range(batch_count), desc="Training")
        for step in pbar:
            ids = np.random.randint(n, size=self.args.batch_size)
            boards, pis, vs = zip(*[examples[i] for i in ids])

            boards_t = torch.tensor(np.array(boards), dtype=torch.float32,
                                    device=self.device)
            pis_t = torch.tensor(np.array(pis), dtype=torch.float32,
                                 device=self.device)
            vs_t = torch.tensor(np.array(vs), dtype=torch.float32,
                                device=self.device)

            with torch.amp.autocast(device_type=self.device.type,
                                    enabled=self.use_amp):
                log_pi, v = self.nnet(boards_t)
                loss_v = F.mse_loss(v.view(-1), vs_t.view(-1))
                loss_pi = -torch.sum(pis_t * log_pi) / pis_t.size(0)
                loss = loss_v + loss_pi

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{total_loss / (step + 1):.4f}")

    # ---- Evaluation ------------------------------------------------ #
    def _evaluate_model(self, iteration):
        """Pit current model vs best model using GPU inference."""
        logging.info(f"--- Evaluating at iteration {iteration} ---")

        mcts_new = MCTS(self.game, self.nnet, self.args,
                        device=self.device)
        mcts_best = MCTS(self.game, self.best_nnet, self.args,
                         device=self.device)

        wins = losses = draws = 0
        for g in range(self.args.eval_games):
            board = self.game.get_init_board()
            cur_player = 1
            new_player = 1 if g % 2 == 0 else -1

            while True:
                mcts_active = (mcts_new if cur_player == new_player
                               else mcts_best)
                canonical = self.game.get_canonical_form(board, cur_player)
                pi = mcts_active.get_action_prob(canonical, temp=0)
                action = int(np.argmax(pi))
                board, cur_player = self.game.get_next_state(
                    board, cur_player, action)
                r = self.game.get_game_ended(board, 1)

                if r != 0:
                    if ((r == 1 and new_player == 1)
                            or (r == -1 and new_player == -1)):
                        wins += 1
                    elif r == 1e-4:
                        draws += 1
                    else:
                        losses += 1
                    break

        win_rate = (wins + 0.5 * draws) / self.args.eval_games
        logging.info(
            f"Eval: W{wins} L{losses} D{draws}  WinRate={win_rate:.3f}")

        with open(os.path.join(self.args.checkpoint_dir,
                               "eval_log.txt"), "a") as f:
            f.write(f"Iter {iteration}: W{wins} L{losses} D{draws} "
                    f"WR={win_rate:.3f}\n")

        if win_rate >= self.args.update_threshold:
            logging.info("New model accepted as best.")
            self.best_win_rate = win_rate
            self.best_nnet.load_state_dict(self.nnet.state_dict())
            self._save_best_model()
        else:
            logging.info("New model rejected.")

        self.eval_history.append({
            'iteration': iteration, 'wins': wins, 'losses': losses,
            'draws': draws, 'games': self.args.eval_games,
            'win_rate': win_rate,
        })

    # ---- Checkpointing --------------------------------------------- #
    def _save_checkpoint(self, iteration):
        folder = os.path.join(self.args.checkpoint_dir,
                              f'checkpoint_{iteration}')
        os.makedirs(folder, exist_ok=True)

        while len(self.train_examples_history) > self.args.history_len:
            self.train_examples_history.pop(0)

        state = {
            'iteration': iteration,
            'state_dict': self.nnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_examples_history': self.train_examples_history,
            'eval_history': self.eval_history,
            'best_win_rate': self.best_win_rate,
            'args': self.args.to_dict(),
        }
        torch.save(state, os.path.join(folder, 'checkpoint.pth.tar'))
        torch.save(state, os.path.join(self.args.checkpoint_dir,
                                       'latest.pth.tar'))
        torch.save(self.nnet.state_dict(),
                   os.path.join(folder, 'model.pth'))
        logging.info(f"Checkpoint saved: iteration {iteration}")
        self._cleanup_checkpoints()

    def _save_best_model(self):
        state = {
            'state_dict': self.best_nnet.state_dict(),
            'best_win_rate': self.best_win_rate,
            'args': self.args.to_dict(),
        }
        torch.save(state, os.path.join(self.args.checkpoint_dir,
                                       'best.pth.tar'))
        logging.info(
            f"Best model saved (WR={self.best_win_rate:.3f})")

    def _cleanup_checkpoints(self):
        dirs = [d for d in os.listdir(self.args.checkpoint_dir)
                if d.startswith('checkpoint_')
                and os.path.isdir(
                    os.path.join(self.args.checkpoint_dir, d))]
        iters = []
        for d in dirs:
            try:
                iters.append((int(d.split('_')[1]), d))
            except ValueError:
                continue
        iters.sort()
        if len(iters) > self.args.max_checkpoints:
            for _, name in iters[:-self.args.max_checkpoints]:
                p = os.path.join(self.args.checkpoint_dir, name)
                logging.info(f"Removing old checkpoint: {name}")
                try:
                    shutil.rmtree(p)
                except Exception as e:
                    logging.warning(f"Cleanup failed: {e}")

    # ---- Main loop ------------------------------------------------- #
    def train(self):
        for i in range(self.start_iter, self.args.num_iterations + 1):
            logging.info(
                f"=== Iteration {i}/{self.args.num_iterations} ===")

            # 1. Self-play (CPU workers)
            iter_examples = self._execute_self_play()
            self.train_examples_history.append(iter_examples)

            while len(self.train_examples_history) > self.args.history_len:
                self.train_examples_history.pop(0)

            # Flatten + weight latest iteration
            train_data = []
            for idx, e in enumerate(self.train_examples_history):
                train_data.extend(e)
                if (idx == len(self.train_examples_history) - 1
                        and self.args.latest_data_weight > 1):
                    extra = int(
                        len(e) * (self.args.latest_data_weight - 1))
                    if extra > 0:
                        sample_ids = np.random.choice(
                            len(e), extra, replace=(extra > len(e)))
                        train_data.extend([e[j] for j in sample_ids])
                        logging.info(
                            f"Weighted latest: {len(e)} + {extra} extra")
            shuffle(train_data)

            # 2. Train on GPU
            self._train_network(train_data)

            del train_data
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

            self.scheduler.step()

            # 3. Save & evaluate
            if i % self.args.checkpoint_interval == 0:
                self._save_checkpoint(i)
                self._evaluate_model(i)

            # 4. Cooldown
            if (self.args.cooldown_minutes > 0
                    and i < self.args.num_iterations):
                logging.info(
                    f"Cooling down {self.args.cooldown_minutes} min…")
                for m in range(self.args.cooldown_minutes, 0, -1):
                    logging.info(f"  {m} min remaining")
                    time.sleep(60)

        # Final save
        self._save_checkpoint(self.args.num_iterations)
        self._final_report()

    def _final_report(self):
        path = os.path.join(self.args.checkpoint_dir, "FINAL_REPORT.txt")
        with open(path, "w") as f:
            f.write("=== Training Final Report ===\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Iterations: {self.args.num_iterations}\n")
            f.write(f"Games/iter: {self.args.num_self_play_games}\n")
            if self.eval_history:
                wrs = [e['win_rate'] for e in self.eval_history]
                f.write(
                    f"Avg win-rate vs best: {sum(wrs)/len(wrs):.3f}\n")
                f.write(f"Best win-rate: {self.best_win_rate:.3f}\n")
                last = self.eval_history[-1]
                f.write(
                    f"Last eval (Iter {last['iteration']}): "
                    f"W{last['wins']} L{last['losses']} D{last['draws']}\n")
            latest = os.path.join(self.args.checkpoint_dir,
                                  'latest.pth.tar')
            best = os.path.join(self.args.checkpoint_dir, 'best.pth.tar')
            if os.path.exists(latest):
                f.write(f"\nResume point: {latest}\n")
            if os.path.exists(best):
                f.write(
                    f"Best model (WR {self.best_win_rate:.3f}): {best}\n")
        logging.info(f"Report saved: {path}")
