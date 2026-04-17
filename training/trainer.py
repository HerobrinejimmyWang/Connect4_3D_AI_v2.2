import os
import sys
import time
import shutil
import logging
import json
from datetime import datetime
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import multiprocessing
import gc
import copy
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from collections import deque

from game_rules import GameRules
from model import Connect4Net, board_to_channels, NUM_INPUT_CHANNELS, extract_model_config
from parallel_games import execute_evaluation_parallel, execute_self_play_parallel
from teacher_data import (
    build_history_entry,
    build_teacher_bootstrap_args,
    compose_training_data,
    estimate_teacher_sample_budget,
    flatten_history_examples,
    get_teacher_replay_ratio,
    get_teacher_warmup_iterations,
    is_teacher_warmup_iteration,
    load_teacher_bootstrap_buffer,
    resolve_teacher_buffer_path,
    save_teacher_bootstrap_buffer,
)
from model_compat import (
    load_checkpoint_payload,
    load_compatible_model,
)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Connect4Dataset(torch.utils.data.Dataset):
    """Dataset wrapping training examples for use with DataLoader."""
    def __init__(self, examples):
        if not examples:
            self.raw_boards = torch.empty((0, 6, 5, 5), dtype=torch.int16)
            self.boards = torch.empty((0, NUM_INPUT_CHANNELS, 6, 5, 5), dtype=torch.float32)
            self.pis = torch.empty((0, 0), dtype=torch.float32)
            self.vs = torch.empty((0, 1), dtype=torch.float32)
            return

        boards, pis, vs = zip(*examples)
        boards_np = np.asarray(boards, dtype=np.int16)
        encoded_boards = np.stack([board_to_channels(board) for board in boards_np], axis=0).astype(np.float32, copy=False)
        pis_np = np.asarray(pis, dtype=np.float32)
        vs_np = np.asarray(vs, dtype=np.float32).reshape(-1, 1)

        self.raw_boards = torch.from_numpy(boards_np)
        self.boards = torch.from_numpy(encoded_boards)
        self.pis = torch.from_numpy(pis_np)
        self.vs = torch.from_numpy(vs_np)

    def __len__(self):
        return len(self.vs)

    def __getitem__(self, idx):
        return (
            self.raw_boards[idx],
            self.boards[idx],
            self.pis[idx],
            self.vs[idx],
        )


class TrainerArgs:
    def __init__(self):
        has_cuda = torch.cuda.is_available()
        cpu_count = max(1, multiprocessing.cpu_count())

        self.num_iterations = 200     # Total training iterations
        self.num_self_play_games = 100 # Games per iteration (Parallelized)
        self.num_channels = 256       # Neural Network channels
        self.num_mcts_sims = 64       # MCTS simulations per move
        self.cpuct = 1.0              # PUCT exploration constant
        self.dirichlet_alpha = 0.3    # Alpha for Dirichlet Noise
        self.dirichlet_epsilon = 0.25 # Mixing weight for Dirichlet Noise
        self.batch_size = 512 if has_cuda else 64  # Training batch size
        self.epochs = 10              # Training epochs per iteration
        self.checkpoint_dir = './checkpoints'
        self.learning_rate = 0.001
        self.weight_decay = 1e-4      # L2 Regularization
        self.self_play_exploration_strength = 1.0
        self.self_play_phase_schedule = [
            {
                'name': 'opening_stable',
                'max_step': 7,
                'temperature': 0.05,
                'dirichlet_alpha': 0.35,
                'dirichlet_epsilon': 0.03,
            },
            {
                'name': 'early_midgame_probe',
                'max_step': 20,
                'temperature': 0.18,
                'dirichlet_alpha': 0.24,
                'dirichlet_epsilon': 0.08,
            },
            {
                'name': 'mid_lategame_greedy',
                'max_step': None,
                'temperature': 0.0,
                'dirichlet_alpha': 0.0,
                'dirichlet_epsilon': 0.0,
            },
        ]
        self.exploration_iteration_schedule = [
            {'start_iter': 1, 'end_iter': 30, 'temperature_scale': 1.0, 'noise_scale': 1.0},
            {'start_iter': 31, 'end_iter': 80, 'temperature_scale': 0.85, 'noise_scale': 0.75},
            {'start_iter': 81, 'end_iter': None, 'temperature_scale': 0.65, 'noise_scale': 0.45},
        ]
        self.history_len = 20             # Number of iterations to keep history
        self.min_game_steps = 8          # Filter games shorter than this
        self.min_game_steps_start_iteration = 11  # Enable short-game filtering from iteration 11 onward
        self.latest_data_weight = 1.0     # Disable latest-iteration oversampling temporarily
        self.checkpoint_interval = 5      # Checkpoint every X iterations
        self.eval_interval = 5            # Evaluate every X iterations
        self.lr_decay_step_size = 50      # Reduce LR every X iterations
        self.lr_decay_gamma = 0.8         # LR decay factor
        self.min_learning_rate = 0.0      # Optional lower bound for LR decay
        self.max_checkpoints = 3          # Number of old checkpoints to keep (excluding best/latest)
        self.update_threshold = 0.55      # Win rate required to replace the 'best' model
        self.eval_games = 10              # Number of games to play during evaluation
        self.best_eval_games_per_generation = 30
        self.best_eval_required_generations = 2
        self.best_update_threshold = 0.55
        self.best_eval_parallelize_generations = True
        self.best_recent_filename = 'best_new.pth.tar'
        self.best_older_filename = 'best_old.pth.tar'
        self.best_legacy_filename = 'best.pth.tar'
        self.train_device = 'cuda' if has_cuda else 'cpu'  # Device used for training
        self.infer_device = 'cpu'         # Device used for self-play workers
        self.shared_inference_device = 'cuda' if has_cuda else 'cpu'
        self.data_loader_workers = max(2, cpu_count // 2) if has_cuda else 0
        self.self_play_cpu_ratio = 0.75   # Ratio of CPU cores used for self-play
        self.self_play_workers = min(4, cpu_count)
        self.max_self_play_workers = cpu_count
        self.shared_inference_server_count = 2 if has_cuda else 1
        self.high_mcts_shared_inference_server_threshold = 1024
        self.high_mcts_shared_inference_server_count = self.shared_inference_server_count
        self.compatible_inference_server_count = 1
        self.num_mcts_threads = 8
        self.virtual_loss = 1.0
        self.inference_batch_size = 32
        self.inference_timeout_s = 0.003
        self.enable_tf32 = has_cuda       # Allow TF32 on Ampere+ for faster matmul/convolution
        self.mcts_schedule = [
            (1, 20, 64),
            (21, 40, 128),
            (41, None, 256),
        ]
        self.mcts_sim_candidates = [64, 128, 256, 512, 1024]
        self.mcts_promotion_improve_count = 2
        self.eval_interval_after_best = 3
        self.eval_boost_rounds_after_improve = 2
        self.random_baseline_stability_threshold = 0.60
        self.always_evaluate_random_baseline = False
        self.random_baseline_eval_min_mcts_sims = 256
        self.enable_random_baseline_eval = False
        self.loss_increase_patience = 3   # Early stop if iteration loss strictly increases for N iterations
        self.no_improve_eval_patience = 3 # Early stop if evaluation fails to beat best for N times
        self.info_log_name = 'train_info.log'
        self.auxiliary_model_path = None
        self.auxiliary_model_label = 'legacy_teacher'
        self.auxiliary_eval_games = 12
        self.auxiliary_eval_interval = 2
        self.teacher_bootstrap_num_games = 64
        self.teacher_bootstrap_buffer_path = None
        self.teacher_bootstrap_regenerate = False
        self.teacher_bootstrap_mcts_sims = 128
        self.teacher_bootstrap_temperature = 0.06
        self.teacher_bootstrap_dirichlet_alpha = 0.20
        self.teacher_bootstrap_dirichlet_epsilon = 0.04
        self.teacher_bootstrap_warmup_iterations = 2
        self.teacher_replay_initial_ratio = 1.0
        self.teacher_replay_final_ratio = 0.0
        self.teacher_replay_decay_iterations = 20
        self.teacher_opponent_history_len = 40
        self.teacher_replay_drift_threshold = 0.35
        self.teacher_replay_drift_ratio = 0.60
        self.teacher_replay_relax_start_iteration = 0
        self.teacher_replay_relax_end_iteration = 0
        self.teacher_replay_relaxed_drift_ratio = 0.0
        self.step_stagnation_window = 5
        self.step_stagnation_mean_tolerance = 0.01
        self.step_stagnation_variance_tolerance = 0.01
        self.tactical_override_max_step = 12
        self.tactical_override_prefer_win = True
        self.tactical_override_prefer_block = True
        self.enable_option_a_recovery = False
        self.option_a_policy_only_iterations = 8
        self.option_a_low_temp_min = 0.3
        self.option_a_low_temp_max = 0.5
        self.option_a_low_temp_bridge_iterations = 3
        self.option_a_freeze_value_prefixes = ['val_']
        self.option_a_freeze_shared_prefixes = []
        self.value_loss_weight = 1.0
        self.policy_loss_weight = 1.0
        self.policy_head_lr_scale = 1.0
        self.value_head_lr_scale = 0.8

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('__')}


def _prune_history_to_limit(history, limit):
    if limit <= 0:
        return []
    if len(history) <= limit:
        return history
    return history[-limit:]

class Trainer:
    def __init__(self, args, resume_path=None):
        self.args = args
        self._normalize_devices()
        self._configure_runtime_for_devices()

        self.game = GameRules()
        self.nnet = Connect4Net(num_channels=args.num_channels).to(args.train_device)
        self.optimizer = self._build_optimizer_with_head_lr_scales()
        # Learning rate scheduler: reduce LR every step_size iterations
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=getattr(args, 'lr_decay_step_size', 50), 
            gamma=getattr(args, 'lr_decay_gamma', 0.8)
        )
        
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
            
        self.train_examples_history = []  # history of examples
        self.start_iter = 1
        self.eval_history = []  # list of dicts: {'iteration', 'wins','losses','draws','games'}
        self.iteration_metrics_history = []
        self.best_win_rate = -1.0 # Track best performance
        self.best_model_iteration = 0
        self.best_model_label = 'random_model'
        self.older_best_win_rate = -1.0
        self.older_best_model_iteration = 0
        self.older_best_model_label = 'none'
        self.consecutive_no_improve_evals = 0
        self.iteration_loss_history = deque(maxlen=self.args.loss_increase_patience)
        self.stop_reason = None
        self.info_log_path = os.path.join(self.args.checkpoint_dir, self.args.info_log_name)
        self.current_mcts_stage_index = self._resolve_mcts_stage_index(int(self.args.num_mcts_sims))
        self.mcts_promotion_progress_count = 0
        self.boosted_eval_rounds_remaining = 0
        self.next_eval_iteration = max(1, int(self.args.eval_interval))
        self.option_a_policy_only_end_iter = None
        self.option_a_low_temp_bridge_end_iter = None
        self.option_a_state = 'disabled'
        self._base_self_play_phase_schedule = copy.deepcopy(getattr(self.args, 'self_play_phase_schedule', []))
        
        # Best model for pitting
        self.best_nnet = None
        self.older_best_nnet = None
        self.best_model_iteration = 0
        self.best_model_label = 'random_model'
        self.auxiliary_nnet = None
        self.auxiliary_model_config = None
        self.auxiliary_model_metadata = {}
        self.auxiliary_eval_history = []
        self.teacher_opponent_history = []
        self.teacher_bootstrap_examples = []
        self.teacher_bootstrap_metadata = {}
        self.teacher_bootstrap_buffer_path = None
        self.latest_auxiliary_win_rate = None
        self.last_self_play_game_results = []
        self.teacher_logging_enabled = False
        
        # Resume functionality
        if resume_path:
            self.load_checkpoint(resume_path)
        else:
            logging.info("Starting training from scratch. Evaluation will be against random policy.")

        self._load_best_generations_from_disk()

        self._load_auxiliary_model()
        self._initialize_teacher_bootstrap_buffer()
        self.teacher_logging_enabled = self._is_teacher_pipeline_enabled()

        self._initialize_option_a_recovery_state()

        self._initialize_info_log(resume_path)

    def _normalize_devices(self):
        if not torch.cuda.is_available():
            self.args.train_device = 'cpu'
            self.args.infer_device = 'cpu'
            return

        # Avoid oversubscribing GPU from many self-play workers unless explicitly requested.
        if isinstance(self.args.infer_device, str) and self.args.infer_device.startswith('cuda'):
            logging.warning(
                "infer_device is set to CUDA. Multiprocess self-play can under-utilize GPU and cause OOM."
            )

    def _configure_runtime_for_devices(self):
        if isinstance(self.args.train_device, str) and self.args.train_device.startswith('cuda'):
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = bool(self.args.enable_tf32)
            torch.backends.cudnn.allow_tf32 = bool(self.args.enable_tf32)
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

    def _initialize_info_log(self, resume_path=None):
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        if not os.path.exists(self.info_log_path):
            with open(self.info_log_path, 'w', encoding='utf-8') as f:
                f.write("=== Connect4 Training Info Log ===\n")
        mode = 'Resume' if resume_path else 'Start'
        self._append_info_log([
            f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {mode} training session",
            (
                f"train_device={self.args.train_device}, infer_device={self.args.infer_device}, "
                f"shared_inference_device={self.args.shared_inference_device}, num_channels={self.args.num_channels}, "
                f"batch_size={self.args.batch_size}, epochs={self.args.epochs}, "
                f"shared_inference_servers={int(getattr(self.args, 'shared_inference_server_count', 1))}, "
                f"compatible_inference_servers={int(getattr(self.args, 'compatible_inference_server_count', 1))}"
            ),
            f"self_play_exploration_strength={getattr(self.args, 'self_play_exploration_strength', 1.0):.3f}",
            f"self_play_phase_schedule={self._format_phase_schedule()}",
            f"exploration_iteration_schedule={self._format_iteration_schedule(getattr(self.args, 'exploration_iteration_schedule', []), 'temperature_scale', 'noise_scale')}",
        ])
        if self.teacher_logging_enabled:
            self._append_info_log(
                (
                    f"teacher_bootstrap_games={int(getattr(self.args, 'teacher_bootstrap_num_games', 0))}, "
                    f"teacher_warmup_iters={int(getattr(self.args, 'teacher_bootstrap_warmup_iterations', 0))}, "
                    f"teacher_replay_ratio={float(getattr(self.args, 'teacher_replay_initial_ratio', 0.0)):.3f}"
                )
            )
            self._append_info_log(
                (
                    f"teacher_replay_schedule=initial={float(getattr(self.args, 'teacher_replay_initial_ratio', 0.0)):.3f}, "
                    f"final={float(getattr(self.args, 'teacher_replay_final_ratio', 0.0)):.3f}, "
                    f"decay_iters={int(getattr(self.args, 'teacher_replay_decay_iterations', 1))}, "
                    f"drift_threshold={float(getattr(self.args, 'teacher_replay_drift_threshold', 0.0)):.3f}, "
                    f"drift_ratio={float(getattr(self.args, 'teacher_replay_drift_ratio', 0.0)):.3f}, "
                    f"drift_relax={int(getattr(self.args, 'teacher_replay_relax_start_iteration', 0))}"
                    f"->{int(getattr(self.args, 'teacher_replay_relax_end_iteration', 0))}, "
                    f"relaxed_drift_ratio={float(getattr(self.args, 'teacher_replay_relaxed_drift_ratio', 0.0)):.3f}"
                )
            )
        if self.teacher_logging_enabled and self.auxiliary_nnet is not None:
            self._append_info_log(
                f"auxiliary_model={self.args.auxiliary_model_label} @ {self.args.auxiliary_model_path}"
            )
        if self.teacher_logging_enabled and self.teacher_bootstrap_buffer_path:
            self._append_info_log(
                f"teacher_bootstrap_buffer={self.teacher_bootstrap_buffer_path} | samples={len(self.teacher_bootstrap_examples)}"
            )
        if bool(getattr(self.args, 'enable_option_a_recovery', False)):
            self._append_info_log(
                "option_a_recovery="
                f"enabled | policy_only_iters={int(getattr(self.args, 'option_a_policy_only_iterations', 0) or 0)} | "
                f"freeze_value_prefixes={list(getattr(self.args, 'option_a_freeze_value_prefixes', ['val_']))} | "
                f"freeze_shared_prefixes={list(getattr(self.args, 'option_a_freeze_shared_prefixes', []))} | "
                f"low_temp_bridge_iters={int(getattr(self.args, 'option_a_low_temp_bridge_iterations', 0) or 0)} | "
                f"low_temp=[{float(getattr(self.args, 'option_a_low_temp_min', 0.3)):.3f}, "
                f"{float(getattr(self.args, 'option_a_low_temp_max', 0.5)):.3f}]"
            )
        if self.teacher_logging_enabled:
            teacher_budget = self._get_teacher_sample_budget()
            if teacher_budget.get('estimated_total_teacher_samples_before_pure') is not None:
                self._append_info_log(
                    "teacher_sample_budget="
                    f"warmup={teacher_budget.get('warmup_teacher_samples', 0)} | "
                    f"replay_before_pure={teacher_budget.get('estimated_teacher_replay_samples_before_pure', 0)} | "
                    f"total_before_pure={teacher_budget.get('estimated_total_teacher_samples_before_pure', 0)}"
                )

    def _append_info_log(self, lines):
        if isinstance(lines, str):
            lines = [lines]
        with open(self.info_log_path, 'a', encoding='utf-8') as f:
            for line in lines:
                f.write(f"{line}\n")

    def _format_duration(self, seconds):
        if seconds < 60:
            return f"{seconds:.2f}s"
        minutes, sec = divmod(seconds, 60)
        if minutes < 60:
            return f"{int(minutes):02d}:{sec:05.2f}"
        hours, minutes = divmod(minutes, 60)
        return f"{int(hours):02d}:{int(minutes):02d}:{sec:05.2f}"

    def _get_learning_rate(self):
        return float(self.optimizer.param_groups[0]['lr'])

    def _apply_min_learning_rate(self):
        min_learning_rate = float(getattr(self.args, 'min_learning_rate', 0.0) or 0.0)
        if min_learning_rate <= 0:
            return False

        any_clamped = False
        for param_group in self.optimizer.param_groups:
            group_scale = float(param_group.get('lr_scale', 1.0) or 1.0)
            floor_lr = min_learning_rate * group_scale
            current_lr = float(param_group['lr'])
            if current_lr < floor_lr:
                param_group['lr'] = floor_lr
                any_clamped = True
        return any_clamped

    def _get_head_lr_scales(self):
        policy_scale = float(getattr(self.args, 'policy_head_lr_scale', 1.0) or 1.0)
        value_scale = float(getattr(self.args, 'value_head_lr_scale', 1.0) or 1.0)

        if policy_scale <= 0:
            logging.warning("policy_head_lr_scale<=0 is invalid. Falling back to 1.0")
            policy_scale = 1.0
        if value_scale <= 0:
            logging.warning("value_head_lr_scale<=0 is invalid. Falling back to 1.0")
            value_scale = 1.0
        return policy_scale, value_scale

    def _build_optimizer_with_head_lr_scales(self):
        base_lr = float(self.args.learning_rate)
        policy_scale, value_scale = self._get_head_lr_scales()

        shared_and_policy_params = []
        value_head_params = []
        for name, param in self.nnet.named_parameters():
            if name.startswith('val_'):
                value_head_params.append(param)
            else:
                shared_and_policy_params.append(param)

        if not shared_and_policy_params or not value_head_params:
            logging.warning(
                "Could not split policy/value parameter groups as expected. Falling back to single-group optimizer."
            )
            return optim.Adam(self.nnet.parameters(), lr=base_lr, weight_decay=self.args.weight_decay)

        optimizer = optim.Adam(
            [
                {
                    'params': shared_and_policy_params,
                    'lr': base_lr * policy_scale,
                    'group_name': 'policy_and_shared',
                    'lr_scale': policy_scale,
                },
                {
                    'params': value_head_params,
                    'lr': base_lr * value_scale,
                    'group_name': 'value_head',
                    'lr_scale': value_scale,
                },
            ],
            weight_decay=self.args.weight_decay,
        )
        logging.info(
            "Head LR scales enabled: base_lr=%.6f, policy/shared=%.6f (x%.3f), value=%.6f (x%.3f)",
            base_lr,
            base_lr * policy_scale,
            policy_scale,
            base_lr * value_scale,
            value_scale,
        )
        return optimizer

    def _get_optimizer_group_scales(self):
        scales = []
        for idx, group in enumerate(self.optimizer.param_groups):
            scale = group.get('lr_scale')
            if scale is None:
                if idx == 0:
                    scale = float(getattr(self.args, 'policy_head_lr_scale', 1.0) or 1.0)
                elif idx == 1:
                    scale = float(getattr(self.args, 'value_head_lr_scale', 1.0) or 1.0)
                else:
                    scale = 1.0
            scales.append(float(scale))
        return scales

    def _initialize_option_a_recovery_state(self):
        if not bool(getattr(self.args, 'enable_option_a_recovery', False)):
            self.option_a_state = 'disabled'
            return

        bridge_iters = max(0, int(getattr(self.args, 'option_a_low_temp_bridge_iterations', 0) or 0))
        policy_only_iters = max(0, int(getattr(self.args, 'option_a_policy_only_iterations', 0) or 0))
        if policy_only_iters <= 0:
            self._set_loss_weights(value_weight=1.0, policy_weight=1.0)
            self._apply_option_a_freeze_plan(policy_only=False)
            if bridge_iters > 0:
                self.option_a_state = 'active_low_temp'
                self.option_a_low_temp_bridge_end_iter = max(self.start_iter, self.start_iter + bridge_iters - 1)
                self._apply_option_a_low_temperature_schedule()
                logging.info(
                    "Option A recovery enabled without policy-only stage: low-temperature bridge from iter %s to %s, then restore base schedule.",
                    self.start_iter,
                    self.option_a_low_temp_bridge_end_iter,
                )
            else:
                self.option_a_state = 'completed'
                self._restore_option_a_base_temperature_schedule()
            return

        self.option_a_policy_only_end_iter = max(self.start_iter, self.start_iter + policy_only_iters - 1)
        self.option_a_state = 'policy_only'
        self._set_loss_weights(value_weight=0.0, policy_weight=1.0)
        self._apply_option_a_freeze_plan(policy_only=True)
        logging.info(
            "Option A recovery enabled: policy-only training from iter %s to %s, then unfreeze and apply low temperature.",
            self.start_iter,
            self.option_a_policy_only_end_iter,
        )

    def _set_loss_weights(self, value_weight, policy_weight):
        self.args.value_loss_weight = float(value_weight)
        self.args.policy_loss_weight = float(policy_weight)

    def _name_matches_prefixes(self, name, prefixes):
        return any(name.startswith(prefix) for prefix in prefixes)

    def _apply_option_a_freeze_plan(self, policy_only):
        value_prefixes = list(getattr(self.args, 'option_a_freeze_value_prefixes', ['val_']))
        shared_prefixes = list(getattr(self.args, 'option_a_freeze_shared_prefixes', []))

        for name, param in self.nnet.named_parameters():
            if not policy_only:
                param.requires_grad = True
                continue

            freeze_param = self._name_matches_prefixes(name, value_prefixes) or self._name_matches_prefixes(name, shared_prefixes)
            param.requires_grad = not freeze_param

    def _apply_option_a_low_temperature_schedule(self):
        schedule = copy.deepcopy(self._base_self_play_phase_schedule)
        if not schedule:
            return

        low_min = float(getattr(self.args, 'option_a_low_temp_min', 0.3))
        low_max = float(getattr(self.args, 'option_a_low_temp_max', 0.5))
        if low_min > low_max:
            low_min, low_max = low_max, low_min

        for entry in schedule:
            original_temp = float(entry.get('temperature', 0.0))
            entry['temperature'] = max(low_min, min(low_max, original_temp))

        self.args.self_play_phase_schedule = schedule
        logging.info(
            "Option A recovery: low-temperature schedule applied (clamped to [%.3f, %.3f]).",
            low_min,
            low_max,
        )

    def _restore_option_a_base_temperature_schedule(self):
        self.args.self_play_phase_schedule = copy.deepcopy(self._base_self_play_phase_schedule)
        logging.info("Option A recovery: restored original self-play temperature schedule.")

    def _update_option_a_stage(self, iteration):
        if not bool(getattr(self.args, 'enable_option_a_recovery', False)):
            return None

        if self.option_a_state == 'policy_only' and self.option_a_policy_only_end_iter is not None and int(iteration) > int(self.option_a_policy_only_end_iter):
            self._set_loss_weights(value_weight=1.0, policy_weight=1.0)
            self._apply_option_a_freeze_plan(policy_only=False)
            bridge_iters = max(0, int(getattr(self.args, 'option_a_low_temp_bridge_iterations', 0) or 0))
            if bridge_iters > 0:
                self.option_a_state = 'active_low_temp'
                self.option_a_low_temp_bridge_end_iter = max(int(iteration), int(iteration) + bridge_iters - 1)
                self._apply_option_a_low_temperature_schedule()
                return (
                    "Option A switched to joint training: value network unfrozen, "
                    "low-temperature bridge enabled before restoring base schedule."
                )

            self.option_a_state = 'completed'
            self._restore_option_a_base_temperature_schedule()
            return "Option A switched to joint training and directly restored base temperature schedule."

        if self.option_a_state == 'active_low_temp' and self.option_a_low_temp_bridge_end_iter is not None and int(iteration) > int(self.option_a_low_temp_bridge_end_iter):
            self.option_a_state = 'completed'
            self._restore_option_a_base_temperature_schedule()
            return "Option A low-temperature bridge finished; resumed original temperature schedule."

        return None

    def _get_option_a_progress_note(self, iteration):
        if not bool(getattr(self.args, 'enable_option_a_recovery', False)):
            return None

        if self.option_a_state == 'policy_only':
            if self.option_a_policy_only_end_iter is None:
                return "Option A policy-only stage is active."
            return (
                f"Option A policy-only stage active ({int(iteration)}/{int(self.option_a_policy_only_end_iter)}), "
                "value loss weight=0."
            )

        if self.option_a_state == 'active_low_temp':
            if self.option_a_low_temp_bridge_end_iter is None:
                return "Option A low-temperature bridge stage is active."
            return (
                f"Option A low-temperature bridge active ({int(iteration)}/{int(self.option_a_low_temp_bridge_end_iter)}), "
                "will restore base schedule automatically."
            )

        if self.option_a_state == 'completed':
            return "Option A handoff completed; running with base temperature schedule."

        return (
            f"Option A low-temperature joint stage active, value loss weight={float(getattr(self.args, 'value_loss_weight', 1.0)):.2f}."
        )

    def _format_phase_schedule(self):
        schedule = getattr(self.args, 'self_play_phase_schedule', [])
        if not schedule:
            return '[]'

        parts = []
        for entry in schedule:
            max_step = entry.get('max_step')
            step_label = f"<= {int(max_step)}" if max_step is not None else 'rest'
            parts.append(
                f"{entry.get('name', 'phase')}({step_label}, temp={float(entry.get('temperature', 0.0)):.3f}, "
                f"alpha={float(entry.get('dirichlet_alpha', 0.0)):.3f}, eps={float(entry.get('dirichlet_epsilon', 0.0)):.3f})"
            )
        return ' | '.join(parts)

    def _format_iteration_schedule(self, schedule, *value_keys):
        if not schedule:
            return '[]'

        parts = []
        for entry in schedule:
            start_iter = int(entry.get('start_iter', 1))
            end_iter = entry.get('end_iter')
            iter_label = f"{start_iter}-{int(end_iter)}" if end_iter is not None else f">={start_iter}"
            values = ', '.join(f"{key}={float(entry.get(key, 1.0)):.3f}" for key in value_keys)
            parts.append(f"{iter_label}({values})")
        return ' | '.join(parts)

    def _load_auxiliary_model(self):
        model_path = getattr(self.args, 'auxiliary_model_path', None)
        if not model_path:
            return
        if not os.path.isfile(model_path):
            logging.warning(f"Auxiliary model not found: {model_path}")
            return

        try:
            model, config, metadata = load_compatible_model(model_path, device=self.args.train_device)
        except Exception as e:
            logging.warning(f"Failed to load auxiliary model {model_path}: {e}")
            return

        self.auxiliary_nnet = model
        self.auxiliary_model_config = config
        self.auxiliary_model_metadata = metadata
        logging.info(
            "Loaded auxiliary teacher model %s (%s, input_channels=%s)",
            getattr(self.args, 'auxiliary_model_label', 'legacy_teacher'),
            metadata.get('architecture', 'unknown'),
            metadata.get('input_channels', 'unknown'),
        )

    def _get_auxiliary_parallel_spec(self):
        if self.auxiliary_nnet is None or self.auxiliary_model_config is None:
            return None
        return {
            'state_dict': {k: v.detach().cpu() for k, v in self.auxiliary_nnet.state_dict().items()},
            'config': dict(self.auxiliary_model_config),
            'label': getattr(self.args, 'auxiliary_model_label', 'legacy_teacher'),
        }

    def _resolve_teacher_buffer_path(self):
        return resolve_teacher_buffer_path(self.args)

    def _load_teacher_bootstrap_buffer(self):
        try:
            examples, metadata, buffer_path = load_teacher_bootstrap_buffer(self.args)
        except Exception as e:
            logging.warning(f"Failed to load teacher bootstrap buffer {self._resolve_teacher_buffer_path()}: {e}")
            return False

        self.teacher_bootstrap_examples = list(examples)
        self.teacher_bootstrap_metadata = dict(metadata)
        self.teacher_bootstrap_buffer_path = buffer_path
        if not self.teacher_bootstrap_examples:
            return False
        logging.info(
            "Loaded teacher bootstrap buffer: samples=%s path=%s",
            len(self.teacher_bootstrap_examples),
            buffer_path,
        )
        return True

    def _save_teacher_bootstrap_buffer(self):
        if not self.teacher_bootstrap_examples:
            return

        buffer_path = save_teacher_bootstrap_buffer(
            self.args,
            self.teacher_bootstrap_examples,
            self.teacher_bootstrap_metadata,
        )
        self.teacher_bootstrap_buffer_path = buffer_path
        logging.info(
            "Saved teacher bootstrap buffer: samples=%s path=%s",
            len(self.teacher_bootstrap_examples),
            buffer_path,
        )

    def _build_teacher_bootstrap_args(self):
        return build_teacher_bootstrap_args(self.args)

    def _generate_teacher_bootstrap_buffer(self):
        opponent_spec = self._get_auxiliary_parallel_spec()
        if opponent_spec is None:
            return False

        num_games = max(0, int(getattr(self.args, 'teacher_bootstrap_num_games', 0) or 0))
        if num_games <= 0:
            return False

        logging.info(
            "Generating teacher bootstrap samples from %s for %s games...",
            getattr(self.args, 'auxiliary_model_label', 'legacy_teacher'),
            num_games,
        )
        bootstrap_args = self._build_teacher_bootstrap_args()
        num_workers = self._get_parallel_worker_count(num_games)
        logging.info(
            "Spawning %s teacher bootstrap workers with shared inference on %s...",
            num_workers,
            self.args.shared_inference_device,
        )
        generated_examples, game_results = execute_self_play_parallel(
            args=bootstrap_args,
            num_games=num_games,
            num_workers=num_workers,
            shared_inference_device=self.args.shared_inference_device,
            inference_batch_size=self.args.inference_batch_size,
            inference_timeout_s=self.args.inference_timeout_s,
            compatible_model_spec=opponent_spec,
            progress_desc='Teacher Bootstrap',
        )

        if not generated_examples:
            logging.warning("Teacher bootstrap generation produced no usable samples.")
            return False

        self.teacher_bootstrap_examples = generated_examples
        self.teacher_bootstrap_metadata = {
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'teacher_label': getattr(self.args, 'auxiliary_model_label', 'legacy_teacher'),
            'teacher_model_path': getattr(self.args, 'auxiliary_model_path', None),
            'games': num_games,
            'samples': len(generated_examples),
            'mcts_sims': int(bootstrap_args.num_mcts_sims),
            'temperature': float(getattr(self.args, 'teacher_bootstrap_temperature', 0.0)),
            'dirichlet_alpha': float(getattr(self.args, 'teacher_bootstrap_dirichlet_alpha', 0.0)),
            'dirichlet_epsilon': float(getattr(self.args, 'teacher_bootstrap_dirichlet_epsilon', 0.0)),
            'self_play_stats': self._summarize_self_play_lengths(game_results),
        }
        self._save_teacher_bootstrap_buffer()
        return True

    def _initialize_teacher_bootstrap_buffer(self):
        if self.auxiliary_nnet is None:
            return

        bootstrap_enabled = any([
            int(getattr(self.args, 'teacher_bootstrap_num_games', 0) or 0) > 0,
            int(getattr(self.args, 'teacher_bootstrap_warmup_iterations', 0) or 0) > 0,
            float(getattr(self.args, 'teacher_replay_initial_ratio', 0.0) or 0.0) > 0.0,
            float(getattr(self.args, 'teacher_replay_final_ratio', 0.0) or 0.0) > 0.0,
            float(getattr(self.args, 'teacher_replay_drift_ratio', 0.0) or 0.0) > 0.0,
        ])
        if not bootstrap_enabled:
            self.teacher_bootstrap_examples = []
            self.teacher_bootstrap_metadata = {}
            self.teacher_bootstrap_buffer_path = None
            return

        self.teacher_bootstrap_buffer_path = self._resolve_teacher_buffer_path()

        loaded = False
        if not bool(getattr(self.args, 'teacher_bootstrap_regenerate', False)):
            loaded = self._load_teacher_bootstrap_buffer()

        if not loaded:
            self._generate_teacher_bootstrap_buffer()

    def _get_teacher_warmup_iterations(self):
        return get_teacher_warmup_iterations(self.args)

    def _is_teacher_warmup_iteration(self, iteration):
        return is_teacher_warmup_iteration(self.args, self.teacher_bootstrap_examples, iteration)

    def _get_teacher_replay_ratio(self, iteration):
        return get_teacher_replay_ratio(
            self.args,
            self.teacher_bootstrap_examples,
            self.latest_auxiliary_win_rate,
            iteration,
            teacher_opponent_history=self.teacher_opponent_history,
        )

    def _get_teacher_opponent_history_limit(self):
        configured = int(getattr(self.args, 'teacher_opponent_history_len', 0) or 0)
        if configured > 0:
            return configured
        return max(0, int(getattr(self.args, 'history_len', 0) or 0))

    def _append_teacher_opponent_history(self, examples, source, iteration=None, metadata=None):
        examples = list(examples or [])
        if not examples:
            return False

        self.teacher_opponent_history.append(
            build_history_entry(examples, source=source, iteration=iteration, metadata=metadata)
        )
        history_before_prune = len(self.teacher_opponent_history)
        self.teacher_opponent_history = _prune_history_to_limit(
            self.teacher_opponent_history,
            self._get_teacher_opponent_history_limit(),
        )
        return len(self.teacher_opponent_history) < history_before_prune

    def _compose_training_data(self, iteration):
        return compose_training_data(
            self.args,
            self.train_examples_history,
            self.teacher_bootstrap_examples,
            self.teacher_opponent_history,
            self.latest_auxiliary_win_rate,
            iteration,
        )

    def _get_teacher_sample_budget(self):
        return estimate_teacher_sample_budget(
            self.args,
            teacher_bootstrap_examples=self.teacher_bootstrap_examples,
            teacher_bootstrap_metadata=self.teacher_bootstrap_metadata,
        )

    def _is_teacher_pipeline_enabled(self):
        auxiliary_eval_enabled = (
            self.auxiliary_nnet is not None
            and int(getattr(self.args, 'auxiliary_eval_games', 0) or 0) > 0
            and int(getattr(self.args, 'auxiliary_eval_interval', 0) or 0) > 0
        )
        teacher_data_enabled = any([
            int(getattr(self.args, 'teacher_bootstrap_num_games', 0) or 0) > 0,
            int(getattr(self.args, 'teacher_bootstrap_warmup_iterations', 0) or 0) > 0,
            float(getattr(self.args, 'teacher_replay_initial_ratio', 0.0) or 0.0) > 0.0,
            float(getattr(self.args, 'teacher_replay_final_ratio', 0.0) or 0.0) > 0.0,
            float(getattr(self.args, 'teacher_replay_drift_ratio', 0.0) or 0.0) > 0.0,
        ])
        has_teacher_data = bool(self.teacher_bootstrap_examples) or bool(self.teacher_opponent_history)
        return bool(auxiliary_eval_enabled or teacher_data_enabled or has_teacher_data)

    def _evaluate_against_auxiliary_model(self, iteration):
        if self.auxiliary_nnet is None or self.auxiliary_model_config is None:
            return None
        eval_interval = max(1, int(getattr(self.args, 'auxiliary_eval_interval', 1)))
        if int(iteration) % eval_interval != 0:
            return None

        num_games = max(2, int(getattr(self.args, 'auxiliary_eval_games', 0)))
        if num_games <= 0:
            return None

        logging.info(
            "--- Evaluating model vs auxiliary baseline %s at iteration %s ---",
            getattr(self.args, 'auxiliary_model_label', 'legacy_teacher'),
            iteration,
        )

        opponent_spec = self._get_auxiliary_parallel_spec()
        if opponent_spec is None:
            return None

        wins, losses, draws = self.execute_evaluation_parallel(
            num_games,
            opponent_model_spec=opponent_spec,
        )

        win_rate = (wins + 0.5 * draws) / num_games
        teacher_result = {
            'iteration': iteration,
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'games': num_games,
            'win_rate': win_rate,
            'opponent_label': getattr(self.args, 'auxiliary_model_label', 'legacy_teacher'),
        }
        self.latest_auxiliary_win_rate = win_rate
        teacher_result['teacher_replay_ratio'] = self._get_teacher_replay_ratio(iteration + 1)
        self.auxiliary_eval_history.append(teacher_result)
        logging.info(
            "Auxiliary Baseline Result - Wins: %s, Losses: %s, Draws: %s | WinRate: %.3f | NextTeacherReplay=%.3f",
            wins,
            losses,
            draws,
            win_rate,
            teacher_result['teacher_replay_ratio'],
        )
        return teacher_result

    def _resolve_mcts_stage_index(self, sims):
        candidates = [int(value) for value in getattr(self.args, 'mcts_sim_candidates', [int(sims)])]
        if not candidates:
            candidates = [int(sims)]
        if int(sims) in candidates:
            return candidates.index(int(sims))
        candidates.append(int(sims))
        candidates = sorted(set(candidates))
        self.args.mcts_sim_candidates = candidates
        return candidates.index(int(sims))

    def _get_current_mcts_sims(self):
        candidates = [int(value) for value in getattr(self.args, 'mcts_sim_candidates', [int(self.args.num_mcts_sims)])]
        if not candidates:
            return int(self.args.num_mcts_sims)
        stage_index = min(max(0, int(self.current_mcts_stage_index)), len(candidates) - 1)
        return int(candidates[stage_index])

    def _sync_scheduler_after_resume(self, scheduler_state=None, completed_iterations=0):
        base_lr = float(self.args.learning_rate)
        decay_steps = 0
        step_size = max(1, int(getattr(self.args, 'lr_decay_step_size', 50)))
        gamma = float(getattr(self.args, 'lr_decay_gamma', 0.8))

        # Always enforce scheduler hyperparameters from current runtime args.
        # This avoids inheriting stale step_size/gamma from older checkpoints.
        if scheduler_state is not None:
            saved_step_size = scheduler_state.get('step_size', step_size)
            saved_gamma = scheduler_state.get('gamma', gamma)
            if int(saved_step_size) != int(step_size) or float(saved_gamma) != float(gamma):
                logging.warning(
                    "Scheduler config mismatch on resume: checkpoint(step_size=%s, gamma=%s) "
                    "!= current(step_size=%s, gamma=%s). Using current config.",
                    saved_step_size,
                    saved_gamma,
                    step_size,
                    gamma,
                )

        # Rebuild scheduler to guarantee step_size/gamma exactly match current args.
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=step_size,
            gamma=gamma,
        )
        
        # Correct calculation: step should be max(0, completed_iterations - 1)
        # to match StepLR's behavior where first step occurs AFTER first iteration.
        effective_iterations = max(0, int(completed_iterations) - 1)
        if step_size > 0:
            decay_steps = effective_iterations // step_size

        decay_factor = gamma ** decay_steps
        group_scales = self._get_optimizer_group_scales()
        min_learning_rate = float(getattr(self.args, 'min_learning_rate', 0.0) or 0.0)
        current_lrs = []
        base_lrs = []
        for idx, param_group in enumerate(self.optimizer.param_groups):
            group_base_lr = base_lr * group_scales[idx]
            group_current_lr = group_base_lr * decay_factor
            if min_learning_rate > 0:
                group_current_lr = max(group_current_lr, min_learning_rate * group_scales[idx])
            base_lrs.append(group_base_lr)
            current_lrs.append(group_current_lr)
            param_group['lr'] = group_current_lr
            param_group['lr_scale'] = group_scales[idx]
        
        # Ensure scheduler internal state is also consistent
        self.scheduler.last_epoch = max(-1, int(completed_iterations) - 1)
        self.scheduler.base_lrs = base_lrs
        self.scheduler._last_lr = current_lrs
        logging.info(
            "Scheduler aligned after resume: completed_iterations=%s, step_size=%s, gamma=%.6f, decay_steps=%s, lrs=%s",
            int(completed_iterations),
            step_size,
            gamma,
            decay_steps,
            [f"{lr:.9f}" for lr in current_lrs],
        )

    def _get_active_eval_interval(self):
        if self.boosted_eval_rounds_remaining > 0 and self.best_nnet is not None:
            return max(1, int(getattr(self.args, 'eval_interval_after_best', self.args.eval_interval)))
        return max(1, int(self.args.eval_interval))

    def _get_best_file_paths(self):
        recent_path = os.path.join(
            self.args.checkpoint_dir,
            getattr(self.args, 'best_recent_filename', 'best_new.pth.tar'),
        )
        older_path = os.path.join(
            self.args.checkpoint_dir,
            getattr(self.args, 'best_older_filename', 'best_old.pth.tar'),
        )
        legacy_path = os.path.join(
            self.args.checkpoint_dir,
            getattr(self.args, 'best_legacy_filename', 'best.pth.tar'),
        )
        return recent_path, older_path, legacy_path

    def _load_checkpoint_payload_safely(self, path):
        try:
            with torch.serialization.safe_globals([np._core.multiarray._reconstruct]):
                return torch.load(path, map_location='cpu')
        except Exception:
            try:
                with torch.serialization.safe_globals([np.core.multiarray._reconstruct]):
                    return torch.load(path, map_location='cpu')
            except Exception:
                return torch.load(path, map_location='cpu', weights_only=False)

    def _build_model_from_state_dict(self, state_dict):
        model = Connect4Net(num_channels=self.args.num_channels).to(self.args.train_device)
        model.load_state_dict(state_dict)
        return model

    def _load_best_model_file(self, path):
        if not os.path.exists(path):
            return None, {}
        payload = self._load_checkpoint_payload_safely(path)
        state_dict = payload.get('state_dict') if isinstance(payload, dict) else payload
        if state_dict is None:
            return None, {}
        model = self._build_model_from_state_dict(state_dict)
        metadata = payload if isinstance(payload, dict) else {}
        return model, metadata

    def _load_best_generations_from_disk(self):
        recent_path, older_path, legacy_path = self._get_best_file_paths()

        self.best_nnet = None
        self.older_best_nnet = None

        recent_model, recent_meta = self._load_best_model_file(recent_path)
        if recent_model is None and os.path.exists(legacy_path):
            logging.info(
                "No recent best file found (%s). Falling back to legacy best file (%s).",
                recent_path,
                legacy_path,
            )
            recent_model, recent_meta = self._load_best_model_file(legacy_path)

        if recent_model is not None:
            self.best_nnet = recent_model
            self.best_model_iteration = recent_meta.get('best_model_iteration', self.best_model_iteration)
            self.best_model_label = recent_meta.get('best_model_label', self.best_model_label)
            self.best_win_rate = recent_meta.get('best_win_rate', self.best_win_rate)

        older_model, older_meta = self._load_best_model_file(older_path)
        if older_model is not None:
            self.older_best_nnet = older_model
            self.older_best_model_iteration = older_meta.get(
                'older_best_model_iteration',
                older_meta.get('best_model_iteration', self.older_best_model_iteration),
            )
            self.older_best_model_label = older_meta.get(
                'older_best_model_label',
                older_meta.get('best_model_label', self.older_best_model_label),
            )
            self.older_best_win_rate = older_meta.get(
                'older_best_win_rate',
                older_meta.get('best_win_rate', self.older_best_win_rate),
            )

        if self.best_nnet is not None:
            logging.info(
                "Loaded best generations for evaluation: recent=%s, older=%s",
                self.best_model_label,
                self.older_best_model_label if self.older_best_nnet is not None else 'none',
            )
        else:
            logging.info("No best model found on disk. Evaluation will use random baseline until first promotion.")

    def _should_run_evaluation(self, iteration):
        return int(iteration) >= int(self.next_eval_iteration)

    def _schedule_next_evaluation(self, iteration):
        next_interval = self._get_active_eval_interval()
        if self.boosted_eval_rounds_remaining > 0:
            self.boosted_eval_rounds_remaining -= 1
        self.next_eval_iteration = int(iteration) + next_interval

    def _maybe_promote_mcts_sims(self, eval_metrics):
        if not eval_metrics or not eval_metrics.get('improved'):
            return None

        random_baseline = eval_metrics.get('random_baseline')
        stability_threshold = float(getattr(self.args, 'random_baseline_stability_threshold', 0.0))
        if random_baseline is not None and random_baseline.get('win_rate', 1.0) < stability_threshold:
            self.mcts_promotion_progress_count = 0
            return (
                "Adaptive MCTS: 随机基线胜率 "
                f"{random_baseline['win_rate']:.3f} 低于稳定阈值 {stability_threshold:.3f}，暂停继续升档"
            )

        self.mcts_promotion_progress_count += 1
        required_improves = max(1, int(getattr(self.args, 'mcts_promotion_improve_count', 2)))
        candidates = [int(value) for value in getattr(self.args, 'mcts_sim_candidates', [int(self.args.num_mcts_sims)])]
        if self.mcts_promotion_progress_count < required_improves:
            return None
        if self.current_mcts_stage_index >= len(candidates) - 1:
            return None

        previous_sims = self._get_current_mcts_sims()
        self.current_mcts_stage_index += 1
        promoted_sims = self._get_current_mcts_sims()
        self.args.num_mcts_sims = promoted_sims
        self.mcts_promotion_progress_count = 0
        self.boosted_eval_rounds_remaining = max(
            self.boosted_eval_rounds_remaining,
            max(1, int(getattr(self.args, 'eval_boost_rounds_after_improve', 2))),
        )
        self.next_eval_iteration = min(self.next_eval_iteration, int(eval_metrics['iteration']) + self._get_active_eval_interval())
        return (
            f"Adaptive MCTS: 新 best 已累计出现 {required_improves} 次，搜索量从 {previous_sims} 提升到 {promoted_sims}"
        )

    def _log_iteration_summary(self, summary):
        train_metrics = summary['train_metrics']
        self_play_stats = summary.get('self_play_stats') or {}
        train_line = (
            f"  Train: total_samples={summary['train_samples']} | epochs={self.args.epochs} | "
            f"duration={self._format_duration(train_metrics['duration_sec'])} | "
            f"source={summary.get('train_data_mode', 'unknown')} | "
            f"self_play_samples={summary.get('self_play_train_samples', 0)} | "
            f"loss(total={train_metrics['total_loss']:.6f}, value={train_metrics['value_loss']:.6f}, policy={train_metrics['policy_loss']:.6f})"
        )
        show_teacher_train_fields = self.teacher_logging_enabled or summary.get('teacher_replay_samples', 0) > 0
        if show_teacher_train_fields:
            train_line = (
                train_line + " | "
                + f"teacher_replay_samples={summary.get('teacher_replay_samples', 0)} | "
                + f"teacher_bootstrap={summary.get('teacher_bootstrap_replay_samples', 0)} | "
                + f"teacher_opponent={summary.get('teacher_opponent_replay_samples', 0)} | "
                + f"teacher_replay_ratio={summary.get('teacher_replay_ratio', 0.0):.3f}"
            )

        lines = [
            (
                f"Iter {summary['iteration']}/{self.args.num_iterations} | start={summary['start_time']} | "
                f"duration={self._format_duration(summary['iteration_duration_sec'])} | "
                f"mcts_sims={summary['mcts_sims']} | lr={summary['learning_rate']:.6f}"
            ),
            (
                f"  Self-Play: games={summary['self_play_games']} | new_samples={summary['new_samples']} | "
                f"avg_steps={self_play_stats.get('mean_steps', 0.0):.2f} | var_steps={self_play_stats.get('variance_steps', 0.0):.2f} | "
                f"policy_entropy={self_play_stats.get('mean_policy_entropy', 0.0):.4f} | "
                f"min/max={self_play_stats.get('min_steps', 0)}/{self_play_stats.get('max_steps', 0)} | "
                f"long_games={self_play_stats.get('long_games', 0)} | short_games={self_play_stats.get('short_games', 0)} | "
                f"filtered={self_play_stats.get('filtered_games', 0)} | "
                f"duration={self._format_duration(summary['self_play_duration_sec'])}"
            ),
            train_line,
        ]

        if self.teacher_logging_enabled and summary.get('teacher_opponent_history_samples', 0) > 0:
            lines.append(
                f"  Teacher-History: replay_pool_samples={summary.get('teacher_opponent_history_samples', 0)}"
            )

        eval_metrics = summary.get('eval_metrics')
        if eval_metrics:
            lines.append(
                (
                    f"  Eval: opponent={eval_metrics['opponent_label']} | games={eval_metrics['games']} | "
                    f"W/L/D={eval_metrics['wins']}/{eval_metrics['losses']}/{eval_metrics['draws']} | "
                    f"win_rate={eval_metrics['win_rate']:.3f} | improved={'yes' if eval_metrics['improved'] else 'no'} | "
                    f"no_improve_streak={eval_metrics['no_improve_streak']}"
                )
            )
            opponent_results = eval_metrics.get('opponent_results') or []
            if opponent_results:
                required_count = int(eval_metrics.get('required_opponent_count', len(opponent_results)))
                required_threshold = float(eval_metrics.get('required_threshold', 0.0))
                compact = []
                for idx, item in enumerate(opponent_results):
                    marker = 'required' if idx < required_count else 'optional'
                    compact.append(
                        f"{item.get('label', 'unknown')}[{marker}]={item.get('win_rate', 0.0):.3f}"
                    )
                lines.append(
                    "  Eval-Best-Generations: "
                    + " | ".join(compact)
                    + f" | threshold={required_threshold:.3f} | decision={'pass' if eval_metrics.get('improved') else 'fail'}"
                )

            auxiliary_teacher = eval_metrics.get('auxiliary_teacher')
            if auxiliary_teacher is not None:
                lines.append(
                    (
                        f"  Eval-Teacher: opponent={auxiliary_teacher['opponent_label']} | games={auxiliary_teacher['games']} | "
                        f"W/L/D={auxiliary_teacher['wins']}/{auxiliary_teacher['losses']}/{auxiliary_teacher['draws']} | "
                        f"win_rate={auxiliary_teacher['win_rate']:.3f} | next_teacher_replay={auxiliary_teacher.get('teacher_replay_ratio', 0.0):.3f}"
                    )
                )
            random_eval_metrics = eval_metrics.get('random_baseline')
            if random_eval_metrics is not None:
                lines.append(
                    (
                        f"  Eval-Random: games={random_eval_metrics['games']} | "
                        f"W/L/D={random_eval_metrics['wins']}/{random_eval_metrics['losses']}/{random_eval_metrics['draws']} | "
                        f"win_rate={random_eval_metrics['win_rate']:.3f}"
                    )
                )

        if summary.get('stop_reason'):
            lines.append(f"  Early Stop: {summary['stop_reason']}")

        for note in summary.get('adaptive_notes', []):
            lines.append(f"  Adaptive: {note}")

        for line in lines:
            logging.info(line)
        self._append_info_log(lines)

    def _summarize_self_play_lengths(self, game_results):
        if not game_results:
            return {
                'mean_steps': 0.0,
                'variance_steps': 0.0,
                'mean_policy_entropy': 0.0,
                'variance_policy_entropy': 0.0,
                'min_policy_entropy': 0.0,
                'max_policy_entropy': 0.0,
                'min_steps': 0,
                'max_steps': 0,
                'long_games': 0,
                'short_games': 0,
                'filtered_games': 0,
                'used_games': 0,
            }

        lengths = np.asarray([item.get('steps', 0) for item in game_results], dtype=np.float64)
        entropies = np.asarray([item.get('policy_entropy_mean', 0.0) for item in game_results], dtype=np.float64)
        filtered_games = sum(0 if item.get('used_for_training', True) else 1 for item in game_results)
        long_games = sum(
            1
            for item in game_results
            if bool(item.get('used_for_training', True))
            and float(item.get('long_game_weight', 1.0) or 1.0) > 1.0
        )
        short_games = sum(1 for item in game_results if int(item.get('steps', 0) or 0) < 10)
        return {
            'mean_steps': float(np.mean(lengths)),
            'variance_steps': float(np.var(lengths)),
            'mean_policy_entropy': float(np.mean(entropies)),
            'variance_policy_entropy': float(np.var(entropies)),
            'min_policy_entropy': float(np.min(entropies)),
            'max_policy_entropy': float(np.max(entropies)),
            'min_steps': int(np.min(lengths)),
            'max_steps': int(np.max(lengths)),
            'long_games': int(long_games),
            'short_games': int(short_games),
            'filtered_games': int(filtered_games),
            'used_games': int(len(game_results) - filtered_games),
        }

    def _check_step_stagnation(self, current_summary):
        window = max(2, int(getattr(self.args, 'step_stagnation_window', 0)))
        history = [
            item for item in (self.iteration_metrics_history + [current_summary])
            if item.get('self_play_stats') is not None
        ]
        if len(history) < window:
            return None

        recent = history[-window:]
        mean_steps = [float(item['self_play_stats'].get('mean_steps', 0.0)) for item in recent]
        variance_steps = [float(item['self_play_stats'].get('variance_steps', 0.0)) for item in recent]
        mean_tol = max(0.0, float(getattr(self.args, 'step_stagnation_mean_tolerance', 0.0)))
        variance_tol = max(0.0, float(getattr(self.args, 'step_stagnation_variance_tolerance', 0.0)))

        if (max(mean_steps) - min(mean_steps)) <= mean_tol and max(variance_steps) <= variance_tol:
            return (
                f"最近 {window} 次迭代的平均步数几乎不变（约 {mean_steps[-1]:.2f}），且方差持续很低，训练可能进入停滞"
            )
        return None

    def _check_loss_early_stop(self, iteration, total_loss):
        if not np.isfinite(total_loss):
            self.iteration_loss_history.clear()
            return None

        self.iteration_loss_history.append((iteration, float(total_loss)))
        if len(self.iteration_loss_history) < self.args.loss_increase_patience:
            return None

        losses = [entry[1] for entry in self.iteration_loss_history]
        if all(losses[idx] < losses[idx + 1] for idx in range(len(losses) - 1)):
            iter_range = [entry[0] for entry in self.iteration_loss_history]
            return (
                f"总 loss 在连续 {self.args.loss_increase_patience} 次迭代中严格递增："
                f"迭代 {iter_range[0]} -> {iter_range[-1]} 为 {losses[0]:.6f} -> {losses[-1]:.6f}"
            )
        return None

    def _check_eval_early_stop(self, eval_metrics):
        if not eval_metrics:
            return None
        if eval_metrics['improved']:
            self.consecutive_no_improve_evals = 0
            eval_metrics['no_improve_streak'] = 0
            return None

        self.consecutive_no_improve_evals += 1
        eval_metrics['no_improve_streak'] = self.consecutive_no_improve_evals
        if self.consecutive_no_improve_evals >= self.args.no_improve_eval_patience:
            return (
                f"连续 {self.consecutive_no_improve_evals} 次评估未能战胜历史 best 模型，"
                f"最近一次对手为 {eval_metrics['opponent_label']}"
            )
        return None

    def validate_model(self):
        """
        Performs a dummy forward pass to ensure model and weights are compatible.
        """
        try:
            self.nnet.eval()
            device = next(self.nnet.parameters()).device
            dummy_input = torch.randn(
                1,
                NUM_INPUT_CHANNELS,
                self.nnet.board_layers,
                self.nnet.board_size,
                self.nnet.board_size,
            ).to(device)
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
            
            # Case 1: Full training state (dictionary with iteration metadata)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint and 'iteration' in checkpoint:
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
                restored_history_count = len(self.train_examples_history)
                self.train_examples_history = _prune_history_to_limit(
                    self.train_examples_history,
                    self.args.history_len,
                )
                self.teacher_opponent_history = checkpoint.get('teacher_opponent_history', [])
                restored_teacher_history_count = len(self.teacher_opponent_history)
                self.teacher_opponent_history = _prune_history_to_limit(
                    self.teacher_opponent_history,
                    self._get_teacher_opponent_history_limit(),
                )
                self.eval_history = checkpoint.get('eval_history', [])
                self.iteration_metrics_history = checkpoint.get('iteration_metrics_history', [])
                self.best_win_rate = checkpoint.get('best_win_rate', -1.0)
                self.best_model_iteration = checkpoint.get('best_model_iteration', self.best_model_iteration)
                self.best_model_label = checkpoint.get('best_model_label', self.best_model_label)
                self.older_best_win_rate = checkpoint.get('older_best_win_rate', self.older_best_win_rate)
                self.older_best_model_iteration = checkpoint.get('older_best_model_iteration', self.older_best_model_iteration)
                self.older_best_model_label = checkpoint.get('older_best_model_label', self.older_best_model_label)
                self.consecutive_no_improve_evals = checkpoint.get('consecutive_no_improve_evals', 0)
                recent_losses = checkpoint.get('iteration_loss_history', [])
                self.iteration_loss_history = deque(recent_losses, maxlen=self.args.loss_increase_patience)
                self.stop_reason = checkpoint.get('stop_reason')
                self.auxiliary_eval_history = checkpoint.get('auxiliary_eval_history', [])
                self.teacher_bootstrap_metadata = checkpoint.get('teacher_bootstrap_metadata', self.teacher_bootstrap_metadata)
                self.latest_auxiliary_win_rate = checkpoint.get('latest_auxiliary_win_rate', self.latest_auxiliary_win_rate)
                saved_buffer_path = checkpoint.get('teacher_bootstrap_buffer_path')
                if saved_buffer_path and not getattr(self.args, 'teacher_bootstrap_buffer_path', None):
                    self.args.teacher_bootstrap_buffer_path = saved_buffer_path
                self.current_mcts_stage_index = checkpoint.get(
                    'current_mcts_stage_index',
                    self._resolve_mcts_stage_index(checkpoint.get('args', {}).get('num_mcts_sims', self.args.num_mcts_sims)),
                )
                self.mcts_promotion_progress_count = checkpoint.get(
                    'mcts_promotion_progress_count',
                    checkpoint.get('best_stability_eval_count', 0),
                )
                self.boosted_eval_rounds_remaining = checkpoint.get('boosted_eval_rounds_remaining', 0)
                self.next_eval_iteration = checkpoint.get('next_eval_iteration', self.start_iter + self._get_active_eval_interval() - 1)
                if self.best_win_rate < 0 and self.best_model_iteration == 0:
                    self.best_model_label = 'random_model'
                self._sync_scheduler_after_resume(
                    checkpoint.get('scheduler'),
                    completed_iterations=max(0, self.start_iter - 1),
                )
                
                # Optional: Restore/Compare parameters
                saved_args = checkpoint.get('args', {})
                if saved_args:
                    # Check for significant mismatches if needed, but here we just log
                    logging.info(f"Loaded hyperparameters from checkpoint (e.g. lr={saved_args.get('learning_rate')})")

                logging.info(f"Resuming from iteration {self.start_iter}. Previous best win rate: {self.best_win_rate:.3f}")
                if len(self.train_examples_history) > 0:
                    logging.info(f"Restored {len(self.train_examples_history)} iterations of training data.")
                if restored_history_count > len(self.train_examples_history):
                    logging.info(
                        f"Pruned restored history from {restored_history_count} to {len(self.train_examples_history)} "
                        f"to respect history_len={self.args.history_len}."
                    )
                if len(self.teacher_opponent_history) > 0:
                    teacher_history_samples, teacher_history_sources = flatten_history_examples(self.teacher_opponent_history)
                    logging.info(
                        "Restored teacher-opponent history: entries=%s samples=%s sources=%s",
                        len(self.teacher_opponent_history),
                        len(teacher_history_samples),
                        teacher_history_sources,
                    )
                if restored_teacher_history_count > len(self.teacher_opponent_history):
                    logging.info(
                        "Pruned restored teacher-opponent history from %s to %s to respect teacher_opponent_history_len=%s.",
                        restored_teacher_history_count,
                        len(self.teacher_opponent_history),
                        self._get_teacher_opponent_history_limit(),
                    )
            
            # Case 2: Weights-only checkpoint (e.g. best.pth.tar) or plain state_dict
            else:
                state_dict = checkpoint.get('state_dict') if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
                self.nnet.load_state_dict(state_dict)
                if self.validate_model():
                    logging.info(
                        "Loaded weights-only checkpoint without training state. Starting from iteration 1. "
                        "Use latest.pth.tar or checkpoint_*/checkpoint.pth.tar for true resume."
                    )
                else:
                    raise ValueError("Model weights loaded but validation failed.")
                
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}")
            logging.info("Critical error during resume. Ensure your model architecture has not changed.")
            sys.exit(1) # Exit to allow user to investigate architecture issues

    def execute_episode_parallel(self):
        num_workers = self._get_parallel_worker_count(self.args.num_self_play_games)
        logging.info(
            "Preparing %s self-play workers with shared inference on %s...",
            num_workers,
            self.args.shared_inference_device,
        )
        return execute_self_play_parallel(
            args=self.args,
            num_games=self.args.num_self_play_games,
            num_workers=num_workers,
            shared_inference_device=self.args.shared_inference_device,
            inference_batch_size=self.args.inference_batch_size,
            inference_timeout_s=self.args.inference_timeout_s,
            model_state={k: v.detach().cpu() for k, v in self.nnet.state_dict().items()},
            model_config=extract_model_config(self.nnet),
            progress_desc='Self-Play',
        )

    def _get_parallel_worker_count(self, total_games):
        cpu_count = multiprocessing.cpu_count()
        explicit_workers = int(getattr(self.args, 'self_play_workers', 0) or 0)
        if explicit_workers > 0:
            target_workers = explicit_workers
        else:
            target_workers = int(cpu_count * self.args.self_play_cpu_ratio)

        return max(1, min(self.args.max_self_play_workers, target_workers, total_games))

    def execute_evaluation_parallel(self, num_games, opponent_nnet=None, opponent_model_spec=None):
        num_workers = self._get_parallel_worker_count(num_games)
        opponent_state = None
        opponent_config = None
        if opponent_nnet is not None:
            opponent_state = {k: v.detach().cpu() for k, v in opponent_nnet.state_dict().items()}
            opponent_config = extract_model_config(opponent_nnet)
        return execute_evaluation_parallel(
            args=self.args,
            num_games=num_games,
            num_workers=num_workers,
            shared_inference_device=self.args.shared_inference_device,
            inference_batch_size=self.args.inference_batch_size,
            inference_timeout_s=self.args.inference_timeout_s,
            new_model_state={k: v.detach().cpu() for k, v in self.nnet.state_dict().items()},
            new_model_config=extract_model_config(self.nnet),
            opponent_nnet_state=opponent_state,
            opponent_nnet_config=opponent_config,
            opponent_model_spec=opponent_model_spec,
        )

    def train(self):
        last_completed_iteration = self.start_iter - 1

        for i in range(self.start_iter, self.args.num_iterations + 1):
            iteration_start_wall = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            iteration_start = time.perf_counter()
            current_mcts_sims = self._get_current_mcts_sims()
            self.args.num_mcts_sims = current_mcts_sims
            self.args.current_iteration = i

            logging.info(f'Starting Iteration {i}/{self.args.num_iterations} | mcts_sims={current_mcts_sims}')

            if str(self.args.train_device).startswith('cuda'):
                torch.cuda.empty_cache()
            
            # 1. Self-Play / Teacher Bootstrap Warmup
            if self._is_teacher_warmup_iteration(i):
                logging.info(
                    "Iteration %s uses teacher bootstrap buffer only (warmup %s/%s).",
                    i,
                    i,
                    self._get_teacher_warmup_iterations(),
                )
                iter_examples = []
                self_play_game_results = []
                self.last_self_play_game_results = []
                self_play_duration = 0.0
                self_play_stats = self._summarize_self_play_lengths(self_play_game_results)
            else:
                self_play_start = time.perf_counter()
                iter_examples, self_play_game_results = self.execute_episode_parallel()
                self_play_duration = time.perf_counter() - self_play_start
                self_play_stats = self._summarize_self_play_lengths(self_play_game_results)
                self.last_self_play_game_results = list(self_play_game_results)
                self.train_examples_history.append(iter_examples)

                history_before_prune = len(self.train_examples_history)
                self.train_examples_history = _prune_history_to_limit(
                    self.train_examples_history,
                    self.args.history_len,
                )
                if len(self.train_examples_history) < history_before_prune:
                    logging.info(f"Removing oldest history (keep last {self.args.history_len})")

            train_data, train_data_source = self._compose_training_data(i)
            
            # 2. Train Neural Net
            stage_transition_note = self._update_option_a_stage(i)
            train_metrics = self.train_network(train_data, iteration=i)
            
            # Free memory from train_data
            train_sample_count = len(train_data)
            del train_data
            gc.collect()
            
            self.scheduler.step()
            self._apply_min_learning_rate()
            
            # 3. Save Checkpoint & Evaluate
            eval_metrics = None
            checkpoint_saved = False
            adaptive_notes = []
            if i % self.args.checkpoint_interval == 0:
                self.save_checkpoint(i, self.last_self_play_game_results)
                checkpoint_saved = True
            if self._should_run_evaluation(i):
                eval_metrics = self.evaluate_model(i)
                adaptive_note = self._maybe_promote_mcts_sims(eval_metrics)
                if adaptive_note is not None:
                    adaptive_notes.append(adaptive_note)
                self._schedule_next_evaluation(i)

            iteration_duration = time.perf_counter() - iteration_start
            last_completed_iteration = i

            eval_stop_reason = self._check_eval_early_stop(eval_metrics)
            stop_reason = self._check_loss_early_stop(i, train_metrics['total_loss'])
            if stop_reason is None:
                stop_reason = eval_stop_reason

            iteration_summary = {
                'iteration': i,
                'start_time': iteration_start_wall,
                'iteration_duration_sec': iteration_duration,
                'self_play_duration_sec': self_play_duration,
                'self_play_games': 0 if self._is_teacher_warmup_iteration(i) else self.args.num_self_play_games,
                'self_play_stats': self_play_stats,
                'new_samples': len(iter_examples),
                'train_samples': train_sample_count,
                'train_data_mode': train_data_source['mode'],
                'self_play_train_samples': train_data_source['self_play_samples'],
                'teacher_replay_samples': train_data_source['teacher_replay_samples'],
                'teacher_replay_ratio': train_data_source['teacher_replay_ratio'],
                'teacher_bootstrap_replay_samples': train_data_source.get('teacher_bootstrap_replay_samples', 0),
                'teacher_opponent_replay_samples': train_data_source.get('teacher_opponent_replay_samples', 0),
                'teacher_opponent_history_samples': train_data_source.get('teacher_opponent_history_samples', 0),
                'train_metrics': train_metrics,
                'eval_metrics': eval_metrics,
                'mcts_sims': current_mcts_sims,
                'learning_rate': self._get_learning_rate(),
                'stop_reason': stop_reason,
                'adaptive_notes': adaptive_notes,
            }
            if stage_transition_note is not None:
                iteration_summary['adaptive_notes'].append(stage_transition_note)
            option_a_note = self._get_option_a_progress_note(i)
            if option_a_note is not None:
                iteration_summary['adaptive_notes'].append(option_a_note)
            stagnation_note = self._check_step_stagnation(iteration_summary)
            if stagnation_note is not None:
                iteration_summary['adaptive_notes'].append(stagnation_note)
            self.iteration_metrics_history.append(iteration_summary)
            self.stop_reason = stop_reason
            self._log_iteration_summary(iteration_summary)

            if stop_reason:
                logging.info(f"Early stopping triggered at iteration {i}: {stop_reason}")
                if not checkpoint_saved:
                    self.save_checkpoint(i, self.last_self_play_game_results)
                break

        # 训练循环结束后：确保最终模型被保存一次
        logging.info("Saving final checkpoint...")
        if last_completed_iteration >= self.start_iter:
            self.save_checkpoint(last_completed_iteration, self.last_self_play_game_results)
        self.final_report(last_completed_iteration)

    def train_network(self, examples, iteration):
        """
        Train the network on examples using DataLoader for proper epoch-based sampling.
        examples: list of (board, policy, value)
        """
        self.nnet.train()
        device = next(self.nnet.parameters()).device
        use_cuda = (device.type == 'cuda')
        
        # 混合精度训练初始化
        scaler = torch.amp.GradScaler('cuda', enabled=True) if use_cuda else None
        
        dataset_build_start = time.perf_counter()
        dataset = Connect4Dataset(examples)
        dataset_build_duration = time.perf_counter() - dataset_build_start

        mp_start_method = multiprocessing.get_start_method(allow_none=True)
        requested_loader_workers = int(getattr(self.args, 'data_loader_workers', 0) or 0)
        effective_loader_workers = requested_loader_workers
        if requested_loader_workers > 0 and mp_start_method == 'spawn':
            effective_loader_workers = 0
            logging.info(
                "DataLoader workers reduced from %s to 0 because multiprocessing start method is 'spawn'; "
                "this avoids duplicating the full training dataset into each worker before epoch 1.",
                requested_loader_workers,
            )

        logging.info(
            "Packed training dataset in %.2fs | samples=%s | loader_workers=%s",
            dataset_build_duration,
            len(dataset),
            effective_loader_workers,
        )
        
        # 自适应 DataLoader：CUDA 时并行加载并启用 pinned memory。
        loader_kwargs = {
            'batch_size': self.args.batch_size,
            'shuffle': True,
            'num_workers': effective_loader_workers,
            'pin_memory': use_cuda,
        }
        if effective_loader_workers > 0:
            loader_kwargs['persistent_workers'] = True
            loader_kwargs['prefetch_factor'] = 2

        loader = torch.utils.data.DataLoader(
            dataset, 
            **loader_kwargs
        )

        metrics = {
            'samples': len(dataset),
            'duration_sec': 0.0,
            'total_loss': 0.0,
            'value_loss': 0.0,
            'policy_loss': 0.0,
            'epoch_metrics': [],
        }
        if len(dataset) == 0:
            logging.warning("No training examples available for this iteration. Skipping optimizer step.")
            return metrics

        train_start = time.perf_counter()
        total_batches = 0
        total_loss_sum = 0.0
        total_value_loss_sum = 0.0
        total_policy_loss_sum = 0.0

        for epoch in range(self.args.epochs):
            total_loss = 0.0
            total_value_loss = 0.0
            total_policy_loss = 0.0
            batch_count = 0
            pbar = tqdm(loader, desc=f"Training Epoch {epoch + 1}/{self.args.epochs}")
            for batch_idx, (_, boards, target_pis, target_vs) in enumerate(pbar):
                boards = boards.to(device, non_blocking=True)
                target_pis = target_pis.to(device, non_blocking=True)
                target_vs = target_vs.to(device, non_blocking=True)

                self.optimizer.zero_grad(set_to_none=True)
                
                # 使用 autocast 进行自动混合精度
                with torch.amp.autocast(device_type='cuda', enabled=use_cuda):
                    out_pi, out_v = self.nnet(boards)

                    # Value loss: mean squared error
                    l_v = F.mse_loss(out_v.view(-1), target_vs.view(-1))
                    # Policy loss: cross-entropy (out_pi is log_softmax output)
                    l_pi = -(target_pis * out_pi).sum(dim=1).mean()

                    # 添加正则化项 (L2 weight decay 已经在 optimizer 中，但显式加权可以微调)
                    value_weight = float(getattr(self.args, 'value_loss_weight', 1.0))
                    policy_weight = float(getattr(self.args, 'policy_loss_weight', 1.0))
                    total_l = value_weight * l_v + policy_weight * l_pi

                if scaler is not None:
                    # 缩放梯度以防止下溢
                    scaler.scale(total_l).backward()

                    # 在更新之前先解缩放以应用裁剪
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), max_norm=5.0)

                    # 尝试步进并更新缩放器
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    total_l.backward()
                    torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), max_norm=5.0)
                    self.optimizer.step()

                total_loss += total_l.item()
                total_value_loss += l_v.item()
                total_policy_loss += l_pi.item()
                batch_count += 1
                total_batches += 1

                total_loss_sum += total_l.item()
                total_value_loss_sum += l_v.item()
                total_policy_loss_sum += l_pi.item()

                pbar.set_postfix(
                    total=f"{total_loss / batch_count:.4f}",
                    value=f"{total_value_loss / batch_count:.4f}",
                    policy=f"{total_policy_loss / batch_count:.4f}"
                )

            epoch_metrics = {
                'epoch': epoch + 1,
                'total_loss': total_loss / max(1, batch_count),
                'value_loss': total_value_loss / max(1, batch_count),
                'policy_loss': total_policy_loss / max(1, batch_count),
                'batches': batch_count,
            }
            metrics['epoch_metrics'].append(epoch_metrics)

        metrics['duration_sec'] = time.perf_counter() - train_start
        metrics['total_loss'] = total_loss_sum / max(1, total_batches)
        metrics['value_loss'] = total_value_loss_sum / max(1, total_batches)
        metrics['policy_loss'] = total_policy_loss_sum / max(1, total_batches)
        return metrics

    def save_checkpoint(self, iteration, self_play_game_results=None):
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
        self.train_examples_history = _prune_history_to_limit(
            self.train_examples_history,
            self.args.history_len,
        )
        self.teacher_opponent_history = _prune_history_to_limit(
            self.teacher_opponent_history,
            self._get_teacher_opponent_history_limit(),
        )

        state = self._build_training_state(iteration)

        # Save full state
        torch.save(state, filepath)
        # Also save as latest for easy resume
        torch.save(state, latest_filepath)
        
        # Legacy/Simple model save (weights only)
        torch.save(self.nnet.state_dict(), model_filepath)
        self._save_self_play_sample_json(iteration, folder, self_play_game_results)
        
        logging.info(f"Checkpoint saved: {filepath}")

    def _select_equally_spaced_indices(self, start_idx, end_idx, target_count):
        if end_idx < start_idx:
            return []
        total = end_idx - start_idx + 1
        if total <= target_count:
            return list(range(start_idx, end_idx + 1))

        linspace_values = np.linspace(start_idx, end_idx, num=target_count)
        rounded = [int(round(v)) for v in linspace_values]
        deduped = []
        for idx in rounded:
            if not deduped or idx != deduped[-1]:
                deduped.append(idx)
        if len(deduped) < target_count:
            for idx in range(start_idx, end_idx + 1):
                if idx not in deduped:
                    deduped.append(idx)
                if len(deduped) >= target_count:
                    break
        return sorted(deduped[:target_count])

    def _save_self_play_sample_json(self, iteration, checkpoint_folder, self_play_game_results):
        if not self_play_game_results:
            return

        valid_games = [
            game for game in self_play_game_results
            if int(game.get('steps', 0) or 0) > 0 and isinstance(game.get('trace'), dict)
        ]
        if not valid_games:
            return

        sorted_games = sorted(valid_games, key=lambda item: int(item.get('steps', 0) or 0))
        if len(sorted_games) <= 2:
            sampled_indices = list(range(len(sorted_games)))
        else:
            sampled_indices = self._select_equally_spaced_indices(
                start_idx=1,
                end_idx=len(sorted_games) - 2,
                target_count=min(5, len(sorted_games) - 2),
            )
        if not sampled_indices:
            return

        sampled_games = []
        for rank, idx in enumerate(sampled_indices, start=1):
            game_data = sorted_games[idx]
            trace = game_data.get('trace') or {}
            sampled_games.append({
                'sample_rank': int(rank),
                'game_idx': int(game_data.get('game_idx', -1)),
                'steps': int(game_data.get('steps', 0) or 0),
                'used_for_training': bool(game_data.get('used_for_training', False)),
                'policy_entropy_mean': float(game_data.get('policy_entropy_mean', 0.0) or 0.0),
                'winner': int(trace.get('winner', 0) or 0),
                'is_draw': bool(trace.get('is_draw', False)),
                'result_code': float(trace.get('result_code', 0.0) or 0.0),
                'moves': list(trace.get('moves', [])),
            })

        sample_payload = {
            'iteration': int(iteration),
            'source': 'self_play_stage',
            'sampling_rule': {
                'method': 'equally_spaced_by_steps',
                'sorted_by': 'steps_ascending',
                'range': 'from_second_shortest_to_second_longest',
                'target_games': 5,
                'actual_games': len(sampled_games),
            },
            'total_games_considered': len(sorted_games),
            'samples': sampled_games,
        }

        sample_path = os.path.join(checkpoint_folder, 'self_play_samples.json')
        with open(sample_path, 'w', encoding='utf-8') as f:
            json.dump(sample_payload, f, ensure_ascii=False, indent=2)
        logging.info("Saved self-play sample JSON: %s", sample_path)

    def _build_training_state(self, iteration, model_state=None, checkpoint_kind='checkpoint'):
        if model_state is None:
            model_state = self.nnet.state_dict()

        return {
            'iteration': iteration,
            'state_dict': model_state,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'train_examples_history': self.train_examples_history,
            'teacher_opponent_history': self.teacher_opponent_history,
            'eval_history': self.eval_history,
            'iteration_metrics_history': self.iteration_metrics_history,
            'best_win_rate': self.best_win_rate,
            'best_model_iteration': self.best_model_iteration,
            'best_model_label': self.best_model_label,
            'older_best_win_rate': self.older_best_win_rate,
            'older_best_model_iteration': self.older_best_model_iteration,
            'older_best_model_label': self.older_best_model_label,
            'consecutive_no_improve_evals': self.consecutive_no_improve_evals,
            'iteration_loss_history': list(self.iteration_loss_history),
            'stop_reason': self.stop_reason,
            'current_mcts_stage_index': self.current_mcts_stage_index,
            'mcts_promotion_progress_count': self.mcts_promotion_progress_count,
            'boosted_eval_rounds_remaining': self.boosted_eval_rounds_remaining,
            'next_eval_iteration': self.next_eval_iteration,
            'auxiliary_eval_history': self.auxiliary_eval_history,
            'teacher_bootstrap_buffer_path': self.teacher_bootstrap_buffer_path,
            'teacher_bootstrap_metadata': self.teacher_bootstrap_metadata,
            'latest_auxiliary_win_rate': self.latest_auxiliary_win_rate,
            'checkpoint_kind': checkpoint_kind,
            'args': self.args.to_dict(),
        }

    def save_best_model(self):
        """
        Rotates and saves two best generations (recent + older).
        """
        recent_path, older_path, legacy_path = self._get_best_file_paths()

        if os.path.exists(older_path):
            os.remove(older_path)
        if os.path.exists(recent_path):
            shutil.move(recent_path, older_path)

        self.best_nnet = self._build_model_from_state_dict(self.nnet.state_dict())

        recent_state = self._build_training_state(
            self.best_model_iteration,
            model_state=self.best_nnet.state_dict(),
            checkpoint_kind='best_recent',
        )
        torch.save(recent_state, recent_path)
        # Keep legacy alias for backward compatibility.
        torch.save(recent_state, legacy_path)

        if self.older_best_nnet is not None:
            older_state = self._build_training_state(
                self.older_best_model_iteration,
                model_state=self.older_best_nnet.state_dict(),
                checkpoint_kind='best_older',
            )
            torch.save(older_state, older_path)

        logging.info(
            "Updated BEST rotation: recent=%s (win_rate=%.3f), older=%s",
            recent_path,
            self.best_win_rate,
            self.older_best_model_label if self.older_best_nnet is not None else 'none',
        )
        self.cleanup_checkpoints()

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
        Evaluate the current model against recent/older best generations.
        The model is promoted only if it passes all required opponent thresholds.
        """
        games_per_best = max(2, int(getattr(self.args, 'best_eval_games_per_generation', self.args.eval_games)))
        threshold = float(getattr(self.args, 'best_update_threshold', getattr(self.args, 'update_threshold', 0.55)))
        required_generations = max(1, int(getattr(self.args, 'best_eval_required_generations', 2)))

        opponents = []
        if self.best_nnet is not None:
            opponents.append(('recent_best', self.best_model_label, self.best_nnet))
        if self.older_best_nnet is not None:
            opponents.append(('older_best', self.older_best_model_label, self.older_best_nnet))
        if not opponents:
            opponents.append(('random_model', 'random_model', None))

        logging.info(
            "--- Evaluating model at iteration %s against %s opponent generations (games each=%s) ---",
            iteration,
            len(opponents),
            games_per_best,
        )

        def _run_one_eval(opponent_item):
            key, label, model_ref = opponent_item
            wins_i, losses_i, draws_i = self.execute_evaluation_parallel(games_per_best, opponent_nnet=model_ref)
            win_rate_i = (wins_i + 0.5 * draws_i) / games_per_best
            return {
                'key': key,
                'label': label,
                'wins': wins_i,
                'losses': losses_i,
                'draws': draws_i,
                'games': games_per_best,
                'win_rate': win_rate_i,
            }

        parallel_eval = bool(getattr(self.args, 'best_eval_parallelize_generations', True)) and len(opponents) > 1
        if parallel_eval:
            with ThreadPoolExecutor(max_workers=len(opponents)) as executor:
                opponent_results = list(executor.map(_run_one_eval, opponents))
        else:
            opponent_results = [_run_one_eval(item) for item in opponents]

        for result in opponent_results:
            logging.info(
                "Pitting Result vs %s - Wins: %s, Losses: %s, Draws: %s | WinRate: %.3f",
                result['label'],
                result['wins'],
                result['losses'],
                result['draws'],
                result['win_rate'],
            )

        required_count = required_generations if len(opponent_results) >= required_generations else len(opponent_results)
        required_results = opponent_results[:required_count]
        improved = bool(required_results) and all(item['win_rate'] >= threshold for item in required_results)

        primary_result = opponent_results[0]
        opponent_label = primary_result['label']
        wins, losses, draws = primary_result['wins'], primary_result['losses'], primary_result['draws']
        num_games = primary_result['games']
        win_rate = primary_result['win_rate']

        if improved:
            passed_labels = ', '.join(item['label'] for item in required_results)
            logging.info(
                "SUCCESS: New model passed all required best generations (threshold=%.3f): %s. Updating best.",
                threshold,
                passed_labels,
            )
            previous_best_model = self.best_nnet
            previous_best_iteration = self.best_model_iteration
            previous_best_label = self.best_model_label
            previous_best_win_rate = self.best_win_rate

            self.best_model_iteration = iteration
            self.best_model_label = f'checkpoint_{iteration}'
            self.best_win_rate = min(item['win_rate'] for item in required_results)

            self.older_best_nnet = previous_best_model
            self.older_best_model_iteration = previous_best_iteration
            self.older_best_model_label = previous_best_label
            self.older_best_win_rate = previous_best_win_rate

            self.boosted_eval_rounds_remaining = max(
                self.boosted_eval_rounds_remaining,
                max(1, int(getattr(self.args, 'eval_boost_rounds_after_improve', 2))),
            )
            self.save_best_model()
        else:
            logging.info(
                "REJECTED: New model rejected. Need >= %.3f against required best generations.",
                threshold,
            )

        random_baseline = None
        auxiliary_teacher = self._evaluate_against_auxiliary_model(iteration)
        should_eval_random_baseline = bool(getattr(self.args, 'enable_random_baseline_eval', False)) and self.best_nnet is not None and (
            not improved or (
                bool(getattr(self.args, 'always_evaluate_random_baseline', False))
                and self._get_current_mcts_sims() >= int(getattr(self.args, 'random_baseline_eval_min_mcts_sims', 0))
            )
        )
        if should_eval_random_baseline:
            logging.info(f"--- Evaluating model vs random_model at iteration {iteration} ---")
            random_wins, random_losses, random_draws = self.execute_evaluation_parallel(num_games, opponent_nnet=None)
            random_win_rate = (random_wins + 0.5 * random_draws) / num_games
            logging.info(
                "Random Baseline Result - Wins: %s, Losses: %s, Draws: %s | WinRate: %.3f",
                random_wins,
                random_losses,
                random_draws,
                random_win_rate,
            )
            random_baseline = {
                'wins': random_wins,
                'losses': random_losses,
                'draws': random_draws,
                'games': num_games,
                'win_rate': random_win_rate,
            }
            stability_threshold = float(getattr(self.args, 'random_baseline_stability_threshold', 0.0))
            if random_win_rate < stability_threshold:
                logging.warning(
                    "Random baseline win rate %.3f fell below stability threshold %.3f at iteration %s.",
                    random_win_rate,
                    stability_threshold,
                    iteration,
                )

        eval_result = {
            'iteration': iteration,
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'games': num_games,
            'win_rate': win_rate,
            'opponent_label': opponent_label,
            'opponent_results': opponent_results,
            'required_opponent_count': required_count,
            'required_threshold': threshold,
            'improved': improved,
            'no_improve_streak': 0,
            'random_baseline': random_baseline,
            'auxiliary_teacher': auxiliary_teacher,
        }
        self.eval_history.append(eval_result)
        return eval_result

    def final_report(self, completed_iterations):
        report_path = os.path.join(self.args.checkpoint_dir, "FINAL_REPORT.txt")
        with open(report_path, "w") as f:
            f.write("=== Training Final Report ===\n")
            f.write(f"Planned Iterations: {self.args.num_iterations}\n")
            f.write(f"Completed Iterations: {max(0, completed_iterations)}\n")
            f.write(f"Self-Play Games per Iter: {self.args.num_self_play_games}\n")
            f.write("Model trained successfully.\n")
            if self.stop_reason:
                f.write(f"Early Stop Reason: {self.stop_reason}\n")

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
                f.write(
                    f"Last Eval (Iter {last_entry['iteration']} vs {last_entry['opponent_label']}): "
                    f"Wins {last_entry['wins']}, Losses {last_entry['losses']}, Draws {last_entry['draws']}\n"
                )
            self_play_summaries = [item.get('self_play_stats') for item in self.iteration_metrics_history if item.get('self_play_stats')]
            if len(self_play_summaries) > 0:
                f.write("\n=== Self-Play Length Summary ===\n")
                mean_series = [float(item.get('mean_steps', 0.0)) for item in self_play_summaries]
                variance_series = [float(item.get('variance_steps', 0.0)) for item in self_play_summaries]
                entropy_series = [float(item.get('mean_policy_entropy', 0.0)) for item in self_play_summaries]
                last_steps = self_play_summaries[-1]
                f.write(f"Average of Iteration Mean Steps: {sum(mean_series) / len(mean_series):.3f}\n")
                f.write(f"Average of Iteration Step Variance: {sum(variance_series) / len(variance_series):.3f}\n")
                f.write(f"Average of Iteration Policy Entropy: {sum(entropy_series) / len(entropy_series):.4f}\n")
                f.write(
                    f"Last Iteration Steps: mean {last_steps.get('mean_steps', 0.0):.3f}, "
                    f"variance {last_steps.get('variance_steps', 0.0):.3f}, "
                    f"min/max {last_steps.get('min_steps', 0)}/{last_steps.get('max_steps', 0)}\n"
                )
                f.write(
                    f"Last Iteration Policy Entropy: mean {last_steps.get('mean_policy_entropy', 0.0):.4f}, "
                    f"variance {last_steps.get('variance_policy_entropy', 0.0):.4f}, "
                    f"min/max {last_steps.get('min_policy_entropy', 0.0):.4f}/{last_steps.get('max_policy_entropy', 0.0):.4f}\n"
                )
            if len(self.auxiliary_eval_history) > 0:
                f.write("\n=== Auxiliary Teacher Summary ===\n")
                last_teacher = self.auxiliary_eval_history[-1]
                avg_teacher_win = sum(item['win_rate'] for item in self.auxiliary_eval_history) / len(self.auxiliary_eval_history)
                f.write(f"Teacher Opponent: {last_teacher['opponent_label']}\n")
                f.write(f"Number of Teacher Match Snapshots: {len(self.auxiliary_eval_history)}\n")
                f.write(f"Average Win Rate vs Teacher: {avg_teacher_win:.3f}\n")
                f.write(
                    f"Last Teacher Snapshot (Iter {last_teacher['iteration']}): "
                    f"Wins {last_teacher['wins']}, Losses {last_teacher['losses']}, Draws {last_teacher['draws']}\n"
                )
            if self.teacher_bootstrap_buffer_path and len(self.teacher_bootstrap_examples) > 0:
                f.write("\n=== Teacher Bootstrap Buffer ===\n")
                f.write(f"Buffer Path: {self.teacher_bootstrap_buffer_path}\n")
                f.write(f"Saved Samples: {len(self.teacher_bootstrap_examples)}\n")
                if self.teacher_bootstrap_metadata:
                    f.write(
                        f"Created At: {self.teacher_bootstrap_metadata.get('created_at', 'unknown')} | "
                        f"Teacher: {self.teacher_bootstrap_metadata.get('teacher_label', 'unknown')} | "
                        f"Games: {self.teacher_bootstrap_metadata.get('games', 0)}\n"
                    )
                teacher_budget = self._get_teacher_sample_budget()
                if teacher_budget.get('estimated_total_teacher_samples_before_pure') is not None:
                    f.write(
                        "Teacher Samples Before Pure Self-Play: "
                        f"warmup {teacher_budget.get('warmup_teacher_samples', 0)} + "
                        f"replay {teacher_budget.get('estimated_teacher_replay_samples_before_pure', 0)} = "
                        f"{teacher_budget.get('estimated_total_teacher_samples_before_pure', 0)}\n"
                    )
            
            # 最终模型路径信息
            latest_model_path = os.path.join(self.args.checkpoint_dir, 'latest.pth.tar')
            best_model_path = os.path.join(self.args.checkpoint_dir, 'best.pth.tar')
            
            if os.path.exists(latest_model_path):
                f.write(f"\nLatest training state (Resume point): {latest_model_path}\n")
            if os.path.exists(best_model_path):
                f.write(
                    f"Best model tracked (Iter {self.best_model_iteration}, Win rate {self.best_win_rate:.3f}): "
                    f"{best_model_path}\n"
                )
            
            last_checkpoint_model = os.path.join(self.args.checkpoint_dir, f'checkpoint_{completed_iterations}', 'model.pth')
            if os.path.exists(last_checkpoint_model):
                f.write(f"Final model weights (weights only): {last_checkpoint_model}\n")
            else:
                f.write("\nCheckpoints available in the checkpoint directory.\n")
        logging.info(f"Training Complete. Report saved to {report_path}")
