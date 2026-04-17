import os
import sys
import gc
import time
import copy
import math
import re
import random
import logging
import subprocess
from datetime import datetime

import numpy as np
import torch
import torch.multiprocessing as mp
import multiprocessing

from parallel_games import execute_self_play_parallel, execute_teacher_failure_parallel
from model import Connect4Net
from trainer import Trainer, TrainerArgs, _prune_history_to_limit
from model_compat import load_checkpoint_payload, extract_state_dict_and_metadata, load_compatible_model
from teacher_data import flatten_history_examples


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _load_example_buffer(buffer_path, label):
    if not buffer_path:
        return [], {}
    if not os.path.isfile(buffer_path):
        logging.warning('%s buffer not found: %s', label, buffer_path)
        return [], {}

    payload = load_checkpoint_payload(buffer_path)
    if isinstance(payload, dict) and 'examples' in payload:
        examples = list(payload.get('examples') or [])
        metadata = dict(payload.get('metadata') or {})
    elif isinstance(payload, list):
        examples = list(payload)
        metadata = {}
    else:
        raise ValueError(f'Unsupported {label} buffer format: {buffer_path}')

    metadata.update({
        'buffer_path': buffer_path,
        'label': label,
        'samples': len(examples),
    })
    return examples, metadata


def _extract_iteration_from_checkpoint(checkpoint_path):
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return 0

    try:
        checkpoint = load_checkpoint_payload(checkpoint_path)
    except Exception as exc:
        logging.warning('Failed to inspect checkpoint iteration from %s: %s', checkpoint_path, exc)
        return 0

    if isinstance(checkpoint, dict) and 'iteration' in checkpoint:
        return int(checkpoint.get('iteration', 0) or 0)

    path_norm = str(checkpoint_path).replace('\\', '/')
    match = re.search(r'checkpoint_(\d+)/model\.pth$', path_norm)
    if match:
        return int(match.group(1))
    return 0


def _get_env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in ('1', 'true', 'y', 'yes', 'on')


def _align_next_eval_iteration(start_iteration, eval_interval):
    start_iteration = max(1, int(start_iteration))
    eval_interval = max(1, int(eval_interval))
    remainder = start_iteration % eval_interval
    if remainder == 0:
        return start_iteration
    return start_iteration + (eval_interval - remainder)


class TeacherFailureRLTrainer(Trainer):
    def __init__(self, args, resume_path=None, bootstrap_weights_path=None, source_iteration=0):
        self.bootstrap_weights_path = bootstrap_weights_path
        self.bootstrap_source_iteration = int(source_iteration or 0)
        self.human_guidance_examples = []
        self.human_guidance_metadata = {}
        self.teacher_failure_opponent_specs = []
        self._latest_teacher_failure_eval = None
        super().__init__(args, resume_path=resume_path)
        if resume_path is not None:
            self._apply_runtime_config_overrides_after_resume(resume_path)
        self._initialize_human_guidance_buffer()
        self._initialize_teacher_failure_opponents()
        if resume_path is None and bootstrap_weights_path:
            self._load_bootstrap_weights(bootstrap_weights_path)
            if self.bootstrap_source_iteration > 0:
                self.start_iter = self.bootstrap_source_iteration + 1
                logging.info(
                    'Targeted RL iteration numbering continues from source checkpoint: start_iter=%s',
                    self.start_iter,
                )

    def _apply_runtime_config_overrides_after_resume(self, resume_path):
        if not bool(getattr(self.args, 'force_runtime_overrides_on_resume', True)):
            logging.info('Resume runtime overrides disabled. Keeping checkpoint runtime state from %s', resume_path)
            return

        old_next_eval_iteration = int(getattr(self, 'next_eval_iteration', self.start_iter))
        old_boosted_eval_rounds = int(getattr(self, 'boosted_eval_rounds_remaining', 0))
        old_stage_index = int(getattr(self, 'current_mcts_stage_index', 0))

        self.next_eval_iteration = _align_next_eval_iteration(self.start_iter, self.args.eval_interval)
        self.boosted_eval_rounds_remaining = 0
        self.current_mcts_stage_index = self._resolve_mcts_stage_index(int(self.args.num_mcts_sims))
        self.mcts_promotion_progress_count = 0
        self.stop_reason = None

        logging.info(
            'Applied runtime overrides after resume from %s | next_eval_iteration %s -> %s | '
            'boosted_eval_rounds %s -> 0 | mcts_stage_index %s -> %s',
            resume_path,
            old_next_eval_iteration,
            self.next_eval_iteration,
            old_boosted_eval_rounds,
            old_stage_index,
            self.current_mcts_stage_index,
        )

    def _build_training_state(self, iteration, model_state=None, checkpoint_kind='checkpoint'):
        state = super()._build_training_state(iteration, model_state=model_state, checkpoint_kind=checkpoint_kind)
        state['targeted_rl_metadata'] = {
            'bootstrap_weights_path': self.bootstrap_weights_path,
            'bootstrap_source_iteration': self.bootstrap_source_iteration,
            'human_guidance_buffer_path': getattr(self.args, 'human_guidance_buffer_path', None),
            'human_guidance_metadata': self.human_guidance_metadata,
            'targeted_rl_rounds': int(getattr(self.args, 'targeted_rl_rounds', 0) or 0),
        }
        return state

    def _load_bootstrap_weights(self, bootstrap_weights_path):
        checkpoint = load_checkpoint_payload(bootstrap_weights_path)
        state_dict, _ = extract_state_dict_and_metadata(checkpoint)
        self.nnet.load_state_dict(state_dict, strict=True)
        if self.best_nnet is None:
            self.best_nnet = Connect4Net(num_channels=self.args.num_channels).to(self.args.train_device)
        self.best_nnet.load_state_dict(state_dict, strict=True)
        self.best_model_iteration = self.bootstrap_source_iteration
        self.best_model_label = os.path.basename(bootstrap_weights_path)
        self.best_win_rate = 0.0
        logging.info('Loaded bootstrap weights for targeted RL from %s', bootstrap_weights_path)

    def _initialize_human_guidance_buffer(self):
        buffer_path = getattr(self.args, 'human_guidance_buffer_path', None)
        try:
            examples, metadata = _load_example_buffer(buffer_path, 'human_guidance')
        except Exception as exc:
            logging.warning('Failed to load human guidance buffer %s: %s', buffer_path, exc)
            examples, metadata = [], {}

        self.human_guidance_examples = examples
        self.human_guidance_metadata = metadata
        if self.human_guidance_examples:
            logging.info(
                'Loaded human guidance buffer: samples=%s path=%s',
                len(self.human_guidance_examples),
                buffer_path,
            )

    def _initialize_teacher_failure_opponents(self):
        configured_paths = list(getattr(self.args, 'teacher_failure_opponent_paths', []) or [])
        fallback_auxiliary_path = getattr(self.args, 'auxiliary_model_path', None)
        if not configured_paths and fallback_auxiliary_path:
            configured_paths = [fallback_auxiliary_path]

        configured_labels = list(getattr(self.args, 'teacher_failure_opponent_labels', []) or [])
        loaded_specs = []
        for index, path in enumerate(configured_paths):
            if not path:
                continue
            if not os.path.isfile(path):
                logging.warning('Teacher failure opponent not found, skipping: %s', path)
                continue
            try:
                model, config, metadata = load_compatible_model(path, device='cpu')
            except Exception as exc:
                logging.warning('Failed to load teacher failure opponent %s: %s', path, exc)
                continue

            label = os.path.basename(path)
            if index < len(configured_labels) and configured_labels[index]:
                label = str(configured_labels[index])
            loaded_specs.append(
                {
                    'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    'config': dict(config),
                    'label': str(label),
                    'model_path': str(path),
                    'metadata': dict(metadata),
                }
            )

        if not loaded_specs:
            fallback_spec = self._get_auxiliary_parallel_spec()
            if fallback_spec is not None:
                loaded_specs = [dict(fallback_spec)]

        self.teacher_failure_opponent_specs = loaded_specs
        if loaded_specs:
            logging.info(
                'Loaded %s teacher-failure opponent(s): %s',
                len(loaded_specs),
                [spec.get('label', 'unknown') for spec in loaded_specs],
            )

    def _get_target_mix_ratios(self):
        teacher_ratio = float(getattr(self.args, 'target_teacher_failure_mix_ratio', 0.4) or 0.0)
        self_play_ratio = float(getattr(self.args, 'target_self_play_mix_ratio', 0.6) or 0.0)
        teacher_ratio = max(0.0, teacher_ratio)
        self_play_ratio = max(0.0, self_play_ratio)
        total = teacher_ratio + self_play_ratio
        if total <= 0.0:
            return 0.4, 0.6
        return teacher_ratio / total, self_play_ratio / total

    def _build_weighted_teacher_failure_pool(self, current_iteration):
        raw_pool, raw_by_source = flatten_history_examples(self.teacher_opponent_history)
        history_weight = float(getattr(self.args, 'teacher_failure_history_weight', 0.5) or 0.0)
        history_weight = max(0.0, min(history_weight, 1.0))
        fresh_weight = float(getattr(self.args, 'teacher_failure_fresh_weight', 1.0) or 1.0)
        fresh_weight = max(0.0, min(fresh_weight, 1.0))

        weighted_pool = []
        fresh_samples = 0
        history_samples = 0
        weighted_fresh_samples = 0
        weighted_history_samples = 0

        for entry in self.teacher_opponent_history:
            if isinstance(entry, dict) and 'examples' in entry:
                examples = list(entry.get('examples') or [])
                entry_iteration = int(entry.get('iteration', 0) or 0)
            else:
                examples = list(entry or [])
                entry_iteration = 0

            if not examples:
                continue

            is_fresh_entry = entry_iteration == int(current_iteration)
            sample_weight = fresh_weight if is_fresh_entry else history_weight

            if is_fresh_entry:
                fresh_samples += len(examples)
            else:
                history_samples += len(examples)

            if sample_weight <= 0.0:
                continue

            take_count = int(round(len(examples) * sample_weight))
            if take_count <= 0:
                continue
            take_count = min(take_count, len(examples))
            if take_count < len(examples):
                selected = random.sample(examples, take_count)
            else:
                selected = list(examples)

            weighted_pool.extend(selected)
            if is_fresh_entry:
                weighted_fresh_samples += len(selected)
            else:
                weighted_history_samples += len(selected)

        return weighted_pool, {
            'raw_samples': len(raw_pool),
            'raw_by_source': raw_by_source,
            'fresh_samples': int(fresh_samples),
            'history_samples': int(history_samples),
            'weighted_fresh_samples': int(weighted_fresh_samples),
            'weighted_history_samples': int(weighted_history_samples),
            'weighted_samples': int(len(weighted_pool)),
            'history_weight': float(history_weight),
            'fresh_weight': float(fresh_weight),
        }

    def _sample_human_guidance_examples(self, reference_sample_count):
        if not self.human_guidance_examples:
            return []

        mix_ratio = max(0.0, float(getattr(self.args, 'human_guidance_mix_ratio', 0.0) or 0.0))
        if mix_ratio <= 0.0:
            return []

        reference_sample_count = max(0, int(reference_sample_count))
        if reference_sample_count <= 0:
            reference_sample_count = len(self.human_guidance_examples)

        target_count = int(round(reference_sample_count * mix_ratio))
        max_samples = int(getattr(self.args, 'human_guidance_max_samples_per_iteration', 0) or 0)
        if max_samples > 0:
            target_count = min(target_count, max_samples)
        if target_count <= 0:
            return []

        if len(self.human_guidance_examples) >= target_count:
            return random.sample(self.human_guidance_examples, target_count)

        indices = np.random.choice(len(self.human_guidance_examples), target_count, replace=True)
        return [self.human_guidance_examples[int(index)] for index in np.atleast_1d(indices)]

    def _estimate_self_play_examples_per_game(self):
        configured_games = max(0, int(getattr(self.args, 'num_self_play_games', 0) or 0))
        if configured_games <= 0:
            return None

        recent_histories = [examples for examples in self.train_examples_history[-3:] if examples]
        if recent_histories:
            total_examples = sum(len(examples) for examples in recent_histories)
            total_games = len(recent_histories) * configured_games
            if total_examples > 0 and total_games > 0:
                return total_examples / float(total_games)

        recent_iterations = [
            item for item in self.iteration_metrics_history[-3:]
            if int(item.get('self_play_games', 0) or 0) > 0 and int(item.get('new_samples', 0) or 0) > 0
        ]
        if recent_iterations:
            total_examples = sum(int(item.get('new_samples', 0) or 0) for item in recent_iterations)
            total_games = sum(int(item.get('self_play_games', 0) or 0) for item in recent_iterations)
            if total_examples > 0 and total_games > 0:
                return total_examples / float(total_games)

        return None

    def _estimate_existing_self_play_training_samples(self):
        if not self.train_examples_history:
            return 0

        total_samples = sum(len(examples) for examples in self.train_examples_history)
        latest_weight = float(getattr(self.args, 'latest_data_weight', 1.0) or 1.0)
        if latest_weight > 1.0 and self.train_examples_history[-1]:
            total_samples += int(len(self.train_examples_history[-1]) * (latest_weight - 1.0))
        return max(0, int(total_samples))

    def _compose_targeted_self_play_data(self, iter_examples, target_self_play_samples):
        iter_examples = list(iter_examples or [])
        target_self_play_samples = max(0, int(target_self_play_samples or 0))

        if not iter_examples:
            return [], {
                'mode': 'fresh_self_play_empty',
                'self_play_samples': 0,
                'fresh_self_play_samples': 0,
                'target_self_play_samples': target_self_play_samples,
                'used_replacement': False,
            }

        if target_self_play_samples <= 0:
            selected_self_play = list(iter_examples)
            return selected_self_play, {
                'mode': 'fresh_self_play_only',
                'self_play_samples': len(selected_self_play),
                'fresh_self_play_samples': len(iter_examples),
                'target_self_play_samples': len(selected_self_play),
                'used_replacement': False,
            }

        if len(iter_examples) >= target_self_play_samples:
            selected_self_play = random.sample(iter_examples, target_self_play_samples)
            used_replacement = False
        else:
            indices = np.random.choice(len(iter_examples), target_self_play_samples, replace=True)
            selected_self_play = [iter_examples[int(index)] for index in np.atleast_1d(indices)]
            used_replacement = True

        return selected_self_play, {
            'mode': 'fresh_self_play_targeted_budget',
            'self_play_samples': len(selected_self_play),
            'fresh_self_play_samples': len(iter_examples),
            'target_self_play_samples': target_self_play_samples,
            'used_replacement': used_replacement,
        }

    def _plan_self_play_game_count(self, teacher_failure_pool_size):
        max_games = max(0, int(getattr(self.args, 'num_self_play_games', 0) or 0))
        min_games = max(0, int(getattr(self.args, 'min_self_play_games_per_iteration', 1) or 0))
        min_games = min(max_games, min_games)
        teacher_mix_ratio, self_play_mix_ratio = self._get_target_mix_ratios()
        if max_games <= 0:
            return 0, {
                'mode': 'self_play_disabled',
                'estimated_self_play_samples_per_game': None,
                'existing_self_play_samples': self._estimate_existing_self_play_training_samples(),
                'target_teacher_failure_samples': int(teacher_failure_pool_size or 0),
                'target_self_play_samples': 0,
            }

        teacher_failure_pool_size = max(0, int(teacher_failure_pool_size or 0))
        if teacher_failure_pool_size <= 0:
            return max_games, {
                'mode': 'teacher_failure_empty_use_full_self_play',
                'estimated_self_play_samples_per_game': self._estimate_self_play_examples_per_game(),
                'existing_self_play_samples': self._estimate_existing_self_play_training_samples(),
                'target_teacher_failure_samples': 0,
                'target_self_play_samples': None,
            }

        if teacher_mix_ratio <= 0.0 or self_play_mix_ratio <= 0.0:
            return max_games, {
                'mode': 'invalid_mix_ratio_use_full_self_play',
                'estimated_self_play_samples_per_game': self._estimate_self_play_examples_per_game(),
                'existing_self_play_samples': self._estimate_existing_self_play_training_samples(),
                'target_teacher_failure_samples': teacher_failure_pool_size,
                'target_self_play_samples': None,
            }

        target_self_play_samples = int(math.ceil(teacher_failure_pool_size * (self_play_mix_ratio / teacher_mix_ratio)))

        estimated_self_play_samples_per_game = self._estimate_self_play_examples_per_game()
        if estimated_self_play_samples_per_game is None or estimated_self_play_samples_per_game <= 0.0:
            return max_games, {
                'mode': 'no_self_play_estimate_use_full_self_play',
                'estimated_self_play_samples_per_game': estimated_self_play_samples_per_game,
                'existing_self_play_samples': self._estimate_existing_self_play_training_samples(),
                'target_teacher_failure_samples': teacher_failure_pool_size,
                'target_self_play_samples': target_self_play_samples,
            }

        count_history_for_budget = bool(getattr(self.args, 'targeted_rl_use_self_play_history_for_budget', False))
        existing_self_play_samples = self._estimate_existing_self_play_training_samples() if count_history_for_budget else 0
        remaining_target_samples = max(0, target_self_play_samples - existing_self_play_samples)
        if remaining_target_samples <= 0:
            return min_games, {
                'mode': 'history_satisfies_self_play_budget_keep_floor',
                'estimated_self_play_samples_per_game': estimated_self_play_samples_per_game,
                'existing_self_play_samples': existing_self_play_samples,
                'target_teacher_failure_samples': teacher_failure_pool_size,
                'target_self_play_samples': target_self_play_samples,
            }

        planned_games = int(math.ceil(remaining_target_samples / float(estimated_self_play_samples_per_game)))
        planned_games = max(min_games, min(max_games, planned_games))
        return planned_games, {
            'mode': 'teacher_failure_capped_self_play',
            'estimated_self_play_samples_per_game': estimated_self_play_samples_per_game,
            'existing_self_play_samples': existing_self_play_samples,
            'target_teacher_failure_samples': teacher_failure_pool_size,
            'target_self_play_samples': target_self_play_samples,
        }

    def _build_teacher_failure_eval_result(self, iteration, game_results):
        game_results = list(game_results or [])
        total_games = len(game_results)
        if total_games <= 0:
            return None

        wins = 0
        losses = 0
        draws = 0
        captured_loss_games = 0
        captured_positions = 0

        for item in game_results:
            outcome_for_model = float(item.get('outcome_for_model', 0.0) or 0.0)
            if outcome_for_model > 0.0:
                wins += 1
            elif outcome_for_model < 0.0:
                losses += 1
            else:
                draws += 1

            if item.get('used_for_training', False):
                captured_loss_games += 1
            captured_positions += int(item.get('captured_positions', 0) or 0)

        win_rate = (wins + 0.5 * draws) / float(total_games)
        loss_rate = losses / float(total_games)
        opponent_labels = sorted({str(item.get('opponent_label', 'unknown')) for item in game_results})
        if len(opponent_labels) == 1:
            opponent_label = opponent_labels[0]
        else:
            opponent_label = ' + '.join(opponent_labels) if opponent_labels else 'mixed_historical_best'
        return {
            'iteration': int(iteration),
            'wins': int(wins),
            'losses': int(losses),
            'draws': int(draws),
            'games': int(total_games),
            'win_rate': float(win_rate),
            'loss_rate': float(loss_rate),
            'captured_loss_games': int(captured_loss_games),
            'captured_positions': int(captured_positions),
            'opponent_label': opponent_label,
            'source': 'teacher_failure_collection',
            'teacher_replay_ratio': self._get_teacher_replay_ratio(int(iteration) + 1),
        }

    def _record_teacher_failure_eval(self, iteration, game_results):
        teacher_eval = self._build_teacher_failure_eval_result(iteration, game_results)
        self._latest_teacher_failure_eval = teacher_eval
        if teacher_eval is None:
            return None

        self.latest_auxiliary_win_rate = teacher_eval['win_rate']
        self.auxiliary_eval_history.append(dict(teacher_eval))
        return teacher_eval

    def _evaluate_against_auxiliary_model(self, iteration):
        latest = self._latest_teacher_failure_eval
        if latest is None:
            return None
        if int(latest.get('iteration', -1)) != int(iteration):
            return None
        return dict(latest)

    def execute_episode_parallel(self, num_games=None):
        if num_games is None:
            return super().execute_episode_parallel()

        num_games = max(0, int(num_games or 0))
        if num_games <= 0:
            return [], []

        num_workers = self._get_parallel_worker_count(num_games)
        logging.info(
            'Preparing %s self-play workers with shared inference on %s for %s games...',
            num_workers,
            self.args.shared_inference_device,
            num_games,
        )
        return execute_self_play_parallel(
            args=self.args,
            num_games=num_games,
            num_workers=num_workers,
            shared_inference_device=self.args.shared_inference_device,
            inference_batch_size=self.args.inference_batch_size,
            inference_timeout_s=self.args.inference_timeout_s,
            model_state={k: v.detach().cpu() for k, v in self.nnet.state_dict().items()},
            progress_desc='Self-Play',
        )

    def collect_teacher_failure_examples(self):
        opponent_specs = list(self.teacher_failure_opponent_specs or [])
        if not opponent_specs:
            raise RuntimeError('Auxiliary teacher model is required for teacher failure RL.')

        total_games = max(2, int(getattr(self.args, 'teacher_failure_num_games', self.args.num_self_play_games) or 0))
        base_games = total_games // len(opponent_specs)
        remainder = total_games % len(opponent_specs)

        total_examples = []
        total_game_results = []
        for index, opponent_spec in enumerate(opponent_specs):
            num_games = base_games + (1 if index < remainder else 0)
            if num_games <= 0:
                continue

            num_workers = self._get_parallel_worker_count(num_games)
            collection_args = copy.deepcopy(self.args)
            collection_args.current_iteration = int(getattr(self.args, 'current_iteration', 1))
            collection_args.tactical_override_max_step = 0
            collection_args.min_game_steps = 0
            collection_args.min_game_steps_start_iteration = 10 ** 9

            opponent_examples, opponent_game_results = execute_teacher_failure_parallel(
                args=collection_args,
                num_games=num_games,
                num_workers=num_workers,
                shared_inference_device=self.args.shared_inference_device,
                inference_batch_size=self.args.inference_batch_size,
                inference_timeout_s=self.args.inference_timeout_s,
                model_state={k: v.detach().cpu() for k, v in self.nnet.state_dict().items()},
                teacher_model_spec=opponent_spec,
                progress_desc=f"Teacher Failure [{opponent_spec.get('label', 'teacher')}]",
            )
            for item in opponent_game_results:
                item['opponent_label'] = opponent_spec.get('label', 'teacher')

            total_examples.extend(opponent_examples)
            total_game_results.extend(opponent_game_results)

        return total_examples, total_game_results

    def _mix_teacher_failure_examples(self, self_play_train_data, teacher_failure_examples):
        self_play_train_data = list(self_play_train_data)
        teacher_failure_examples = list(teacher_failure_examples)
        teacher_mix_ratio, self_play_mix_ratio = self._get_target_mix_ratios()

        if not self_play_train_data and not teacher_failure_examples:
            return [], {
                'mode': 'empty_training_set',
                'self_play_samples': 0,
                'teacher_failure_samples': 0,
                'teacher_failure_mix_ratio': 0.0,
                'human_guidance_samples': 0,
            }

        if not self_play_train_data:
            mixed = teacher_failure_examples
            human_guidance = self._sample_human_guidance_examples(len(mixed))
            mixed.extend(human_guidance)
            random.shuffle(mixed)
            return mixed, {
                'mode': 'teacher_failure_only_fallback_with_human' if human_guidance else 'teacher_failure_only_fallback',
                'self_play_samples': 0,
                'teacher_failure_samples': len(teacher_failure_examples),
                'teacher_failure_mix_ratio': 1.0,
                'target_teacher_mix_ratio': teacher_mix_ratio,
                'human_guidance_samples': len(human_guidance),
            }

        if not teacher_failure_examples:
            human_guidance = self._sample_human_guidance_examples(len(self_play_train_data))
            self_play_train_data.extend(human_guidance)
            random.shuffle(self_play_train_data)
            return self_play_train_data, {
                'mode': 'self_play_only_fallback_with_human' if human_guidance else 'self_play_only_fallback',
                'self_play_samples': len(self_play_train_data) - len(human_guidance),
                'teacher_failure_samples': 0,
                'teacher_failure_mix_ratio': 0.0,
                'target_teacher_mix_ratio': teacher_mix_ratio,
                'human_guidance_samples': len(human_guidance),
            }

        if teacher_mix_ratio <= 0.0:
            selected_teacher = []
            selected_self_play = list(self_play_train_data)
        elif self_play_mix_ratio <= 0.0:
            selected_self_play = []
            selected_teacher = list(teacher_failure_examples)
        else:
            max_teacher_by_self_play = int(round(len(self_play_train_data) * (teacher_mix_ratio / self_play_mix_ratio)))
            target_teacher_count = min(len(teacher_failure_examples), max_teacher_by_self_play)
            selected_teacher = random.sample(teacher_failure_examples, target_teacher_count) if target_teacher_count > 0 else []

            if selected_teacher:
                target_self_play_count = int(round(len(selected_teacher) * (self_play_mix_ratio / teacher_mix_ratio)))
            else:
                target_self_play_count = len(self_play_train_data)
            target_self_play_count = min(len(self_play_train_data), max(0, target_self_play_count))
            selected_self_play = random.sample(self_play_train_data, target_self_play_count) if target_self_play_count > 0 else []

        human_guidance = self._sample_human_guidance_examples(max(len(selected_self_play), len(selected_teacher)))
        mixed = selected_self_play + selected_teacher + human_guidance
        random.shuffle(mixed)
        actual_teacher_ratio = len(selected_teacher) / float(max(1, len(selected_self_play) + len(selected_teacher)))
        return mixed, {
            'mode': 'teacher_failure_self_play_target_mix_with_human' if human_guidance else 'teacher_failure_self_play_target_mix',
            'self_play_samples': len(selected_self_play),
            'teacher_failure_samples': len(selected_teacher),
            'teacher_failure_mix_ratio': actual_teacher_ratio,
            'target_teacher_mix_ratio': teacher_mix_ratio,
            'human_guidance_samples': len(human_guidance),
        }

    def train(self):
        last_completed_iteration = self.start_iter - 1

        for i in range(self.start_iter, self.args.num_iterations + 1):
            iteration_start_wall = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            iteration_start = time.perf_counter()
            self.args.current_iteration = i
            self.args.num_mcts_sims = int(self.args.num_mcts_sims)

            logging.info(
                'Starting Targeted Teacher-Failure RL Iteration %s/%s | mcts_sims=%s',
                i,
                self.args.num_iterations,
                self.args.num_mcts_sims,
            )

            if str(self.args.train_device).startswith('cuda'):
                torch.cuda.empty_cache()

            teacher_failure_start = time.perf_counter()
            teacher_failure_examples, teacher_failure_game_results = self.collect_teacher_failure_examples()
            teacher_failure_duration = time.perf_counter() - teacher_failure_start
            teacher_failure_stats = self._summarize_self_play_lengths(teacher_failure_game_results)
            teacher_failure_eval = self._record_teacher_failure_eval(i, teacher_failure_game_results)
            teacher_history_pruned = self._append_teacher_opponent_history(
                teacher_failure_examples,
                source='teacher_failure',
                iteration=i,
                metadata={
                    'games': int(getattr(self.args, 'teacher_failure_num_games', self.args.num_self_play_games)),
                    'stats': teacher_failure_stats,
                },
            )
            if teacher_history_pruned:
                logging.info(
                    'Pruned oldest teacher-opponent history (keep last %s)',
                    self._get_teacher_opponent_history_limit(),
                )
            teacher_failure_pool, teacher_failure_pool_info = self._build_weighted_teacher_failure_pool(i)

            planned_self_play_games, self_play_plan = self._plan_self_play_game_count(len(teacher_failure_pool))
            self_play_start = time.perf_counter()
            iter_examples, self_play_game_results = self.execute_episode_parallel(planned_self_play_games)
            self_play_duration = time.perf_counter() - self_play_start
            self_play_stats = self._summarize_self_play_lengths(self_play_game_results)

            if iter_examples:
                self.train_examples_history.append(iter_examples)

                history_before_prune = len(self.train_examples_history)
                self.train_examples_history = _prune_history_to_limit(
                    self.train_examples_history,
                    self.args.history_len,
                )
                if len(self.train_examples_history) < history_before_prune:
                    logging.info('Removing oldest history (keep last %s)', self.args.history_len)

            self_play_train_data, self_play_source = self._compose_targeted_self_play_data(
                iter_examples,
                self_play_plan.get('target_self_play_samples', 0),
            )

            train_data, train_data_source = self._mix_teacher_failure_examples(
                self_play_train_data,
                teacher_failure_pool,
            )

            train_metrics = self.train_network(train_data, iteration=i)

            teacher_failure_fresh_sample_count = len(teacher_failure_examples)
            teacher_failure_history_sample_count = len(teacher_failure_pool)
            train_sample_count = len(train_data)
            del self_play_train_data
            del teacher_failure_examples
            del train_data
            gc.collect()

            self.scheduler.step()

            eval_metrics = None
            checkpoint_saved = False
            if i % self.args.checkpoint_interval == 0:
                self.save_checkpoint(i)
                checkpoint_saved = True
            if self._should_run_evaluation(i):
                eval_metrics = self.evaluate_model(i)
                self._schedule_next_evaluation(i)

            iteration_duration = time.perf_counter() - iteration_start
            last_completed_iteration = i
            stop_reason = self._check_loss_early_stop(i, train_metrics['total_loss'])
            if stop_reason is None:
                stop_reason = self._check_eval_early_stop(eval_metrics)

            iteration_summary = {
                'iteration': i,
                'start_time': iteration_start_wall,
                'iteration_duration_sec': iteration_duration,
                'self_play_duration_sec': self_play_duration,
                'self_play_games': planned_self_play_games,
                'self_play_stats': self_play_stats,
                'new_samples': len(iter_examples),
                'train_samples': train_sample_count,
                'train_data_mode': train_data_source['mode'],
                'self_play_train_samples': train_data_source['self_play_samples'],
                'teacher_replay_samples': train_data_source['teacher_failure_samples'],
                'teacher_replay_ratio': train_data_source['teacher_failure_mix_ratio'],
                'human_guidance_samples': train_data_source.get('human_guidance_samples', 0),
                'teacher_failure_games': int(getattr(self.args, 'teacher_failure_num_games', self.args.num_self_play_games)),
                'teacher_failure_duration_sec': teacher_failure_duration,
                'teacher_failure_stats': teacher_failure_stats,
                'teacher_failure_eval': teacher_failure_eval,
                'teacher_failure_self_play_samples': self_play_source['self_play_samples'],
                'teacher_failure_collected_samples': train_data_source['teacher_failure_samples'],
                'teacher_failure_fresh_samples': teacher_failure_fresh_sample_count,
                'teacher_failure_history_samples': teacher_failure_history_sample_count,
                'teacher_failure_history_pool_raw_samples': teacher_failure_pool_info['raw_samples'],
                'teacher_failure_history_pool_by_source': teacher_failure_pool_info['raw_by_source'],
                'teacher_failure_history_weight': teacher_failure_pool_info['history_weight'],
                'teacher_failure_fresh_weight': teacher_failure_pool_info['fresh_weight'],
                'self_play_plan': self_play_plan,
                'train_metrics': train_metrics,
                'eval_metrics': eval_metrics,
                'mcts_sims': int(self.args.num_mcts_sims),
                'learning_rate': self._get_learning_rate(),
                'stop_reason': stop_reason,
                'adaptive_notes': [
                    (
                        'TeacherFailureMix: '
                        f"self_play_pool={self_play_source['self_play_samples']} | "
                        f"fresh_self_play={self_play_source.get('fresh_self_play_samples', len(iter_examples))} | "
                        f"teacher_failure_pool={teacher_failure_stats.get('used_games', 0)} games fresh / "
                        f"history_raw={teacher_failure_pool_info['raw_samples']} samples {teacher_failure_pool_info['raw_by_source']} / "
                        f"history_weighted={len(teacher_failure_pool)} (fresh={teacher_failure_pool_info['weighted_fresh_samples']}, "
                        f"old={teacher_failure_pool_info['weighted_history_samples']}, old_w={teacher_failure_pool_info['history_weight']:.2f}) / "
                        f"{train_data_source['teacher_failure_samples']} samples | "
                        f"teacher_wld={(teacher_failure_eval or {}).get('wins', 0)}/{(teacher_failure_eval or {}).get('losses', 0)}/{(teacher_failure_eval or {}).get('draws', 0)} | "
                        f"teacher_win_rate={(teacher_failure_eval or {}).get('win_rate', 0.0):.3f} | "
                        f"target_teacher_mix={train_data_source.get('target_teacher_mix_ratio', 0.0):.2f} | "
                        f"actual_teacher_mix={train_data_source['teacher_failure_mix_ratio']:.2f} | "
                        f"human_guidance={train_data_source.get('human_guidance_samples', 0)} | "
                        f"teacher_failure_duration={self._format_duration(teacher_failure_duration)}"
                    ),
                    (
                        'SelfPlayBudget: '
                        f"mode={self_play_plan['mode']} | "
                        f"planned_games={planned_self_play_games}/{self.args.num_self_play_games} | "
                        f"existing_self_play_samples={self_play_plan['existing_self_play_samples']} | "
                        f"target_teacher_failure_samples={self_play_plan['target_teacher_failure_samples']} | "
                        f"target_self_play_samples={self_play_plan.get('target_self_play_samples')} | "
                        f"used_replacement={'yes' if self_play_source.get('used_replacement', False) else 'no'} | "
                        f"estimated_samples_per_game="
                        f"{0.0 if self_play_plan['estimated_self_play_samples_per_game'] is None else self_play_plan['estimated_self_play_samples_per_game']:.2f}"
                    ),
                ],
            }
            self.iteration_metrics_history.append(iteration_summary)
            self.stop_reason = stop_reason
            self._log_iteration_summary(iteration_summary)

            if stop_reason:
                logging.info('Early stopping triggered at iteration %s: %s', i, stop_reason)
                if not checkpoint_saved:
                    self.save_checkpoint(i)
                break

        logging.info('Saving final checkpoint...')
        if last_completed_iteration >= self.start_iter:
            self.save_checkpoint(last_completed_iteration)
        self.final_report(last_completed_iteration)


def _resolve_resume_checkpoint(checkpoint_dir):
    latest_path = os.path.join(checkpoint_dir, 'latest.pth.tar')
    best_path = os.path.join(checkpoint_dir, 'best.pth.tar')
    if os.path.exists(latest_path):
        return latest_path
    if os.path.exists(best_path):
        return best_path
    return None


def _resolve_bootstrap_weights_path(training_root, checkpoint_dir):
    preferred_override = os.environ.get('TARGETED_RL_BOOTSTRAP_PATH')
    if preferred_override and os.path.exists(preferred_override):
        return preferred_override

    own_resume = _resolve_resume_checkpoint(checkpoint_dir)
    if own_resume:
        return None

    main_checkpoint_dir = os.path.join(training_root, 'checkpoints')
    preferred = [
        os.path.join(main_checkpoint_dir, 'checkpoint_213', 'model.pth'),
        os.path.join(main_checkpoint_dir, 'best.pth.tar'),
        os.path.join(main_checkpoint_dir, 'latest.pth.tar'),
        os.path.join(os.path.dirname(training_root), 'save_model', 'best.pth.tar'),
    ]
    for path in preferred:
        if os.path.exists(path):
            return path
    return None


def _launch_main_train_after_rl(training_root, rl_args):
    if not bool(getattr(rl_args, 'auto_switch_to_main_train', False)):
        return

    main_train_entry = os.path.abspath(getattr(rl_args, 'main_train_entry', os.path.join(training_root, 'main_train.py')))
    if not os.path.isfile(main_train_entry):
        raise FileNotFoundError(f'main_train entry not found: {main_train_entry}')

    latest_rl_checkpoint = os.path.join(rl_args.checkpoint_dir, 'latest.pth.tar')
    if not os.path.isfile(latest_rl_checkpoint):
        raise FileNotFoundError(f'RL latest checkpoint not found: {latest_rl_checkpoint}')

    completed_iteration = _extract_iteration_from_checkpoint(latest_rl_checkpoint)
    required_rl_iterations = int(getattr(rl_args, 'num_iterations', 0) or 0)
    if required_rl_iterations > 0 and completed_iteration < required_rl_iterations:
        logging.info(
            'Skipping auto switch to main_train.py because RL stopped at iteration %s before planned iteration %s.',
            completed_iteration,
            required_rl_iterations,
        )
        return

    target_total_iterations = int(getattr(rl_args, 'main_train_total_iterations', 0) or 0)
    if target_total_iterations > 0 and completed_iteration >= target_total_iterations:
        logging.info(
            'Skipping auto switch to main_train.py because RL already reached target_total_iterations=%s (completed=%s).',
            target_total_iterations,
            completed_iteration,
        )
        return

    env = os.environ.copy()
    env['CUSTOM_RESUME_PATH'] = latest_rl_checkpoint
    env['RESUME_CHECKPOINT_PATH'] = latest_rl_checkpoint
    if target_total_iterations > 0:
        env['MAIN_TRAIN_NUM_ITERATIONS'] = str(target_total_iterations)

    logging.info(
        'Auto switching to main_train.py | resume=%s | target_total_iterations=%s',
        latest_rl_checkpoint,
        target_total_iterations if target_total_iterations > 0 else 'default',
    )
    subprocess.run(
        [sys.executable, main_train_entry],
        cwd=training_root,
        env=env,
        check=True,
    )


if __name__ == '__main__':
    multiprocessing.freeze_support()
    mp.set_start_method('spawn', force=True)

    global_seed = 42
    random.seed(global_seed)
    np.random.seed(global_seed)
    torch.manual_seed(global_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(global_seed)

    args = TrainerArgs()
    training_root = os.path.dirname(os.path.abspath(__file__))
    targeted_rl_rounds = 4

    args.checkpoint_dir = os.path.join(training_root, 'checkpoints_teacher_failure_rl')
    args.num_iterations = targeted_rl_rounds
    args.targeted_rl_rounds = targeted_rl_rounds
    args.num_self_play_games = 160
    args.min_self_play_games_per_iteration = 10
    args.targeted_rl_use_self_play_history_for_budget = False
    args.target_teacher_failure_mix_ratio = float(os.environ.get('TARGET_TEACHER_FAILURE_MIX_RATIO', '0.4') or 0.4)
    args.target_self_play_mix_ratio = float(os.environ.get('TARGET_SELF_PLAY_MIX_RATIO', '0.6') or 0.6)
    args.teacher_failure_num_games = 160
    args.teacher_failure_history_weight = float(os.environ.get('TEACHER_FAILURE_HISTORY_WEIGHT', '0.5') or 0.5)
    args.teacher_failure_fresh_weight = float(os.environ.get('TEACHER_FAILURE_FRESH_WEIGHT', '1.0') or 1.0)
    args.num_channels = 256
    args.checkpoint_interval = 2
    args.eval_interval = 4
    args.force_runtime_overrides_on_resume = _get_env_flag('RL_FORCE_RUNTIME_OVERRIDES_ON_RESUME', True)
    args.auto_switch_to_main_train = _get_env_flag('RL_AUTO_SWITCH_TO_MAIN_TRAIN', True)
    args.main_train_total_iterations = int(os.environ.get('MAIN_TRAIN_TOTAL_ITERATIONS', '300') or 300)
    args.main_train_entry = os.environ.get('MAIN_TRAIN_ENTRY', os.path.join(training_root, 'main_train.py'))
    args.epochs = 4
    args.learning_rate = 0.0002
    args.policy_head_lr_scale = 1.0
    args.value_head_lr_scale = 1.0
    args.lr_decay_step_size = 200
    args.lr_decay_gamma = 0.8
    args.max_checkpoints = 3
    args.eval_games = 40
    args.update_threshold = 0.55
    args.self_play_workers = 64
    args.shared_inference_server_count = 6
    args.high_mcts_shared_inference_server_threshold = 0
    args.high_mcts_shared_inference_server_count = 6
    args.compatible_inference_server_count = 1
    args.num_mcts_threads = 8
    args.inference_batch_size = 128
    args.inference_timeout_s = 0.003
    args.virtual_loss = 1.2
    args.tactical_override_max_step = 0
    args.tactical_override_prefer_win = False
    args.tactical_override_prefer_block = False
    args.dirichlet_alpha = 0.18
    args.dirichlet_epsilon = 0.10
    args.self_play_exploration_strength = 1.0
    args.self_play_phase_schedule = [
        {
            'name': 'targeted_rl_opening',
            'max_step': 9,
            'temperature': 0.5,
            'dirichlet_alpha': 0.0,
            'dirichlet_epsilon': 0.0,
        },
        {
            'name': 'targeted_rl_mid',
            'max_step': 25,
            'temperature': 0.8,
            'dirichlet_alpha': 0.0,
            'dirichlet_epsilon': 0.0,
        },
        {
            'name': 'targeted_rl_late',
            'max_step': None,
            'temperature': 0.0,
            'dirichlet_alpha': 0.0,
            'dirichlet_epsilon': 0.0,
        },
    ]
    args.exploration_iteration_schedule = [
        {'start_iter': 1, 'end_iter': None, 'temperature_scale': 1.0, 'noise_scale': 1.0},
    ]
    args.history_len = 6
    args.teacher_opponent_history_len = 18
    args.latest_data_weight = 1.0
    args.min_game_steps = 0
    args.min_game_steps_start_iteration = 10 ** 9
    args.num_mcts_sims = 1024
    args.mcts_sim_candidates = [1024]
    args.mcts_promotion_improve_count = 0
    args.eval_interval_after_best = 2
    args.eval_boost_rounds_after_improve = 1
    args.random_baseline_stability_threshold = 0.60
    args.always_evaluate_random_baseline = False
    args.random_baseline_eval_min_mcts_sims = 512
    args.enable_random_baseline_eval = False
    args.loss_increase_patience = 300
    args.no_improve_eval_patience = 8
    args.teacher_bootstrap_num_games = 0
    args.teacher_bootstrap_warmup_iterations = 0
    args.teacher_replay_initial_ratio = 0.0
    args.teacher_replay_final_ratio = 0.0
    args.teacher_replay_decay_iterations = 1
    args.teacher_replay_drift_threshold = -1.0
    args.teacher_replay_drift_ratio = 0.0
    args.teacher_replay_relax_start_iteration = 0
    args.teacher_replay_relax_end_iteration = 0
    args.teacher_replay_relaxed_drift_ratio = 0.0
    args.teacher_failure_target_temperature = 0.0
    args.teacher_failure_game_temperature = 0.0
    args.human_guidance_buffer_path = os.environ.get('HUMAN_GUIDANCE_BUFFER_PATH')
    args.human_guidance_mix_ratio = float(os.environ.get('HUMAN_GUIDANCE_MIX_RATIO', '0.0') or 0.0)
    args.human_guidance_max_samples_per_iteration = int(os.environ.get('HUMAN_GUIDANCE_MAX_SAMPLES', '0') or 0)

    default_checkpoint_dir = os.path.join(training_root, 'checkpoints')
    default_teacher_paths = [
        os.path.join(default_checkpoint_dir, 'best_new.pth.tar'),
        os.path.join(default_checkpoint_dir, 'best_old.pth.tar'),
    ]
    configured_teacher_paths = os.environ.get('TEACHER_FAILURE_OPPONENT_PATHS', '').strip()
    if configured_teacher_paths:
        args.teacher_failure_opponent_paths = [
            item.strip() for item in configured_teacher_paths.split(os.pathsep) if item.strip()
        ]
    else:
        args.teacher_failure_opponent_paths = list(default_teacher_paths)

    configured_teacher_labels = os.environ.get('TEACHER_FAILURE_OPPONENT_LABELS', '').strip()
    if configured_teacher_labels:
        args.teacher_failure_opponent_labels = [
            item.strip() for item in configured_teacher_labels.split(',') if item.strip()
        ]
    else:
        args.teacher_failure_opponent_labels = ['best_new', 'best_old']

    default_auxiliary_model = args.teacher_failure_opponent_paths[0] if args.teacher_failure_opponent_paths else None
    args.auxiliary_model_path = os.environ.get('AUXILIARY_MODEL_PATH', default_auxiliary_model)
    args.auxiliary_model_label = 'historical_best_teacher'
    args.auxiliary_eval_games = 40
    args.auxiliary_eval_interval = 1

    if torch.cuda.is_available():
        args.train_device = 'cuda'
        args.infer_device = 'cpu'
        args.shared_inference_device = 'cuda'
        args.batch_size = max(args.batch_size, 512)
    else:
        allow_cpu_training = os.environ.get('ALLOW_CPU_TRAINING', '').strip().lower()
        interactive_stdin = bool(getattr(sys.stdin, 'isatty', lambda: False)())

        if allow_cpu_training in ('1', 'true', 'y', 'yes'):
            resp = 'y'
        elif interactive_stdin:
            try:
                resp = input('未检测到 NVIDIA GPU (CUDA)。是否使用 CPU 继续训练？ [y/N]: ').strip().lower()
            except Exception:
                resp = 'n'
        else:
            resp = 'y'
            print('未检测到 NVIDIA GPU (CUDA)，当前为非交互环境，自动降级为 CPU 训练。')

        if resp in ('y', 'yes'):
            args.train_device = 'cpu'
            args.infer_device = 'cpu'
            args.shared_inference_device = 'cpu'
            print('已选择使用 CPU 进行训练。')
        else:
            print('未同意使用 CPU 训练，程序终止。')
            sys.exit(0)

    resume_checkpoint = _resolve_resume_checkpoint(args.checkpoint_dir)
    bootstrap_weights_path = _resolve_bootstrap_weights_path(training_root, args.checkpoint_dir)
    bootstrap_source_iteration = _extract_iteration_from_checkpoint(bootstrap_weights_path)
    resume_iteration = _extract_iteration_from_checkpoint(resume_checkpoint)

    if resume_checkpoint:
        args.num_iterations = resume_iteration + targeted_rl_rounds
    elif bootstrap_source_iteration > 0:
        args.num_iterations = bootstrap_source_iteration + targeted_rl_rounds

    if resume_checkpoint:
        print(f'Resume Training requested. Using checkpoint: {resume_checkpoint}')
    elif bootstrap_weights_path:
        print(f'No targeted RL checkpoint found. Bootstrapping from: {bootstrap_weights_path}')
    else:
        print('No targeted RL checkpoint or bootstrap weights found. Starting from scratch.')

    if resume_checkpoint:
        try:
            checkpoint_data = load_checkpoint_payload(resume_checkpoint)
            if isinstance(checkpoint_data, dict) and 'iteration' in checkpoint_data:
                last_iter = checkpoint_data['iteration']
                if last_iter >= args.num_iterations:
                    print(f'WARNING: Checkpoint is at iteration {last_iter}, but num_iterations is set to {args.num_iterations}.')
                    print(f'Increasing num_iterations to {last_iter + targeted_rl_rounds} to continue training.')
                    args.num_iterations = last_iter + targeted_rl_rounds
            del checkpoint_data
        except Exception as exc:
            print(f'警告：读取 targeted RL checkpoint 轮次失败，将继续按默认轮次训练。错误：{exc}')

    print(
        'RL iteration plan: '
        f"targeted_rl_rounds={targeted_rl_rounds}, "
        f"resume_iteration={resume_iteration}, "
        f"final_num_iterations={args.num_iterations}"
    )

    print('Initializing Targeted Teacher-Failure RL Training...')
    if args.auxiliary_model_path and os.path.exists(args.auxiliary_model_path):
        print(f'Auxiliary teacher enabled: {args.auxiliary_model_label} -> {args.auxiliary_model_path}')
    elif args.auxiliary_model_path:
        print(f'Auxiliary teacher not found, skipping: {args.auxiliary_model_path}')

    trainer = TeacherFailureRLTrainer(
        args,
        resume_path=resume_checkpoint,
        bootstrap_weights_path=bootstrap_weights_path,
        source_iteration=bootstrap_source_iteration,
    )

    print('Starting Targeted Teacher-Failure RL Loop...')
    trainer.train()
    _launch_main_train_after_rl(training_root, args)