import copy
import os
from random import shuffle

import numpy as np
import torch

from model_compat import load_checkpoint_payload


def build_history_entry(examples, source, iteration=None, metadata=None):
    return {
        'examples': list(examples or []),
        'source': str(source),
        'iteration': None if iteration is None else int(iteration),
        'metadata': dict(metadata or {}),
    }


def flatten_history_examples(history):
    flat_examples = []
    source_sample_counts = {}

    for entry in history or []:
        if isinstance(entry, dict) and 'examples' in entry:
            examples = list(entry.get('examples') or [])
            source = str(entry.get('source') or 'unknown')
        else:
            examples = list(entry or [])
            source = 'self_play'

        if not examples:
            continue

        flat_examples.extend(examples)
        source_sample_counts[source] = source_sample_counts.get(source, 0) + len(examples)

    return flat_examples, source_sample_counts


def _sample_examples(examples, sample_count):
    if not examples:
        return []

    sample_count = int(sample_count)
    if sample_count <= 0:
        return []

    total_available = len(examples)
    replace = sample_count > total_available
    indices = np.random.choice(total_available, sample_count, replace=replace)
    return [examples[int(index)] for index in np.atleast_1d(indices)]


def resolve_teacher_buffer_path(args):
    configured = getattr(args, 'teacher_bootstrap_buffer_path', None)
    if configured:
        return configured
    return os.path.join(args.checkpoint_dir, 'teacher_bootstrap_examples.pth.tar')


def load_teacher_bootstrap_buffer(args):
    buffer_path = resolve_teacher_buffer_path(args)
    if not os.path.isfile(buffer_path):
        return [], {}, buffer_path

    payload = load_checkpoint_payload(buffer_path)
    if isinstance(payload, dict) and 'examples' in payload:
        return list(payload.get('examples') or []), dict(payload.get('metadata') or {}), buffer_path
    if isinstance(payload, list):
        return list(payload), {}, buffer_path
    raise ValueError(f'Unsupported teacher bootstrap buffer format: {buffer_path}')


def save_teacher_bootstrap_buffer(args, examples, metadata):
    buffer_path = resolve_teacher_buffer_path(args)
    folder = os.path.dirname(buffer_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    torch.save({'examples': list(examples), 'metadata': dict(metadata)}, buffer_path)
    return buffer_path


def build_teacher_bootstrap_args(args):
    bootstrap_args = copy.deepcopy(args)
    bootstrap_args.current_iteration = 1
    bootstrap_args.num_mcts_sims = max(
        1,
        int(getattr(args, 'teacher_bootstrap_mcts_sims', 0) or args.num_mcts_sims),
    )
    bootstrap_args.self_play_exploration_strength = 1.0
    bootstrap_args.dirichlet_alpha = float(
        getattr(args, 'teacher_bootstrap_dirichlet_alpha', getattr(args, 'dirichlet_alpha', 0.0))
    )
    bootstrap_args.dirichlet_epsilon = float(
        getattr(args, 'teacher_bootstrap_dirichlet_epsilon', getattr(args, 'dirichlet_epsilon', 0.0))
    )
    bootstrap_args.self_play_phase_schedule = [
        {
            'name': 'teacher_bootstrap',
            'max_step': None,
            'temperature': float(getattr(args, 'teacher_bootstrap_temperature', 0.0)),
            'dirichlet_alpha': bootstrap_args.dirichlet_alpha,
            'dirichlet_epsilon': bootstrap_args.dirichlet_epsilon,
        }
    ]
    bootstrap_args.exploration_iteration_schedule = [
        {'start_iter': 1, 'end_iter': None, 'temperature_scale': 1.0, 'noise_scale': 1.0}
    ]
    bootstrap_args.min_game_steps_start_iteration = 1
    return bootstrap_args


def get_teacher_warmup_iterations(args):
    return max(0, int(getattr(args, 'teacher_bootstrap_warmup_iterations', 0) or 0))


def is_teacher_warmup_iteration(args, teacher_bootstrap_examples, iteration):
    return bool(teacher_bootstrap_examples) and int(iteration) <= get_teacher_warmup_iterations(args)


def get_teacher_drift_replay_ratio(args, iteration):
    base_ratio = max(0.0, float(getattr(args, 'teacher_replay_drift_ratio', 0.0)))
    relaxed_ratio = max(
        0.0,
        min(base_ratio, float(getattr(args, 'teacher_replay_relaxed_drift_ratio', base_ratio))),
    )
    relax_start = int(getattr(args, 'teacher_replay_relax_start_iteration', 0) or 0)
    relax_end = int(getattr(args, 'teacher_replay_relax_end_iteration', 0) or 0)
    iteration = int(iteration)

    if relax_start <= 0 or iteration < relax_start:
        return base_ratio
    if relax_end <= relax_start:
        return relaxed_ratio
    if iteration >= relax_end:
        return relaxed_ratio

    progress = (iteration - relax_start) / float(relax_end - relax_start)
    return base_ratio + progress * (relaxed_ratio - base_ratio)


def get_teacher_replay_ratio(args, teacher_bootstrap_examples, latest_auxiliary_win_rate, iteration, teacher_opponent_history=None):
    teacher_opponent_examples, _ = flatten_history_examples(teacher_opponent_history)
    if not teacher_bootstrap_examples and not teacher_opponent_examples:
        return 0.0

    initial_ratio = max(0.0, float(getattr(args, 'teacher_replay_initial_ratio', 0.0)))
    final_ratio = max(0.0, float(getattr(args, 'teacher_replay_final_ratio', 0.0)))
    decay_iterations = max(1, int(getattr(args, 'teacher_replay_decay_iterations', 1)))
    progress_index = max(0, int(iteration) - get_teacher_warmup_iterations(args) - 1)

    if decay_iterations <= 1:
        ratio = final_ratio
    else:
        progress = min(1.0, progress_index / float(decay_iterations - 1))
        ratio = initial_ratio + progress * (final_ratio - initial_ratio)

    drift_threshold = float(getattr(args, 'teacher_replay_drift_threshold', -1.0))
    drift_ratio = get_teacher_drift_replay_ratio(args, iteration)
    if latest_auxiliary_win_rate is not None and latest_auxiliary_win_rate < drift_threshold:
        ratio = max(ratio, drift_ratio)

    return max(0.0, float(ratio))


def sample_teacher_replay_examples(teacher_bootstrap_examples, sample_count):
    return _sample_examples(teacher_bootstrap_examples, sample_count)


def sample_teacher_replay_examples_by_source(
    teacher_bootstrap_examples,
    teacher_opponent_history,
    sample_count,
):
    sample_count = int(sample_count)
    teacher_opponent_examples, _ = flatten_history_examples(teacher_opponent_history)

    if sample_count <= 0:
        return [], {
            'teacher_bootstrap_replay_samples': 0,
            'teacher_opponent_replay_samples': 0,
            'teacher_opponent_history_samples': len(teacher_opponent_examples),
        }

    bootstrap_available = len(teacher_bootstrap_examples)
    opponent_available = len(teacher_opponent_examples)
    total_available = bootstrap_available + opponent_available
    if total_available <= 0:
        return [], {
            'teacher_bootstrap_replay_samples': 0,
            'teacher_opponent_replay_samples': 0,
            'teacher_opponent_history_samples': 0,
        }

    if bootstrap_available <= 0:
        bootstrap_target = 0
        opponent_target = sample_count
    elif opponent_available <= 0:
        bootstrap_target = sample_count
        opponent_target = 0
    else:
        opponent_target = int(round(sample_count * (opponent_available / float(total_available))))
        opponent_target = max(0, min(sample_count, opponent_target))
        bootstrap_target = sample_count - opponent_target

        if sample_count > 1:
            if opponent_target == 0:
                opponent_target = 1
                bootstrap_target = sample_count - 1
            elif bootstrap_target == 0:
                bootstrap_target = 1
                opponent_target = sample_count - 1

    sampled_bootstrap = _sample_examples(teacher_bootstrap_examples, bootstrap_target)
    sampled_opponent = _sample_examples(teacher_opponent_examples, opponent_target)
    teacher_examples = sampled_bootstrap + sampled_opponent
    shuffle(teacher_examples)

    return teacher_examples, {
        'teacher_bootstrap_replay_samples': len(sampled_bootstrap),
        'teacher_opponent_replay_samples': len(sampled_opponent),
        'teacher_opponent_history_samples': opponent_available,
    }


def compose_training_data(
    args,
    train_examples_history,
    teacher_bootstrap_examples,
    teacher_opponent_history,
    latest_auxiliary_win_rate,
    iteration,
):
    if is_teacher_warmup_iteration(args, teacher_bootstrap_examples, iteration):
        train_data = list(teacher_bootstrap_examples)
        shuffle(train_data)
        return train_data, {
            'mode': 'teacher_bootstrap_only',
            'self_play_samples': 0,
            'teacher_replay_samples': len(train_data),
            'teacher_replay_ratio': 1.0,
            'latest_self_play_extra_samples': 0,
        }

    train_data = []
    latest_extra_samples = 0
    for idx, examples in enumerate(train_examples_history):
        if idx == len(train_examples_history) - 1:
            train_data.extend(examples)
            if args.latest_data_weight > 1 and len(examples) > 0:
                extra_count = int(len(examples) * (args.latest_data_weight - 1))
                if extra_count > 0:
                    extra_indices = np.random.choice(len(examples), extra_count, replace=extra_count > len(examples))
                    train_data.extend([examples[i] for i in extra_indices])
                    latest_extra_samples = extra_count
        else:
            train_data.extend(examples)

    self_play_samples = len(train_data)
    teacher_opponent_examples, _ = flatten_history_examples(teacher_opponent_history)
    teacher_replay_ratio = get_teacher_replay_ratio(
        args,
        teacher_bootstrap_examples,
        latest_auxiliary_win_rate,
        iteration,
        teacher_opponent_history=teacher_opponent_history,
    )
    teacher_replay_target = int(round(self_play_samples * teacher_replay_ratio))
    if self_play_samples == 0 and (teacher_bootstrap_examples or teacher_opponent_examples):
        teacher_replay_target = len(teacher_bootstrap_examples) + len(teacher_opponent_examples)

    teacher_examples, teacher_sample_stats = sample_teacher_replay_examples_by_source(
        teacher_bootstrap_examples,
        teacher_opponent_history,
        teacher_replay_target,
    )
    train_data.extend(teacher_examples)
    shuffle(train_data)

    return train_data, {
        'mode': 'mixed_self_play_teacher' if teacher_examples else 'self_play_only',
        'self_play_samples': self_play_samples,
        'teacher_replay_samples': len(teacher_examples),
        'teacher_replay_ratio': teacher_replay_ratio,
        'latest_self_play_extra_samples': latest_extra_samples,
        'teacher_bootstrap_replay_samples': teacher_sample_stats['teacher_bootstrap_replay_samples'],
        'teacher_opponent_replay_samples': teacher_sample_stats['teacher_opponent_replay_samples'],
        'teacher_opponent_history_samples': teacher_sample_stats['teacher_opponent_history_samples'],
    }


def estimate_teacher_sample_budget(args, teacher_bootstrap_examples=None, teacher_bootstrap_metadata=None):
    teacher_bootstrap_examples = teacher_bootstrap_examples or []
    teacher_bootstrap_metadata = teacher_bootstrap_metadata or {}

    buffer_samples = len(teacher_bootstrap_examples)
    if buffer_samples <= 0:
        buffer_samples = int(teacher_bootstrap_metadata.get('samples', 0) or 0)

    buffer_games = int(teacher_bootstrap_metadata.get('games', 0) or 0)
    avg_samples_per_game = None
    if buffer_samples > 0 and buffer_games > 0:
        avg_samples_per_game = buffer_samples / float(buffer_games)

    warmup_iterations = get_teacher_warmup_iterations(args)
    warmup_teacher_samples = buffer_samples * warmup_iterations if buffer_samples > 0 else None

    initial_ratio = max(0.0, float(getattr(args, 'teacher_replay_initial_ratio', 0.0)))
    final_ratio = max(0.0, float(getattr(args, 'teacher_replay_final_ratio', 0.0)))
    decay_iterations = max(1, int(getattr(args, 'teacher_replay_decay_iterations', 1)))
    if final_ratio == 0.0 and decay_iterations > 1:
        mixed_iterations_before_pure = decay_iterations - 1
    else:
        mixed_iterations_before_pure = decay_iterations

    ratio_schedule = []
    for offset in range(mixed_iterations_before_pure):
        if decay_iterations <= 1:
            ratio = final_ratio
        else:
            progress = min(1.0, offset / float(decay_iterations - 1))
            ratio = initial_ratio + progress * (final_ratio - initial_ratio)
        ratio_schedule.append(float(max(0.0, ratio)))

    estimated_self_play_samples_per_iteration = None
    estimated_teacher_replay_samples = None
    if avg_samples_per_game is not None:
        per_iteration_examples = avg_samples_per_game * float(getattr(args, 'num_self_play_games', 0))
        estimated_self_play_samples_per_iteration = per_iteration_examples
        latest_weight = float(getattr(args, 'latest_data_weight', 1.0))
        history_len = max(1, int(getattr(args, 'history_len', 1)))
        estimated_teacher_replay_samples = 0
        for mixed_index, ratio in enumerate(ratio_schedule, start=1):
            history_size = min(mixed_index, history_len)
            effective_self_play_samples = per_iteration_examples * ((history_size - 1) + latest_weight)
            estimated_teacher_replay_samples += int(round(effective_self_play_samples * ratio))

    estimated_total_teacher_samples_before_pure = None
    if warmup_teacher_samples is not None:
        estimated_total_teacher_samples_before_pure = warmup_teacher_samples
        if estimated_teacher_replay_samples is not None:
            estimated_total_teacher_samples_before_pure += estimated_teacher_replay_samples

    return {
        'buffer_samples': buffer_samples,
        'buffer_games': buffer_games,
        'avg_samples_per_game': avg_samples_per_game,
        'warmup_iterations': warmup_iterations,
        'warmup_teacher_samples': warmup_teacher_samples,
        'mixed_iterations_before_pure': mixed_iterations_before_pure,
        'ratio_schedule': ratio_schedule,
        'estimated_self_play_samples_per_iteration': estimated_self_play_samples_per_iteration,
        'estimated_teacher_replay_samples_before_pure': estimated_teacher_replay_samples,
        'estimated_total_teacher_samples_before_pure': estimated_total_teacher_samples_before_pure,
    }