import logging
import multiprocessing
import queue
import time

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from game_rules import GameRules
from mcts import GlobalInferenceServer, MCTS, RemoteBatchInferenceClient
from model_compat import CompatibleGlobalInferenceServer


def _match_iteration_schedule_entry(schedule, iteration):
    if not schedule:
        return None

    iteration = int(iteration)
    for entry in schedule:
        start_iter = int(entry.get('start_iter', 1))
        end_iter = entry.get('end_iter')
        if iteration < start_iter:
            continue
        if end_iter is not None and iteration > int(end_iter):
            continue
        return entry
    return schedule[-1]


def _match_step_schedule_entry(schedule, step):
    if not schedule:
        return None

    step = int(step)
    for entry in schedule:
        max_step = entry.get('max_step')
        if max_step is None or step <= int(max_step):
            return entry
    return schedule[-1]


def _get_iteration_exploration_scales(args, iteration):
    strength = max(0.0, float(getattr(args, 'self_play_exploration_strength', 1.0)))
    entry = _match_iteration_schedule_entry(getattr(args, 'exploration_iteration_schedule', None), iteration) or {}
    return {
        'temperature_scale': float(strength * max(0.0, float(entry.get('temperature_scale', 1.0)))),
        'noise_scale': float(strength * max(0.0, float(entry.get('noise_scale', 1.0)))),
    }


def _get_self_play_exploration_config(args, iteration, episode_step):
    phase_entry = _match_step_schedule_entry(getattr(args, 'self_play_phase_schedule', None), episode_step) or {}
    scales = _get_iteration_exploration_scales(args, iteration)

    temperature = max(0.0, float(phase_entry.get('temperature', 0.0))) * scales['temperature_scale']
    dirichlet_alpha = max(0.0, float(phase_entry.get('dirichlet_alpha', getattr(args, 'dirichlet_alpha', 0.0))))
    dirichlet_epsilon = max(0.0, float(phase_entry.get('dirichlet_epsilon', getattr(args, 'dirichlet_epsilon', 0.0))))
    dirichlet_epsilon = min(1.0, dirichlet_epsilon * scales['noise_scale'])

    if temperature < 1e-3:
        temperature = 0.0
    if dirichlet_epsilon < 1e-4:
        dirichlet_epsilon = 0.0
        dirichlet_alpha = 0.0

    return {
        'phase_name': phase_entry.get('name', 'default'),
        'temperature': float(temperature),
        'dirichlet_alpha': float(dirichlet_alpha),
        'dirichlet_epsilon': float(dirichlet_epsilon),
    }


def mask_policy_to_valid_moves(game, canonical_board, policy):
    valid_moves = game.get_valid_moves(canonical_board).astype(np.float64)
    valid_sum = float(np.sum(valid_moves))
    if valid_sum <= 0.0:
        return np.full(game.get_action_size(), 1.0 / game.get_action_size(), dtype=np.float64)

    masked_policy = np.asarray(policy, dtype=np.float64)
    masked_policy = np.clip(masked_policy, 0.0, None) * valid_moves
    masked_sum = float(np.sum(masked_policy))
    if not np.isfinite(masked_sum) or masked_sum <= 0.0:
        return valid_moves / valid_sum
    return masked_policy / masked_sum


def _compute_policy_entropy(policy):
    probs = np.asarray(policy, dtype=np.float64)
    probs = np.clip(probs, 0.0, None)
    total = float(np.sum(probs))
    if not np.isfinite(total) or total <= 0.0:
        return 0.0
    probs = probs / total
    non_zero = probs[probs > 1e-12]
    if non_zero.size == 0:
        return 0.0
    return float(-np.sum(non_zero * np.log(non_zero)))


def _find_immediate_winning_actions(game, board, player):
    winning_actions = []
    for action in np.flatnonzero(game.get_valid_moves(board) > 0):
        next_board, _ = game.get_next_state(board, int(player), int(action))
        if game.check_win(next_board, int(player)):
            winning_actions.append(int(action))
    return winning_actions


def _build_forced_action_policy(game, actions):
    policy = np.zeros(game.get_action_size(), dtype=np.float64)
    if actions:
        weight = 1.0 / len(actions)
        for action in actions:
            policy[int(action)] = weight
    return policy


def _get_tactical_override_policy(game, board, player, episode_step, args):
    max_step = int(getattr(args, 'tactical_override_max_step', 0) or 0)
    if max_step <= 0 or int(episode_step) > max_step:
        return None

    if bool(getattr(args, 'tactical_override_prefer_win', True)):
        winning_actions = _find_immediate_winning_actions(game, board, player)
        if winning_actions:
            return _build_forced_action_policy(game, winning_actions)

    if bool(getattr(args, 'tactical_override_prefer_block', True)):
        blocking_actions = _find_immediate_winning_actions(game, board, -int(player))
        if blocking_actions:
            return _build_forced_action_policy(game, blocking_actions)

    return None


def play_self_play_game(game, predictor, args, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    mcts = MCTS(game, predictor, args)

    train_examples = []
    policy_entropy_series = []
    move_trace = []
    board = game.get_init_board()
    cur_player = 1
    episode_step = 0
    current_iteration = int(getattr(args, 'current_iteration', 1))
    filter_start_iteration = int(getattr(args, 'min_game_steps_start_iteration', 1))
    use_min_step_filter = current_iteration >= filter_start_iteration

    while True:
        episode_step += 1
        canonical_board = game.get_canonical_form(board, cur_player)

        exploration_config = _get_self_play_exploration_config(args, current_iteration, episode_step)
        temp = exploration_config['temperature']

        tactical_pi = _get_tactical_override_policy(game, board, cur_player, episode_step, args)
        if tactical_pi is not None:
            pi = tactical_pi
        else:
            pi = np.asarray(
                mcts.get_action_prob(
                    canonical_board,
                    temp=temp,
                    training=(exploration_config['dirichlet_epsilon'] > 0.0),
                    dirichlet_alpha=exploration_config['dirichlet_alpha'],
                    dirichlet_epsilon=exploration_config['dirichlet_epsilon'],
                ),
                dtype=np.float64,
            )
        pi = mask_policy_to_valid_moves(game, canonical_board, pi)
        policy_entropy_series.append(_compute_policy_entropy(pi))
        train_examples.append([canonical_board, cur_player, pi, None])

        action = np.random.choice(len(pi), p=pi)
        move_trace.append({
            'step': int(episode_step),
            'player': int(cur_player),
            'action': int(action),
        })
        board, cur_player = game.get_next_state(board, cur_player, action)

        result = game.get_game_ended(board, cur_player)
        if result == 0:
            mcts.advance_to_action(action, game.get_canonical_form(board, cur_player))
            continue

        if use_min_step_filter and episode_step < args.min_game_steps:
            search_stats = mcts.get_search_stats()
            mcts.close()
            return {
                'examples': [],
                'steps': int(episode_step),
                'used_for_training': False,
                'policy_entropy_mean': float(np.mean(policy_entropy_series)) if policy_entropy_series else 0.0,
                'policy_entropy_min': float(np.min(policy_entropy_series)) if policy_entropy_series else 0.0,
                'policy_entropy_max': float(np.max(policy_entropy_series)) if policy_entropy_series else 0.0,
                'trace': {
                    'moves': move_trace,
                    'winner': 0,
                    'is_draw': bool(result == 1e-4),
                    'result_code': float(result),
                },
                'search_stats': search_stats,
            }

        winner = 0
        if game.check_win(board, 1):
            winner = 1
        elif game.check_win(board, -1):
            winner = -1

        return_data = []
        for canonical_state, state_player, policy, _ in train_examples:
            reward = result * (1 if state_player == cur_player else -1)
            for board_sym, policy_sym in game.get_symmetries(canonical_state, policy):
                return_data.append((board_sym.astype(np.int8), policy_sym.astype(np.float32), float(reward)))
        search_stats = mcts.get_search_stats()
        mcts.close()
        return {
            'examples': return_data,
            'steps': int(episode_step),
            'used_for_training': True,
            'policy_entropy_mean': float(np.mean(policy_entropy_series)) if policy_entropy_series else 0.0,
            'policy_entropy_min': float(np.min(policy_entropy_series)) if policy_entropy_series else 0.0,
            'policy_entropy_max': float(np.max(policy_entropy_series)) if policy_entropy_series else 0.0,
            'trace': {
                'moves': move_trace,
                'winner': int(winner),
                'is_draw': bool(result == 1e-4),
                'result_code': float(result),
            },
            'search_stats': search_stats,
        }


def self_play_worker_loop(worker_id, task_queue, result_queue, request_queue, response_conn, args, base_seed):
    game = GameRules()
    predictor = RemoteBatchInferenceClient(worker_id, request_queue, response_conn)

    try:
        while True:
            game_idx = task_queue.get()
            if game_idx is None:
                break
            game_seed = int(base_seed) + int(game_idx)
            game_data = play_self_play_game(game, predictor, args, game_seed)
            result_queue.put((int(game_idx), game_data))
    finally:
        predictor.close()
        result_queue.close()
        result_queue.join_thread()


def play_evaluation_game(game, new_predictor, best_predictor, args, game_idx, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    new_model_player = 1 if int(game_idx) % 2 == 0 else -1
    board = game.get_init_board()
    cur_player = 1

    mcts_new = MCTS(game, new_predictor, args)
    mcts_best = MCTS(game, best_predictor, args) if best_predictor is not None else None

    while True:
        if cur_player == new_model_player:
            canonical_board = game.get_canonical_form(board, cur_player)
            pi = mask_policy_to_valid_moves(
                game,
                canonical_board,
                np.asarray(mcts_new.get_action_prob(canonical_board, temp=0, training=False), dtype=np.float64),
            )
            action = int(np.argmax(pi))
        elif mcts_best is not None:
            canonical_board = game.get_canonical_form(board, cur_player)
            pi = mask_policy_to_valid_moves(
                game,
                canonical_board,
                np.asarray(mcts_best.get_action_prob(canonical_board, temp=0, training=False), dtype=np.float64),
            )
            action = int(np.argmax(pi))
        else:
            valid_moves = game.get_valid_moves(game.get_canonical_form(board, cur_player))
            probabilities = valid_moves / np.sum(valid_moves)
            action = np.random.choice(len(probabilities), p=probabilities)

        board, cur_player = game.get_next_state(board, cur_player, action)
        result = game.get_game_ended(board, 1)
        if result == 0:
            next_canonical = game.get_canonical_form(board, cur_player)
            mcts_new.advance_to_action(action, next_canonical)
            if mcts_best is not None:
                mcts_best.advance_to_action(action, next_canonical)
            continue
        mcts_new.close()
        if mcts_best is not None:
            mcts_best.close()
        if result == 1e-4:
            return 'draw'
        if (result == 1 and new_model_player == 1) or (result == -1 and new_model_player == -1):
            return 'win'
        return 'loss'


def _select_action_from_policy(policy):
    policy = np.asarray(policy, dtype=np.float64)
    policy = np.clip(policy, 0.0, None)
    total = float(np.sum(policy))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError('Policy distribution is invalid for action selection.')
    policy = policy / total
    return int(np.random.choice(len(policy), p=policy))


def _get_teacher_failure_collection_config(args):
    target_temperature = float(getattr(args, 'teacher_failure_target_temperature', 0.0))
    game_temperature = float(getattr(args, 'teacher_failure_game_temperature', 0.0))
    return {
        'target_temperature': max(0.0, target_temperature),
        'game_temperature': max(0.0, game_temperature),
    }


def play_teacher_failure_game(game, model_predictor, teacher_predictor, args, game_idx, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

    model_player = 1 if int(game_idx) % 2 == 0 else -1
    board = game.get_init_board()
    cur_player = 1
    episode_step = 0
    model_mcts = MCTS(game, model_predictor, args)
    teacher_mcts = MCTS(game, teacher_predictor, args)
    collection_config = _get_teacher_failure_collection_config(args)
    collected_positions = []

    while True:
        episode_step += 1
        canonical_board = game.get_canonical_form(board, cur_player)

        if cur_player == model_player:
            teacher_policy = mask_policy_to_valid_moves(
                game,
                canonical_board,
                np.asarray(
                    teacher_mcts.get_action_prob(
                        canonical_board,
                        temp=collection_config['target_temperature'],
                        training=False,
                    ),
                    dtype=np.float64,
                ),
            )
            model_policy = mask_policy_to_valid_moves(
                game,
                canonical_board,
                np.asarray(
                    model_mcts.get_action_prob(
                        canonical_board,
                        temp=collection_config['game_temperature'],
                        training=False,
                    ),
                    dtype=np.float64,
                ),
            )
            collected_positions.append((np.array(canonical_board, copy=True), teacher_policy.astype(np.float32, copy=False)))
            action = _select_action_from_policy(model_policy)
        else:
            teacher_policy = mask_policy_to_valid_moves(
                game,
                canonical_board,
                np.asarray(
                    teacher_mcts.get_action_prob(
                        canonical_board,
                        temp=collection_config['game_temperature'],
                        training=False,
                    ),
                    dtype=np.float64,
                ),
            )
            action = _select_action_from_policy(teacher_policy)

        board, cur_player = game.get_next_state(board, cur_player, action)
        result = game.get_game_ended(board, cur_player)
        if result == 0:
            next_canonical = game.get_canonical_form(board, cur_player)
            model_mcts.advance_to_action(action, next_canonical)
            teacher_mcts.advance_to_action(action, next_canonical)
            continue

        if result == 1e-4:
            outcome_for_model = 0.0
        elif cur_player == model_player:
            outcome_for_model = float(result)
        else:
            outcome_for_model = -float(result)

        failure_examples = []
        used_for_training = bool(outcome_for_model < 0.0 and collected_positions)
        if used_for_training:
            for canonical_state, teacher_policy in collected_positions:
                for board_sym, policy_sym in game.get_symmetries(canonical_state, teacher_policy):
                    failure_examples.append(
                        (
                            board_sym.astype(np.int8),
                            policy_sym.astype(np.float32),
                            float(outcome_for_model),
                        )
                    )

        search_stats = {
            'model': model_mcts.get_search_stats(),
            'teacher': teacher_mcts.get_search_stats(),
        }
        model_mcts.close()
        teacher_mcts.close()
        return {
            'examples': failure_examples,
            'steps': int(episode_step),
            'used_for_training': used_for_training,
            'model_player': int(model_player),
            'outcome_for_model': float(outcome_for_model),
            'captured_positions': len(collected_positions),
            'search_stats': search_stats,
        }


def evaluation_worker_loop(
    worker_id,
    task_queue,
    result_queue,
    new_request_queue,
    new_response_conn,
    best_request_queue,
    best_response_conn,
    args,
    base_seed,
):
    game = GameRules()
    new_predictor = RemoteBatchInferenceClient(worker_id, new_request_queue, new_response_conn)
    best_predictor = RemoteBatchInferenceClient(worker_id, best_request_queue, best_response_conn) if best_request_queue is not None else None

    try:
        while True:
            game_idx = task_queue.get()
            if game_idx is None:
                break
            game_seed = int(base_seed) + int(game_idx)
            result = play_evaluation_game(game, new_predictor, best_predictor, args, game_idx, game_seed)
            result_queue.put((int(game_idx), result))
    finally:
        new_predictor.close()
        if best_predictor is not None:
            best_predictor.close()
        result_queue.close()
        result_queue.join_thread()


def teacher_failure_worker_loop(
    worker_id,
    task_queue,
    result_queue,
    model_request_queue,
    model_response_conn,
    teacher_request_queue,
    teacher_response_conn,
    args,
    base_seed,
):
    game = GameRules()
    model_predictor = RemoteBatchInferenceClient(worker_id, model_request_queue, model_response_conn)
    teacher_predictor = RemoteBatchInferenceClient(worker_id, teacher_request_queue, teacher_response_conn)

    try:
        while True:
            game_idx = task_queue.get()
            if game_idx is None:
                break
            game_seed = int(base_seed) + int(game_idx)
            game_data = play_teacher_failure_game(
                game,
                model_predictor,
                teacher_predictor,
                args,
                game_idx,
                game_seed,
            )
            result_queue.put((int(game_idx), game_data))
    finally:
        model_predictor.close()
        teacher_predictor.close()
        result_queue.close()
        result_queue.join_thread()


def _close_processes(worker_processes):
    for process in worker_processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)


def _resolve_self_play_server_count(args, num_workers, compatible_model_spec):
    if compatible_model_spec is not None:
        configured = int(getattr(args, 'compatible_inference_server_count', 1) or 1)
        return max(1, min(configured, num_workers))

    configured = int(getattr(args, 'shared_inference_server_count', 2) or 2)
    threshold = int(getattr(args, 'high_mcts_shared_inference_server_threshold', 0) or 0)
    boosted = int(getattr(args, 'high_mcts_shared_inference_server_count', configured) or configured)
    current_sims = int(getattr(args, 'num_mcts_sims', 0) or 0)

    if threshold > 0 and current_sims >= threshold:
        configured = max(configured, boosted)

    return max(1, min(configured, num_workers))


def _resolve_effective_batch_size(args, requested_batch_size, num_workers):
    threads = max(
        1,
        int(getattr(args, 'num_mcts_threads', 1) or 1),
        int(getattr(args, 'model_num_mcts_threads', 1) or 1),
        int(getattr(args, 'teacher_num_mcts_threads', 1) or 1),
    )
    active_request_limit = max(1, int(num_workers) * threads)
    return max(1, min(int(requested_batch_size), active_request_limit))


def _build_self_play_servers(
    model_state,
    model_config,
    compatible_model_spec,
    shared_inference_device,
    inference_batch_size,
    inference_timeout_s,
    server_count,
    inference_precision,
):
    server_count = max(1, int(server_count))
    if compatible_model_spec is None:
        server_response_conns = [dict() for _ in range(server_count)]
        servers = [
            GlobalInferenceServer(
                model_state=model_state,
                worker_response_conns=response_conns,
                model_config=model_config,
                device=shared_inference_device,
                batch_size=inference_batch_size,
                batch_timeout_s=inference_timeout_s,
                inference_precision=inference_precision,
            )
            for response_conns in server_response_conns
        ]
    else:
        server_response_conns = [dict() for _ in range(server_count)]
        action_size = GameRules().get_action_size()
        servers = [
            CompatibleGlobalInferenceServer(
                model_state=compatible_model_spec['state_dict'],
                model_config=compatible_model_spec['config'],
                action_size=action_size,
                worker_response_conns=response_conns,
                device=shared_inference_device,
                batch_size=inference_batch_size,
                batch_timeout_s=inference_timeout_s,
                inference_precision=inference_precision,
            )
            for response_conns in server_response_conns
        ]

    return servers, server_response_conns


def execute_self_play_parallel(
    args,
    num_games,
    num_workers,
    shared_inference_device,
    inference_batch_size,
    inference_timeout_s,
    model_state=None,
    model_config=None,
    compatible_model_spec=None,
    progress_desc='Self-Play',
):
    if model_state is not None and compatible_model_spec is not None:
        raise ValueError('model_state and compatible_model_spec are mutually exclusive.')
    if model_state is None and compatible_model_spec is None:
        raise ValueError('Either model_state or compatible_model_spec must be provided.')

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    worker_processes = []
    worker_specs = []
    server_count = _resolve_self_play_server_count(args, num_workers, compatible_model_spec)
    if server_count > 1 and str(shared_inference_device).startswith('cuda'):
        logging.warning(
            'Using %s inference servers for one model on a single GPU can increase VRAM use and CUDA context contention.',
            server_count,
        )
    inference_batch_size = _resolve_effective_batch_size(args, inference_batch_size, num_workers)
    servers, server_response_conns = _build_self_play_servers(
        model_state,
        model_config,
        compatible_model_spec,
        shared_inference_device,
        inference_batch_size,
        inference_timeout_s,
        server_count,
        str(getattr(args, 'inference_precision', 'fp32')),
    )

    for game_idx in range(num_games):
        task_queue.put(game_idx)
    for _ in range(num_workers):
        task_queue.put(None)

    base_seed = int(time.time())
    iteration_examples = []
    game_results = []
    server_count = len(servers)

    logging.info(
        'Spawning %s self-play workers with %s shared inference server(s) on %s (mcts_sims=%s)...',
        num_workers,
        server_count,
        shared_inference_device,
        int(getattr(args, 'num_mcts_sims', 0) or 0),
    )

    try:
        for worker_id in range(num_workers):
            worker_recv_conn, server_send_conn = mp.Pipe(duplex=False)
            server_index = worker_id % server_count
            server_response_conns[server_index][worker_id] = server_send_conn
            worker_specs.append((worker_id, worker_recv_conn, servers[server_index].request_queue))

        for server in servers:
            server.start()

        for worker_id, worker_recv_conn, server_send_queue in worker_specs:
            process = mp.Process(
                target=self_play_worker_loop,
                args=(
                    worker_id,
                    task_queue,
                    result_queue,
                    server_send_queue,
                    worker_recv_conn,
                    args,
                    base_seed + worker_id * 100000,
                ),
            )
            process.start()
            worker_processes.append(process)
            worker_recv_conn.close()

        for response_conns in server_response_conns:
            for conn in response_conns.values():
                conn.close()

        completed_games = 0
        progress_bar = tqdm(total=num_games, desc=progress_desc)
        while completed_games < num_games:
            try:
                game_idx, game_data = result_queue.get(timeout=5.0)
            except queue.Empty:
                failed_workers = [p.exitcode for p in worker_processes if p.exitcode not in (0, None)]
                if failed_workers:
                    raise RuntimeError(f'{progress_desc} worker failed with exit codes: {failed_workers}')
                continue

            game_data['game_idx'] = int(game_idx)
            game_results.append(game_data)
            completed_games += 1
            progress_bar.update(1)
        progress_bar.close()

        # Long/short-game weighting by current-iteration steps distribution.
        if game_results:
            total_games = len(game_results)
            mean_steps = float(np.mean([int(item.get('steps', 0) or 0) for item in game_results]))
            short_game_step_threshold = 10
            short_game_weight = 0.9
            long_game_indices = [
                idx
                for idx, item in enumerate(game_results)
                if bool(item.get('used_for_training', True))
                and len(item.get('examples') or []) > 0
                and int(item.get('steps', 0) or 0) >= short_game_step_threshold
                and int(item.get('steps', 0) or 0) > mean_steps
            ]
            long_game_ratio = len(long_game_indices) / float(max(1, total_games))
            if long_game_ratio > 0.5:
                long_game_weight_cap = 1.2
            elif long_game_ratio > 0.4:
                long_game_weight_cap = 1.4
            elif long_game_ratio > 0.25:
                long_game_weight_cap = 1.6
            else:
                long_game_weight_cap = 1.8

            long_game_index_set = set(long_game_indices)
            max_long_steps = max(
                [int(game_results[idx].get('steps', 0) or 0) for idx in long_game_indices],
                default=int(round(mean_steps)),
            )
            weighted_game_count = 0
            weighted_extra_samples = 0
            short_downweighted_game_count = 0
            short_reduced_samples = 0

            for idx, game_data in enumerate(game_results):
                examples = list(game_data.get('examples') or [])
                steps = int(game_data.get('steps', 0) or 0)
                used_for_training = bool(game_data.get('used_for_training', True))
                effective_examples = examples

                # Keep filter behavior independent: only downweight games that are already trainable.
                if used_for_training and examples and steps < short_game_step_threshold:
                    keep_count = int(round(len(examples) * short_game_weight))
                    keep_count = max(1, min(len(examples), keep_count))
                    if keep_count < len(examples):
                        keep_indices = np.random.choice(
                            len(examples),
                            keep_count,
                            replace=False,
                        )
                        effective_examples = [examples[int(i)] for i in np.atleast_1d(keep_indices)]
                        short_downweighted_game_count += 1
                        short_reduced_samples += (len(examples) - keep_count)

                iteration_examples.extend(effective_examples)

                long_game_weight = 1.0
                if idx in long_game_index_set and max_long_steps > mean_steps:
                    normalized = (steps - mean_steps) / float(max_long_steps - mean_steps)
                    normalized = min(1.0, max(0.0, normalized))
                    long_game_weight = 1.0 + normalized * (long_game_weight_cap - 1.0)
                    extra_count = int(round(len(effective_examples) * (long_game_weight - 1.0)))
                    if extra_count > 0:
                        extra_indices = np.random.choice(
                            len(effective_examples),
                            extra_count,
                            replace=extra_count > len(effective_examples),
                        )
                        iteration_examples.extend([effective_examples[int(i)] for i in np.atleast_1d(extra_indices)])
                        weighted_game_count += 1
                        weighted_extra_samples += extra_count

                game_data['long_game_weight'] = float(long_game_weight)
                game_data['long_game_weight_cap'] = float(long_game_weight_cap)

            short_game_count = sum(
                1
                for item in game_results
                if int(item.get('steps', 0) or 0) < short_game_step_threshold
            )
            for game_data in game_results:
                game_data['weighted_game_count'] = int(weighted_game_count)
                game_data['short_game_count'] = int(short_game_count)
                game_data['long_game_ratio'] = float(long_game_ratio)
                game_data['long_game_weighted_extra_samples'] = int(weighted_extra_samples)
                game_data['short_game_weight'] = float(short_game_weight)
                game_data['short_downweighted_game_count'] = int(short_downweighted_game_count)
                game_data['short_reduced_samples'] = int(short_reduced_samples)

        for process in worker_processes:
            process.join()
            if process.exitcode not in (0, None):
                raise RuntimeError(f'{progress_desc} worker exited with code {process.exitcode}')
    finally:
        _close_processes(worker_processes)
        for server in servers:
            server.close()
        task_queue.close()
        task_queue.join_thread()
        result_queue.close()
        result_queue.join_thread()

    return iteration_examples, game_results


def execute_evaluation_parallel(
    args,
    num_games,
    num_workers,
    shared_inference_device,
    inference_batch_size,
    inference_timeout_s,
    new_model_state,
    new_model_config=None,
    opponent_nnet_state=None,
    opponent_nnet_config=None,
    opponent_model_spec=None,
):
    if opponent_nnet_state is not None and opponent_model_spec is not None:
        raise ValueError('opponent_nnet_state and opponent_model_spec are mutually exclusive.')

    use_random_opponent = opponent_nnet_state is None and opponent_model_spec is None
    use_compatible_opponent = opponent_model_spec is not None
    logging.info(
        'Spawning %s evaluation workers with %s on %s...',
        num_workers,
        (
            'random baseline'
            if use_random_opponent
            else 'compatible shared inference' if use_compatible_opponent else 'dual shared inference'
        ),
        shared_inference_device,
    )

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    new_response_conns = {}
    best_response_conns = {}
    worker_processes = []
    inference_batch_size = _resolve_effective_batch_size(args, inference_batch_size, num_workers)
    inference_precision = str(getattr(args, 'inference_precision', 'fp32'))

    for game_idx in range(num_games):
        task_queue.put(game_idx)
    for _ in range(num_workers):
        task_queue.put(None)

    new_inference_server = GlobalInferenceServer(
        model_state=new_model_state,
        worker_response_conns=new_response_conns,
        model_config=new_model_config,
        device=shared_inference_device,
        batch_size=inference_batch_size,
        batch_timeout_s=inference_timeout_s,
        inference_precision=inference_precision,
    )

    best_inference_server = None
    if opponent_nnet_state is not None:
        best_inference_server = GlobalInferenceServer(
            model_state=opponent_nnet_state,
            worker_response_conns=best_response_conns,
            model_config=opponent_nnet_config,
            device=shared_inference_device,
            batch_size=inference_batch_size,
            batch_timeout_s=inference_timeout_s,
            inference_precision=inference_precision,
        )
    elif opponent_model_spec is not None:
        best_inference_server = CompatibleGlobalInferenceServer(
            model_state=opponent_model_spec['state_dict'],
            model_config=opponent_model_spec['config'],
            action_size=GameRules().get_action_size(),
            worker_response_conns=best_response_conns,
            device=shared_inference_device,
            batch_size=inference_batch_size,
            batch_timeout_s=inference_timeout_s,
            inference_precision=inference_precision,
        )

    base_seed = int(time.time())
    wins = 0
    losses = 0
    draws = 0

    try:
        new_worker_recv_conns = []
        best_worker_recv_conns = []
        for worker_id in range(num_workers):
            new_worker_recv_conn, new_server_send_conn = mp.Pipe(duplex=False)
            new_response_conns[worker_id] = new_server_send_conn
            new_worker_recv_conns.append(new_worker_recv_conn)

            if best_inference_server is not None:
                best_worker_recv_conn, best_server_send_conn = mp.Pipe(duplex=False)
                best_response_conns[worker_id] = best_server_send_conn
                best_worker_recv_conns.append(best_worker_recv_conn)
            else:
                best_worker_recv_conns.append(None)

        new_inference_server.start()
        if best_inference_server is not None:
            best_inference_server.start()

        for worker_id, (new_worker_recv_conn, best_worker_recv_conn) in enumerate(zip(new_worker_recv_conns, best_worker_recv_conns)):
            process = mp.Process(
                target=evaluation_worker_loop,
                args=(
                    worker_id,
                    task_queue,
                    result_queue,
                    new_inference_server.request_queue,
                    new_worker_recv_conn,
                    best_inference_server.request_queue if best_inference_server is not None else None,
                    best_worker_recv_conn,
                    args,
                    base_seed + worker_id * 100000,
                ),
            )
            process.start()
            worker_processes.append(process)
            new_worker_recv_conn.close()
            if best_worker_recv_conn is not None:
                best_worker_recv_conn.close()

        for server_conn in new_response_conns.values():
            server_conn.close()
        for server_conn in best_response_conns.values():
            server_conn.close()

        completed_games = 0
        progress_bar = tqdm(total=num_games, desc='Evaluation')
        while completed_games < num_games:
            try:
                _, result = result_queue.get(timeout=5.0)
            except queue.Empty:
                failed_workers = [p.exitcode for p in worker_processes if p.exitcode not in (0, None)]
                if failed_workers:
                    raise RuntimeError(f'Evaluation worker failed with exit codes: {failed_workers}')
                continue

            if result == 'win':
                wins += 1
            elif result == 'loss':
                losses += 1
            else:
                draws += 1
            completed_games += 1
            progress_bar.update(1)
        progress_bar.close()

        for process in worker_processes:
            process.join()
            if process.exitcode not in (0, None):
                raise RuntimeError(f'Evaluation worker exited with code {process.exitcode}')
    finally:
        _close_processes(worker_processes)
        new_inference_server.close()
        if best_inference_server is not None:
            best_inference_server.close()
        task_queue.close()
        task_queue.join_thread()
        result_queue.close()
        result_queue.join_thread()

    return wins, losses, draws


def execute_evaluation_group_parallel(
    args,
    matches,
    total_workers,
    shared_inference_device,
    inference_batch_size,
    inference_timeout_s,
    new_model_state,
    new_model_config=None,
):
    """Run several opponents with one shared candidate-model inference service."""
    matches = [dict(match) for match in matches if int(match.get('num_games', 0)) > 0]
    if not matches:
        return []

    group_count = len(matches)
    workers_per_match = max(1, int(total_workers) // group_count)
    inference_batch_size = _resolve_effective_batch_size(
        args,
        inference_batch_size,
        workers_per_match * group_count,
    )
    inference_precision = str(getattr(args, 'inference_precision', 'fp32'))
    candidate_response_conns = {}
    candidate_server = GlobalInferenceServer(
        model_state=new_model_state,
        worker_response_conns=candidate_response_conns,
        model_config=new_model_config,
        device=shared_inference_device,
        batch_size=inference_batch_size,
        batch_timeout_s=inference_timeout_s,
        inference_precision=inference_precision,
    )

    runtimes = []
    worker_processes = []
    next_worker_id = 0
    base_seed = int(time.time())
    try:
        for match_index, match in enumerate(matches):
            games = int(match['num_games'])
            local_workers = min(games, workers_per_match)
            task_queue = mp.Queue()
            result_queue = mp.Queue()
            opponent_response_conns = {}
            for game_idx in range(games):
                task_queue.put(game_idx)
            for _ in range(local_workers):
                task_queue.put(None)

            opponent_state = match.get('opponent_nnet_state')
            opponent_spec = match.get('opponent_model_spec')
            if opponent_state is not None and opponent_spec is not None:
                raise ValueError('Grouped evaluation opponent formats are mutually exclusive.')
            if opponent_state is not None:
                opponent_server = GlobalInferenceServer(
                    model_state=opponent_state,
                    worker_response_conns=opponent_response_conns,
                    model_config=match.get('opponent_nnet_config'),
                    device=shared_inference_device,
                    batch_size=inference_batch_size,
                    batch_timeout_s=inference_timeout_s,
                    inference_precision=inference_precision,
                )
            elif opponent_spec is not None:
                opponent_server = CompatibleGlobalInferenceServer(
                    model_state=opponent_spec['state_dict'],
                    model_config=opponent_spec['config'],
                    action_size=GameRules().get_action_size(),
                    worker_response_conns=opponent_response_conns,
                    device=shared_inference_device,
                    batch_size=inference_batch_size,
                    batch_timeout_s=inference_timeout_s,
                    inference_precision=inference_precision,
                )
            else:
                opponent_server = None

            worker_specs = []
            for local_worker in range(local_workers):
                worker_id = next_worker_id
                next_worker_id += 1
                candidate_recv, candidate_send = mp.Pipe(duplex=False)
                candidate_response_conns[worker_id] = candidate_send
                opponent_recv = None
                if opponent_server is not None:
                    opponent_recv, opponent_send = mp.Pipe(duplex=False)
                    opponent_response_conns[worker_id] = opponent_send
                worker_specs.append((worker_id, candidate_recv, opponent_recv))
            runtimes.append(
                {
                    'match': match,
                    'games': games,
                    'task_queue': task_queue,
                    'result_queue': result_queue,
                    'opponent_server': opponent_server,
                    'opponent_response_conns': opponent_response_conns,
                    'worker_specs': worker_specs,
                }
            )

        candidate_server.start()
        for runtime in runtimes:
            opponent_server = runtime['opponent_server']
            if opponent_server is not None:
                opponent_server.start()
            for worker_id, candidate_recv, opponent_recv in runtime['worker_specs']:
                process = mp.Process(
                    target=evaluation_worker_loop,
                    args=(
                        worker_id,
                        runtime['task_queue'],
                        runtime['result_queue'],
                        candidate_server.request_queue,
                        candidate_recv,
                        opponent_server.request_queue if opponent_server is not None else None,
                        opponent_recv,
                        args,
                        base_seed + worker_id * 100000,
                    ),
                )
                process.start()
                worker_processes.append(process)
                candidate_recv.close()
                if opponent_recv is not None:
                    opponent_recv.close()

        for conn in candidate_response_conns.values():
            conn.close()
        for runtime in runtimes:
            for conn in runtime['opponent_response_conns'].values():
                conn.close()

        grouped_results = []
        for runtime in runtimes:
            wins = losses = draws = completed = 0
            games = runtime['games']
            progress = tqdm(total=games, desc=f"Evaluation {runtime['match'].get('label', '')}")
            while completed < games:
                try:
                    _, result = runtime['result_queue'].get(timeout=5.0)
                except queue.Empty:
                    failures = [p.exitcode for p in worker_processes if p.exitcode not in (0, None)]
                    if failures:
                        raise RuntimeError(f'Grouped evaluation worker failed: {failures}')
                    continue
                wins += int(result == 'win')
                losses += int(result == 'loss')
                draws += int(result == 'draw')
                completed += 1
                progress.update(1)
            progress.close()
            grouped_results.append((wins, losses, draws))

        for process in worker_processes:
            process.join()
            if process.exitcode not in (0, None):
                raise RuntimeError(f'Grouped evaluation worker exited with code {process.exitcode}')
        return grouped_results
    finally:
        _close_processes(worker_processes)
        candidate_server.close()
        for runtime in runtimes:
            opponent_server = runtime['opponent_server']
            if opponent_server is not None:
                opponent_server.close()
            runtime['task_queue'].close()
            runtime['task_queue'].join_thread()
            runtime['result_queue'].close()
            runtime['result_queue'].join_thread()


def execute_teacher_failure_parallel(
    args,
    num_games,
    num_workers,
    shared_inference_device,
    inference_batch_size,
    inference_timeout_s,
    model_state,
    teacher_model_spec,
    progress_desc='Teacher Failure Collection',
    model_config=None,
    model_inference_batch_size=None,
    model_inference_timeout_s=None,
    teacher_inference_batch_size=None,
    teacher_inference_timeout_s=None,
    model_inference_server_count=1,
    teacher_inference_server_count=1,
):
    if teacher_model_spec is None:
        raise ValueError('teacher_model_spec must be provided for teacher failure collection.')

    task_queue = mp.Queue()
    result_queue = mp.Queue()
    worker_processes = []
    model_batch_size = _resolve_effective_batch_size(
        args, model_inference_batch_size or inference_batch_size, num_workers
    )
    teacher_batch_size = _resolve_effective_batch_size(
        args, teacher_inference_batch_size or inference_batch_size, num_workers
    )
    inference_precision = str(getattr(args, 'inference_precision', 'fp32'))

    for game_idx in range(num_games):
        task_queue.put(game_idx)
    for _ in range(num_workers):
        task_queue.put(None)

    model_servers, model_response_conns = _build_self_play_servers(
        model_state,
        model_config,
        None,
        shared_inference_device,
        model_batch_size,
        model_inference_timeout_s if model_inference_timeout_s is not None else inference_timeout_s,
        max(1, min(int(model_inference_server_count), num_workers)),
        inference_precision,
    )
    teacher_servers, teacher_response_conns = _build_self_play_servers(
        None,
        None,
        teacher_model_spec,
        shared_inference_device,
        teacher_batch_size,
        teacher_inference_timeout_s if teacher_inference_timeout_s is not None else inference_timeout_s,
        max(1, min(int(teacher_inference_server_count), num_workers)),
        inference_precision,
    )
    if (len(model_servers) > 1 or len(teacher_servers) > 1) and str(shared_inference_device).startswith('cuda'):
        logging.warning('Multiple teacher-failure inference servers may contend for single-GPU memory and contexts.')

    base_seed = int(time.time())
    iteration_examples = []
    game_results = []

    logging.info(
        'Spawning %s teacher-failure workers with dual shared inference on %s (mcts_sims=%s)...',
        num_workers,
        shared_inference_device,
        int(getattr(args, 'num_mcts_sims', 0) or 0),
    )

    try:
        worker_specs = []
        for worker_id in range(num_workers):
            model_worker_recv_conn, model_server_send_conn = mp.Pipe(duplex=False)
            teacher_worker_recv_conn, teacher_server_send_conn = mp.Pipe(duplex=False)
            model_index = worker_id % len(model_servers)
            teacher_index = worker_id % len(teacher_servers)
            model_response_conns[model_index][worker_id] = model_server_send_conn
            teacher_response_conns[teacher_index][worker_id] = teacher_server_send_conn
            worker_specs.append((
                worker_id,
                model_worker_recv_conn,
                model_servers[model_index].request_queue,
                teacher_worker_recv_conn,
                teacher_servers[teacher_index].request_queue,
            ))

        for server in model_servers + teacher_servers:
            server.start()

        for worker_id, model_worker_recv_conn, model_request_queue, teacher_worker_recv_conn, teacher_request_queue in worker_specs:
            process = mp.Process(
                target=teacher_failure_worker_loop,
                args=(
                    worker_id,
                    task_queue,
                    result_queue,
                    model_request_queue,
                    model_worker_recv_conn,
                    teacher_request_queue,
                    teacher_worker_recv_conn,
                    args,
                    base_seed + worker_id * 100000,
                ),
            )
            process.start()
            worker_processes.append(process)
            model_worker_recv_conn.close()
            teacher_worker_recv_conn.close()

        for response_conns in model_response_conns + teacher_response_conns:
            for server_conn in response_conns.values():
                server_conn.close()

        completed_games = 0
        progress_bar = tqdm(total=num_games, desc=progress_desc)
        while completed_games < num_games:
            try:
                _, game_data = result_queue.get(timeout=5.0)
            except queue.Empty:
                failed_workers = [p.exitcode for p in worker_processes if p.exitcode not in (0, None)]
                if failed_workers:
                    raise RuntimeError(f'{progress_desc} worker failed with exit codes: {failed_workers}')
                continue

            game_results.append(game_data)
            iteration_examples.extend(game_data.get('examples', []))
            completed_games += 1
            progress_bar.update(1)
        progress_bar.close()

        for process in worker_processes:
            process.join()
            if process.exitcode not in (0, None):
                raise RuntimeError(f'{progress_desc} worker exited with code {process.exitcode}')
    finally:
        _close_processes(worker_processes)
        for server in model_servers + teacher_servers:
            server.close()
        task_queue.close()
        task_queue.join_thread()
        result_queue.close()
        result_queue.join_thread()

    return iteration_examples, game_results
