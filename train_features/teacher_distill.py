from __future__ import annotations

import concurrent.futures
import multiprocessing
from types import SimpleNamespace
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CURRENT_DIR.parent
TRAINING_DIR = WORKSPACE_ROOT / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from game_rules import GameRules
from mcts import AsyncBatchInferenceManager, MCTS
from model_compat import CompatibleModelPredictor, encode_board_for_model, load_checkpoint_payload, load_compatible_model

try:
    from .feature_extractor import CandidateFeatureExtractor
except ImportError:
    from feature_extractor import CandidateFeatureExtractor


@dataclass
class TeacherSelfPlayConfig:
    model_path: str
    num_games: int = 40
    max_steps_per_game: int = 160
    temperature: float = 0.8
    opening_random_moves: int = 6
    sample_weight: float = 1.0
    device: str | None = None
    mcts_sims: int = 256
    mcts_threads: int = 4
    cpuct: float = 1.0
    virtual_loss: float = 1.0
    inference_batch_size: int = 32
    inference_timeout_s: float = 0.003
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.10
    parallel_workers: int = 0
    cpu_worker_ratio: float = 0.5


def _resolve_device(requested: str | None = None) -> str:
    text = (requested or "").strip().lower()
    has_cuda = bool(torch.cuda.is_available())
    if not text:
        return "cuda" if has_cuda else "cpu"
    if text.startswith("cuda") and not has_cuda:
        return "cpu"
    return text


def load_teacher_cache_samples(
    feature_extractor: CandidateFeatureExtractor,
    cache_path: str | Path,
    sample_weight: float = 1.0,
    max_samples: int = 0,
) -> Tuple[List[dict], Dict[str, float]]:
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Teacher cache file not found: {path}")

    payload = load_checkpoint_payload(str(path))
    metadata = {}
    games_len = 0
    if isinstance(payload, dict) and "examples" in payload:
        raw_examples = list(payload.get("examples") or [])
        metadata = dict(payload.get("metadata") or {})
        games_len = len(payload.get("games") or [])
    elif isinstance(payload, list):
        raw_examples = list(payload)
    else:
        raise ValueError(f"Unsupported teacher cache format: {path}")

    if max_samples > 0 and len(raw_examples) > int(max_samples):
        indices = np.random.choice(len(raw_examples), int(max_samples), replace=False)
        raw_examples = [raw_examples[int(i)] for i in np.atleast_1d(indices)]

    converted: List[dict] = []
    dropped = 0
    for item in raw_examples:
        sample = _convert_cache_example_to_tiny_sample(
            feature_extractor=feature_extractor,
            example=item,
            sample_weight=sample_weight,
        )
        if sample is None:
            dropped += 1
            continue
        converted.append(sample)

    stats = {
        "cache_path": str(path),
        "raw_examples": float(len(raw_examples)),
        "usable_examples": float(len(converted)),
        "dropped_examples": float(dropped),
        "cache_games": float(games_len),
        "cache_meta_samples": float(metadata.get("samples", 0) or 0),
        "cache_meta_games": float(metadata.get("games", 0) or 0),
    }
    return converted, stats


def _convert_cache_example_to_tiny_sample(
    feature_extractor: CandidateFeatureExtractor,
    example,
    sample_weight: float,
) -> dict | None:
    if not isinstance(example, (list, tuple)) or len(example) < 3:
        return None

    board = np.asarray(example[0], dtype=np.int8)
    policy = np.asarray(example[1], dtype=np.float32).reshape(-1)
    try:
        value = float(example[2])
    except Exception:
        value = 0.0

    if board.ndim != 3:
        return None

    # Distillation cache stores canonical boards from current player's perspective.
    player = 1
    feat = feature_extractor.extract(board, player)
    valid_mask = np.asarray(feat["valid_mask"], dtype=np.float32)
    action_map = np.asarray(feat["candidate_action_map"], dtype=np.int64)
    valid_indices = np.flatnonzero(valid_mask > 0.5)
    if len(valid_indices) == 0:
        return None

    candidate_probs = np.zeros((feature_extractor.candidate_count,), dtype=np.float32)
    for idx in valid_indices:
        action = int(action_map[idx])
        if 0 <= action < len(policy):
            candidate_probs[int(idx)] = float(max(0.0, policy[action]))

    candidate_probs = _normalize_with_mask(candidate_probs, valid_mask)
    target_idx = int(np.argmax(candidate_probs))

    return {
        "board": np.array(board, copy=True, dtype=np.int8),
        "player": int(player),
        "target_idx": int(target_idx),
        "weight": float(sample_weight),
        "teacher_probs": np.array(candidate_probs, copy=True, dtype=np.float32),
        "teacher_value": float(value),
    }


class TeacherPolicyOracle:
    def __init__(self, model_path: str, action_size: int, device: str | None = None) -> None:
        resolved_device = _resolve_device(device)
        self.model, self.model_config, self.metadata = load_compatible_model(model_path, device=resolved_device)
        self.device = resolved_device
        self.action_size = int(action_size)

    def predict(self, canonical_board: np.ndarray) -> Tuple[np.ndarray, float]:
        encoded = encode_board_for_model(canonical_board, self.model_config)
        tensor = torch.from_numpy(np.asarray(encoded, dtype=np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            log_pi, value = self.model(tensor)

        policy = torch.exp(log_pi).squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        policy = np.asarray(policy[: self.action_size], dtype=np.float32)
        policy = np.clip(policy, 0.0, None)
        total = float(np.sum(policy))
        if not np.isfinite(total) or total <= 0.0:
            policy = np.full((self.action_size,), 1.0 / float(self.action_size), dtype=np.float32)
        else:
            policy = policy / total

        value_scalar = float(value.squeeze().detach().cpu().item())
        return policy, value_scalar


def _generate_teacher_self_play_samples_serial(
    feature_extractor: CandidateFeatureExtractor,
    config: TeacherSelfPlayConfig,
) -> Tuple[List[dict], Dict[str, float]]:
    game = GameRules()
    oracle = TeacherPolicyOracle(
        model_path=str(config.model_path),
        action_size=int(game.get_action_size()),
        device=config.device,
    )

    if str(oracle.device).startswith("cpu") and int(config.mcts_threads) > multiprocessing.cpu_count():
        mcts_threads = max(1, multiprocessing.cpu_count())
    elif str(oracle.device).startswith("cpu"):
        mcts_threads = max(1, int(config.mcts_threads))
    else:
        mcts_threads = max(1, int(config.mcts_threads))

    predictor = CompatibleModelPredictor(oracle.model, oracle.model_config, action_size=game.get_action_size())
    inference_manager = AsyncBatchInferenceManager(
        predictor,
        batch_size=max(1, int(config.inference_batch_size)),
        batch_timeout_s=max(0.001, float(config.inference_timeout_s)),
    )
    mcts_args = SimpleNamespace(
        cpuct=float(config.cpuct),
        num_mcts_sims=max(1, int(config.mcts_sims)),
        num_mcts_threads=int(mcts_threads),
        virtual_loss=float(config.virtual_loss),
        inference_batch_size=max(1, int(config.inference_batch_size)),
        inference_timeout_s=max(0.001, float(config.inference_timeout_s)),
        dirichlet_alpha=float(config.dirichlet_alpha),
        dirichlet_epsilon=float(config.dirichlet_epsilon),
    )

    samples: List[dict] = []
    total_steps = 0
    completed_games = 0
    try:
        for _ in range(max(0, int(config.num_games))):
            board = game.get_init_board()
            player = 1
            game_steps = 0

            while game_steps < int(config.max_steps_per_game):
                canonical = game.get_canonical_form(board, player)
                mcts = MCTS(game, inference_manager, mcts_args)
                action_policy = np.asarray(
                    mcts.get_action_prob(
                        canonical,
                        temp=1.0,
                        training=True,
                        dirichlet_alpha=float(config.dirichlet_alpha),
                        dirichlet_epsilon=float(config.dirichlet_epsilon),
                    ),
                    dtype=np.float32,
                )
                _, teacher_value = oracle.predict(canonical)

                feat = feature_extractor.extract(board, player)
                valid_mask = feat["valid_mask"]
                action_map = feat["candidate_action_map"]
                valid_indices = np.flatnonzero(valid_mask > 0.5)
                if len(valid_indices) == 0:
                    break

                candidate_probs = np.zeros((feature_extractor.candidate_count,), dtype=np.float32)
                for idx in valid_indices:
                    action = int(action_map[idx])
                    if 0 <= action < len(action_policy):
                        candidate_probs[int(idx)] = float(action_policy[action])
                candidate_probs = _normalize_with_mask(candidate_probs, valid_mask)

                if game_steps < int(config.opening_random_moves):
                    chosen_idx = int(np.random.choice(valid_indices))
                else:
                    chosen_idx = _sample_candidate(
                        probs=candidate_probs,
                        valid_mask=valid_mask,
                        temperature=float(config.temperature),
                    )

                action = int(action_map[chosen_idx])
                if action < 0:
                    action = int(action_map[int(np.random.choice(valid_indices))])

                target_idx = int(chosen_idx)
                samples.append(
                    {
                        "board": np.array(board, copy=True, dtype=np.int8),
                        "player": int(player),
                        "target_idx": int(target_idx),
                        "weight": float(config.sample_weight),
                        "teacher_probs": np.array(candidate_probs, copy=True, dtype=np.float32),
                        "teacher_value": float(teacher_value),
                    }
                )

                board, player = game.get_next_state(board, player, action)
                total_steps += 1
                game_steps += 1
                ended = game.get_game_ended(board, player)
                if ended != 0:
                    completed_games += 1
                    break
    finally:
        inference_manager.close()

    stats = {
        "requested_games": float(max(0, int(config.num_games))),
        "completed_games": float(completed_games),
        "samples": float(len(samples)),
        "mcts_sims": float(max(1, int(config.mcts_sims))),
        "mcts_threads": float(max(1, int(mcts_threads))),
        "teacher_device": str(oracle.device),
        "parallel_workers": 1.0,
        "cpu_worker_ratio": float(config.cpu_worker_ratio),
        "total_steps": float(total_steps),
        "avg_steps_per_game": float(total_steps / max(1, completed_games)) if completed_games > 0 else 0.0,
    }
    return samples, stats


def _teacher_worker_generate_chunk(
    config_dict: dict,
    worker_games: int,
    board_size: int,
    max_layers: int,
    connect_n: int,
    worker_seed: int,
) -> Tuple[List[dict], Dict[str, float]]:
    np.random.seed(int(worker_seed))
    worker_cfg = TeacherSelfPlayConfig(**dict(config_dict))
    worker_cfg.num_games = int(worker_games)
    worker_cfg.parallel_workers = 1

    extractor = CandidateFeatureExtractor(
        board_size=int(board_size),
        max_layers=int(max_layers),
        connect_n=int(connect_n),
    )
    samples, stats = _generate_teacher_self_play_samples_serial(extractor, worker_cfg)
    stats["worker_games"] = float(worker_games)
    stats["worker_seed"] = float(worker_seed)
    return samples, stats


def _resolve_parallel_plan(config: TeacherSelfPlayConfig, resolved_device: str) -> Tuple[int, int, int]:
    cpu_total = max(1, int(multiprocessing.cpu_count()))
    cpu_budget = max(1, int(round(cpu_total * max(0.1, float(config.cpu_worker_ratio)))))
    total_games = max(0, int(config.num_games))
    if total_games <= 0:
        return 1, max(1, int(config.mcts_threads)), cpu_budget

    if str(resolved_device).startswith("cuda"):
        workers = 1
        mcts_threads = max(1, int(config.mcts_threads))
        return workers, mcts_threads, cpu_budget

    requested_workers = int(config.parallel_workers)
    if requested_workers > 0:
        workers = min(total_games, requested_workers)
    else:
        workers = min(total_games, cpu_budget)
    workers = max(1, workers)

    per_worker_budget = max(1, cpu_budget // workers)
    mcts_threads = max(1, min(int(config.mcts_threads), per_worker_budget))
    return workers, mcts_threads, cpu_budget


def generate_teacher_self_play_samples(
    feature_extractor: CandidateFeatureExtractor,
    config: TeacherSelfPlayConfig,
) -> Tuple[List[dict], Dict[str, float]]:
    total_games = max(0, int(config.num_games))
    if total_games <= 0:
        return [], {
            "requested_games": 0.0,
            "completed_games": 0.0,
            "samples": 0.0,
            "mcts_sims": float(max(1, int(config.mcts_sims))),
            "mcts_threads": float(max(1, int(config.mcts_threads))),
            "teacher_device": str(_resolve_device(config.device)),
            "parallel_workers": 0.0,
            "cpu_worker_ratio": float(config.cpu_worker_ratio),
            "cpu_budget_threads": float(max(1, int(multiprocessing.cpu_count()))),
            "avg_steps_per_game": 0.0,
        }

    resolved_device = _resolve_device(config.device)
    workers, mcts_threads, cpu_budget = _resolve_parallel_plan(config, resolved_device)

    effective_cfg = TeacherSelfPlayConfig(**dict(config.__dict__))
    effective_cfg.device = resolved_device
    effective_cfg.mcts_threads = int(mcts_threads)

    if workers <= 1:
        samples, stats = _generate_teacher_self_play_samples_serial(feature_extractor, effective_cfg)
        stats["parallel_workers"] = 1.0
        stats["cpu_budget_threads"] = float(cpu_budget)
        stats["cpu_total_threads"] = float(max(1, int(multiprocessing.cpu_count())))
        return samples, stats

    game_chunks = [total_games // workers for _ in range(workers)]
    for idx in range(total_games % workers):
        game_chunks[idx] += 1

    config_payload = dict(effective_cfg.__dict__)
    seed_base = int(np.random.randint(0, 2_000_000_000))
    all_samples: List[dict] = []
    worker_stats: List[Dict[str, float]] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures = []
        for worker_idx, chunk_games in enumerate(game_chunks):
            if chunk_games <= 0:
                continue
            futures.append(
                pool.submit(
                    _teacher_worker_generate_chunk,
                    config_payload,
                    int(chunk_games),
                    int(feature_extractor.board_size),
                    int(feature_extractor.max_layers),
                    int(feature_extractor.connect_n),
                    int(seed_base + worker_idx * 9973),
                )
            )

        for future in concurrent.futures.as_completed(futures):
            samples_chunk, stats_chunk = future.result()
            all_samples.extend(samples_chunk)
            worker_stats.append(stats_chunk)

    completed_games = float(sum(float(item.get("completed_games", 0.0)) for item in worker_stats))
    total_steps = float(sum(float(item.get("total_steps", 0.0)) for item in worker_stats))

    stats = {
        "requested_games": float(total_games),
        "completed_games": float(completed_games),
        "samples": float(len(all_samples)),
        "mcts_sims": float(max(1, int(effective_cfg.mcts_sims))),
        "mcts_threads": float(max(1, int(effective_cfg.mcts_threads))),
        "teacher_device": str(effective_cfg.device),
        "parallel_workers": float(workers),
        "cpu_worker_ratio": float(effective_cfg.cpu_worker_ratio),
        "cpu_budget_threads": float(cpu_budget),
        "cpu_total_threads": float(max(1, int(multiprocessing.cpu_count()))),
        "avg_steps_per_game": float(total_steps / max(1.0, completed_games)),
        "worker_stats": worker_stats,
    }
    return all_samples, stats


class TeacherSelfPlayDataset(Dataset):
    def __init__(self, samples: Sequence[dict], feature_extractor: CandidateFeatureExtractor) -> None:
        self.samples = list(samples)
        self.feature_extractor = feature_extractor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        feat = self.feature_extractor.extract(sample["board"], sample["player"])

        return {
            "global_features": torch.from_numpy(feat["global"]),
            "candidate_features": torch.from_numpy(feat["candidate"]),
            "valid_mask": torch.from_numpy(feat["valid_mask"]),
            "target_idx": torch.tensor(int(sample["target_idx"]), dtype=torch.long),
            "sample_weight": torch.tensor(float(sample.get("weight", 1.0)), dtype=torch.float32),
            "teacher_probs": torch.tensor(np.asarray(sample["teacher_probs"], dtype=np.float32)),
            "teacher_valid": torch.tensor(1.0, dtype=torch.float32),
            "teacher_value": torch.tensor(float(sample.get("teacher_value", 0.0)), dtype=torch.float32),
            "teacher_value_valid": torch.tensor(1.0, dtype=torch.float32),
        }


def _normalize_with_mask(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    masked = np.asarray(values, dtype=np.float64) * np.asarray(valid_mask, dtype=np.float64)
    masked = np.clip(masked, 0.0, None)
    total = float(np.sum(masked))
    if not np.isfinite(total) or total <= 0.0:
        valid_sum = float(np.sum(valid_mask))
        if valid_sum <= 0.0:
            return np.full((len(values),), 1.0 / float(len(values)), dtype=np.float32)
        return (np.asarray(valid_mask, dtype=np.float64) / valid_sum).astype(np.float32)
    return (masked / total).astype(np.float32)


def _sample_candidate(probs: np.ndarray, valid_mask: np.ndarray, temperature: float) -> int:
    valid_indices = np.flatnonzero(valid_mask > 0.5)
    if len(valid_indices) == 0:
        return 0

    if temperature <= 1e-6:
        return int(valid_indices[np.argmax(probs[valid_indices])])

    selected = np.asarray(probs[valid_indices], dtype=np.float64)
    selected = np.clip(selected, 1e-10, None)
    selected = selected ** (1.0 / float(temperature))
    selected = selected / np.sum(selected)
    return int(np.random.choice(valid_indices, p=selected))
