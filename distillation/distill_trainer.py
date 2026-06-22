import copy
import hashlib
import importlib.util
import json
import logging
import multiprocessing
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = PROJECT_ROOT / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from game_rules import BOARD_SIZE, MAX_LAYERS, GameRules  # noqa: E402
from model import Connect4Net, board_to_channels, build_model_config  # noqa: E402
from model_compat import load_checkpoint_payload, load_compatible_model  # noqa: E402
from parallel_games import (  # noqa: E402
    execute_evaluation_group_parallel,
    execute_evaluation_parallel,
    execute_self_play_parallel,
    execute_teacher_failure_parallel,
)


SOURCE_SELF_PLAY = 0
SOURCE_TEACHER_WARMUP = 1
SOURCE_TEACHER_LOSS = 2
SOURCE_STUDENT_WIN = 3
SOURCE_TEACHER_RESPONSE = 4

CHECKPOINT_FORMAT_VERSION = 2
PACKED_HISTORY_FORMAT = "distillation_history_v1"


def _pack_history_entries(entries: object) -> Dict[str, object]:
    """Pack per-iteration examples into contiguous tensors for compact serialization."""
    packed_entries = []
    for entry_index, entry in enumerate(list(entries or [])):
        if not isinstance(entry, dict):
            raise ValueError(f"History entry {entry_index} must be a dict.")

        iteration = max(1, int(entry.get("iteration", 1) or 1))
        examples = list(entry.get("examples") or [])
        if examples:
            try:
                boards, policies, values, sources = zip(*examples)
            except ValueError as exc:
                raise ValueError(f"History entry {entry_index} contains malformed examples.") from exc

            boards_raw = np.asarray(boards)
            if boards_raw.size and not np.isin(boards_raw, (-1, 0, 1)).all():
                raise ValueError(f"History entry {entry_index} contains invalid board cell values.")
            boards_np = boards_raw.astype(np.int8, copy=False)
            policies_np = np.asarray(policies, dtype=np.float32)
            values_np = np.asarray(values, dtype=np.float32).reshape(-1)
            sources_np = np.asarray(sources, dtype=np.int64).reshape(-1)
        else:
            boards_np = np.empty((0, MAX_LAYERS, BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
            policies_np = np.empty((0, MAX_LAYERS * BOARD_SIZE * BOARD_SIZE), dtype=np.float32)
            values_np = np.empty((0,), dtype=np.float32)
            sources_np = np.empty((0,), dtype=np.int64)

        expected_board_shape = (len(examples), MAX_LAYERS, BOARD_SIZE, BOARD_SIZE)
        expected_policy_shape = (len(examples), MAX_LAYERS * BOARD_SIZE * BOARD_SIZE)
        if boards_np.shape != expected_board_shape:
            raise ValueError(
                f"History entry {entry_index} board shape mismatch: "
                f"expected {expected_board_shape}, got {boards_np.shape}."
            )
        if policies_np.shape != expected_policy_shape:
            raise ValueError(
                f"History entry {entry_index} policy shape mismatch: "
                f"expected {expected_policy_shape}, got {policies_np.shape}."
            )
        if values_np.shape != (len(examples),) or sources_np.shape != (len(examples),):
            raise ValueError(f"History entry {entry_index} value/source count mismatch.")
        if policies_np.size and not np.isfinite(policies_np).all():
            raise ValueError(f"History entry {entry_index} contains non-finite policy values.")
        if values_np.size and not np.isfinite(values_np).all():
            raise ValueError(f"History entry {entry_index} contains non-finite outcome values.")
        if sources_np.size and (np.any(sources_np < 0) or np.any(sources_np > 255)):
            raise ValueError(f"History entry {entry_index} contains an invalid source id.")

        packed_entries.append(
            {
                "iteration": iteration,
                "boards": torch.from_numpy(np.ascontiguousarray(boards_np)),
                "policies": torch.from_numpy(np.ascontiguousarray(policies_np)),
                "values": torch.from_numpy(np.ascontiguousarray(values_np)),
                "sources": torch.from_numpy(np.ascontiguousarray(sources_np.astype(np.uint8, copy=False))),
            }
        )

    return {"format": PACKED_HISTORY_FORMAT, "entries": packed_entries}


def _unpack_history_entries(payload: object) -> List[Dict[str, object]]:
    """Restore packed history using NumPy row views backed by contiguous tensors."""
    if not isinstance(payload, dict) or payload.get("format") != PACKED_HISTORY_FORMAT:
        return list(payload or [])

    restored = []
    for entry_index, packed in enumerate(list(payload.get("entries") or [])):
        if not isinstance(packed, dict):
            raise ValueError(f"Packed history entry {entry_index} must be a dict.")

        boards = torch.as_tensor(packed.get("boards"), device="cpu").to(dtype=torch.int8).contiguous()
        policies = torch.as_tensor(packed.get("policies"), device="cpu").to(dtype=torch.float32).contiguous()
        values = torch.as_tensor(packed.get("values"), device="cpu").to(dtype=torch.float32).reshape(-1).contiguous()
        sources = torch.as_tensor(packed.get("sources"), device="cpu").to(dtype=torch.uint8).reshape(-1).contiguous()
        sample_count = int(values.shape[0])

        expected_board_shape = (sample_count, MAX_LAYERS, BOARD_SIZE, BOARD_SIZE)
        expected_policy_shape = (sample_count, MAX_LAYERS * BOARD_SIZE * BOARD_SIZE)
        if tuple(boards.shape) != expected_board_shape:
            raise ValueError(
                f"Packed history entry {entry_index} board shape mismatch: "
                f"expected {expected_board_shape}, got {tuple(boards.shape)}."
            )
        if tuple(policies.shape) != expected_policy_shape:
            raise ValueError(
                f"Packed history entry {entry_index} policy shape mismatch: "
                f"expected {expected_policy_shape}, got {tuple(policies.shape)}."
            )
        if int(sources.shape[0]) != sample_count:
            raise ValueError(f"Packed history entry {entry_index} source count mismatch.")
        if policies.numel() and not bool(torch.isfinite(policies).all()):
            raise ValueError(f"Packed history entry {entry_index} contains non-finite policy values.")
        if values.numel() and not bool(torch.isfinite(values).all()):
            raise ValueError(f"Packed history entry {entry_index} contains non-finite outcome values.")

        boards_np = boards.numpy()
        policies_np = policies.numpy()
        values_np = values.numpy()
        sources_np = sources.numpy()
        examples = [
            (boards_np[index], policies_np[index], float(values_np[index]), int(sources_np[index]))
            for index in range(sample_count)
        ]
        restored.append(
            {
                "iteration": max(1, int(packed.get("iteration", 1) or 1)),
                "examples": examples,
            }
        )
    return restored


def _strip_json_comments(text: str) -> str:
    result = []
    i = 0
    in_string = False
    escaped = False

    while i < len(text):
        char = text[i]
        next_char = text[i + 1] if i + 1 < len(text) else ""

        if in_string:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            i += 1
            continue

        if char == '"':
            in_string = True
            result.append(char)
            i += 1
            continue

        if char == "/" and next_char == "/":
            i += 2
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue

        if char == "/" and next_char == "*":
            i += 2
            while i + 1 < len(text) and not (text[i] == "*" and text[i + 1] == "/"):
                i += 1
            i += 2 if i + 1 < len(text) else 1
            continue

        result.append(char)
        i += 1

    return "".join(result)


def _default_teacher_mix_schedule() -> List[Dict[str, float]]:
    return [
        {"start_iter": 1, "end_iter": 3, "teacher_ratio": 0.90},
        {"start_iter": 4, "end_iter": 10, "teacher_ratio": 0.50},
        {"start_iter": 11, "end_iter": None, "teacher_ratio": 0.20},
    ]


def _default_temperature_schedule() -> List[Dict[str, float]]:
    return [
        {"start_iter": 1, "end_iter": 3, "temperature": 0.06},
        {"start_iter": 4, "end_iter": 10, "temperature": 0.18},
        {"start_iter": 11, "end_iter": None, "temperature": 0.28},
    ]


def _default_student_mcts_schedule() -> List[Tuple[int, Optional[int], int]]:
    return [
        (1, 3, 128),
        (4, 10, 192),
        (11, None, 256),
    ]


def _default_self_play_mcts_schedule() -> List[Tuple[int, Optional[int], int]]:
    return [
        (1, None, 256),
    ]


def _default_student_self_play_search() -> Dict[str, object]:
    return {}


def _default_teacher_cache_search() -> Dict[str, object]:
    return {}


def _default_teacher_adversarial_search() -> Dict[str, object]:
    return {}


def _default_learning_rate_schedule() -> List[Dict[str, float]]:
    return []


def _prune_history_to_limit(history: List[object], limit: int) -> List[object]:
    limit = int(limit)
    if limit <= 0:
        return []
    if len(history) <= limit:
        return list(history)
    return list(history[-limit:])


@dataclass
class DistillationArgs:
    seed: int = 42

    run_name: str = "distill_run"
    checkpoint_root: str = "distillation/checkpoints"
    checkpoint_interval: int = 2
    eval_interval: int = 2
    max_checkpoints: int = 4

    num_iterations: int = 50
    num_self_play_games: int = 200
    batch_size: int = 1024 if torch.cuda.is_available() else 256
    epochs: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    lr_decay_step_size: int = 20
    lr_decay_gamma: float = 0.8
    learning_rate_schedule: List[Dict[str, float]] = field(default_factory=_default_learning_rate_schedule)
    history_window_len: int = 20
    self_play_history_window_len: int = 0
    adversarial_history_window_len: int = 0
    teacher_history_window_len: int = 0

    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    self_play_loss_weight: float = 0.8
    teacher_loss_decay_start: int = 4
    teacher_loss_decay_end: int = 12
    teacher_loss_floor: float = 0.2

    teacher_mix_schedule: List[Dict[str, float]] = field(default_factory=_default_teacher_mix_schedule)
    exploration_temperature_schedule: List[Dict[str, float]] = field(default_factory=_default_temperature_schedule)
    self_play_mcts_schedule: List[Tuple[int, Optional[int], int]] = field(default_factory=_default_self_play_mcts_schedule)
    student_mcts_schedule: List[Tuple[int, Optional[int], int]] = field(default_factory=_default_student_mcts_schedule)
    eval_mcts_sims: int = 256
    student_self_play_search: Dict[str, object] = field(default_factory=_default_student_self_play_search)
    teacher_cache_search: Dict[str, object] = field(default_factory=_default_teacher_cache_search)
    teacher_adversarial_search: Dict[str, object] = field(default_factory=_default_teacher_adversarial_search)

    balanced_model_preset: Dict[str, int] = field(default_factory=lambda: {"num_channels": 224, "num_res_blocks": 4})
    fast_model_preset: Dict[str, int] = field(default_factory=lambda: {"num_channels": 128, "num_res_blocks": 3})
    active_model_preset: str = "balanced"

    teacher_model_path: str = "training/checkpoints/best_new.pth.tar"
    v21_high_model_path: str = "save_model (old v2.1)/High/best.pth.tar"

    teacher_num_mcts_sims: int = 1024
    teacher_label_temperature: float = 0.06
    hot_start_iterations: int = 4
    hot_start_teacher_games_per_iteration: int = 250
    adversarial_loss_teacher_weight: float = 1.0
    adversarial_win_student_weight: float = 0.35
    adversarial_win_teacher_response_weight: float = 0.35
    teacher_sample_ratio: float = 0.6
    teacher_key_position_ratio: float = 0.5
    teacher_replay_decay_iterations: int = 20

    teacher_data_generation_enabled: bool = False
    teacher_data_generation_games: int = 1000
    teacher_data_generate_mode: bool = False
    teacher_data_cache_path: str = "distillation/cache/teacher_examples.pth.tar"
    teacher_data_batch_size: int = 4096
    teacher_data_regenerate: bool = False
    teacher_data_refresh_policy: str = "manual"

    resume: bool = False
    resume_path: Optional[str] = None
    resume_weights_only: bool = False
    rollback_iteration: Optional[int] = None
    continue_from_iteration: Optional[int] = None
    restore_optimizer_state: bool = True
    restore_schedule_state: bool = True
    force_overwrite: bool = False

    best_eval_games_per_generation: int = 30
    best_eval_required_generations: int = 2
    best_update_threshold: float = 0.55
    best_eval_parallelize_generations: bool = True
    baseline_eval_parallelize: bool = True
    shared_evaluation_services: bool = True
    eval_games_vs_teacher: int = 30
    eval_games_vs_v21_high: int = 30
    enable_best_refresh: bool = True
    info_log_name: str = "train_info.log"

    drift_no_refresh_patience: int = 4
    drift_min_win_rate: float = 0.35
    recovery_teacher_mix_ratio: float = 0.60
    recovery_teacher_weight: float = 0.80
    recovery_temperature_cap: float = 0.15
    recovery_boost_iterations: int = 3

    enable_speed_check: bool = False
    speed_check_iterations: List[int] = field(default_factory=lambda: [10, -1])

    train_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    shared_inference_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    self_play_workers: int = min(64, max(1, multiprocessing.cpu_count()))
    max_self_play_workers: int = max(1, multiprocessing.cpu_count())
    self_play_cpu_ratio: float = 0.75
    search_thread_budget: int = 0

    cpuct: float = 1.0
    num_mcts_threads: int = 4
    virtual_loss: float = 1.0
    reuse_mcts_tree: bool = True
    persistent_mcts_threads: bool = True
    enable_mcts_search_stats: bool = True
    inference_batch_size: int = 64
    inference_timeout_s: float = 0.003
    inference_precision: str = "fp32"
    compact_training_dataset: bool = True
    shared_inference_server_count: int = 1
    compatible_inference_server_count: int = 1
    high_mcts_shared_inference_server_threshold: int = 1024
    high_mcts_shared_inference_server_count: int = 1

    dirichlet_alpha: float = 0.30
    dirichlet_epsilon: float = 0.25
    self_play_exploration_strength: float = 1.0
    min_game_steps: int = 1
    min_game_steps_start_iteration: int = 999999

    def to_dict(self) -> Dict[str, object]:
        return {k: copy.deepcopy(v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, values: Dict[str, object]) -> "DistillationArgs":
        args = cls()
        unknown_keys = [
            key for key in values
            if not hasattr(args, key) and not str(key).startswith("_comment")
        ]
        if unknown_keys:
            raise ValueError(f"Unknown distillation config keys: {', '.join(sorted(unknown_keys))}")
        for key, value in values.items():
            if not hasattr(args, key):
                continue
            setattr(args, key, value)
        return args

    @classmethod
    def from_json_file(cls, file_path: str) -> "DistillationArgs":
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        values = json.loads(_strip_json_comments(raw_text))
        return cls.from_dict(values)

    @classmethod
    def from_python_file(cls, file_path: str) -> "DistillationArgs":
        module_name = f"distill_config_{abs(hash(str(file_path)))}"
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Unable to import python config: {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        values = None
        for key in ("CONFIG", "config", "DISTILL_CONFIG", "distill_config"):
            candidate = getattr(module, key, None)
            if isinstance(candidate, dict):
                values = candidate
                break

        if values is None and hasattr(module, "get_config"):
            candidate = module.get_config()
            if isinstance(candidate, dict):
                values = candidate

        if values is None:
            raise ValueError(
                "Python config must provide a dict via CONFIG/config/DISTILL_CONFIG/distill_config or get_config()."
            )

        return cls.from_dict(values)

    @classmethod
    def from_config_file(cls, file_path: str) -> "DistillationArgs":
        suffix = Path(file_path).suffix.lower()
        if suffix == ".py":
            return cls.from_python_file(file_path)
        return cls.from_json_file(file_path)


class DistillationDataset(Dataset):
    def __init__(self, examples: List[Tuple[np.ndarray, np.ndarray, float, int]], compact_boards: bool = True):
        if not examples:
            board_shape = (0, MAX_LAYERS, BOARD_SIZE, BOARD_SIZE)
            if not compact_boards:
                board_shape = (0, 2, MAX_LAYERS, BOARD_SIZE, BOARD_SIZE)
            self.boards = torch.empty(board_shape, dtype=torch.int8 if compact_boards else torch.float32)
            self.pis = torch.empty((0, MAX_LAYERS * BOARD_SIZE * BOARD_SIZE), dtype=torch.float32)
            self.vs = torch.empty((0, 1), dtype=torch.float32)
            self.sources = torch.empty((0,), dtype=torch.int64)
            return

        boards, pis, vs, sources = zip(*examples)
        boards_np = np.ascontiguousarray(np.asarray(boards, dtype=np.int8))
        pis_np = np.asarray(pis, dtype=np.float32)
        vs_np = np.asarray(vs, dtype=np.float32).reshape(-1, 1)
        src_np = np.asarray(sources, dtype=np.int64)

        if compact_boards:
            self.boards = torch.from_numpy(boards_np)
        else:
            encoded = np.stack([board_to_channels(board) for board in boards_np], axis=0).astype(np.float32)
            self.boards = torch.from_numpy(encoded)
        self.pis = torch.from_numpy(pis_np)
        self.vs = torch.from_numpy(vs_np)
        self.sources = torch.from_numpy(src_np)

    def __len__(self) -> int:
        return int(self.vs.shape[0])

    def __getitem__(self, idx: int):
        return self.boards[idx], self.pis[idx], self.vs[idx], self.sources[idx]


class DistillationTrainer:
    def __init__(self, args: DistillationArgs):
        self.args = args
        self._setup_logging()
        self._set_seed(self.args.seed)

        if self.args.train_device.startswith("cuda") and not torch.cuda.is_available():
            logging.warning("CUDA unavailable, fallback to CPU for training and shared inference.")
            self.args.train_device = "cpu"
            self.args.shared_inference_device = "cpu"
        self._runtime_preflight()

        self.game = GameRules()
        self.student_model_config = self._resolve_student_model_config()
        self.student = self._build_student_model().to(self.args.train_device)
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=float(self.args.learning_rate),
            weight_decay=float(self.args.weight_decay),
        )
        self.grad_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.args.train_device.startswith("cuda"),
        )
        self._parsed_learning_rate_schedule = self._parse_learning_rate_schedule(
            getattr(self.args, "learning_rate_schedule", None)
        )
        self._use_learning_rate_schedule_table = bool(self._parsed_learning_rate_schedule)
        self.scheduler = None
        if not self._use_learning_rate_schedule_table:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=max(1, int(self.args.lr_decay_step_size)),
                gamma=float(self.args.lr_decay_gamma),
            )

        self.project_root = PROJECT_ROOT
        self.run_dir = self._resolve_run_dir()
        self._prepare_run_dir()
        self.info_log_path = self.run_dir / str(getattr(self.args, "info_log_name", "train_info.log"))

        self.teacher_model_spec = self._load_compatible_model_spec(self.args.teacher_model_path, "teacher", required=False)
        self.v21_model_spec = self._load_compatible_model_spec(self.args.v21_high_model_path, "v2.1_high", required=False)

        self.teacher_cache_examples: List[Tuple[np.ndarray, np.ndarray, float]] = []
        self.teacher_cache_games: List[List[Tuple[np.ndarray, np.ndarray, float]]] = []
        self.teacher_cache_game_summaries: List[Dict[str, object]] = []
        self.teacher_cache_metadata: Dict[str, object] = {}
        self.teacher_history_pool: List[Tuple[np.ndarray, np.ndarray, float, int]] = []
        self.self_play_examples_history: List[Dict[str, object]] = []
        self.adversarial_examples_history: List[Dict[str, object]] = []
        self.teacher_pure_examples_history: List[Dict[str, object]] = []

        self.start_iteration = 1
        self.last_checkpoint_iteration = 0
        self.eval_history: List[Dict[str, object]] = []
        self.iteration_metrics_history: List[Dict[str, object]] = []

        self.best_recent_state: Optional[Dict[str, object]] = None
        self.best_older_state: Optional[Dict[str, object]] = None
        self.no_refresh_streak: int = 0
        self.recovery_until_iteration: int = 0

        self._ensure_teacher_cache_ready()
        resumed = self._maybe_resume_from_checkpoint()
        self._initialize_info_log(resumed=resumed)

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def _append_info_log(self, lines):
        if isinstance(lines, str):
            lines = [lines]
        if not lines:
            return
        self.info_log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.info_log_path, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(str(line).rstrip("\n") + "\n")

    def _initialize_info_log(self, resumed: bool):
        self.info_log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.info_log_path.exists():
            with open(self.info_log_path, "w", encoding="utf-8") as f:
                f.write("=== Distillation Training Info Log ===\n")

        mode = "Resume" if resumed else "Start"
        self._append_info_log(
            [
                f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {mode} distillation session",
                (
                    f"run_dir={self.run_dir} | train_device={self.args.train_device} | "
                    f"shared_inference_device={self.args.shared_inference_device} | "
                    f"model_preset={self.args.active_model_preset} | "
                    f"iterations={self.args.num_iterations} | self_play_games={self.args.num_self_play_games}"
                ),
                (
                    f"hot_start_iterations={int(self.args.hot_start_iterations)} | "
                    f"hot_start_teacher_games_per_iteration={int(self.args.hot_start_teacher_games_per_iteration)} | "
                    f"teacher_cache={self._teacher_cache_file()}"
                ),
                (
                    f"history_window_len={int(getattr(self.args, 'history_window_len', 20) or 20)} | "
                    f"self_play_window={self._resolve_history_window_limit('self_play')} | "
                    f"adversarial_window={self._resolve_history_window_limit('adversarial')} | "
                    f"teacher_window={self._resolve_history_window_limit('teacher')}"
                ),
                (
                    f"best_eval_games_per_generation={int(self.args.best_eval_games_per_generation)} | "
                    f"best_eval_required_generations={int(self.args.best_eval_required_generations)} | "
                    f"best_update_threshold={float(self.args.best_update_threshold):.3f} | "
                    f"eval_games_vs_teacher={int(self.args.eval_games_vs_teacher)} | "
                    f"eval_games_vs_v21_high={int(self.args.eval_games_vs_v21_high)}"
                ),
            ]
        )

    def _set_seed(self, seed: int):
        seed = int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _runtime_preflight(self):
        precision = str(getattr(self.args, "inference_precision", "fp32")).lower()
        if precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError(f"Unsupported inference_precision: {precision}")
        inference_device = str(self.args.shared_inference_device)
        if precision != "fp32" and not inference_device.startswith("cuda"):
            raise ValueError(f"inference_precision={precision} requires a CUDA inference device.")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 inference was requested but this CUDA device does not support BF16.")

        cpu_count = max(1, multiprocessing.cpu_count())
        thread_budget = int(getattr(self.args, "search_thread_budget", 0) or 0) or cpu_count
        mcts_threads = max(1, int(getattr(self.args, "num_mcts_threads", 1) or 1))
        worker_cap = max(1, thread_budget // mcts_threads)
        server_count = int(getattr(self.args, "shared_inference_server_count", 1) or 1)
        if server_count > 1 and inference_device.startswith("cuda"):
            logging.warning(
                "Single-GPU preflight: shared_inference_server_count=%s may waste VRAM and contend for CUDA contexts.",
                server_count,
            )
        logging.info(
            "Runtime preflight | cpu=%s search_thread_budget=%s mcts_threads=%s worker_cap=%s "
            "inference_device=%s precision=%s servers=%s tree_reuse=%s persistent_threads=%s",
            cpu_count,
            thread_budget,
            mcts_threads,
            worker_cap,
            inference_device,
            precision,
            server_count,
            bool(getattr(self.args, "reuse_mcts_tree", True)),
            bool(getattr(self.args, "persistent_mcts_threads", True)),
        )

    @staticmethod
    def _file_fingerprint(path_value: object) -> Optional[str]:
        path = Path(str(path_value)) if path_value else None
        if path is None or not path.exists() or not path.is_file():
            return None
        digest = hashlib.sha256()
        with open(path, "rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _resolve_student_model_config(self) -> Dict[str, int]:
        active = str(self.args.active_model_preset).strip().lower()
        if active not in {"balanced", "fast"}:
            raise ValueError("active_model_preset must be 'balanced' or 'fast'.")

        preset = self.args.balanced_model_preset if active == "balanced" else self.args.fast_model_preset
        num_channels = int(preset.get("num_channels", 256))
        num_res_blocks = int(preset.get("num_res_blocks", 8))
        return build_model_config(
            board_layers=MAX_LAYERS,
            board_size=BOARD_SIZE,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
        )

    def _build_student_model(self) -> Connect4Net:
        return Connect4Net(
            board_layers=int(self.student_model_config["board_layers"]),
            board_size=int(self.student_model_config["board_size"]),
            num_channels=int(self.student_model_config["num_channels"]),
            num_res_blocks=int(self.student_model_config["num_res_blocks"]),
        )

    def _resolve_run_dir(self) -> Path:
        checkpoint_root = Path(self.args.checkpoint_root)
        if not checkpoint_root.is_absolute():
            checkpoint_root = self.project_root / checkpoint_root

        run_dir_name = f"{self.args.run_name}_{self.args.active_model_preset}"
        return checkpoint_root / run_dir_name

    def _prepare_run_dir(self):
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            has_resume_intent = bool(self.args.resume or self.args.resume_path or self.args.rollback_iteration)
            if not has_resume_intent:
                if self.args.force_overwrite:
                    shutil.rmtree(self.run_dir)
                else:
                    raise FileExistsError(
                        f"Run directory already exists and is not empty: {self.run_dir}. "
                        "Use resume flags or set force_overwrite=true."
                    )
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def _to_abs_path(self, path_value: Optional[str]) -> Optional[Path]:
        if not path_value:
            return None
        path = Path(path_value)
        if not path.is_absolute():
            path = self.project_root / path
        return path

    def _load_compatible_model_spec(self, model_path: str, label: str, required: bool) -> Optional[Dict[str, object]]:
        path = self._to_abs_path(model_path)
        if path is None or not path.exists():
            if required:
                raise FileNotFoundError(f"{label} model not found: {path}")
            logging.warning("%s model path missing, skip: %s", label, path)
            return None

        model, config, metadata = load_compatible_model(str(path), device="cpu")
        state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        return {
            "label": label,
            "path": str(path),
            "state_dict": state_dict,
            "config": dict(config),
            "metadata": dict(metadata),
        }

    def _resolve_worker_count(self, total_games: int, num_mcts_threads: Optional[int] = None) -> int:
        explicit = int(getattr(self.args, "self_play_workers", 0) or 0)
        cpu_count = max(1, multiprocessing.cpu_count())
        if explicit > 0:
            target = explicit
        else:
            target = int(cpu_count * float(self.args.self_play_cpu_ratio))
        search_budget = int(getattr(self.args, "search_thread_budget", 0) or 0)
        if search_budget <= 0:
            search_budget = cpu_count
        threads_per_game = max(1, int(num_mcts_threads or getattr(self.args, "num_mcts_threads", 1) or 1))
        search_limited_workers = max(1, search_budget // threads_per_game)
        return max(
            1,
            min(
                int(self.args.max_self_play_workers),
                target,
                int(total_games),
                search_limited_workers,
            ),
        )

    def _resolve_history_window_limit(self, source_bucket: str) -> int:
        global_limit = int(getattr(self.args, "history_window_len", 0) or 0)
        if source_bucket == "self_play":
            bucket_limit = int(getattr(self.args, "self_play_history_window_len", 0) or 0)
        elif source_bucket == "adversarial":
            bucket_limit = int(getattr(self.args, "adversarial_history_window_len", 0) or 0)
        elif source_bucket == "teacher":
            bucket_limit = int(getattr(self.args, "teacher_history_window_len", 0) or 0)
        else:
            bucket_limit = 0

        effective = bucket_limit if bucket_limit > 0 else global_limit
        return max(1, int(effective))

    def _normalize_history_entries(
        self,
        entries: object,
        checkpoint_iteration: Optional[int] = None,
        inferred_iterations: Optional[List[int]] = None,
    ) -> List[Dict[str, object]]:
        raw_entries = list(entries or [])
        normalized: List[Dict[str, object]] = []
        estimated_start_iter = None
        if checkpoint_iteration is not None:
            estimated_start_iter = int(checkpoint_iteration) - len(raw_entries) + 1
        inferred = list(inferred_iterations or [])

        for idx, item in enumerate(raw_entries):
            if isinstance(item, dict) and "examples" in item:
                chunk = list(item.get("examples") or [])
                entry_iter_raw = item.get("iteration")
            else:
                chunk = list(item or [])
                entry_iter_raw = None

            if not chunk:
                continue

            if entry_iter_raw is None:
                if idx < len(inferred):
                    entry_iter = int(inferred[idx])
                else:
                    entry_iter = (estimated_start_iter + idx) if estimated_start_iter is not None else 0
            else:
                try:
                    entry_iter = int(entry_iter_raw)
                except Exception:
                    if idx < len(inferred):
                        entry_iter = int(inferred[idx])
                    else:
                        entry_iter = (estimated_start_iter + idx) if estimated_start_iter is not None else 0

            normalized.append(
                {
                    "iteration": max(0, int(entry_iter)),
                    "examples": chunk,
                }
            )
        return normalized

    def _infer_legacy_history_iterations(
        self,
        payload: Dict[str, object],
        source_bucket: str,
        entry_count: int,
        checkpoint_iteration: int,
    ) -> List[int]:
        entry_count = max(0, int(entry_count))
        if entry_count <= 0:
            return []

        key_map = {
            "self_play": "current_self_play_samples",
            "adversarial": "current_adversarial_samples",
            "teacher": "current_teacher_pure_samples",
        }
        sample_key = key_map.get(source_bucket)
        if sample_key is None:
            return []

        inferred: List[int] = []
        metrics_history = list(payload.get("iteration_metrics_history") or [])
        for item in metrics_history:
            if not isinstance(item, dict):
                continue
            iter_id_raw = item.get("iteration")
            try:
                iter_id = int(iter_id_raw)
            except Exception:
                continue

            data_mix = item.get("data_mix") or {}
            if not isinstance(data_mix, dict):
                continue
            try:
                added = int(data_mix.get(sample_key, 0) or 0)
            except Exception:
                added = 0
            if added > 0:
                inferred.append(iter_id)

        if inferred:
            if len(inferred) >= entry_count:
                return inferred[-entry_count:]

            # If metrics were partially missing, left-pad with best-effort recent range.
            pad_count = entry_count - len(inferred)
            start = max(1, checkpoint_iteration - entry_count + 1)
            pad = list(range(start, start + pad_count))
            return pad + inferred

        # Fallback when metrics history is unavailable.
        start = max(1, checkpoint_iteration - entry_count + 1)
        return list(range(start, start + entry_count))

    def _prune_history_entries(
        self,
        history: object,
        source_bucket: str,
        current_iteration: Optional[int] = None,
    ) -> List[Dict[str, object]]:
        limit = self._resolve_history_window_limit(source_bucket)
        normalized = self._normalize_history_entries(history)

        if current_iteration is None:
            return _prune_history_to_limit(normalized, limit)

        now_iter = max(1, int(current_iteration))
        kept: List[Dict[str, object]] = []
        for entry in normalized:
            entry_iter = int(entry.get("iteration", 0) or 0)
            age = (now_iter - entry_iter + 1) if entry_iter > 0 else now_iter
            if age <= limit:
                kept.append(entry)

        return _prune_history_to_limit(kept, limit)

    def _flatten_tagged_history(
        self,
        history: List[Dict[str, object]],
    ) -> List[Tuple[np.ndarray, np.ndarray, float, int]]:
        merged: List[Tuple[np.ndarray, np.ndarray, float, int]] = []
        for entry in list(history or []):
            merged.extend(list(entry.get("examples") or []))
        return merged

    def _history_iteration_span(self, history: List[Dict[str, object]]) -> str:
        if not history:
            return "-"
        values = []
        for entry in history:
            try:
                values.append(int(entry.get("iteration", 0) or 0))
            except Exception:
                continue
        values = [v for v in values if v > 0]
        if not values:
            return "-"
        return f"{min(values)}-{max(values)}"

    def _refresh_teacher_history_pool(self):
        self.teacher_history_pool = self._flatten_tagged_history(self.teacher_pure_examples_history)

    def _prune_history_windows(self, current_iteration: Optional[int] = None):
        self.self_play_examples_history = self._prune_history_entries(
            self.self_play_examples_history,
            "self_play",
            current_iteration=current_iteration,
        )
        self.adversarial_examples_history = self._prune_history_entries(
            self.adversarial_examples_history,
            "adversarial",
            current_iteration=current_iteration,
        )
        self.teacher_pure_examples_history = self._prune_history_entries(
            self.teacher_pure_examples_history,
            "teacher",
            current_iteration=current_iteration,
        )
        self._refresh_teacher_history_pool()

    def _append_tagged_history(
        self,
        source_bucket: str,
        examples: List[Tuple[np.ndarray, np.ndarray, float, int]],
        iteration: int,
    ) -> bool:
        tagged = list(examples or [])
        if not tagged:
            return False

        if source_bucket == "self_play":
            target = self.self_play_examples_history
        elif source_bucket == "adversarial":
            target = self.adversarial_examples_history
        elif source_bucket == "teacher":
            target = self.teacher_pure_examples_history
        else:
            raise ValueError(f"Unknown history bucket: {source_bucket}")

        target.append(
            {
                "iteration": max(1, int(iteration)),
                "examples": tagged,
            }
        )
        before_prune = len(target)
        pruned = self._prune_history_entries(target, source_bucket, current_iteration=iteration)

        if source_bucket == "self_play":
            self.self_play_examples_history = pruned
        elif source_bucket == "adversarial":
            self.adversarial_examples_history = pruned
        else:
            self.teacher_pure_examples_history = pruned
            self._refresh_teacher_history_pool()

        return len(pruned) < before_prune

    def _collect_windowed_training_examples(self, iteration: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float, int]], Dict[str, int]]:
        # Age-based pruning makes sparse buckets (teacher/adversarial) naturally expire by iteration.
        self._prune_history_windows(current_iteration=iteration)

        self_play_pool = self._flatten_tagged_history(self.self_play_examples_history)
        adversarial_pool = self._flatten_tagged_history(self.adversarial_examples_history)
        teacher_pool = self._flatten_tagged_history(self.teacher_pure_examples_history)

        merged = list(self_play_pool)
        merged.extend(adversarial_pool)
        merged.extend(teacher_pool)

        fallback_samples = 0
        if not merged and self.teacher_history_pool:
            fallback_count = min(len(self.teacher_history_pool), max(1, int(self.args.teacher_data_batch_size)))
            indices = np.random.choice(
                len(self.teacher_history_pool),
                fallback_count,
                replace=fallback_count > len(self.teacher_history_pool),
            )
            merged.extend(self.teacher_history_pool[int(i)] for i in np.atleast_1d(indices))
            fallback_samples = int(fallback_count)

        random.shuffle(merged)
        return merged, {
            "window_self_play_entries": int(len(self.self_play_examples_history)),
            "window_adversarial_entries": int(len(self.adversarial_examples_history)),
            "window_teacher_entries": int(len(self.teacher_pure_examples_history)),
            "window_self_play_iter_span": self._history_iteration_span(self.self_play_examples_history),
            "window_adversarial_iter_span": self._history_iteration_span(self.adversarial_examples_history),
            "window_teacher_iter_span": self._history_iteration_span(self.teacher_pure_examples_history),
            "window_self_play_samples": int(len(self_play_pool)),
            "window_adversarial_samples": int(len(adversarial_pool)),
            "window_teacher_samples": int(len(teacher_pool)),
            "window_total_samples": int(len(merged)),
            "fallback_history_samples": int(fallback_samples),
        }

    def _count_examples_by_source(self, examples: List[Tuple[np.ndarray, np.ndarray, float, int]]) -> Dict[int, int]:
        counts = {
            SOURCE_SELF_PLAY: 0,
            SOURCE_TEACHER_WARMUP: 0,
            SOURCE_TEACHER_LOSS: 0,
            SOURCE_STUDENT_WIN: 0,
            SOURCE_TEACHER_RESPONSE: 0,
        }
        for _, _, _, source in list(examples or []):
            source_id = int(source)
            counts[source_id] = counts.get(source_id, 0) + 1
        return counts

    def _match_schedule_entry(self, schedule: List[Dict[str, float]], iteration: int) -> Optional[Dict[str, float]]:
        if not schedule:
            return None
        # If the current iteration is before the first configured stage,
        # use the first stage instead of accidentally falling back to the last one.
        first_start = int(schedule[0].get("start_iter", 1))
        if iteration < first_start:
            return schedule[0]
        for entry in schedule:
            start_iter = int(entry.get("start_iter", 1))
            end_iter = entry.get("end_iter")
            if iteration < start_iter:
                continue
            if end_iter is not None and iteration > int(end_iter):
                continue
            return entry
        return schedule[-1]

    def _get_search_profile(self, profile_name: str) -> Dict[str, object]:
        profile = getattr(self.args, profile_name, None)
        if isinstance(profile, dict):
            return profile
        return {}

    def _resolve_profile_value(
        self,
        profile_name: str,
        key: str,
        *,
        fallback_attr: Optional[str] = None,
        default: Optional[object] = None,
    ) -> object:
        profile = self._get_search_profile(profile_name)
        value = profile.get(key, None)
        if value is None and fallback_attr is not None:
            value = getattr(self.args, fallback_attr, None)
        if value is None:
            value = default
        return value

    def _resolve_mcts_sims_from_schedule(self, schedule: object, iteration: int, default_value: int) -> int:
        entries = list(schedule or [])
        if not entries:
            return int(default_value)

        parsed = []
        for item in entries:
            if isinstance(item, dict):
                start_iter = int(item.get("start_iter", 1))
                end_iter = item.get("end_iter")
                sims = item.get("mcts_sims", item.get("num_mcts_sims", item.get("sims", default_value)))
            else:
                try:
                    start_iter, end_iter, sims = item
                except Exception:
                    continue
            parsed.append((int(start_iter), None if end_iter is None else int(end_iter), int(sims)))

        if not parsed:
            return int(default_value)

        first_start = int(parsed[0][0])
        if iteration < first_start:
            return int(parsed[0][2])

        for start_iter, end_iter, sims in parsed:
            if iteration < int(start_iter):
                continue
            if end_iter is not None and iteration > int(end_iter):
                continue
            return int(sims)
        return int(parsed[-1][2])

    def _resolve_temperature_from_schedule(self, schedule: object, iteration: int, default_value: float) -> float:
        entries = list(schedule or [])
        if not entries:
            return max(0.0, float(default_value))

        matched = None
        if isinstance(entries[0], dict):
            matched = self._match_schedule_entry(entries, iteration)

        if matched is not None:
            return max(0.0, float(matched.get("temperature", default_value)))

        parsed = []
        for item in entries:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            start_iter, end_iter, temperature = item[0], item[1], item[2]
            parsed.append((int(start_iter), None if end_iter is None else int(end_iter), float(temperature)))

        if not parsed:
            return max(0.0, float(default_value))

        first_start = int(parsed[0][0])
        if iteration < first_start:
            return max(0.0, float(parsed[0][2]))

        for start_iter, end_iter, temperature in parsed:
            if iteration < int(start_iter):
                continue
            if end_iter is not None and iteration > int(end_iter):
                continue
            return max(0.0, float(temperature))
        return max(0.0, float(parsed[-1][2]))

    def _parse_learning_rate_schedule(self, schedule: object) -> List[Dict[str, float]]:
        parsed: List[Dict[str, float]] = []
        base_lr = float(self.args.learning_rate)
        for item in list(schedule or []):
            if not isinstance(item, dict):
                continue

            start_iter = int(item.get("start_iter", 1))
            end_iter = item.get("end_iter")

            direct_lr = item.get("lr", item.get("learning_rate", None))
            lr_scale = item.get("lr_scale", item.get("scale", None))

            if direct_lr is not None:
                lr_value = float(direct_lr)
            elif lr_scale is not None:
                lr_value = float(base_lr) * float(lr_scale)
            else:
                continue

            parsed.append(
                {
                    "start_iter": int(start_iter),
                    "end_iter": None if end_iter is None else int(end_iter),
                    "lr": float(max(0.0, lr_value)),
                }
            )
        return parsed

    def _resolve_learning_rate(self, iteration: int) -> float:
        if not self._use_learning_rate_schedule_table:
            return float(self.optimizer.param_groups[0]["lr"])

        entry = self._match_schedule_entry(self._parsed_learning_rate_schedule, int(iteration))
        if entry is None:
            return float(self.args.learning_rate)
        return float(max(0.0, float(entry.get("lr", self.args.learning_rate))))

    def _apply_learning_rate_for_iteration(self, iteration: int):
        if not self._use_learning_rate_schedule_table:
            return
        lr_value = self._resolve_learning_rate(iteration)
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr_value)

    def _resolve_teacher_ratio(self, iteration: int) -> float:
        entry = self._match_schedule_entry(self.args.teacher_mix_schedule, iteration)
        ratio = float(entry.get("teacher_ratio", 0.0) if entry else 0.0)
        ratio = max(0.0, min(1.0, ratio))

        if iteration <= self.recovery_until_iteration:
            ratio = max(ratio, float(self.args.recovery_teacher_mix_ratio))
        return ratio

    def _resolve_temperature(self, iteration: int) -> float:
        profile_schedule = self._get_search_profile("student_self_play_search").get("temperature_schedule")
        schedule = profile_schedule if profile_schedule else self.args.exploration_temperature_schedule
        temperature = self._resolve_temperature_from_schedule(schedule, iteration, 0.0)
        temperature = max(0.0, temperature)
        if iteration <= self.recovery_until_iteration:
            temperature = min(temperature, float(self.args.recovery_temperature_cap))
        return temperature

    def _resolve_student_further_train_start_iter(self) -> int:
        profile = self._get_search_profile("student_self_play_search")
        value = profile.get("further_train_start_iter", None)
        if value is None:
            return 10**9
        try:
            return max(1, int(value))
        except Exception:
            return 10**9

    def _resolve_student_self_play_exploration_enabled(self, iteration: int) -> bool:
        return int(iteration) >= int(self._resolve_student_further_train_start_iter())

    def _should_persist_teacher_histories(self, iteration: int) -> bool:
        further_train_start_iter = int(self._resolve_student_further_train_start_iter())
        history_window_len = max(0, int(getattr(self.args, "history_window_len", 0) or 0))
        persist_until_iter = further_train_start_iter + history_window_len
        return int(iteration) <= int(persist_until_iter)

    def _compact_iteration_metrics_entry(self, entry: object) -> object:
        if not isinstance(entry, dict):
            return entry

        compacted = dict(entry)
        compacted.pop("self_play_game_results", None)
        compacted.pop("adversarial_game_results", None)
        compacted.pop("game_results", None)
        return compacted

    def _compact_iteration_metrics_history(self, keep_last_full: bool = False):
        if not self.iteration_metrics_history:
            return

        last_index = len(self.iteration_metrics_history) - 1
        compacted = []
        for index, item in enumerate(self.iteration_metrics_history):
            if keep_last_full and index == last_index:
                compacted.append(item)
            else:
                compacted.append(self._compact_iteration_metrics_entry(item))

        self.iteration_metrics_history = compacted

    def _link_or_copy_file(self, source: Path, target: Path):
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_target = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
        try:
            try:
                os.link(str(source), str(temp_target))
            except OSError:
                shutil.copy2(str(source), str(temp_target))
            os.replace(str(temp_target), str(target))
        finally:
            if temp_target.exists():
                temp_target.unlink()

    def _atomic_torch_save(self, payload: object, target: Path):
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_target = target.with_name(f".{target.name}.{os.getpid()}.{time.time_ns()}.tmp")
        try:
            torch.save(payload, str(temp_target))
            os.replace(str(temp_target), str(target))
        finally:
            if temp_target.exists():
                temp_target.unlink()

    def _capture_rng_state(self) -> Dict[str, object]:
        cuda_state = None
        if torch.cuda.is_available():
            cuda_state = [state.cpu() for state in torch.cuda.get_rng_state_all()]
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().cpu(),
            "torch_cuda": cuda_state,
        }

    def _restore_rng_state(self, state: object):
        if not isinstance(state, dict):
            logging.info("Checkpoint has no RNG state; continuing from the current seeded state.")
            return

        try:
            random.setstate(state["python"])
            np.random.set_state(state["numpy"])
            torch.set_rng_state(torch.as_tensor(state["torch_cpu"], dtype=torch.uint8, device="cpu"))
            cuda_state = state.get("torch_cuda")
            if cuda_state is not None and torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if len(cuda_state) != device_count:
                    logging.warning(
                        "CUDA RNG device count mismatch: checkpoint=%s runtime=%s; CUDA RNG state was not restored.",
                        len(cuda_state),
                        device_count,
                    )
                else:
                    torch.cuda.set_rng_state_all(
                        [torch.as_tensor(item, dtype=torch.uint8, device="cpu") for item in cuda_state]
                    )
        except Exception as exc:
            logging.warning("Failed to restore complete RNG state: %s", exc)

    def _parse_exploration_iteration_schedule(self, schedule: object) -> List[Dict[str, float]]:
        entries = []
        for item in list(schedule or []):
            if not isinstance(item, dict):
                continue
            start_iter = int(item.get("start_iter", 1))
            end_iter = item.get("end_iter")
            entries.append(
                {
                    "start_iter": int(start_iter),
                    "end_iter": None if end_iter is None else int(end_iter),
                    "temperature_scale": float(max(0.0, float(item.get("temperature_scale", 1.0)))),
                    "noise_scale": float(max(0.0, float(item.get("noise_scale", 1.0)))),
                }
            )
        return entries

    def _parse_self_play_phase_schedule(
        self,
        schedule: object,
        *,
        default_temperature: float,
        default_dirichlet_alpha: float,
        default_dirichlet_epsilon: float,
    ) -> List[Dict[str, object]]:
        phases: List[Dict[str, object]] = []
        for idx, item in enumerate(list(schedule or []), start=1):
            if not isinstance(item, dict):
                continue

            temperature = float(max(0.0, float(item.get("temperature", default_temperature))))
            dirichlet_alpha = float(max(0.0, float(item.get("dirichlet_alpha", default_dirichlet_alpha))))
            dirichlet_epsilon = float(max(0.0, float(item.get("dirichlet_epsilon", default_dirichlet_epsilon))))

            if temperature <= 1e-6:
                dirichlet_epsilon = 0.0
            if dirichlet_epsilon <= 1e-6:
                dirichlet_alpha = 0.0

            max_step = item.get("max_step", None)
            phase_entry: Dict[str, object] = {
                "name": str(item.get("name", f"phase_{idx}")),
                "max_step": None if max_step is None else int(max_step),
                "temperature": float(temperature),
                "dirichlet_alpha": float(dirichlet_alpha),
                "dirichlet_epsilon": float(dirichlet_epsilon),
            }
            phases.append(phase_entry)
        return phases

    def _resolve_self_play_mcts_sims(self, iteration: int) -> int:
        profile_schedule = self._get_search_profile("student_self_play_search").get("mcts_schedule")
        schedule = list(profile_schedule or getattr(self.args, "self_play_mcts_schedule", None) or getattr(self.args, "student_mcts_schedule", []) or [])
        return self._resolve_mcts_sims_from_schedule(schedule, iteration, 256)

    def _resolve_teacher_cache_num_mcts_sims(self) -> int:
        value = self._resolve_profile_value(
            "teacher_cache_search",
            "num_mcts_sims",
            fallback_attr="teacher_num_mcts_sims",
            default=1024,
        )
        return max(1, int(value))

    def _resolve_teacher_cache_temperature(self) -> float:
        value = self._resolve_profile_value(
            "teacher_cache_search",
            "label_temperature",
            fallback_attr="teacher_label_temperature",
            default=0.0,
        )
        return max(0.0, float(value))

    def _resolve_teacher_cache_noise_scale(self) -> float:
        value = self._resolve_profile_value(
            "teacher_cache_search",
            "noise_scale",
            default=0.2,
        )
        return max(0.0, float(value))

    def _resolve_adversarial_game_temperature(self, iteration: int) -> float:
        schedule = self._get_search_profile("teacher_adversarial_search").get("game_temperature_schedule")
        if schedule:
            return self._resolve_temperature_from_schedule(schedule, iteration, self._resolve_temperature(iteration))
        return self._resolve_temperature(iteration)

    def _resolve_adversarial_model_mcts_sims(self, iteration: int) -> int:
        schedule = self._get_search_profile("teacher_adversarial_search").get("model_mcts_schedule")
        if schedule:
            return self._resolve_mcts_sims_from_schedule(schedule, iteration, self._resolve_self_play_mcts_sims(iteration))
        return self._resolve_self_play_mcts_sims(iteration)

    def _resolve_adversarial_teacher_mcts_sims(self, iteration: int) -> int:
        profile = self._get_search_profile("teacher_adversarial_search")
        schedule = profile.get("teacher_mcts_schedule")
        if schedule:
            return self._resolve_mcts_sims_from_schedule(schedule, iteration, self._resolve_teacher_cache_num_mcts_sims())
        value = profile.get("teacher_num_mcts_sims", None)
        if value is None:
            value = self._resolve_teacher_cache_num_mcts_sims()
        return max(1, int(value))

    def _resolve_eval_mcts_sims(self, iteration: int) -> int:
        configured = int(getattr(self.args, "eval_mcts_sims", 0) or 0)
        if configured > 0:
            return configured
        return max(128, self._resolve_self_play_mcts_sims(iteration))

    def _resolve_teacher_weight(self, iteration: int) -> float:
        start = int(self.args.teacher_loss_decay_start)
        end = max(start + 1, int(self.args.teacher_loss_decay_end))
        floor = max(0.0, min(1.0, float(self.args.teacher_loss_floor)))

        if iteration <= start:
            weight = 1.0
        elif iteration >= end:
            weight = floor
        else:
            progress = (iteration - start) / float(end - start)
            weight = 1.0 + progress * (floor - 1.0)

        if iteration <= self.recovery_until_iteration:
            weight = max(weight, float(self.args.recovery_teacher_weight))

        return float(max(0.0, weight))

    def _build_parallel_args(
        self,
        iteration: int,
        num_mcts_sims: int,
        temperature: float,
        noise_scale: float,
        search_profile: str = "student_self_play_search",
        use_student_exploration_table: bool = False,
    ) -> SimpleNamespace:
        profile = self._get_search_profile(search_profile)

        cpuct = float(profile.get("cpuct", self.args.cpuct))
        num_mcts_threads = int(profile.get("num_mcts_threads", self.args.num_mcts_threads))
        virtual_loss = float(profile.get("virtual_loss", self.args.virtual_loss))
        inference_batch_size = int(profile.get("inference_batch_size", self.args.inference_batch_size))
        inference_timeout_s = float(profile.get("inference_timeout_s", self.args.inference_timeout_s))
        compatible_inference_server_count = int(
            profile.get("compatible_inference_server_count", self.args.compatible_inference_server_count)
        )
        shared_inference_server_count = int(profile.get("shared_inference_server_count", self.args.shared_inference_server_count))
        high_threshold = int(
            profile.get("high_mcts_shared_inference_server_threshold", self.args.high_mcts_shared_inference_server_threshold)
        )
        high_server_count = int(
            profile.get("high_mcts_shared_inference_server_count", self.args.high_mcts_shared_inference_server_count)
        )
        exploration_strength = float(profile.get("self_play_exploration_strength", self.args.self_play_exploration_strength))

        dirichlet_alpha = float(profile.get("dirichlet_alpha", self.args.dirichlet_alpha))
        dirichlet_epsilon = float(profile.get("dirichlet_epsilon", self.args.dirichlet_epsilon)) * float(noise_scale)
        if temperature <= 1e-6:
            dirichlet_epsilon = 0.0

        phase_schedule: List[Dict[str, object]] = [
            {
                "name": "distillation_phase",
                "max_step": None,
                "temperature": float(temperature),
                "dirichlet_alpha": float(dirichlet_alpha),
                "dirichlet_epsilon": float(dirichlet_epsilon),
            }
        ]
        exploration_iteration_schedule: List[Dict[str, float]] = [
            {"start_iter": 1, "end_iter": None, "temperature_scale": 1.0, "noise_scale": 1.0}
        ]

        if use_student_exploration_table and search_profile == "student_self_play_search":
            profile_phases = self._parse_self_play_phase_schedule(
                profile.get("self_play_phase_schedule"),
                default_temperature=float(temperature),
                default_dirichlet_alpha=float(dirichlet_alpha),
                default_dirichlet_epsilon=float(dirichlet_epsilon),
            )
            if profile_phases:
                phase_schedule = profile_phases

            profile_iteration_schedule = self._parse_exploration_iteration_schedule(
                profile.get("exploration_iteration_schedule")
            )
            if profile_iteration_schedule:
                exploration_iteration_schedule = profile_iteration_schedule

        return SimpleNamespace(
            cpuct=float(cpuct),
            num_mcts_sims=int(num_mcts_sims),
            num_mcts_threads=int(num_mcts_threads),
            virtual_loss=float(virtual_loss),
            reuse_mcts_tree=bool(getattr(self.args, "reuse_mcts_tree", True)),
            persistent_mcts_threads=bool(getattr(self.args, "persistent_mcts_threads", True)),
            enable_mcts_search_stats=bool(getattr(self.args, "enable_mcts_search_stats", True)),
            inference_batch_size=int(inference_batch_size),
            inference_timeout_s=float(inference_timeout_s),
            inference_precision=str(getattr(self.args, "inference_precision", "fp32")),
            dirichlet_alpha=float(dirichlet_alpha),
            dirichlet_epsilon=float(dirichlet_epsilon),
            self_play_exploration_strength=float(exploration_strength),
            exploration_iteration_schedule=exploration_iteration_schedule,
            self_play_phase_schedule=phase_schedule,
            current_iteration=int(iteration),
            min_game_steps=int(self.args.min_game_steps),
            min_game_steps_start_iteration=int(self.args.min_game_steps_start_iteration),
            compatible_inference_server_count=int(compatible_inference_server_count),
            shared_inference_server_count=int(shared_inference_server_count),
            high_mcts_shared_inference_server_threshold=int(high_threshold),
            high_mcts_shared_inference_server_count=int(high_server_count),
        )

    def _teacher_cache_file(self) -> Path:
        path = self._to_abs_path(self.args.teacher_data_cache_path)
        if path is None:
            raise ValueError("teacher_data_cache_path must not be empty.")
        return path

    def _build_game_summary(self, game_data: Dict[str, object]) -> Dict[str, object]:
        trace = game_data.get("trace") or {}
        used_for_training = game_data.get("used_for_training_split")
        if used_for_training is None:
            used_for_training = game_data.get("used_for_training", True)
        return {
            "steps": int(game_data.get("steps", 0) or 0),
            "used_for_training": bool(used_for_training),
            "policy_entropy_mean": float(game_data.get("policy_entropy_mean", 0.0) or 0.0),
            "winner": int(trace.get("winner", 0) or 0),
            "is_draw": bool(trace.get("is_draw", False)),
            "result_code": float(trace.get("result_code", 0.0) or 0.0),
            "game_idx": int(game_data.get("game_idx", -1) or -1),
            "outcome_for_model": float(game_data.get("outcome_for_model", 0.0) or 0.0),
        }

    def _summarize_game_quality(self, game_summaries: List[Dict[str, object]]) -> Dict[str, object]:
        if not game_summaries:
            return {
                "games": 0,
                "used_games": 0,
                "avg_steps": 0.0,
                "var_steps": 0.0,
                "min_steps": 0,
                "max_steps": 0,
                "short_games": 0,
                "long_games": 0,
                "policy_entropy_mean": 0.0,
                "policy_entropy_var": 0.0,
                "policy_entropy_min": 0.0,
                "policy_entropy_max": 0.0,
                "shortest_game": {},
                "longest_game": {},
            }

        steps = np.asarray([int(item.get("steps", 0) or 0) for item in game_summaries], dtype=np.float64)
        entropies = np.asarray([float(item.get("policy_entropy_mean", 0.0) or 0.0) for item in game_summaries], dtype=np.float64)
        avg_steps = float(np.mean(steps))
        shortest_idx = int(np.argmin(steps))
        longest_idx = int(np.argmax(steps))

        short_games = sum(1 for item in game_summaries if int(item.get("steps", 0) or 0) < 10)
        long_games = sum(1 for item in game_summaries if int(item.get("steps", 0) or 0) > avg_steps)
        used_games = sum(1 for item in game_summaries if bool(item.get("used_for_training", True)))

        return {
            "games": int(len(game_summaries)),
            "used_games": int(used_games),
            "avg_steps": avg_steps,
            "var_steps": float(np.var(steps)),
            "min_steps": int(np.min(steps)),
            "max_steps": int(np.max(steps)),
            "short_games": int(short_games),
            "long_games": int(long_games),
            "policy_entropy_mean": float(np.mean(entropies)),
            "policy_entropy_var": float(np.var(entropies)),
            "policy_entropy_min": float(np.min(entropies)),
            "policy_entropy_max": float(np.max(entropies)),
            "shortest_game": {
                "index": shortest_idx,
                **dict(game_summaries[shortest_idx]),
            },
            "longest_game": {
                "index": longest_idx,
                **dict(game_summaries[longest_idx]),
            },
        }

    def _log_teacher_cache_quality(self, event: str):
        stats = self._summarize_game_quality(self.teacher_cache_game_summaries)
        self.teacher_cache_metadata["quality"] = dict(stats)
        logging.info(
            "%s teacher cache quality | games=%s used=%s avg_steps=%.2f var_steps=%.2f min/max=%s/%s short_games=%s long_games=%s entropy(mean/var/min/max)=%.4f/%.4f/%.4f/%.4f",
            event,
            stats["games"],
            stats["used_games"],
            stats["avg_steps"],
            stats["var_steps"],
            stats["min_steps"],
            stats["max_steps"],
            stats["short_games"],
            stats["long_games"],
            stats["policy_entropy_mean"],
            stats["policy_entropy_var"],
            stats["policy_entropy_min"],
            stats["policy_entropy_max"],
        )
        logging.info(
            "%s teacher cache shortest_game=%s longest_game=%s",
            event,
            json.dumps(stats.get("shortest_game", {}), ensure_ascii=False),
            json.dumps(stats.get("longest_game", {}), ensure_ascii=False),
        )

    def _rebuild_teacher_cache_games_from_examples(self):
        if self.teacher_cache_games:
            return
        if not self.teacher_cache_examples:
            return

        total_games = int(self.teacher_cache_metadata.get("used_games_for_training") or 0)
        if total_games <= 0:
            total_games = int(self.teacher_cache_metadata.get("games") or 0)
        total_games = max(1, min(total_games, len(self.teacher_cache_examples)))

        indices = np.array_split(np.arange(len(self.teacher_cache_examples)), total_games)
        rebuilt_games = []
        rebuilt_summaries = []
        for game_indices in indices:
            if len(game_indices) == 0:
                continue
            game_examples = [self.teacher_cache_examples[int(i)] for i in game_indices]
            rebuilt_games.append(game_examples)
            rebuilt_summaries.append(
                {
                    "steps": int(len(game_indices)),
                    "used_for_training": True,
                    "policy_entropy_mean": 0.0,
                    "winner": 0,
                    "is_draw": False,
                    "result_code": 0.0,
                }
            )

        self.teacher_cache_games = rebuilt_games
        if not self.teacher_cache_game_summaries:
            self.teacher_cache_game_summaries = rebuilt_summaries

    def _load_teacher_cache(self) -> bool:
        cache_path = self._teacher_cache_file()
        if not cache_path.exists():
            return False

        payload = load_checkpoint_payload(str(cache_path))
        if isinstance(payload, dict) and "examples" in payload:
            self.teacher_cache_examples = list(payload.get("examples") or [])
            self.teacher_cache_games = [list(item or []) for item in list(payload.get("games") or [])]
            self.teacher_cache_metadata = dict(payload.get("metadata") or {})
            self.teacher_cache_game_summaries = list(self.teacher_cache_metadata.get("game_summaries") or [])
        elif isinstance(payload, list):
            self.teacher_cache_examples = list(payload)
            self.teacher_cache_games = []
            self.teacher_cache_game_summaries = []
            self.teacher_cache_metadata = {}
        else:
            raise ValueError(f"Unsupported teacher cache format: {cache_path}")

        cached_fingerprint = self.teacher_cache_metadata.get("teacher_model_sha256")
        current_fingerprint = self._file_fingerprint(
            self.teacher_model_spec.get("path") if self.teacher_model_spec else self.args.teacher_model_path
        )
        if cached_fingerprint and current_fingerprint and cached_fingerprint != current_fingerprint:
            logging.warning("Teacher cache fingerprint does not match the configured teacher model.")
        elif not cached_fingerprint:
            logging.info("Legacy teacher cache has no model fingerprint; path/architecture compatibility only.")
        else:
            logging.info("Teacher model/cache fingerprint verified: %s", str(current_fingerprint)[:12])

        self._rebuild_teacher_cache_games_from_examples()

        logging.info(
            "Loaded teacher cache: samples=%s games=%s path=%s",
            len(self.teacher_cache_examples),
            len(self.teacher_cache_games),
            cache_path,
        )
        self._log_teacher_cache_quality("Loaded")
        return True

    def _summarize_runtime_games(self, game_results: List[Dict[str, object]]) -> Dict[str, object]:
        summaries = [self._build_game_summary(item) for item in list(game_results or [])]
        quality = self._summarize_game_quality(summaries)
        search_totals: Dict[str, int] = {}

        def _accumulate(stats: object):
            if not isinstance(stats, dict):
                return
            for key, value in stats.items():
                if isinstance(value, dict):
                    _accumulate(value)
                elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
                    search_totals[str(key)] = search_totals.get(str(key), 0) + int(value)

        for game_result in list(game_results or []):
            _accumulate(game_result.get("search_stats"))
        return {
            "mean_steps": float(quality.get("avg_steps", 0.0)),
            "variance_steps": float(quality.get("var_steps", 0.0)),
            "mean_policy_entropy": float(quality.get("policy_entropy_mean", 0.0)),
            "variance_policy_entropy": float(quality.get("policy_entropy_var", 0.0)),
            "min_policy_entropy": float(quality.get("policy_entropy_min", 0.0)),
            "max_policy_entropy": float(quality.get("policy_entropy_max", 0.0)),
            "min_steps": int(quality.get("min_steps", 0)),
            "max_steps": int(quality.get("max_steps", 0)),
            "long_games": int(quality.get("long_games", 0)),
            "short_games": int(quality.get("short_games", 0)),
            "filtered_games": int(max(0, quality.get("games", 0) - quality.get("used_games", 0))),
            "used_games": int(quality.get("used_games", 0)),
            "games": int(quality.get("games", 0)),
            "shortest_game": dict(quality.get("shortest_game") or {}),
            "longest_game": dict(quality.get("longest_game") or {}),
            "search_stats": search_totals,
        }

    def _save_teacher_cache(self):
        cache_path = self._teacher_cache_file()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "examples": list(self.teacher_cache_examples),
            "games": list(self.teacher_cache_games),
            "metadata": dict(self.teacher_cache_metadata),
        }
        torch.save(payload, str(cache_path))
        logging.info(
            "Saved teacher cache: samples=%s games=%s path=%s",
            len(self.teacher_cache_examples),
            len(self.teacher_cache_games),
            cache_path,
        )

    def generate_teacher_data_cache(self):
        if self.teacher_model_spec is None:
            raise RuntimeError("teacher_model_path is required for teacher data generation.")

        num_games = max(1, int(self.args.teacher_data_generation_games))
        teacher_cache_sims = self._resolve_teacher_cache_num_mcts_sims()
        teacher_cache_temp = self._resolve_teacher_cache_temperature()
        teacher_cache_noise = self._resolve_teacher_cache_noise_scale()
        parallel_args = self._build_parallel_args(
            iteration=1,
            num_mcts_sims=teacher_cache_sims,
            temperature=teacher_cache_temp,
            noise_scale=teacher_cache_noise,
            search_profile="teacher_cache_search",
        )
        workers = self._resolve_worker_count(num_games, parallel_args.num_mcts_threads)

        logging.info(
            "Generating teacher cache from model=%s with games=%s sims=%s",
            self.teacher_model_spec["path"],
            num_games,
            int(teacher_cache_sims),
        )
        examples, game_results = execute_self_play_parallel(
            args=parallel_args,
            num_games=num_games,
            num_workers=workers,
            shared_inference_device=self.args.shared_inference_device,
            inference_batch_size=int(parallel_args.inference_batch_size),
            inference_timeout_s=float(parallel_args.inference_timeout_s),
            model_state=None,
            model_config=None,
            compatible_model_spec={
                "state_dict": self.teacher_model_spec["state_dict"],
                "config": self.teacher_model_spec["config"],
            },
            progress_desc="Teacher Data Generation",
        )

        usable_games = [
            item
            for item in game_results
            if bool(item.get("used_for_training", True)) and len(item.get("examples") or []) > 0
        ]
        self.teacher_cache_games = [list(item.get("examples") or []) for item in usable_games]
        self.teacher_cache_examples = [example for group in self.teacher_cache_games for example in group]
        self.teacher_cache_game_summaries = [self._build_game_summary(item) for item in game_results]
        quality = self._summarize_game_quality(self.teacher_cache_game_summaries)
        self.teacher_cache_metadata = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "games": int(num_games),
            "samples": int(len(self.teacher_cache_examples)),
            "teacher_model_path": self.teacher_model_spec["path"],
            "teacher_model_sha256": self._file_fingerprint(self.teacher_model_spec["path"]),
            "teacher_model_architecture": self.teacher_model_spec["config"].get("architecture"),
            "teacher_num_mcts_sims": int(teacher_cache_sims),
            "teacher_label_temperature": float(teacher_cache_temp),
            "teacher_data_refresh_policy": str(self.args.teacher_data_refresh_policy),
            "used_games_for_training": int(sum(1 for item in game_results if item.get("used_for_training", False))),
            "usable_games": int(len(self.teacher_cache_games)),
            "game_summaries": list(self.teacher_cache_game_summaries),
            "quality": quality,
        }
        self._save_teacher_cache()
        self._log_teacher_cache_quality("Generated")

    def _ensure_teacher_cache_ready(self):
        loaded = self._load_teacher_cache()
        if bool(self.args.teacher_data_generate_mode):
            self.generate_teacher_data_cache()
            return

        if loaded and not bool(self.args.teacher_data_regenerate):
            return

        need_generation = bool(self.args.teacher_data_generation_enabled)
        if need_generation:
            self.generate_teacher_data_cache()
            return

        if not loaded:
            logging.warning("Teacher cache unavailable and generation disabled; training will run self-play only.")

    def _is_hot_start_iteration(self, iteration: int) -> bool:
        return bool(self.teacher_cache_games) and int(iteration) <= max(0, int(self.args.hot_start_iterations))

    def _select_warmup_teacher_games(self, iteration: int) -> List[List[Tuple[np.ndarray, np.ndarray, float]]]:
        if not self.teacher_cache_games:
            return []

        per_iter_games = max(1, int(self.args.hot_start_teacher_games_per_iteration))
        start = max(0, int(iteration - 1) * per_iter_games)
        end = start + per_iter_games

        selected = self.teacher_cache_games[start:end]
        if len(selected) >= per_iter_games:
            return [list(item) for item in selected]

        shortage = per_iter_games - len(selected)
        if shortage <= 0:
            return [list(item) for item in selected]

        fallback_indices = np.random.choice(len(self.teacher_cache_games), shortage, replace=True)
        selected.extend(self.teacher_cache_games[int(i)] for i in np.atleast_1d(fallback_indices))
        return [list(item) for item in selected]

    def _build_hot_start_training_examples(self, iteration: int) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float, int]], Dict[str, object]]:
        selected_games = self._select_warmup_teacher_games(iteration)
        selected_examples = [example for game_examples in selected_games for example in game_examples]

        tagged = [(board, policy, value, SOURCE_TEACHER_WARMUP) for (board, policy, value) in selected_examples]
        self._append_tagged_history("teacher", tagged, iteration)

        merged, window_stats = self._collect_windowed_training_examples(iteration)
        source_counts = self._count_examples_by_source(merged)
        teacher_samples = (
            source_counts.get(SOURCE_TEACHER_WARMUP, 0)
            + source_counts.get(SOURCE_TEACHER_LOSS, 0)
            + source_counts.get(SOURCE_TEACHER_RESPONSE, 0)
        )
        adversarial_samples = (
            source_counts.get(SOURCE_TEACHER_LOSS, 0)
            + source_counts.get(SOURCE_STUDENT_WIN, 0)
            + source_counts.get(SOURCE_TEACHER_RESPONSE, 0)
        )

        return merged, {
            "mode": "hot_start_teacher_cache_window",
            "teacher_ratio": 1.0,
            "self_play_samples": int(source_counts.get(SOURCE_SELF_PLAY, 0)),
            "teacher_samples": int(teacher_samples),
            "teacher_weight": float(self._resolve_teacher_weight(iteration)),
            "hot_start_games": int(len(selected_games)),
            "adversarial_training_samples": int(adversarial_samples),
            "current_self_play_samples": 0,
            "current_adversarial_samples": 0,
            "current_teacher_pure_samples": int(len(tagged)),
            "history_pool_samples": int(len(self.teacher_history_pool)),
            **window_stats,
        }

    def _latest_checkpoint_path(self) -> Path:
        return self.run_dir / "latest.pth.tar"

    def _checkpoint_path_for_iteration(self, iteration: int) -> Path:
        return self.run_dir / f"checkpoint_{int(iteration)}" / "checkpoint.pth.tar"

    def _resolve_resume_path(self) -> Optional[Path]:
        explicit = self._to_abs_path(self.args.resume_path)
        if explicit is not None:
            return explicit

        if self.args.rollback_iteration is not None:
            return self._checkpoint_path_for_iteration(int(self.args.rollback_iteration))

        if self.args.resume:
            return self._latest_checkpoint_path()

        return None

    def _maybe_resume_from_checkpoint(self) -> bool:
        resume_path = self._resolve_resume_path()
        if resume_path is None:
            return False
        if not resume_path.exists():
            logging.warning("Resume path does not exist: %s", resume_path)
            return False

        self.load_checkpoint(str(resume_path), weights_only=bool(self.args.resume_weights_only))
        return True

    def _build_checkpoint_state(self, iteration: int) -> Dict[str, object]:
        persist_teacher_histories = self._should_persist_teacher_histories(iteration)
        return {
            "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
            "iteration": int(iteration),
            "state_dict": {k: v.detach().cpu() for k, v in self.student.state_dict().items()},
            "student_model_config": dict(self.student_model_config),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "grad_scaler": self.grad_scaler.state_dict(),
            "rng_state": self._capture_rng_state(),
            "eval_history": list(self.eval_history),
            "iteration_metrics_history": list(self.iteration_metrics_history),
            "teacher_cache_path": str(self._teacher_cache_file()),
            "teacher_cache_metadata": dict(self.teacher_cache_metadata),
            "teacher_history_pool_size": int(len(self.teacher_history_pool)),
            "self_play_examples_history": _pack_history_entries(self.self_play_examples_history),
            "adversarial_examples_history": _pack_history_entries(
                self.adversarial_examples_history if persist_teacher_histories else []
            ),
            "teacher_pure_examples_history": _pack_history_entries(
                self.teacher_pure_examples_history if persist_teacher_histories else []
            ),
            "best_recent_state": copy.deepcopy(self.best_recent_state),
            "best_older_state": copy.deepcopy(self.best_older_state),
            "no_refresh_streak": int(self.no_refresh_streak),
            "recovery_until_iteration": int(self.recovery_until_iteration),
            "args": self.args.to_dict(),
        }

    def save_checkpoint(self, iteration: int):
        folder = self.run_dir / f"checkpoint_{int(iteration)}"
        folder.mkdir(parents=True, exist_ok=True)
        checkpoint_path = folder / "checkpoint.pth.tar"
        model_path = folder / "model.pth"

        self._prune_history_windows(current_iteration=iteration)
        self._save_iteration_samples(iteration, folder)
        self._compact_iteration_metrics_history()

        if not self._should_persist_teacher_histories(iteration):
            persist_until_iter = int(self._resolve_student_further_train_start_iter()) + max(
                0,
                int(getattr(self.args, "history_window_len", 0) or 0),
            )
            logging.info(
                "Checkpoint %s omits adversarial/teacher histories because the retention cutoff is iteration %s.",
                iteration,
                persist_until_iter,
            )

        state = self._build_checkpoint_state(iteration)
        self._atomic_torch_save(state, checkpoint_path)
        self._link_or_copy_file(checkpoint_path, self._latest_checkpoint_path())
        self._atomic_torch_save(state["state_dict"], model_path)
        self.last_checkpoint_iteration = int(iteration)

        self._cleanup_old_checkpoints()
        logging.info("Saved checkpoint: %s", checkpoint_path)

    def _save_iteration_samples(self, iteration: int, folder: Path):
        if not self.iteration_metrics_history:
            return
        current = self.iteration_metrics_history[-1]
        game_results = list(current.get("self_play_game_results") or [])
        if not game_results:
            return

        sampled = []
        for game in game_results[: min(5, len(game_results))]:
            trace = game.get("trace") or {}
            sampled.append(
                {
                    "game_idx": int(game.get("game_idx", -1)),
                    "steps": int(game.get("steps", 0)),
                    "used_for_training": bool(game.get("used_for_training", False)),
                    "winner": int(trace.get("winner", 0)),
                    "is_draw": bool(trace.get("is_draw", False)),
                    "result_code": float(trace.get("result_code", 0.0)),
                    "moves": list(trace.get("moves", [])),
                }
            )

        sample_payload = {
            "iteration": int(iteration),
            "samples": sampled,
            "sampling_rule": "first_n_games",
        }
        with open(folder / "self_play_samples.json", "w", encoding="utf-8") as f:
            json.dump(sample_payload, f, ensure_ascii=False, indent=2)

    def _cleanup_old_checkpoints(self):
        all_folders = []
        for child in self.run_dir.iterdir():
            if not child.is_dir() or not child.name.startswith("checkpoint_"):
                continue
            try:
                iter_id = int(child.name.split("_")[1])
            except Exception:
                continue
            all_folders.append((iter_id, child))

        all_folders.sort(key=lambda item: item[0])
        keep = int(self.args.max_checkpoints)
        if len(all_folders) <= keep:
            return

        for _, folder in all_folders[:-keep]:
            shutil.rmtree(folder, ignore_errors=True)

    def _load_best_from_state(self, state: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
        if not state:
            return None
        required = {"state_dict", "model_config", "iteration", "label"}
        if not required.issubset(set(state.keys())):
            return None
        return {
            "state_dict": {k: v.detach().cpu() for k, v in state["state_dict"].items()},
            "model_config": dict(state["model_config"]),
            "iteration": int(state["iteration"]),
            "label": str(state["label"]),
            "win_rate": float(state.get("win_rate", 0.0)),
        }

    def load_checkpoint(self, checkpoint_path: str, weights_only: bool = False):
        payload = load_checkpoint_payload(checkpoint_path)
        if not isinstance(payload, dict):
            self.student.load_state_dict(payload, strict=True)
            self.start_iteration = max(1, int(self.args.continue_from_iteration or 1))
            self._apply_learning_rate_for_iteration(self.start_iteration)
            logging.info("Loaded weights-only checkpoint: %s", checkpoint_path)
            return

        state_dict = payload.get("state_dict", payload)
        self.student.load_state_dict(state_dict, strict=True)

        if weights_only:
            target_iter = int(self.args.continue_from_iteration or (payload.get("iteration", 0) + 1))
            self.start_iteration = max(1, target_iter)
            self._apply_learning_rate_for_iteration(self.start_iteration)
            logging.info(
                "Loaded weights only from %s, continue from iteration %s",
                checkpoint_path,
                self.start_iteration,
            )
            return

        if self.args.restore_optimizer_state and "optimizer" in payload:
            try:
                self.optimizer.load_state_dict(payload["optimizer"])
            except Exception as exc:
                logging.warning("Failed to load optimizer state: %s", exc)

        if self.args.restore_schedule_state and self.scheduler is not None and payload.get("scheduler") is not None:
            try:
                self.scheduler.load_state_dict(payload["scheduler"])
            except Exception as exc:
                logging.warning("Failed to load scheduler state: %s", exc)

        scaler_state = payload.get("grad_scaler")
        if scaler_state is not None:
            try:
                self.grad_scaler.load_state_dict(scaler_state)
            except Exception as exc:
                logging.warning("Failed to load AMP GradScaler state: %s", exc)
        else:
            logging.info("Legacy checkpoint has no AMP GradScaler state; using a fresh scaler.")

        self.eval_history = list(payload.get("eval_history", []))
        self.iteration_metrics_history = list(payload.get("iteration_metrics_history", []))
        self._compact_iteration_metrics_history()
        self.teacher_cache_metadata = dict(payload.get("teacher_cache_metadata", self.teacher_cache_metadata))
        raw_self_play_history = _unpack_history_entries(payload.get("self_play_examples_history", []))
        raw_adversarial_history = _unpack_history_entries(payload.get("adversarial_examples_history", []))
        raw_teacher_history = _unpack_history_entries(payload.get("teacher_pure_examples_history", []))
        restored_self_play_history_count = len(raw_self_play_history)
        restored_adversarial_history_count = len(raw_adversarial_history)
        restored_teacher_history_count = len(raw_teacher_history)

        checkpoint_iteration = int(payload.get("iteration", 0) or 0)
        inferred_self_play_iters = self._infer_legacy_history_iterations(
            payload,
            "self_play",
            restored_self_play_history_count,
            checkpoint_iteration,
        )
        inferred_adversarial_iters = self._infer_legacy_history_iterations(
            payload,
            "adversarial",
            restored_adversarial_history_count,
            checkpoint_iteration,
        )
        inferred_teacher_iters = self._infer_legacy_history_iterations(
            payload,
            "teacher",
            restored_teacher_history_count,
            checkpoint_iteration,
        )

        self.self_play_examples_history = self._prune_history_entries(
            self._normalize_history_entries(
                raw_self_play_history,
                checkpoint_iteration=checkpoint_iteration,
                inferred_iterations=inferred_self_play_iters,
            ),
            "self_play",
            current_iteration=checkpoint_iteration if checkpoint_iteration > 0 else None,
        )
        self.adversarial_examples_history = self._prune_history_entries(
            self._normalize_history_entries(
                raw_adversarial_history,
                checkpoint_iteration=checkpoint_iteration,
                inferred_iterations=inferred_adversarial_iters,
            ),
            "adversarial",
            current_iteration=checkpoint_iteration if checkpoint_iteration > 0 else None,
        )
        self.teacher_pure_examples_history = self._prune_history_entries(
            self._normalize_history_entries(
                raw_teacher_history,
                checkpoint_iteration=checkpoint_iteration,
                inferred_iterations=inferred_teacher_iters,
            ),
            "teacher",
            current_iteration=checkpoint_iteration if checkpoint_iteration > 0 else None,
        )
        self._refresh_teacher_history_pool()

        if restored_self_play_history_count > len(self.self_play_examples_history):
            logging.info(
                "Pruned restored self-play history from %s to %s using self_play window=%s.",
                restored_self_play_history_count,
                len(self.self_play_examples_history),
                self._resolve_history_window_limit("self_play"),
            )
        if restored_adversarial_history_count > len(self.adversarial_examples_history):
            logging.info(
                "Pruned restored adversarial history from %s to %s using adversarial window=%s.",
                restored_adversarial_history_count,
                len(self.adversarial_examples_history),
                self._resolve_history_window_limit("adversarial"),
            )
        if restored_teacher_history_count > len(self.teacher_pure_examples_history):
            logging.info(
                "Pruned restored teacher history from %s to %s using teacher window=%s.",
                restored_teacher_history_count,
                len(self.teacher_pure_examples_history),
                self._resolve_history_window_limit("teacher"),
            )

        legacy_teacher_pool_size = int(payload.get("teacher_history_pool_size", 0) or 0)
        if legacy_teacher_pool_size > 0 and not self.teacher_history_pool:
            logging.info(
                "Legacy checkpoint teacher history metadata=%s found, but serialized history samples are unavailable.",
                legacy_teacher_pool_size,
            )
        self.best_recent_state = self._load_best_from_state(payload.get("best_recent_state"))
        self.best_older_state = self._load_best_from_state(payload.get("best_older_state"))
        self.no_refresh_streak = int(payload.get("no_refresh_streak", 0))
        self.recovery_until_iteration = int(payload.get("recovery_until_iteration", 0))

        default_start = int(payload.get("iteration", 0)) + 1
        if self.args.continue_from_iteration is not None:
            default_start = int(self.args.continue_from_iteration)
        self.start_iteration = max(1, default_start)
        self.last_checkpoint_iteration = int(payload.get("iteration", 0) or 0)
        self._apply_learning_rate_for_iteration(self.start_iteration)
        self._restore_rng_state(payload.get("rng_state"))

        logging.info(
            "Resumed distillation checkpoint format v%s from %s at iteration %s",
            int(payload.get("checkpoint_format_version", 1) or 1),
            checkpoint_path,
            self.start_iteration,
        )

    def _resolve_adversarial_game_count(self, iteration: int, self_play_games: int) -> int:
        if self.teacher_model_spec is None:
            return 0
        teacher_ratio = self._resolve_teacher_ratio(iteration)
        if teacher_ratio <= 0.0:
            return 0
        base_games = max(1, int(self_play_games))
        return max(2, int(round(base_games * teacher_ratio)))

    def _run_student_teacher_adversarial(self, iteration: int, self_play_games: int) -> Dict[str, object]:
        if self.teacher_model_spec is None:
            return {
                "games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "loss_examples": [],
                "win_student_examples": [],
                "win_teacher_response_examples": [],
                "game_results": [],
            }

        num_games = self._resolve_adversarial_game_count(iteration, self_play_games)
        if num_games <= 0:
            return {
                "games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "loss_examples": [],
                "win_student_examples": [],
                "win_teacher_response_examples": [],
                "game_results": [],
            }

        profile = self._get_search_profile("teacher_adversarial_search")
        model_mcts_sims = self._resolve_adversarial_model_mcts_sims(iteration)
        teacher_mcts_sims = self._resolve_adversarial_teacher_mcts_sims(iteration)
        game_temperature = max(0.0, self._resolve_adversarial_game_temperature(iteration))
        target_temperature = max(
            0.0,
            float(
                self._resolve_profile_value(
                    "teacher_adversarial_search",
                    "target_temperature",
                    fallback_attr="teacher_label_temperature",
                    default=0.0,
                )
            ),
        )
        student_target_temperature = float(profile.get("student_target_temperature", game_temperature))
        teacher_response_temperature = float(profile.get("teacher_response_temperature", target_temperature))
        adversarial_noise_scale = max(0.0, float(profile.get("noise_scale", 0.0)))

        parallel_args = self._build_parallel_args(
            iteration=iteration,
            num_mcts_sims=model_mcts_sims,
            temperature=game_temperature,
            noise_scale=adversarial_noise_scale,
            search_profile="teacher_adversarial_search",
        )
        parallel_args.model_num_mcts_sims = int(model_mcts_sims)
        parallel_args.teacher_num_mcts_sims = int(teacher_mcts_sims)
        parallel_args.model_cpuct = float(profile.get("model_cpuct", parallel_args.cpuct))
        parallel_args.teacher_cpuct = float(profile.get("teacher_cpuct", profile.get("cpuct", parallel_args.cpuct)))
        parallel_args.model_num_mcts_threads = int(profile.get("model_num_mcts_threads", parallel_args.num_mcts_threads))
        parallel_args.teacher_num_mcts_threads = int(
            profile.get("teacher_num_mcts_threads", profile.get("num_mcts_threads", parallel_args.num_mcts_threads))
        )
        parallel_args.model_virtual_loss = float(profile.get("model_virtual_loss", parallel_args.virtual_loss))
        parallel_args.teacher_virtual_loss = float(
            profile.get("teacher_virtual_loss", profile.get("virtual_loss", parallel_args.virtual_loss))
        )
        parallel_args.teacher_failure_collect_mode = "split_all"
        parallel_args.teacher_failure_target_temperature = float(target_temperature)
        parallel_args.teacher_failure_game_temperature = float(game_temperature)
        parallel_args.teacher_failure_student_target_temperature = float(student_target_temperature)
        parallel_args.teacher_failure_teacher_response_temperature = float(teacher_response_temperature)
        parallel_args.tactical_override_max_step = 0
        parallel_args.min_game_steps = 0
        parallel_args.min_game_steps_start_iteration = 10**9

        model_inference_batch_size = int(profile.get("model_inference_batch_size", parallel_args.inference_batch_size))
        teacher_inference_batch_size = int(
            profile.get("teacher_inference_batch_size", profile.get("inference_batch_size", parallel_args.inference_batch_size))
        )
        model_inference_timeout_s = float(profile.get("model_inference_timeout_s", parallel_args.inference_timeout_s))
        teacher_inference_timeout_s = float(
            profile.get("teacher_inference_timeout_s", profile.get("inference_timeout_s", parallel_args.inference_timeout_s))
        )
        model_inference_server_count = int(profile.get("model_inference_server_count", 1))
        teacher_inference_server_count = int(profile.get("teacher_inference_server_count", 1))

        workers = self._resolve_worker_count(
            num_games,
            max(parallel_args.model_num_mcts_threads, parallel_args.teacher_num_mcts_threads),
        )
        _, game_results = execute_teacher_failure_parallel(
            args=parallel_args,
            num_games=num_games,
            num_workers=workers,
            shared_inference_device=self.args.shared_inference_device,
            inference_batch_size=int(model_inference_batch_size),
            inference_timeout_s=float(model_inference_timeout_s),
            model_state={k: v.detach().cpu() for k, v in self.student.state_dict().items()},
            model_config=dict(self.student_model_config),
            teacher_model_spec={
                "state_dict": self.teacher_model_spec["state_dict"],
                "config": self.teacher_model_spec["config"],
            },
            model_inference_batch_size=int(model_inference_batch_size),
            model_inference_timeout_s=float(model_inference_timeout_s),
            teacher_inference_batch_size=int(teacher_inference_batch_size),
            teacher_inference_timeout_s=float(teacher_inference_timeout_s),
            model_inference_server_count=int(model_inference_server_count),
            teacher_inference_server_count=int(teacher_inference_server_count),
            progress_desc=f"Distill Adversarial Iter {iteration}",
        )

        wins = 0
        losses = 0
        draws = 0
        loss_examples: List[Tuple[np.ndarray, np.ndarray, float]] = []
        win_student_examples: List[Tuple[np.ndarray, np.ndarray, float]] = []
        win_teacher_response_examples: List[Tuple[np.ndarray, np.ndarray, float]] = []

        for item in game_results:
            outcome = float(item.get("outcome_for_model", 0.0) or 0.0)
            if outcome > 0.0:
                wins += 1
            elif outcome < 0.0:
                losses += 1
            else:
                draws += 1

            loss_examples.extend(list(item.get("loss_examples") or []))
            win_student_examples.extend(list(item.get("win_student_examples") or []))
            win_teacher_response_examples.extend(list(item.get("win_teacher_response_examples") or []))

        return {
            "games": int(num_games),
            "wins": int(wins),
            "losses": int(losses),
            "draws": int(draws),
            "loss_examples": loss_examples,
            "win_student_examples": win_student_examples,
            "win_teacher_response_examples": win_teacher_response_examples,
            "game_results": list(game_results),
        }

    def _compose_training_examples(
        self,
        self_play_examples: List[Tuple[np.ndarray, np.ndarray, float]],
        adversarial_data: Dict[str, object],
        iteration: int,
    ) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float, int]], Dict[str, object]]:
        self_play_tagged = [(b, p, v, SOURCE_SELF_PLAY) for (b, p, v) in list(self_play_examples or [])]
        teacher_loss_tagged = [(b, p, v, SOURCE_TEACHER_LOSS) for (b, p, v) in list(adversarial_data.get("loss_examples") or [])]
        student_win_tagged = [(b, p, v, SOURCE_STUDENT_WIN) for (b, p, v) in list(adversarial_data.get("win_student_examples") or [])]
        teacher_response_tagged = [
            (b, p, v, SOURCE_TEACHER_RESPONSE)
            for (b, p, v) in list(adversarial_data.get("win_teacher_response_examples") or [])
        ]
        adversarial_tagged = teacher_loss_tagged + student_win_tagged + teacher_response_tagged

        self._append_tagged_history("self_play", self_play_tagged, iteration)
        self._append_tagged_history("adversarial", adversarial_tagged, iteration)

        merged, window_stats = self._collect_windowed_training_examples(iteration)
        source_counts = self._count_examples_by_source(merged)
        teacher_samples = (
            source_counts.get(SOURCE_TEACHER_WARMUP, 0)
            + source_counts.get(SOURCE_TEACHER_LOSS, 0)
            + source_counts.get(SOURCE_TEACHER_RESPONSE, 0)
        )
        adversarial_samples = (
            source_counts.get(SOURCE_TEACHER_LOSS, 0)
            + source_counts.get(SOURCE_STUDENT_WIN, 0)
            + source_counts.get(SOURCE_TEACHER_RESPONSE, 0)
        )
        return merged, {
            "mode": "self_play_plus_adversarial_teacher_history_window",
            "teacher_ratio": float(self._resolve_teacher_ratio(iteration)),
            "self_play_samples": int(source_counts.get(SOURCE_SELF_PLAY, 0)),
            "teacher_samples": int(teacher_samples),
            "teacher_weight": float(self._resolve_teacher_weight(iteration)),
            "adversarial_games": int(adversarial_data.get("games", 0) or 0),
            "adversarial_wins": int(adversarial_data.get("wins", 0) or 0),
            "adversarial_losses": int(adversarial_data.get("losses", 0) or 0),
            "adversarial_draws": int(adversarial_data.get("draws", 0) or 0),
            "adversarial_loss_examples": int(len(adversarial_data.get("loss_examples") or [])),
            "adversarial_win_student_examples": int(len(adversarial_data.get("win_student_examples") or [])),
            "adversarial_win_teacher_response_examples": int(len(adversarial_data.get("win_teacher_response_examples") or [])),
            "adversarial_training_samples": int(adversarial_samples),
            "current_self_play_samples": int(len(self_play_tagged)),
            "current_adversarial_samples": int(len(adversarial_tagged)),
            "current_teacher_pure_samples": 0,
            "history_pool_samples": int(len(self.teacher_history_pool)),
            **window_stats,
        }

    def _run_student_self_play(self, iteration: int):
        temperature = self._resolve_temperature(iteration)
        mcts_sims = self._resolve_self_play_mcts_sims(iteration)
        use_exploration_table = self._resolve_student_self_play_exploration_enabled(iteration)
        noise_scale = 1.0 if (use_exploration_table or temperature > 1e-6) else 0.0

        parallel_args = self._build_parallel_args(
            iteration=iteration,
            num_mcts_sims=mcts_sims,
            temperature=temperature,
            noise_scale=noise_scale,
            search_profile="student_self_play_search",
            use_student_exploration_table=use_exploration_table,
        )

        num_games = max(1, int(self.args.num_self_play_games))
        workers = self._resolve_worker_count(num_games, parallel_args.num_mcts_threads)
        examples, game_results = execute_self_play_parallel(
            args=parallel_args,
            num_games=num_games,
            num_workers=workers,
            shared_inference_device=self.args.shared_inference_device,
            inference_batch_size=int(parallel_args.inference_batch_size),
            inference_timeout_s=float(parallel_args.inference_timeout_s),
            model_state={k: v.detach().cpu() for k, v in self.student.state_dict().items()},
            model_config=dict(self.student_model_config),
            compatible_model_spec=None,
            progress_desc=f"Distill Self-Play Iter {iteration}",
        )
        return examples, game_results

    def _build_eval_args(self, iteration: int) -> SimpleNamespace:
        eval_sims = self._resolve_eval_mcts_sims(iteration)
        return self._build_parallel_args(
            iteration=iteration,
            num_mcts_sims=eval_sims,
            temperature=0.0,
            noise_scale=0.0,
        )

    def _evaluate_against_best_generations(self, iteration: int) -> List[Dict[str, object]]:
        eval_args = self._build_eval_args(iteration)
        opponents: List[Dict[str, object]] = []

        if self.best_recent_state is not None:
            opponents.append(self.best_recent_state)
        if self.best_older_state is not None:
            opponents.append(self.best_older_state)

        if not opponents:
            return []

        results = []
        games = max(2, int(self.args.best_eval_games_per_generation))
        workers = self._resolve_worker_count(games, eval_args.num_mcts_threads)

        new_model_state = {k: v.detach().cpu() for k, v in self.student.state_dict().items()}

        if bool(getattr(self.args, "shared_evaluation_services", True)) and len(opponents) > 1:
            total_workers = self._resolve_worker_count(games * len(opponents), eval_args.num_mcts_threads)
            match_results = execute_evaluation_group_parallel(
                args=eval_args,
                matches=[
                    {
                        "label": state["label"],
                        "num_games": games,
                        "opponent_nnet_state": {k: v.detach().cpu() for k, v in state["state_dict"].items()},
                        "opponent_nnet_config": dict(state["model_config"]),
                    }
                    for state in opponents
                ],
                total_workers=total_workers,
                shared_inference_device=self.args.shared_inference_device,
                inference_batch_size=int(eval_args.inference_batch_size),
                inference_timeout_s=float(eval_args.inference_timeout_s),
                new_model_state=new_model_state,
                new_model_config=dict(self.student_model_config),
            )
            return [
                {
                    "label": state["label"],
                    "wins": int(counts[0]),
                    "losses": int(counts[1]),
                    "draws": int(counts[2]),
                    "games": int(games),
                    "win_rate": float((counts[0] + 0.5 * counts[2]) / games),
                }
                for state, counts in zip(opponents, match_results)
            ]

        def _run_single(state: Dict[str, object]) -> Dict[str, object]:
            wins, losses, draws = execute_evaluation_parallel(
                args=eval_args,
                num_games=games,
                num_workers=workers,
                shared_inference_device=self.args.shared_inference_device,
                inference_batch_size=int(eval_args.inference_batch_size),
                inference_timeout_s=float(eval_args.inference_timeout_s),
                new_model_state=new_model_state,
                new_model_config=dict(self.student_model_config),
                opponent_nnet_state={k: v.detach().cpu() for k, v in state["state_dict"].items()},
                opponent_nnet_config=dict(state["model_config"]),
                opponent_model_spec=None,
            )
            win_rate = (wins + 0.5 * draws) / float(games)
            return {
                "label": state["label"],
                "wins": int(wins),
                "losses": int(losses),
                "draws": int(draws),
                "games": int(games),
                "win_rate": float(win_rate),
            }

        run_parallel = bool(getattr(self.args, "best_eval_parallelize_generations", True)) and len(opponents) > 1
        if run_parallel:
            with ThreadPoolExecutor(max_workers=len(opponents)) as executor:
                results = list(executor.map(_run_single, opponents))
        else:
            for state in opponents:
                results.append(_run_single(state))

        return results

    def _evaluate_against_model_spec(
        self,
        iteration: int,
        model_spec: Optional[Dict[str, object]],
        games: int,
        label: str,
    ) -> Optional[Dict[str, object]]:
        if model_spec is None or games <= 0:
            return None

        eval_args = self._build_eval_args(iteration)
        workers = self._resolve_worker_count(games, eval_args.num_mcts_threads)
        wins, losses, draws = execute_evaluation_parallel(
            args=eval_args,
            num_games=int(games),
            num_workers=workers,
            shared_inference_device=self.args.shared_inference_device,
            inference_batch_size=int(eval_args.inference_batch_size),
            inference_timeout_s=float(eval_args.inference_timeout_s),
            new_model_state={k: v.detach().cpu() for k, v in self.student.state_dict().items()},
            new_model_config=dict(self.student_model_config),
            opponent_nnet_state=None,
            opponent_nnet_config=None,
            opponent_model_spec={
                "state_dict": model_spec["state_dict"],
                "config": model_spec["config"],
            },
        )

        win_rate = (wins + 0.5 * draws) / float(games)
        return {
            "label": label,
            "wins": int(wins),
            "losses": int(losses),
            "draws": int(draws),
            "games": int(games),
            "win_rate": float(win_rate),
        }

    def _evaluate_baselines(self, iteration: int) -> Dict[str, Optional[Dict[str, object]]]:
        tasks = [
            ("teacher_eval", self.teacher_model_spec, int(self.args.eval_games_vs_teacher), "teacher"),
            ("v21_eval", self.v21_model_spec, int(self.args.eval_games_vs_v21_high), "v2.1_high"),
        ]

        active_tasks = [task for task in tasks if task[1] is not None and task[2] > 0]
        if not active_tasks:
            return {"teacher_eval": None, "v21_eval": None}

        if bool(getattr(self.args, "shared_evaluation_services", True)) and len(active_tasks) > 1:
            eval_args = self._build_eval_args(iteration)
            total_games = sum(task[2] for task in active_tasks)
            total_workers = self._resolve_worker_count(total_games, eval_args.num_mcts_threads)
            counts_list = execute_evaluation_group_parallel(
                args=eval_args,
                matches=[
                    {
                        "label": label,
                        "num_games": games,
                        "opponent_model_spec": {
                            "state_dict": model_spec["state_dict"],
                            "config": model_spec["config"],
                        },
                    }
                    for _, model_spec, games, label in active_tasks
                ],
                total_workers=total_workers,
                shared_inference_device=self.args.shared_inference_device,
                inference_batch_size=int(eval_args.inference_batch_size),
                inference_timeout_s=float(eval_args.inference_timeout_s),
                new_model_state={k: v.detach().cpu() for k, v in self.student.state_dict().items()},
                new_model_config=dict(self.student_model_config),
            )
            result_map: Dict[str, Optional[Dict[str, object]]] = {"teacher_eval": None, "v21_eval": None}
            for (key, _, games, label), counts in zip(active_tasks, counts_list):
                wins, losses, draws = counts
                result_map[key] = {
                    "label": label,
                    "wins": int(wins),
                    "losses": int(losses),
                    "draws": int(draws),
                    "games": int(games),
                    "win_rate": float((wins + 0.5 * draws) / games),
                }
            return result_map

        def _run_single(task):
            key, model_spec, games, label = task
            return key, self._evaluate_against_model_spec(iteration, model_spec, games, label)

        result_map: Dict[str, Optional[Dict[str, object]]] = {"teacher_eval": None, "v21_eval": None}
        run_parallel = bool(getattr(self.args, "baseline_eval_parallelize", True)) and len(active_tasks) > 1
        if run_parallel:
            with ThreadPoolExecutor(max_workers=len(active_tasks)) as executor:
                for key, payload in executor.map(_run_single, active_tasks):
                    result_map[key] = payload
        else:
            for task in active_tasks:
                key, payload = _run_single(task)
                result_map[key] = payload

        return result_map

    def _promote_best_if_needed(self, iteration: int, best_results: List[Dict[str, object]]) -> bool:
        if not bool(self.args.enable_best_refresh):
            return False

        threshold = float(self.args.best_update_threshold)
        required = max(1, int(self.args.best_eval_required_generations))

        if not best_results:
            improved = True
        else:
            required_results = best_results[: min(required, len(best_results))]
            improved = all(item["win_rate"] >= threshold for item in required_results)

        if not improved:
            return False

        previous_recent = copy.deepcopy(self.best_recent_state)
        current_state = {
            "state_dict": {k: v.detach().cpu() for k, v in self.student.state_dict().items()},
            "model_config": dict(self.student_model_config),
            "iteration": int(iteration),
            "label": f"checkpoint_{iteration}",
            "win_rate": float(min([x["win_rate"] for x in best_results], default=1.0)),
        }

        self.best_recent_state = current_state
        self.best_older_state = previous_recent
        self._save_best_files()
        return True

    def _save_best_files(self):
        recent_path = self.run_dir / "best_recent.pth.tar"
        older_path = self.run_dir / "best_older.pth.tar"

        if self.best_recent_state is not None:
            self._atomic_torch_save(self.best_recent_state, recent_path)
        if self.best_older_state is not None:
            self._atomic_torch_save(self.best_older_state, older_path)

    def _run_drift_recovery_if_needed(
        self,
        iteration: int,
        improved: bool,
        teacher_eval: Optional[Dict[str, object]],
        v21_eval: Optional[Dict[str, object]],
    ):
        if improved:
            self.no_refresh_streak = 0
            return

        self.no_refresh_streak += 1
        if self.no_refresh_streak < int(self.args.drift_no_refresh_patience):
            return

        if teacher_eval is None or v21_eval is None:
            return

        drift_threshold = float(self.args.drift_min_win_rate)
        if teacher_eval["win_rate"] < drift_threshold and v21_eval["win_rate"] < drift_threshold:
            self.recovery_until_iteration = max(
                self.recovery_until_iteration,
                iteration + int(self.args.recovery_boost_iterations),
            )
            logging.warning(
                "Drift guard triggered at iteration %s. Recovery mode until iteration %s.",
                iteration,
                self.recovery_until_iteration,
            )

    def evaluate_and_refresh_best(self, iteration: int) -> Dict[str, object]:
        best_results = self._evaluate_against_best_generations(iteration)
        baseline_results = self._evaluate_baselines(iteration)
        teacher_eval = baseline_results.get("teacher_eval")
        v21_eval = baseline_results.get("v21_eval")

        improved = self._promote_best_if_needed(iteration, best_results)
        self._run_drift_recovery_if_needed(iteration, improved, teacher_eval, v21_eval)

        result = {
            "iteration": int(iteration),
            "improved": bool(improved),
            "best_results": best_results,
            "teacher_eval": teacher_eval,
            "v21_eval": v21_eval,
            "no_refresh_streak": int(self.no_refresh_streak),
            "recovery_until_iteration": int(self.recovery_until_iteration),
        }
        self.eval_history.append(result)
        return result

    def _source_loss_scales(self, iteration: int) -> Dict[int, float]:
        teacher_weight = self._resolve_teacher_weight(iteration)
        self_play_weight = float(self.args.self_play_loss_weight)
        return {
            SOURCE_SELF_PLAY: float(self_play_weight),
            SOURCE_TEACHER_WARMUP: float(teacher_weight),
            SOURCE_TEACHER_LOSS: float(teacher_weight * float(self.args.adversarial_loss_teacher_weight)),
            SOURCE_STUDENT_WIN: float(self_play_weight * float(self.args.adversarial_win_student_weight)),
            SOURCE_TEACHER_RESPONSE: float(teacher_weight * float(self.args.adversarial_win_teacher_response_weight)),
        }

    def train_network(self, examples: List[Tuple[np.ndarray, np.ndarray, float, int]], iteration: int) -> Dict[str, object]:
        self.student.train()
        try:
            import psutil

            memory_info = psutil.Process(os.getpid()).memory_info()
            cpu_memory_before = int(memory_info.rss)
        except (ImportError, OSError):
            cpu_memory_before = 0
        dataset_pack_start = time.perf_counter()
        dataset = DistillationDataset(
            examples,
            compact_boards=bool(getattr(self.args, "compact_training_dataset", True)),
        )
        dataset_pack_sec = time.perf_counter() - dataset_pack_start
        dataset_bytes = sum(
            int(tensor.numel() * tensor.element_size())
            for tensor in (dataset.boards, dataset.pis, dataset.vs, dataset.sources)
        )
        try:
            memory_info = psutil.Process(os.getpid()).memory_info()
            cpu_peak_memory = int(getattr(memory_info, "peak_wset", memory_info.rss))
        except (NameError, OSError):
            cpu_peak_memory = 0

        if len(dataset) == 0:
            return {
                "samples": 0,
                "total_loss": 0.0,
                "teacher_policy_loss": 0.0,
                "teacher_value_loss": 0.0,
                "self_policy_loss": 0.0,
                "self_value_loss": 0.0,
                "duration_sec": 0.0,
                "dataset_pack_sec": float(dataset_pack_sec),
                "dataset_cpu_bytes": int(dataset_bytes),
                "cpu_peak_memory_bytes": int(cpu_peak_memory),
                "cpu_dataset_rss_delta_bytes": int(max(0, cpu_peak_memory - cpu_memory_before)),
                "gpu_copy_sec": 0.0,
                "samples_per_sec": 0.0,
            }

        loader = DataLoader(
            dataset,
            batch_size=max(1, int(self.args.batch_size)),
            shuffle=True,
            num_workers=0,
            pin_memory=self.args.train_device.startswith("cuda"),
        )

        use_cuda = self.args.train_device.startswith("cuda")

        teacher_weight = self._resolve_teacher_weight(iteration)
        source_scales = self._source_loss_scales(iteration)
        policy_weight = float(self.args.policy_loss_weight)
        value_weight = float(self.args.value_loss_weight)

        total_loss_sum = 0.0
        teacher_policy_sum = 0.0
        teacher_value_sum = 0.0
        self_policy_sum = 0.0
        self_value_sum = 0.0
        batch_count = 0
        processed_samples = 0
        copy_events = []

        device = torch.device(self.args.train_device)
        start_ts = time.perf_counter()

        for _ in range(max(1, int(self.args.epochs))):
            for boards, target_pi, target_v, source in loader:
                if use_cuda:
                    copy_start = torch.cuda.Event(enable_timing=True)
                    copy_end = torch.cuda.Event(enable_timing=True)
                    copy_start.record()
                boards = boards.to(device, non_blocking=True)
                target_pi = target_pi.to(device, non_blocking=True)
                target_v = target_v.to(device, non_blocking=True).view(-1)
                source = source.to(device, non_blocking=True)
                if use_cuda:
                    copy_end.record()
                    copy_events.append((copy_start, copy_end))
                if boards.ndim == 4:
                    boards = torch.stack((boards > 0, boards < 0), dim=1).to(dtype=torch.float32)

                self.optimizer.zero_grad(set_to_none=True)

                with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
                    out_log_pi, out_v = self.student(boards)
                    out_v = out_v.view(-1)

                    ce = -(target_pi * out_log_pi).sum(dim=1)
                    mse = F.mse_loss(out_v, target_v, reduction="none")

                    teacher_mask = (source == SOURCE_TEACHER_WARMUP) | (source == SOURCE_TEACHER_LOSS) | (source == SOURCE_TEACHER_RESPONSE)
                    self_mask = (source == SOURCE_SELF_PLAY) | (source == SOURCE_STUDENT_WIN)

                    teacher_policy_loss = ce[teacher_mask].mean() if torch.any(teacher_mask) else ce.new_zeros(())
                    teacher_value_loss = mse[teacher_mask].mean() if torch.any(teacher_mask) else mse.new_zeros(())
                    self_policy_loss = ce[self_mask].mean() if torch.any(self_mask) else ce.new_zeros(())
                    self_value_loss = mse[self_mask].mean() if torch.any(self_mask) else mse.new_zeros(())

                    total_loss = ce.new_zeros(())
                    has_weighted_source = False
                    for source_id, source_scale in source_scales.items():
                        if source_scale <= 0.0:
                            continue
                        source_mask = source == int(source_id)
                        if not torch.any(source_mask):
                            continue
                        src_policy = ce[source_mask].mean()
                        src_value = mse[source_mask].mean()
                        total_loss = total_loss + float(source_scale) * (policy_weight * src_policy + value_weight * src_value)
                        has_weighted_source = True

                    if not has_weighted_source:
                        total_loss = policy_weight * ce.mean() + value_weight * mse.mean()

                if use_cuda:
                    self.grad_scaler.scale(total_loss).backward()
                    self.grad_scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=5.0)
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=5.0)
                    self.optimizer.step()

                total_loss_sum += float(total_loss.item())
                teacher_policy_sum += float(teacher_policy_loss.item())
                teacher_value_sum += float(teacher_value_loss.item())
                self_policy_sum += float(self_policy_loss.item())
                self_value_sum += float(self_value_loss.item())
                batch_count += 1
                processed_samples += int(target_v.shape[0])

        duration = time.perf_counter() - start_ts
        gpu_copy_sec = 0.0
        if copy_events:
            torch.cuda.synchronize(device)
            gpu_copy_sec = sum(start.elapsed_time(end) for start, end in copy_events) / 1000.0
        denom = float(max(1, batch_count))
        return {
            "samples": int(len(dataset)),
            "teacher_weight": float(teacher_weight),
            "source_scales": {str(k): float(v) for k, v in source_scales.items()},
            "total_loss": total_loss_sum / denom,
            "teacher_policy_loss": teacher_policy_sum / denom,
            "teacher_value_loss": teacher_value_sum / denom,
            "self_policy_loss": self_policy_sum / denom,
            "self_value_loss": self_value_sum / denom,
            "duration_sec": float(duration),
            "dataset_pack_sec": float(dataset_pack_sec),
            "dataset_cpu_bytes": int(dataset_bytes),
            "cpu_peak_memory_bytes": int(cpu_peak_memory),
            "cpu_dataset_rss_delta_bytes": int(max(0, cpu_peak_memory - cpu_memory_before)),
            "gpu_copy_sec": float(gpu_copy_sec),
            "samples_per_sec": float(processed_samples / max(duration, 1e-9)),
        }

    def _maybe_run_speed_check(self, iteration: int):
        if not bool(self.args.enable_speed_check):
            return None

        checkpoints = set(int(x) for x in self.args.speed_check_iterations)
        final_iter = int(self.args.num_iterations)
        marker = final_iter if -1 in checkpoints else None

        should_run = iteration in checkpoints or (marker is not None and iteration == final_iter)
        if not should_run:
            return None

        self.student.eval()
        device = torch.device("cpu")
        model = self._build_student_model()
        model.load_state_dict({k: v.detach().cpu() for k, v in self.student.state_dict().items()}, strict=True)
        model.to(device)

        dummy = torch.randn(1, 2, MAX_LAYERS, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32, device=device)
        warmup = 5
        trials = 30
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(dummy)
            start = time.perf_counter()
            for _ in range(trials):
                _ = model(dummy)
            elapsed = time.perf_counter() - start

        latency_ms = (elapsed / trials) * 1000.0
        return {"iteration": int(iteration), "cpu_latency_ms": float(latency_ms), "trials": int(trials)}

    def _format_duration(self, seconds: float) -> str:
        total = max(0.0, float(seconds))
        minutes = int(total // 60)
        sec = total - minutes * 60
        if minutes > 0:
            return f"{minutes:02d}:{sec:05.2f}"
        return f"{sec:.2f}s"

    def _log_iteration(self, summary: Dict[str, object]):
        data_mix = summary["data_mix"]
        train = summary["train"]
        self_play_stats = summary.get("self_play_stats") or {}
        adversarial_stats = summary.get("adversarial_stats") or {}

        lines = [
            (
                "Iter %s/%s | duration=%s | mode=%s | lr=%.6f"
                % (
                    summary["iteration"],
                    int(self.args.num_iterations),
                    self._format_duration(summary.get("iteration_duration_sec", 0.0)),
                    data_mix.get("mode", "unknown"),
                    summary["learning_rate"],
                )
            ),
            (
                "  Self-Play: games=%s | new_samples=%s | avg_steps=%.2f | var_steps=%.2f | "
                "policy_entropy=%.4f | min/max=%s/%s | long_games=%s | short_games=%s | filtered=%s | duration=%s"
                % (
                    self_play_stats.get("games", 0),
                    summary.get("self_play_samples", 0),
                    self_play_stats.get("mean_steps", 0.0),
                    self_play_stats.get("variance_steps", 0.0),
                    self_play_stats.get("mean_policy_entropy", 0.0),
                    self_play_stats.get("min_steps", 0),
                    self_play_stats.get("max_steps", 0),
                    self_play_stats.get("long_games", 0),
                    self_play_stats.get("short_games", 0),
                    self_play_stats.get("filtered_games", 0),
                    self._format_duration(summary.get("self_play_duration_sec", 0.0)),
                )
            ),
            (
                "  Self-Play-Extremes: shortest=%s | longest=%s"
                % (
                    json.dumps(self_play_stats.get("shortest_game", {}), ensure_ascii=False),
                    json.dumps(self_play_stats.get("longest_game", {}), ensure_ascii=False),
                )
            ),
            (
                "  Adversarial: games=%s | W/L/D=%s/%s/%s | loss_examples=%s | win_student=%s | "
                "win_teacher_response=%s | avg_steps=%.2f | var_steps=%.2f | policy_entropy=%.4f | "
                "min/max=%s/%s | long_games=%s | short_games=%s | duration=%s"
                % (
                    data_mix.get("adversarial_games", 0),
                    data_mix.get("adversarial_wins", 0),
                    data_mix.get("adversarial_losses", 0),
                    data_mix.get("adversarial_draws", 0),
                    data_mix.get("adversarial_loss_examples", 0),
                    data_mix.get("adversarial_win_student_examples", 0),
                    data_mix.get("adversarial_win_teacher_response_examples", 0),
                    adversarial_stats.get("mean_steps", 0.0),
                    adversarial_stats.get("variance_steps", 0.0),
                    adversarial_stats.get("mean_policy_entropy", 0.0),
                    adversarial_stats.get("min_steps", 0),
                    adversarial_stats.get("max_steps", 0),
                    adversarial_stats.get("long_games", 0),
                    adversarial_stats.get("short_games", 0),
                    self._format_duration(summary.get("adversarial_duration_sec", 0.0)),
                )
            ),
            (
                "  Adversarial-Extremes: shortest=%s | longest=%s"
                % (
                    json.dumps(adversarial_stats.get("shortest_game", {}), ensure_ascii=False),
                    json.dumps(adversarial_stats.get("longest_game", {}), ensure_ascii=False),
                )
            ),
            (
                "  Train: total_samples=%s | self_play_samples=%s | teacher_samples=%s | "
                "teacher_weight=%.3f | loss(total=%.6f, teacher_pi=%.6f, teacher_v=%.6f, self_pi=%.6f, self_v=%.6f) | "
                "duration=%s"
                % (
                    train.get("samples", 0),
                    data_mix.get("self_play_samples", 0),
                    data_mix.get("teacher_samples", 0),
                    train.get("teacher_weight", 0.0),
                    train.get("total_loss", 0.0),
                    train.get("teacher_policy_loss", 0.0),
                    train.get("teacher_value_loss", 0.0),
                    train.get("self_policy_loss", 0.0),
                    train.get("self_value_loss", 0.0),
                    self._format_duration(train.get("duration_sec", 0.0)),
                )
            ),
            (
                "  History: window(self_play/adversarial/teacher)=%s/%s/%s entries=%s/%s/%s | "
                "iter_span=%s/%s/%s | current_add=%s/%s/%s | merged_total=%s | fallback=%s"
                % (
                    data_mix.get("window_self_play_samples", 0),
                    data_mix.get("window_adversarial_samples", 0),
                    data_mix.get("window_teacher_samples", 0),
                    data_mix.get("window_self_play_entries", 0),
                    data_mix.get("window_adversarial_entries", 0),
                    data_mix.get("window_teacher_entries", 0),
                    data_mix.get("window_self_play_iter_span", "-"),
                    data_mix.get("window_adversarial_iter_span", "-"),
                    data_mix.get("window_teacher_iter_span", "-"),
                    data_mix.get("current_self_play_samples", 0),
                    data_mix.get("current_adversarial_samples", 0),
                    data_mix.get("current_teacher_pure_samples", 0),
                    data_mix.get("window_total_samples", train.get("samples", 0)),
                    data_mix.get("fallback_history_samples", 0),
                )
            ),
        ]

        logging.info(
            "Iter %s | mode=%s self_play=%s teacher=%s adv_w/l/d=%s/%s/%s teacher_w=%.3f loss=%.4f lr=%.6f",
            summary["iteration"],
            data_mix.get("mode", "unknown"),
            data_mix.get("self_play_samples", 0),
            data_mix.get("teacher_samples", 0),
            data_mix.get("adversarial_wins", 0),
            data_mix.get("adversarial_losses", 0),
            data_mix.get("adversarial_draws", 0),
            train["teacher_weight"],
            train["total_loss"],
            summary["learning_rate"],
        )

        eval_result = summary.get("eval")
        if eval_result:
            logging.info(
                "Eval iter %s | improved=%s no_refresh=%s",
                summary["iteration"],
                eval_result["improved"],
                eval_result["no_refresh_streak"],
            )
            best_results = list(eval_result.get("best_results") or [])
            if best_results:
                compact = " | ".join(
                    [
                        (
                            f"{item.get('label', 'unknown')}:"
                            f"W/L/D={item.get('wins', 0)}/{item.get('losses', 0)}/{item.get('draws', 0)}"
                            f", wr={item.get('win_rate', 0.0):.3f}"
                        )
                        for item in best_results
                    ]
                )
                lines.append(
                    (
                        f"  Eval-Best-Generations: {compact} | "
                        f"threshold={float(self.args.best_update_threshold):.3f} | "
                        f"decision={'pass' if eval_result.get('improved') else 'fail'}"
                    )
                )

            teacher_eval = eval_result.get("teacher_eval")
            if teacher_eval is not None:
                lines.append(
                    (
                        f"  Eval-Teacher: games={teacher_eval['games']} | "
                        f"W/L/D={teacher_eval['wins']}/{teacher_eval['losses']}/{teacher_eval['draws']} | "
                        f"win_rate={teacher_eval['win_rate']:.3f}"
                    )
                )
            v21_eval = eval_result.get("v21_eval")
            if v21_eval is not None:
                lines.append(
                    (
                        f"  Eval-v2.1-High: games={v21_eval['games']} | "
                        f"W/L/D={v21_eval['wins']}/{v21_eval['losses']}/{v21_eval['draws']} | "
                        f"win_rate={v21_eval['win_rate']:.3f}"
                    )
                )

        speed_check = summary.get("speed_check")
        if speed_check is not None:
            lines.append(
                (
                    f"  Speed: cpu_latency_ms={float(speed_check.get('cpu_latency_ms', 0.0)):.3f} "
                    f"(trials={int(speed_check.get('trials', 0))})"
                )
            )

        for line in lines:
            logging.info(line)
        self._append_info_log(lines)

    def train(self):
        last_iteration = self.start_iteration - 1
        for iteration in range(self.start_iteration, int(self.args.num_iterations) + 1):
            start_ts = time.perf_counter()
            self._apply_learning_rate_for_iteration(iteration)
            logging.info("Starting distillation iteration %s/%s", iteration, self.args.num_iterations)

            self_play_examples: List[Tuple[np.ndarray, np.ndarray, float]] = []
            self_play_game_results: List[Dict[str, object]] = []
            self_play_duration = 0.0
            adversarial_data: Dict[str, object] = {
                "games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "loss_examples": [],
                "win_student_examples": [],
                "win_teacher_response_examples": [],
                "game_results": [],
            }
            adversarial_duration = 0.0

            if self._is_hot_start_iteration(iteration):
                merged_examples, data_mix = self._build_hot_start_training_examples(iteration)
            else:
                self_play_start = time.perf_counter()
                self_play_examples, self_play_game_results = self._run_student_self_play(iteration)
                self_play_duration = time.perf_counter() - self_play_start

                adversarial_start = time.perf_counter()
                adversarial_data = self._run_student_teacher_adversarial(
                    iteration,
                    self_play_games=int(self.args.num_self_play_games),
                )
                adversarial_duration = time.perf_counter() - adversarial_start
                merged_examples, data_mix = self._compose_training_examples(
                    self_play_examples,
                    adversarial_data,
                    iteration,
                )

            self_play_stats = self._summarize_runtime_games(self_play_game_results)
            adversarial_stats = self._summarize_runtime_games(list(adversarial_data.get("game_results") or []))

            train_metrics = self.train_network(merged_examples, iteration)
            if self.scheduler is not None:
                self.scheduler.step()

            eval_result = None
            if iteration % int(self.args.eval_interval) == 0:
                eval_result = self.evaluate_and_refresh_best(iteration)

            speed_check = self._maybe_run_speed_check(iteration)

            iteration_duration = time.perf_counter() - start_ts
            summary = {
                "iteration": int(iteration),
                "iteration_duration_sec": float(iteration_duration),
                "self_play_samples": int(len(self_play_examples)),
                "self_play_duration_sec": float(self_play_duration),
                "self_play_game_results": list(self_play_game_results),
                "self_play_stats": self_play_stats,
                "adversarial_duration_sec": float(adversarial_duration),
                "adversarial_game_results": list(adversarial_data.get("game_results") or []),
                "adversarial_stats": adversarial_stats,
                "data_mix": data_mix,
                "train": train_metrics,
                "eval": eval_result,
                "speed_check": speed_check,
                "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            }
            self.iteration_metrics_history.append(summary)
            self._compact_iteration_metrics_history(keep_last_full=True)
            self._log_iteration(summary)

            if iteration % int(self.args.checkpoint_interval) == 0:
                self.save_checkpoint(iteration)

            last_iteration = iteration

        if last_iteration >= self.start_iteration:
            logging.info(
                "Training loop completed at iteration %s; latest periodic checkpoint is iteration %s.",
                last_iteration,
                self.last_checkpoint_iteration if self.last_checkpoint_iteration > 0 else "none",
            )
            self._write_final_report(last_iteration)

    def _write_final_report(self, final_iteration: int):
        report_path = self.run_dir / "FINAL_REPORT.txt"
        lines = [
            "Distillation Final Report",
            f"final_iteration: {final_iteration}",
            f"latest_checkpoint_iteration: {self.last_checkpoint_iteration}",
            f"active_model_preset: {self.args.active_model_preset}",
            f"student_model_config: {json.dumps(self.student_model_config, ensure_ascii=False)}",
            f"teacher_cache_samples: {len(self.teacher_cache_examples)}",
            f"teacher_cache_games: {len(self.teacher_cache_games)}",
            f"teacher_history_pool_samples: {len(self.teacher_history_pool)}",
            (
                "history_windows: "
                f"self_play_entries={len(self.self_play_examples_history)}, "
                f"adversarial_entries={len(self.adversarial_examples_history)}, "
                f"teacher_entries={len(self.teacher_pure_examples_history)}"
            ),
            f"eval_records: {len(self.eval_history)}",
            f"run_dir: {self.run_dir}",
        ]

        if self.eval_history:
            last_eval = self.eval_history[-1]
            lines.append(f"last_eval_improved: {last_eval.get('improved', False)}")

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        logging.info("Wrote final report: %s", report_path)
