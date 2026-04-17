from __future__ import annotations

from dataclasses import dataclass
import glob
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from .feature_extractor import CandidateFeatureExtractor
except ImportError:
    from feature_extractor import CandidateFeatureExtractor


@dataclass
class HistoryDatasetConfig:
    winner_weight: float = 1.0
    loser_weight: float = 0.35
    draw_weight: float = 0.7
    include_human_moves: bool = True
    include_model_moves: bool = True


class HumanHistoryDataset(Dataset):
    """Converts human-vs-model history JSON files into candidate-policy training samples."""

    def __init__(
        self,
        history_paths: Sequence[str | Path],
        feature_extractor: CandidateFeatureExtractor,
        config: HistoryDatasetConfig | None = None,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.config = config or HistoryDatasetConfig()
        self.samples: List[dict] = []
        self._load_files(history_paths)

    @staticmethod
    def resolve_history_paths(pattern_or_paths: Sequence[str | Path]) -> List[Path]:
        resolved: List[Path] = []
        for item in pattern_or_paths:
            text = str(item)
            if any(ch in text for ch in ("*", "?", "[")):
                for path in sorted(glob.glob(text)):
                    p = Path(path)
                    if p.is_file() and p.suffix.lower() == ".json":
                        resolved.append(p)
            else:
                p = Path(text)
                if p.is_dir():
                    resolved.extend(sorted(x for x in p.glob("*.json") if x.is_file()))
                elif p.is_file() and p.suffix.lower() == ".json":
                    resolved.append(p)
        unique = []
        seen = set()
        for p in resolved:
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            unique.append(p)
        return unique

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        feat = self.feature_extractor.extract(sample["board"], sample["player"])

        global_features = torch.from_numpy(feat["global"])
        candidate_features = torch.from_numpy(feat["candidate"])
        valid_mask = torch.from_numpy(feat["valid_mask"])

        target_idx = torch.tensor(int(sample["target_idx"]), dtype=torch.long)
        sample_weight = torch.tensor(float(sample["weight"]), dtype=torch.float32)

        teacher_probs = sample.get("teacher_probs")
        if teacher_probs is None:
            teacher_probs_tensor = torch.full((self.feature_extractor.candidate_count,), -1.0, dtype=torch.float32)
            teacher_valid = torch.tensor(0.0, dtype=torch.float32)
        else:
            teacher_probs_tensor = torch.tensor(np.asarray(teacher_probs, dtype=np.float32))
            teacher_valid = torch.tensor(1.0, dtype=torch.float32)

        teacher_value = sample.get("teacher_value")
        if teacher_value is None:
            teacher_value_tensor = torch.tensor(0.0, dtype=torch.float32)
            teacher_value_valid = torch.tensor(0.0, dtype=torch.float32)
        else:
            teacher_value_tensor = torch.tensor(float(teacher_value), dtype=torch.float32)
            teacher_value_valid = torch.tensor(1.0, dtype=torch.float32)

        return {
            "global_features": global_features,
            "candidate_features": candidate_features,
            "valid_mask": valid_mask,
            "target_idx": target_idx,
            "sample_weight": sample_weight,
            "teacher_probs": teacher_probs_tensor,
            "teacher_valid": teacher_valid,
            "teacher_value": teacher_value_tensor,
            "teacher_value_valid": teacher_value_valid,
        }

    def _load_files(self, history_paths: Sequence[str | Path]) -> None:
        file_paths = self.resolve_history_paths(history_paths)
        for path in file_paths:
            try:
                payload = json.loads(Path(path).read_text(encoding="utf-8"))
            except Exception:
                continue
            self._load_single_payload(payload)

    def _load_single_payload(self, payload: dict) -> None:
        moves = payload.get("moves") or []
        if not moves:
            return

        winner_actor = str((payload.get("result") or {}).get("winner") or "").strip().lower()
        board = np.zeros(
            (
                self.feature_extractor.max_layers,
                self.feature_extractor.board_size,
                self.feature_extractor.board_size,
            ),
            dtype=np.int8,
        )

        for move in moves:
            actor = str(move.get("actor") or "").strip().lower()
            if actor == "human" and not self.config.include_human_moves:
                self._apply_move_if_possible(board, move)
                continue
            if actor == "model" and not self.config.include_model_moves:
                self._apply_move_if_possible(board, move)
                continue

            player = int(move.get("player", 0))
            coords = move.get("coords") or {}
            row = int(coords.get("row", -1))
            col = int(coords.get("col", -1))
            layer = int(coords.get("layer", -1))

            if player == 0 or row < 0 or col < 0 or layer < 0:
                self._apply_move_if_possible(board, move)
                continue

            target_idx = self.feature_extractor.candidate_index(row, col)
            feat = self.feature_extractor.extract(board, player)
            action_map = feat["candidate_action_map"]
            expected_action = int(action_map[target_idx]) if 0 <= target_idx < len(action_map) else -1
            expected_layer = expected_action // self.feature_extractor.action_plane if expected_action >= 0 else -1

            if expected_action < 0 or expected_layer != layer:
                self._apply_move_if_possible(board, move)
                continue

            weight = self._sample_weight(actor, winner_actor)
            self.samples.append(
                {
                    "board": np.array(board, copy=True, dtype=np.int8),
                    "player": int(player),
                    "target_idx": int(target_idx),
                    "weight": float(weight),
                    "teacher_probs": None,
                }
            )

            board[layer, row, col] = int(player)

    def _sample_weight(self, actor: str, winner_actor: str) -> float:
        if winner_actor == "draw":
            return float(self.config.draw_weight)
        if actor == winner_actor:
            return float(self.config.winner_weight)
        return float(self.config.loser_weight)

    def _apply_move_if_possible(self, board: np.ndarray, move: dict) -> None:
        coords = move.get("coords") or {}
        player = int(move.get("player", 0))
        row = int(coords.get("row", -1))
        col = int(coords.get("col", -1))
        layer = int(coords.get("layer", -1))
        if player == 0:
            return
        if layer < 0 or row < 0 or col < 0:
            return
        if layer >= board.shape[0] or row >= board.shape[1] or col >= board.shape[2]:
            return
        if board[layer, row, col] != 0:
            return
        board[layer, row, col] = int(player)
