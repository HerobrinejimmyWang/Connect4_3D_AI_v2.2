from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = CURRENT_DIR.parent
TRAINING_DIR = WORKSPACE_ROOT / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from game_rules import BOARD_SIZE, MAX_LAYERS, GameRules


@dataclass(frozen=True)
class FeatureShape:
    global_dim: int
    candidate_dim: int
    candidate_count: int


class CandidateFeatureExtractor:
    """Extracts global + per-candidate handcrafted features for a 5x5 top-plane policy."""

    def __init__(
        self,
        board_size: int = BOARD_SIZE,
        max_layers: int = MAX_LAYERS,
        connect_n: int = 4,
    ) -> None:
        self.board_size = int(board_size)
        self.max_layers = int(max_layers)
        self.connect_n = int(connect_n)
        self.candidate_count = self.board_size * self.board_size
        self.action_plane = self.board_size * self.board_size
        self.game = GameRules()

        self._center = (self.board_size - 1) / 2.0
        self._max_dist = np.sqrt(2.0) * self._center if self._center > 0 else 1.0
        self._directions = self._build_unique_directions()
        self._position_prior = self._build_position_prior()

        self._global_dim = 20
        self._candidate_dim = 28

    @property
    def feature_shape(self) -> FeatureShape:
        return FeatureShape(
            global_dim=self._global_dim,
            candidate_dim=self._candidate_dim,
            candidate_count=self.candidate_count,
        )

    def candidate_index(self, row: int, col: int) -> int:
        return int(row) * self.board_size + int(col)

    def candidate_to_row_col(self, candidate_idx: int) -> Tuple[int, int]:
        candidate_idx = int(candidate_idx)
        row = candidate_idx // self.board_size
        col = candidate_idx % self.board_size
        return row, col

    def get_candidate_action_map(self, board: np.ndarray) -> np.ndarray:
        board = np.asarray(board, dtype=np.int8)
        mapping = np.full((self.candidate_count,), -1, dtype=np.int64)
        for row in range(self.board_size):
            for col in range(self.board_size):
                layer = self._next_open_layer(board, row, col)
                if layer is None:
                    continue
                candidate_idx = self.candidate_index(row, col)
                mapping[candidate_idx] = int(layer) * self.action_plane + candidate_idx
        return mapping

    def extract(
        self,
        board: np.ndarray,
        player: int,
    ) -> Dict[str, np.ndarray]:
        board = np.asarray(board, dtype=np.int8)
        player = int(player)

        candidate_action_map = self.get_candidate_action_map(board)
        valid_mask = (candidate_action_map >= 0).astype(np.float32)

        own_wins_now = self._immediate_win_candidates(board, player)
        opp_wins_now = self._immediate_win_candidates(board, -player)

        global_features = self._extract_global_features(
            board=board,
            player=player,
            valid_mask=valid_mask,
            own_wins_now=own_wins_now,
            opp_wins_now=opp_wins_now,
        )

        candidate_features = np.zeros((self.candidate_count, self._candidate_dim), dtype=np.float32)
        for idx in range(self.candidate_count):
            row, col = self.candidate_to_row_col(idx)
            action = int(candidate_action_map[idx])
            is_valid = action >= 0

            if is_valid:
                layer = action // self.action_plane
                move_board = np.array(board, copy=True)
                move_board[layer, row, col] = player

                own_future_wins = self._immediate_win_candidates(move_board, player)
                opp_future_wins = self._immediate_win_candidates(move_board, -player)
                line_stats = self._line_stats_for_move(move_board, player, layer, row, col)
                own_n, opp_n, empty_n = self._neighbor_counts(move_board, player, layer, row, col)

                own_future_count = float(len(own_future_wins))
                opp_future_count = float(len(opp_future_wins))
                future_pressure = (own_future_count - opp_future_count) / float(max(1, self.candidate_count))
                creates_double_threat = 1.0 if own_future_count >= 2.0 else 0.0
                blocks_opp_now = 1.0 if idx in opp_wins_now else 0.0
                immediate_win = 1.0 if idx in own_wins_now else 0.0
            else:
                layer = -1
                line_stats = {
                    "len2_plus": 0.0,
                    "len3_plus": 0.0,
                    "open_two": 0.0,
                    "open_three": 0.0,
                    "open_four": 0.0,
                    "potential": 0.0,
                }
                own_n = opp_n = empty_n = 0.0
                own_future_count = 0.0
                opp_future_count = 0.0
                future_pressure = 0.0
                creates_double_threat = 0.0
                blocks_opp_now = 0.0
                immediate_win = 0.0

            col_view = board[:, row, col]
            own_col = float(np.sum(col_view == player))
            opp_col = float(np.sum(col_view == -player))
            fill_count = own_col + opp_col
            fill_ratio = fill_count / float(self.max_layers)

            row_norm = row / float(max(1, self.board_size - 1))
            col_norm = col / float(max(1, self.board_size - 1))
            dist_center = np.sqrt((row - self._center) ** 2 + (col - self._center) ** 2)
            dist_center_norm = dist_center / float(max(1e-6, self._max_dist))
            layer_norm = (layer / float(max(1, self.max_layers - 1))) if layer >= 0 else 1.0
            support_filled = 1.0 if layer in (0, -1) else float(board[layer - 1, row, col] != 0)
            vertical_stack_after = ((fill_count + (1.0 if is_valid else 0.0)) / float(self.max_layers))
            is_edge = 1.0 if row in (0, self.board_size - 1) or col in (0, self.board_size - 1) else 0.0
            is_corner = 1.0 if row in (0, self.board_size - 1) and col in (0, self.board_size - 1) else 0.0

            candidate_features[idx] = np.asarray(
                [
                    1.0 if is_valid else 0.0,
                    row_norm,
                    col_norm,
                    dist_center_norm,
                    layer_norm,
                    fill_ratio,
                    own_col / float(max(1, self.max_layers)),
                    opp_col / float(max(1, self.max_layers)),
                    support_filled,
                    immediate_win,
                    blocks_opp_now,
                    line_stats["len2_plus"],
                    line_stats["len3_plus"],
                    line_stats["open_two"],
                    line_stats["open_three"],
                    line_stats["open_four"],
                    line_stats["potential"],
                    creates_double_threat,
                    own_future_count / float(max(1, self.candidate_count)),
                    opp_future_count / float(max(1, self.candidate_count)),
                    future_pressure,
                    own_n,
                    opp_n,
                    empty_n,
                    vertical_stack_after,
                    is_edge,
                    is_corner,
                    self._position_prior[row, col],
                ],
                dtype=np.float32,
            )

        return {
            "global": global_features.astype(np.float32),
            "candidate": candidate_features.astype(np.float32),
            "valid_mask": valid_mask.astype(np.float32),
            "candidate_action_map": candidate_action_map.astype(np.int64),
        }

    def _extract_global_features(
        self,
        board: np.ndarray,
        player: int,
        valid_mask: np.ndarray,
        own_wins_now: Set[int],
        opp_wins_now: Set[int],
    ) -> np.ndarray:
        total_cells = float(self.max_layers * self.candidate_count)
        occupied = float(np.sum(board != 0))
        own_count = float(np.sum(board == player))
        opp_count = float(np.sum(board == -player))
        progress = occupied / total_cells

        heights = np.sum(board != 0, axis=0).astype(np.float32)
        mean_height = float(np.mean(heights)) / float(max(1, self.max_layers))
        max_height = float(np.max(heights)) / float(max(1, self.max_layers))
        std_height = float(np.std(heights)) / float(max(1, self.max_layers))

        center_slice = board[:, 1:4, 1:4]
        center_own = float(np.sum(center_slice == player)) / float(max(1, center_slice.size))
        center_opp = float(np.sum(center_slice == -player)) / float(max(1, center_slice.size))

        valid_ratio = float(np.mean(valid_mask))
        own_win_ratio = float(len(own_wins_now)) / float(max(1, self.candidate_count))
        opp_win_ratio = float(len(opp_wins_now)) / float(max(1, self.candidate_count))

        stage_open = 1.0 if progress < 0.25 else 0.0
        stage_mid = 1.0 if 0.25 <= progress < 0.75 else 0.0
        stage_late = 1.0 if progress >= 0.75 else 0.0

        top_layer = self.max_layers - 1
        own_top = float(np.mean(board[top_layer] == player))
        opp_top = float(np.mean(board[top_layer] == -player))

        global_features = np.asarray(
            [
                progress,
                own_count / total_cells,
                opp_count / total_cells,
                (own_count - opp_count) / total_cells,
                valid_ratio,
                own_win_ratio,
                opp_win_ratio,
                own_win_ratio - opp_win_ratio,
                mean_height,
                max_height,
                std_height,
                center_own,
                center_opp,
                center_own - center_opp,
                stage_open,
                stage_mid,
                stage_late,
                own_top,
                opp_top,
                1.0 if player > 0 else 0.0,
            ],
            dtype=np.float32,
        )
        return global_features

    def _immediate_win_candidates(self, board: np.ndarray, player: int) -> Set[int]:
        result: Set[int] = set()
        action_map = self.get_candidate_action_map(board)
        for idx, action in enumerate(action_map):
            if action < 0:
                continue
            row, col = self.candidate_to_row_col(idx)
            layer = int(action) // self.action_plane
            probe = np.array(board, copy=True)
            probe[layer, row, col] = int(player)
            if self.game.check_win(probe, int(player)):
                result.add(int(idx))
        return result

    def _line_stats_for_move(
        self,
        board_after: np.ndarray,
        player: int,
        layer: int,
        row: int,
        col: int,
    ) -> Dict[str, float]:
        len2_plus = 0.0
        len3_plus = 0.0
        open_two = 0.0
        open_three = 0.0
        open_four = 0.0
        potential = 0.0

        for dz, dy, dx in self._directions:
            left = self._count_direction(board_after, player, layer, row, col, -dz, -dy, -dx)
            right = self._count_direction(board_after, player, layer, row, col, dz, dy, dx)
            line_len = left + 1 + right

            end_a = self._line_end(layer, row, col, -dz, -dy, -dx, left + 1)
            end_b = self._line_end(layer, row, col, dz, dy, dx, right + 1)
            open_ends = float(self._is_playable_empty(board_after, *end_a)) + float(
                self._is_playable_empty(board_after, *end_b)
            )

            if line_len >= 2:
                len2_plus += 1.0
            if line_len >= 3:
                len3_plus += 1.0
            if line_len == 2 and open_ends >= 1.0:
                open_two += 1.0
            if line_len == 3 and open_ends >= 1.0:
                open_three += 1.0
            if line_len >= self.connect_n:
                open_four += 1.0

            potential += (line_len / float(self.connect_n)) + 0.25 * open_ends

        normalizer = float(max(1, len(self._directions)))
        return {
            "len2_plus": len2_plus / normalizer,
            "len3_plus": len3_plus / normalizer,
            "open_two": open_two / normalizer,
            "open_three": open_three / normalizer,
            "open_four": open_four / normalizer,
            "potential": potential / normalizer,
        }

    def _neighbor_counts(
        self,
        board: np.ndarray,
        player: int,
        layer: int,
        row: int,
        col: int,
    ) -> Tuple[float, float, float]:
        own = opp = empty = 0.0
        total = 0.0
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    nl, nr, nc = layer + dz, row + dy, col + dx
                    if not self._inside(nl, nr, nc):
                        continue
                    total += 1.0
                    value = int(board[nl, nr, nc])
                    if value == int(player):
                        own += 1.0
                    elif value == int(-player):
                        opp += 1.0
                    else:
                        empty += 1.0
        total = max(1.0, total)
        return own / total, opp / total, empty / total

    def _next_open_layer(self, board: np.ndarray, row: int, col: int) -> Optional[int]:
        for layer in range(self.max_layers):
            if board[layer, row, col] != 0:
                continue
            if layer == 0 or board[layer - 1, row, col] != 0:
                return layer
            return None
        return None

    def _count_direction(
        self,
        board: np.ndarray,
        player: int,
        layer: int,
        row: int,
        col: int,
        dz: int,
        dy: int,
        dx: int,
    ) -> int:
        count = 0
        for step in range(1, self.connect_n):
            nl, nr, nc = layer + step * dz, row + step * dy, col + step * dx
            if not self._inside(nl, nr, nc):
                break
            if int(board[nl, nr, nc]) != int(player):
                break
            count += 1
        return count

    def _line_end(self, layer: int, row: int, col: int, dz: int, dy: int, dx: int, steps: int) -> Tuple[int, int, int]:
        return layer + dz * steps, row + dy * steps, col + dx * steps

    def _is_playable_empty(self, board: np.ndarray, layer: int, row: int, col: int) -> bool:
        if not self._inside(layer, row, col):
            return False
        if int(board[layer, row, col]) != 0:
            return False
        if layer == 0:
            return True
        return bool(board[layer - 1, row, col] != 0)

    def _inside(self, layer: int, row: int, col: int) -> bool:
        return (
            0 <= int(layer) < self.max_layers
            and 0 <= int(row) < self.board_size
            and 0 <= int(col) < self.board_size
        )

    def _build_unique_directions(self) -> List[Tuple[int, int, int]]:
        directions: List[Tuple[int, int, int]] = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    if dz < 0:
                        continue
                    if dz == 0 and dy < 0:
                        continue
                    if dz == 0 and dy == 0 and dx < 0:
                        continue
                    directions.append((dz, dy, dx))
        return directions

    def _build_position_prior(self) -> np.ndarray:
        prior = np.zeros((self.board_size, self.board_size), dtype=np.float32)
        for row in range(self.board_size):
            for col in range(self.board_size):
                dist = np.sqrt((row - self._center) ** 2 + (col - self._center) ** 2)
                score = 1.0 - (dist / float(max(1e-6, self._max_dist)))
                prior[row, col] = float(score)
        min_v = float(np.min(prior))
        max_v = float(np.max(prior))
        if max_v - min_v < 1e-6:
            return np.zeros_like(prior)
        return (prior - min_v) / float(max_v - min_v)
