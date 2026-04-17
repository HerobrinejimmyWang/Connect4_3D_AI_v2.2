from __future__ import annotations

import math

import numpy as np


class GameRules:
    def __init__(self, board_size=5, max_layers=6, connect_n=4):
        self.board_size = int(board_size)
        self.max_layers = int(max_layers)
        self.connect_n = int(connect_n)
        self.board_shape = (self.max_layers, self.board_size, self.board_size)

    def get_init_board(self):
        return np.zeros(self.board_shape, dtype=np.int8)

    def get_board_size(self):
        return self.board_shape

    def get_action_size(self):
        return self.max_layers * self.board_size * self.board_size

    def get_next_state(self, board, player, action):
        action = int(action)
        layer, row, col = self.action_to_coords(action)
        self._validate_move(board, layer, row, col, action=action)
        new_board = np.array(board, copy=True)
        new_board[layer, row, col] = int(player)
        return new_board, -int(player)

    def get_valid_moves(self, board):
        valid_moves = np.zeros(self.get_action_size(), dtype=np.int8)
        for layer in range(self.max_layers):
            for row in range(self.board_size):
                for col in range(self.board_size):
                    if board[layer, row, col] != 0:
                        continue
                    if layer == 0 or board[layer - 1, row, col] != 0:
                        valid_moves[self.coords_to_action(layer, row, col)] = 1
        return valid_moves

    def get_game_ended(self, board, player):
        opponent = -int(player)
        if self.check_win(board, opponent):
            return -1
        if self.check_win(board, int(player)):
            return 1
        if not np.any(board == 0):
            return 1e-4
        return 0

    def check_win(self, board, player):
        occupied = np.argwhere(board == int(player))
        if occupied.size == 0:
            return False

        directions = [
            (dz, dy, dx)
            for dz in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
            if not (dz == 0 and dy == 0 and dx == 0)
        ]

        for layer, row, col in occupied:
            for dz, dy, dx in directions:
                count = 1
                for step in range(1, self.connect_n):
                    next_layer = layer + step * dz
                    next_row = row + step * dy
                    next_col = col + step * dx
                    if not self._is_inside(next_layer, next_row, next_col):
                        break
                    if board[next_layer, next_row, next_col] != player:
                        break
                    count += 1
                if count >= self.connect_n:
                    return True
        return False

    def get_canonical_form(self, board, player):
        return np.asarray(board, dtype=np.int8) * int(player)

    def get_symmetries(self, board, pi):
        pi_board = np.reshape(pi, self.board_shape)
        symmetries = []
        for rotation in range(1, 5):
            rotated_board = np.rot90(board, rotation, axes=(1, 2))
            rotated_pi = np.rot90(pi_board, rotation, axes=(1, 2))
            symmetries.append((rotated_board, rotated_pi.flatten()))

            flipped_board = np.flip(rotated_board, axis=2)
            flipped_pi = np.flip(rotated_pi, axis=2)
            symmetries.append((flipped_board, flipped_pi.flatten()))
        return symmetries

    def string_representation(self, board):
        return np.asarray(board, dtype=np.int8).tobytes()

    def action_to_coords(self, action):
        action = int(action)
        if action < 0 or action >= self.get_action_size():
            raise ValueError(f"Action {action} out of range [0, {self.get_action_size() - 1}].")
        layer = action // (self.board_size * self.board_size)
        remainder = action % (self.board_size * self.board_size)
        row = remainder // self.board_size
        col = remainder % self.board_size
        return layer, row, col

    def coords_to_action(self, layer, row, col):
        return int(layer) * self.board_size * self.board_size + int(row) * self.board_size + int(col)

    def reconstruct_board(self, moves, move_index=None):
        board = self.get_init_board()
        limit = len(moves) if move_index is None else max(0, min(int(move_index), len(moves)))
        for move in moves[:limit]:
            layer, row, col = int(move["coords"]["layer"]), int(move["coords"]["row"]), int(move["coords"]["col"])
            self._validate_move(board, layer, row, col)
            board[layer, row, col] = int(move["player"])
        return board

    def _is_inside(self, layer, row, col):
        return (
            0 <= int(layer) < self.max_layers
            and 0 <= int(row) < self.board_size
            and 0 <= int(col) < self.board_size
        )

    def _validate_move(self, board, layer, row, col, action=None):
        if not self._is_inside(layer, row, col):
            label = f"Action {action}" if action is not None else "Move"
            raise ValueError(f"{label} maps outside the board to ({layer}, {row}, {col}).")
        if board[layer, row, col] != 0:
            label = f"Action {action}" if action is not None else "Move"
            raise ValueError(f"{label} targets occupied position ({layer}, {row}, {col}).")
        if layer > 0 and board[layer - 1, row, col] == 0:
            label = f"Action {action}" if action is not None else "Move"
            raise ValueError(f"{label} violates gravity at ({layer}, {row}, {col}).")


def infer_board_size_from_action_dim(action_dim, preferred_size=None):
    action_dim = int(action_dim)
    if preferred_size is not None:
        preferred_size = int(preferred_size)
        if preferred_size > 0 and action_dim % (preferred_size * preferred_size) == 0:
            return preferred_size, action_dim // (preferred_size * preferred_size)

    candidates = []
    max_side = int(math.sqrt(action_dim)) + 1
    for board_size in range(4, max_side + 1):
        square = board_size * board_size
        if action_dim % square != 0:
            continue
        board_layers = action_dim // square
        if 1 <= board_layers <= 16:
            candidates.append((board_size, board_layers))

    if len(candidates) == 1:
        return candidates[0]

    for board_size, board_layers in candidates:
        if board_size == 5:
            return board_size, board_layers

    if candidates:
        return candidates[0]
    raise ValueError(f"无法从动作维度 {action_dim} 推断棋盘尺寸。")