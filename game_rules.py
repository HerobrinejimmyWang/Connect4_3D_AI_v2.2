import numpy as np

# Game constants
BOARD_SIZE = 5
MAX_LAYERS = 8
BOARD_SHAPE = (MAX_LAYERS, BOARD_SIZE, BOARD_SIZE)

# 13 unique direction vectors (positive half only) for win checking
_WIN_DIRS = [
    (0, 0, 1), (0, 1, 0), (1, 0, 0),          # axis-aligned
    (0, 1, 1), (0, 1, -1),                     # 2D diag in layer
    (1, 1, 0), (1, -1, 0),                     # layer + row
    (1, 0, 1), (1, 0, -1),                     # layer + col
    (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),  # 3D diag
]


class GameRules:
    """
    Game logic for 3D Connect Four (8 layers × 5 rows × 5 cols).
    All operations use numpy arrays on CPU for easy multi-process passing.
    """

    def get_init_board(self):
        return np.zeros(BOARD_SHAPE, dtype=np.int8)

    def get_board_size(self):
        return BOARD_SHAPE

    def get_action_size(self):
        return MAX_LAYERS * BOARD_SIZE * BOARD_SIZE  # 200

    def get_next_state(self, board, player, action):
        layer = action // (BOARD_SIZE * BOARD_SIZE)
        rem = action % (BOARD_SIZE * BOARD_SIZE)
        row = rem // BOARD_SIZE
        col = rem % BOARD_SIZE
        new_board = board.copy()
        new_board[layer, row, col] = player
        return new_board, -player

    def get_valid_moves(self, board):
        """Vectorised valid-move computation respecting gravity."""
        empty = (board == 0)
        supported = np.empty(BOARD_SHAPE, dtype=bool)
        supported[0] = True                     # bottom layer is always supported
        supported[1:] = (board[:-1] != 0)       # above an occupied cell
        return (empty & supported).astype(np.int8).flatten()

    def get_game_ended(self, board, player):
        """
        Returns 0 (not over), 1 (player wins), -1 (player loses), 1e-4 (draw).
        Convention: the opponent is the one who just moved.
        """
        opponent = -player
        if self._check_win(board, opponent):
            return -1
        if self._check_win(board, player):
            return 1
        if not np.any(board == 0):
            return 1e-4
        return 0

    @staticmethod
    def _check_win(board, player):
        occupied = np.argwhere(board == player)
        if len(occupied) < 4:
            return False
        for l, r, c in occupied:
            for dl, dr, dc in _WIN_DIRS:
                count = 1
                for step in range(1, 4):
                    nl, nr, nc = l + step * dl, r + step * dr, c + step * dc
                    if (0 <= nl < MAX_LAYERS and 0 <= nr < BOARD_SIZE
                            and 0 <= nc < BOARD_SIZE
                            and board[nl, nr, nc] == player):
                        count += 1
                    else:
                        break
                if count >= 4:
                    return True
        return False

    def get_canonical_form(self, board, player):
        return board * player

    def get_symmetries(self, board, pi):
        """
        8 symmetries of the square 5×5 base (4 rotations × 2 flips).
        board: (8, 5, 5)   pi: flat vector of length 200
        """
        pi_board = np.asarray(pi).reshape(MAX_LAYERS, BOARD_SIZE, BOARD_SIZE)
        syms = []
        for k in range(1, 5):
            for flip in (False, True):
                b = np.rot90(board, k, axes=(1, 2))
                p = np.rot90(pi_board, k, axes=(1, 2))
                if flip:
                    b = np.flip(b, axis=2)
                    p = np.flip(p, axis=2)
                syms.append((np.ascontiguousarray(b),
                             np.ascontiguousarray(p).flatten()))
        return syms

    def string_representation(self, board):
        return board.tobytes()