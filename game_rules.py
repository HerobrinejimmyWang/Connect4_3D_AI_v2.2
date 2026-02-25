import numpy as np

# Game constants
BOARD_SIZE = 5
MAX_LAYERS = 8
# Board dimensions: (Layers, Rows, Cols)
BOARD_SHAPE = (MAX_LAYERS, BOARD_SIZE, BOARD_SIZE)

class GameRules:
    """
    Logic-only class for 3D Connect Four.
    state is represented as a numpy array of shape (8, 5, 5).
    1 = Player 1, -1 = Player 2 (AI training uses 1 and -1 usually)
    """
    def __init__(self):
        self.board = np.zeros(BOARD_SHAPE, dtype=int)
        self.player = 1 # 1 or -1
        self.last_move = None
        
    def get_init_board(self):
        return np.zeros(BOARD_SHAPE, dtype=int)

    def get_board_size(self):
        return BOARD_SHAPE

    def get_action_size(self):
        return MAX_LAYERS * BOARD_SIZE * BOARD_SIZE

    def get_next_state(self, board, player, action):
        # Action is an integer index 0..199
        layer = action // (BOARD_SIZE * BOARD_SIZE)
        rem = action % (BOARD_SIZE * BOARD_SIZE)
        row = rem // BOARD_SIZE
        col = rem % BOARD_SIZE
        
        new_board = np.copy(board)
        new_board[layer, row, col] = player
        return new_board, -player

    def get_valid_moves(self, board):
        # Return a binary vector of size action_size
        valid_moves = np.zeros(self.get_action_size(), dtype=int)
        
        for layer in range(MAX_LAYERS):
            for row in range(BOARD_SIZE):
                for col in range(BOARD_SIZE):
                    if board[layer, row, col] == 0:
                        # Gravity check
                        if layer == 0 or board[layer-1, row, col] != 0:
                            action_idx = layer * (BOARD_SIZE * BOARD_SIZE) + row * BOARD_SIZE + col
                            valid_moves[action_idx] = 1
        return valid_moves

    def get_game_ended(self, board, player):
        # Return 0 if not ended, 1 if player won, -1 if player lost, 1e-4 for draw
        
        # Check if the PREVIOUS player moved and won. 
        # 'player' is the current player to move. 
        # We usually check if 'player' has lost (meaning opponent won).
        
        # Check win for the opponent (who just moved)
        opponent = -player
        if self.check_win(board, opponent):
            return -1 # Current player lost
            
        if self.check_win(board, player):
            return 1 # Current player won (shouldn't happen turn-based but for safety)
            
        if np.sum(board == 0) == 0:
            return 1e-4 # Draw
            
        return 0

    def check_win(self, board, player):
        # Optimized check win logic for numpy array
        # This is a bit heavy, strictly checking for 'player' pieces
        # Using the logic from your provided code but adapted for 1/-1
        
        player_pieces = (board == player)
        if not np.any(player_pieces):
            return False

        directions = [
            (0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
            (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1),
            (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
            (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
            (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1),
            (-1, 1, 1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)
        ]

        # Optimization: Scan only occupied positions is hard in vectorized, 
        # so we iterate occupied cells.
        occupied = np.argwhere(board == player)
        
        for pos in occupied:
            l, r, c = pos
            for dz, dy, dx in directions:
                # We only need to check forward in one direction to avoid double counting
                # But to keep it simple and robust matching original logic:
                count = 1
                # Forward
                for i in range(1, 4):
                    nl, nr, nc = l + i*dz, r + i*dy, c + i*dx
                    if (0 <= nl < MAX_LAYERS and 0 <= nr < BOARD_SIZE and 
                        0 <= nc < BOARD_SIZE and board[nl, nr, nc] == player):
                        count += 1
                    else:
                        break
                if count >= 4: return True
        return False

    def get_canonical_form(self, board, player):
        # Return state from perspective of player
        # If player is -1, flip signs so AI always thinks it's "1"
        return board * player

    def get_symmetries(self, board, pi):
        """
        Board symmetries for data augmentation (3D: 8 symmetries for square base).
        board: (8, 5, 5)
        pi: vector of length 200 (8*5*5)
        """
        # Reshape pi to match board dimensions for easier transformation
        pi_board = np.reshape(pi, (MAX_LAYERS, BOARD_SIZE, BOARD_SIZE))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                new_b = np.rot90(board, i, axes=(1, 2))
                new_pi = np.rot90(pi_board, i, axes=(1, 2))
                if j:
                    new_b = np.flip(new_b, axis=2) # Horizontal flip
                    new_pi = np.flip(new_pi, axis=2)
                l.append((new_b, new_pi.flatten()))
        return l

    def string_representation(self, board):
        return board.tobytes()