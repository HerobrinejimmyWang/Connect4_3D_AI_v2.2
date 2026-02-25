import math
import numpy as np
import torch

class MCTS:
    def __init__(self, game, model, args):
        self.game = game
        self.model = model
        self.args = args
        self.Qsa = {}  # Stores Q values for s,a
        self.Nsa = {}  # Stores times edge s,a was visited
        self.Ns = {}   # Stores times board s was visited
        self.Ps = {}   # Stores initial policy (returned by neural net)
        self.Es = {}   # Stores game ended for board s
        self.Vs = {}   # Stores valid moves for board s

    def get_action_prob(self, canonical_board, temp=1):
        """
        Runs simulations and returns probabilities.
        """
        # --- Add Dirichlet Noise to the root node for exploration ---
        s = self.game.string_representation(canonical_board)
        
        # Ensure the root node is expanded so we have Ps[s]
        if s not in self.Ps:
            self.search(canonical_board)
            
        # Add noise to Ps[s]
        if temp > 0: # Only add noise during self-play (training)
            # AlphaZero parameters: epsilon=0.25, dirichlet_alpha=0.3 for Go
            # For 3D Connect 4, action space is 200, so 0.3-0.5 is fine.
            epsilon = 0.25
            alpha = 0.5
            
            valid_moves = self.Vs[s]
            noise = np.random.dirichlet([alpha] * int(np.sum(valid_moves)))
            
            # Weighted average of policy and noise
            idx = 0
            for a in range(self.game.get_action_size()):
                if valid_moves[a]:
                    self.Ps[s][a] = (1 - epsilon) * self.Ps[s][a] + epsilon * noise[idx]
                    idx += 1

        for _ in range(self.args.num_mcts_sims):
            self.search(canonical_board)

        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.get_action_size())]

        if temp == 0:
            best_as = np.array(np.argwhere(counts == np.max(counts))).flatten()
            best_a = np.random.choice(best_as)
            probs = [0] * len(counts)
            probs[best_a] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, canonical_board):
        s = self.game.string_representation(canonical_board)

        if s not in self.Es:
            self.Es[s] = self.game.get_game_ended(canonical_board, 1)
        if self.Es[s] != 0:
            return -self.Es[s]

        if s not in self.Ps:
            # Leaf node
            board_tensor = torch.FloatTensor(canonical_board.astype(np.float64))
            # If utilizing GPU, move tensor here. We are on CPU.
            self.model.eval()
            with torch.no_grad():
                pi, v = self.model(board_tensor.unsqueeze(0))
            
            self.Ps[s] = torch.exp(pi).data.cpu().numpy()[0]
            v = v.data.cpu().numpy()[0][0]
            
            valid_moves = self.game.get_valid_moves(canonical_board)
            self.Ps[s] = self.Ps[s] * valid_moves
            sum_Ps_s = np.sum(self.Ps[s])
            
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s
            else:
                # If all valid moves were masked likely due to logic error or extremely rare case
                # Uniform distribution over valid moves
                self.Ps[s] = self.Ps[s] + valid_moves
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valid_moves
            self.Ns[s] = 0
            return -v

        valid_moves = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # Pick the action with the highest upper confidence bound
        for a in range(self.game.get_action_size()):
            if valid_moves[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        next_s, next_player = self.game.get_next_state(canonical_board, 1, a)
        next_s = self.game.get_canonical_form(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v