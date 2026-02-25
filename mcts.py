import math
import numpy as np
import torch


class MCTS:
    """
    Monte-Carlo Tree Search with neural-network leaf evaluation.
    The model may reside on any device (GPU / CPU); tensors are moved
    accordingly and results are brought back to CPU / numpy for the tree.
    """

    def __init__(self, game, model, args, device='cpu'):
        self.game = game
        self.model = model
        self.args = args
        self.device = device
        self.Qsa = {}   # Q(s, a)
        self.Nsa = {}   # N(s, a)
        self.Ns = {}    # N(s)
        self.Ps = {}    # P(s, ·) from the network
        self.Es = {}    # game-ended cache
        self.Vs = {}    # valid-moves cache

    def get_action_prob(self, canonical_board, temp=1):
        s = self.game.string_representation(canonical_board)

        # Ensure root is expanded
        if s not in self.Ps:
            self._search(canonical_board)

        # Dirichlet noise at the root (training only)
        if temp > 0:
            eps, alpha = 0.25, 0.5
            valids = self.Vs[s]
            num_valid = int(np.sum(valids))
            noise = np.random.dirichlet([alpha] * num_valid)
            idx = 0
            for a in range(self.game.get_action_size()):
                if valids[a]:
                    self.Ps[s][a] = (1 - eps) * self.Ps[s][a] + eps * noise[idx]
                    idx += 1

        for _ in range(self.args.num_mcts_sims):
            self._search(canonical_board)

        counts = np.array([self.Nsa.get((s, a), 0)
                           for a in range(self.game.get_action_size())],
                          dtype=np.float64)

        if temp == 0:
            best = np.flatnonzero(counts == counts.max())
            probs = np.zeros_like(counts)
            probs[np.random.choice(best)] = 1.0
            return probs.tolist()

        counts = counts ** (1.0 / temp)
        total = counts.sum()
        probs = (counts / total).tolist() if total > 0 else counts.tolist()
        return probs

    # ------------------------------------------------------------------ #

    def _search(self, canonical_board):
        s = self.game.string_representation(canonical_board)

        # Terminal?
        if s not in self.Es:
            self.Es[s] = self.game.get_game_ended(canonical_board, 1)
        if self.Es[s] != 0:
            return -self.Es[s]

        # Leaf → expand with NN (GPU-aware)
        if s not in self.Ps:
            board_t = torch.tensor(
                canonical_board, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                log_pi, v = self.model(board_t)
            pi = torch.exp(log_pi).cpu().numpy()[0]
            v = v.cpu().item()

            valids = self.game.get_valid_moves(canonical_board)
            pi *= valids
            s_pi = pi.sum()
            if s_pi > 0:
                pi /= s_pi
            else:
                pi = valids.astype(np.float32)
                pi /= pi.sum()

            self.Ps[s] = pi
            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        # Selection (PUCT)
        valids = self.Vs[s]
        best_u = -float('inf')
        best_a = -1
        sqrt_ns = math.sqrt(self.Ns[s] + 1e-8)

        for a in range(self.game.get_action_size()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = (self.Qsa[(s, a)]
                         + self.args.cpuct * self.Ps[s][a] * sqrt_ns
                         / (1 + self.Nsa[(s, a)]))
                else:
                    u = self.args.cpuct * self.Ps[s][a] * sqrt_ns
                if u > best_u:
                    best_u = u
                    best_a = a

        a = best_a
        next_board, next_player = self.game.get_next_state(canonical_board, 1, a)
        next_board = self.game.get_canonical_form(next_board, next_player)

        v = self._search(next_board)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = ((self.Nsa[(s, a)] * self.Qsa[(s, a)] + v)
                                / (self.Nsa[(s, a)] + 1))
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v