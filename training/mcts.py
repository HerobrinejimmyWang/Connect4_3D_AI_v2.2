import math
import itertools
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.multiprocessing as mp

from model import Connect4Net, board_to_channels


class TreeNode:
    """Thread-safe node storing N/W/P and virtual loss state."""

    def __init__(self, state, prior_probability=1.0, parent=None, action_from_parent=None):
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.prior_probability = float(prior_probability)

        self.visit_count = 0
        self.value_sum = 0.0
        self.virtual_loss_count = 0

        self.children = {}
        self.valid_moves = None
        self.is_expanded = False
        self.terminal_outcome = None

        self.lock = threading.Lock()

    def q_value(self):
        with self.lock:
            if self.visit_count == 0:
                return 0.0
            return self.value_sum / self.visit_count

    def add_virtual_loss(self, virtual_loss):
        # Virtual loss is added during selection so other threads are discouraged
        # from choosing the same path until this simulation completes.
        with self.lock:
            self.virtual_loss_count += 1
            self.visit_count += 1
            self.value_sum -= float(virtual_loss)

    def revert_virtual_loss(self, virtual_loss):
        with self.lock:
            if self.virtual_loss_count > 0:
                self.virtual_loss_count -= 1
                self.visit_count -= 1
                self.value_sum += float(virtual_loss)

    def apply_real_backup(self, value):
        with self.lock:
            self.visit_count += 1
            self.value_sum += float(value)


@dataclass
class InferenceRequest:
    state: np.ndarray
    done_event: threading.Event
    policy: np.ndarray = None
    value: float = 0.0


@dataclass
class RemoteInferenceResult:
    done_event: threading.Event
    policy: np.ndarray = None
    value: float = 0.0


def run_global_inference_server(
    model_state,
    model_config,
    worker_response_conns,
    request_queue,
    stop_event,
    device,
    batch_size,
    batch_timeout_s,
):
    model_config = dict(model_config or {})
    model = Connect4Net(
        board_layers=int(model_config.get("board_layers", 6)),
        board_size=int(model_config.get("board_size", 5)),
        num_channels=int(model_config.get("num_channels", 256)),
        num_res_blocks=int(model_config.get("num_res_blocks", 8)),
    )
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    if isinstance(device, str) and device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True

    while True:
        if stop_event.is_set() and request_queue.empty():
            break

        batch = []
        try:
            first_req = request_queue.get(timeout=batch_timeout_s)
            batch.append(first_req)
        except queue.Empty:
            continue

        start = time.perf_counter()
        while len(batch) < batch_size:
            elapsed = time.perf_counter() - start
            if elapsed >= batch_timeout_s:
                break
            try:
                timeout_left = max(0.0, batch_timeout_s - elapsed)
                batch.append(request_queue.get(timeout=timeout_left))
            except queue.Empty:
                break

        states = np.stack([board_to_channels(req[2]) for req in batch], axis=0).astype(np.float32)
        tensor = torch.from_numpy(states).to(device, non_blocking=True)

        with torch.no_grad():
            log_pi, v = model(tensor)

        policies = torch.exp(log_pi).cpu().numpy().astype(np.float32, copy=False)
        values = v.squeeze(1).cpu().numpy().astype(np.float32, copy=False)

        for idx, (worker_id, request_id, _) in enumerate(batch):
            worker_response_conns[int(worker_id)].send(
                (int(request_id), policies[idx], float(values[idx]))
            )

    for conn in worker_response_conns.values():
        try:
            conn.close()
        except Exception:
            pass


class GlobalInferenceServer:
    def __init__(
        self,
        model_state,
        worker_response_conns,
        model_config=None,
        device="cuda",
        batch_size=32,
        batch_timeout_s=0.003,
    ):
        self.request_queue = mp.Queue()
        self.stop_event = mp.Event()
        self.process = mp.Process(
            target=run_global_inference_server,
            args=(
                model_state,
                model_config,
                worker_response_conns,
                self.request_queue,
                self.stop_event,
                device,
                max(1, int(batch_size)),
                float(batch_timeout_s),
            ),
        )

    def start(self):
        self.process.start()

    def close(self):
        self.stop_event.set()
        self.process.join(timeout=5.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2.0)
        self.request_queue.close()
        self.request_queue.join_thread()


class RemoteBatchInferenceClient:
    def __init__(self, worker_id, request_queue, response_conn):
        self.worker_id = int(worker_id)
        self.request_queue = request_queue
        self.response_conn = response_conn
        self.pending = {}
        self.pending_lock = threading.Lock()
        self.request_counter = itertools.count()
        self.stop_event = threading.Event()
        self.failure = None
        self.receiver_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.receiver_thread.start()

    def submit_and_wait(self, state):
        if self.failure is not None:
            raise RuntimeError("Remote inference client is closed.") from self.failure

        request_id = next(self.request_counter)
        result = RemoteInferenceResult(done_event=threading.Event())
        with self.pending_lock:
            self.pending[request_id] = result

        self.request_queue.put((self.worker_id, request_id, np.array(state, copy=True)))
        result.done_event.wait()

        if self.failure is not None:
            raise RuntimeError("Remote inference failed.") from self.failure
        return result.policy, result.value

    def close(self):
        self.stop_event.set()
        try:
            self.response_conn.close()
        except Exception:
            pass
        self.receiver_thread.join(timeout=1.0)
        self._fail_pending(RuntimeError("Remote inference client closed."))

    def _recv_loop(self):
        try:
            while not self.stop_event.is_set():
                if not self.response_conn.poll(0.1):
                    continue
                request_id, policy, value = self.response_conn.recv()
                with self.pending_lock:
                    result = self.pending.pop(int(request_id), None)
                if result is None:
                    continue
                result.policy = np.array(policy, dtype=np.float32, copy=False)
                result.value = float(value)
                result.done_event.set()
        except (EOFError, OSError) as exc:
            self.failure = exc
            self._fail_pending(exc)

    def _fail_pending(self, exc):
        with self.pending_lock:
            pending = list(self.pending.values())
            self.pending.clear()
        for result in pending:
            result.done_event.set()
        if self.failure is None:
            self.failure = exc


class TorchBatchPredictor:
    """Wraps a PyTorch policy-value network with a predict(batch_states) API."""

    def __init__(self, model):
        self.model = model
        self.device = next(self.model.parameters()).device
        self.model_lock = threading.Lock()

    def predict(self, batch_states):
        states = np.stack([board_to_channels(s) for s in batch_states], axis=0).astype(np.float32)
        tensor = torch.from_numpy(states).to(self.device)

        with self.model_lock:
            self.model.eval()
            with torch.no_grad():
                log_pi, v = self.model(tensor)

        pi = torch.exp(log_pi).cpu().numpy()
        values = v.squeeze(1).cpu().numpy().astype(np.float32)
        return pi, values


class AsyncBatchInferenceManager:
    """Collects worker requests and performs one batched forward pass."""

    def __init__(self, neural_network, batch_size=32, batch_timeout_s=0.003):
        self.neural_network = neural_network
        self.batch_size = max(1, int(batch_size))
        self.batch_timeout_s = float(batch_timeout_s)

        self.request_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self._run_loop, daemon=True)
        self.worker_thread.start()

    def submit_and_wait(self, state):
        req = InferenceRequest(state=np.array(state, copy=True), done_event=threading.Event())
        self.request_queue.put(req)
        req.done_event.wait()
        return req.policy, req.value

    def close(self):
        self.stop_event.set()
        self.worker_thread.join(timeout=1.0)

    def _run_loop(self):
        while not self.stop_event.is_set() or not self.request_queue.empty():
            batch = []

            try:
                first_req = self.request_queue.get(timeout=self.batch_timeout_s)
                batch.append(first_req)
            except queue.Empty:
                continue

            start = time.perf_counter()
            while len(batch) < self.batch_size:
                elapsed = time.perf_counter() - start
                if elapsed >= self.batch_timeout_s:
                    break
                try:
                    timeout_left = max(0.0, self.batch_timeout_s - elapsed)
                    batch.append(self.request_queue.get(timeout=timeout_left))
                except queue.Empty:
                    break

            states = [req.state for req in batch]
            policies, values = self.neural_network.predict(states)

            for idx, req in enumerate(batch):
                req.policy = policies[idx]
                req.value = float(values[idx])
                req.done_event.set()


class MCTS:
    """
    Thread-safe MCTS with async batched inference and virtual loss.

    PUCT used at selection:
      U(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
      Q(s,a) = W(s,a) / N(s,a)
    """

    def __init__(self, game, model, args):
        self.game = game
        self.args = args

        self.cpuct = float(getattr(args, "cpuct", 1.0))
        self.num_mcts_sims = int(getattr(args, "num_mcts_sims", 64))
        self.num_worker_threads = int(getattr(args, "num_mcts_threads", 8))
        self.virtual_loss = float(getattr(args, "virtual_loss", 1.0))
        self.inference_batch_size = int(getattr(args, "inference_batch_size", 32))
        self.inference_timeout_s = float(getattr(args, "inference_timeout_s", 0.003))

        # AlphaZero Dirichlet Noise parameters
        self.dirichlet_alpha = float(getattr(args, "dirichlet_alpha", 0.3))
        self.dirichlet_epsilon = float(getattr(args, "dirichlet_epsilon", 0.25))

        if hasattr(model, "submit_and_wait"):
            self.inference_manager = model
            self.owns_inference_manager = False
        else:
            predictor = model if hasattr(model, "predict") else TorchBatchPredictor(model)
            self.inference_manager = AsyncBatchInferenceManager(
                predictor,
                batch_size=self.inference_batch_size,
                batch_timeout_s=self.inference_timeout_s,
            )
            self.owns_inference_manager = True

    def __del__(self):
        try:
            if getattr(self, "owns_inference_manager", False) and hasattr(self, "inference_manager"):
                self.inference_manager.close()
        except Exception:
            pass

    def _normalize_probs(self, probs):
        probs = np.asarray(probs, dtype=np.float64)
        probs = np.clip(probs, 0.0, None)
        total = float(np.sum(probs))
        if not np.isfinite(total) or total <= 0.0:
            return None
        return (probs / total).astype(np.float64, copy=False)

    def get_action_prob(self, canonical_board, temp=1, training=False, dirichlet_alpha=None, dirichlet_epsilon=None):
        root = TreeNode(state=np.array(canonical_board, copy=True), prior_probability=1.0)

        # Pre-expand the root node
        policy, value = self.inference_manager.submit_and_wait(root.state)
        self._expand_if_needed(root, policy, value)

        # Apply Dirichlet noise ONLY at the root node and ONLY during training (self-play)
        if training:
            self._apply_dirichlet_noise(
                root,
                alpha=dirichlet_alpha,
                epsilon=dirichlet_epsilon,
            )

        sim_counter = {"count": 0}
        counter_lock = threading.Lock()

        workers = []
        total_workers = max(1, self.num_worker_threads)
        for _ in range(total_workers):
            worker = threading.Thread(target=self._worker_loop, args=(root, sim_counter, counter_lock), daemon=True)
            workers.append(worker)
            worker.start()

        for worker in workers:
            worker.join()

        counts = np.zeros(self.game.get_action_size(), dtype=np.float32)
        with root.lock:
            child_items = list(root.children.items())
        for action, child in child_items:
            with child.lock:
                counts[action] = max(0, child.visit_count)

        if np.sum(counts) <= 0:
            valid_moves = self.game.get_valid_moves(canonical_board).astype(np.float32)
            valid_sum = np.sum(valid_moves)
            if valid_sum > 0:
                probs = self._normalize_probs(valid_moves / valid_sum)
                return probs.tolist()
            probs = self._normalize_probs(np.ones_like(valid_moves, dtype=np.float64))
            return probs.tolist()

        if temp == 0:
            best_actions = np.flatnonzero(counts == np.max(counts))
            chosen = int(np.random.choice(best_actions))
            probs = np.zeros_like(counts)
            probs[chosen] = 1.0
            return probs.tolist()

        positive_mask = counts > 0
        if not np.any(positive_mask):
            probs = None
        else:
            inv_temp = 1.0 / float(temp)
            scaled_logits = np.full_like(counts, -np.inf, dtype=np.float64)
            scaled_logits[positive_mask] = np.log(counts[positive_mask].astype(np.float64)) * inv_temp
            scaled_logits[positive_mask] -= np.max(scaled_logits[positive_mask])
            exp_logits = np.zeros_like(counts, dtype=np.float64)
            exp_logits[positive_mask] = np.exp(scaled_logits[positive_mask])
            probs = self._normalize_probs(exp_logits)
        if probs is None:
            valid_moves = self.game.get_valid_moves(canonical_board).astype(np.float64)
            probs = self._normalize_probs(valid_moves)
            if probs is None:
                probs = self._normalize_probs(np.ones_like(valid_moves, dtype=np.float64))
        return probs.tolist()

    def _apply_dirichlet_noise(self, node, alpha=None, epsilon=None):
        """Append Dirichlet noise to the root node strategy probabilities during self-play."""
        noise_alpha = self.dirichlet_alpha if alpha is None else float(alpha)
        noise_epsilon = self.dirichlet_epsilon if epsilon is None else float(epsilon)
        if noise_alpha <= 0.0 or noise_epsilon <= 0.0:
            return

        with node.lock:
            actions = list(node.children.keys())
            if not actions:
                return

            noise = np.random.dirichlet([noise_alpha] * len(actions))

            for i, action in enumerate(actions):
                child = node.children[action]
                with child.lock:
                    child.prior_probability = (1 - noise_epsilon) * child.prior_probability + noise_epsilon * noise[i]

    def _worker_loop(self, root, sim_counter, counter_lock):
        while True:
            with counter_lock:
                if sim_counter["count"] >= self.num_mcts_sims:
                    return
                sim_counter["count"] += 1
            self._run_single_simulation(root)

    def _run_single_simulation(self, root):
        node = root
        selected_nodes_with_virtual_loss = []

        while True:
            terminal_result = self.game.get_game_ended(node.state, 1)
            if terminal_result != 0:
                leaf_value = float(terminal_result)
                self._backup(node, leaf_value, selected_nodes_with_virtual_loss)
                return

            with node.lock:
                expanded = node.is_expanded

            if not expanded:
                policy, value = self.inference_manager.submit_and_wait(node.state)
                leaf_value = self._expand_if_needed(node, policy, value)
                self._backup(node, leaf_value, selected_nodes_with_virtual_loss)
                return

            child = self._select_child_with_puct(node)
            child.add_virtual_loss(self.virtual_loss)
            selected_nodes_with_virtual_loss.append(child)
            node = child

    def _expand_if_needed(self, node, raw_policy, value):
        with node.lock:
            if node.is_expanded:
                return float(value)

            valid_moves = self.game.get_valid_moves(node.state).astype(np.float32)
            policy = np.array(raw_policy, dtype=np.float32) * valid_moves
            prob_sum = float(np.sum(policy))

            if prob_sum <= 0.0:
                valid_sum = float(np.sum(valid_moves))
                if valid_sum > 0.0:
                    policy = valid_moves / valid_sum
                else:
                    policy = np.ones_like(valid_moves, dtype=np.float32) / len(valid_moves)
            else:
                policy = policy / prob_sum

            node.valid_moves = valid_moves
            node.children = {}
            for action in np.flatnonzero(valid_moves > 0):
                next_state, next_player = self.game.get_next_state(node.state, 1, int(action))
                next_canonical = self.game.get_canonical_form(next_state, next_player)
                node.children[int(action)] = TreeNode(
                    state=next_canonical,
                    prior_probability=float(policy[action]),
                    parent=node,
                    action_from_parent=int(action),
                )

            node.is_expanded = True
            return float(value)

    def _select_child_with_puct(self, node):
        with node.lock:
            child_items = list(node.children.items())
            total_visits = max(1, node.visit_count)
        sqrt_total = math.sqrt(float(total_visits))

        best_score = -float("inf")
        tied_children = []
        tie_eps = 1e-12

        for action, child in child_items:
            with child.lock:
                child_n = child.visit_count
                child_w = child.value_sum
                child_p = child.prior_probability

            # child stores value from the next player's perspective, so negate it
            # when scoring actions for the current player at the parent node.
            q = -(child_w / child_n) if child_n > 0 else 0.0
            u = q + self.cpuct * child_p * sqrt_total / (1.0 + float(child_n))

            if u > best_score + tie_eps:
                best_score = u
                tied_children = [child]
            elif abs(u - best_score) <= tie_eps:
                tied_children.append(child)

        if not tied_children:
            raise RuntimeError("PUCT selection failed: node has no children.")
        return tied_children[np.random.randint(len(tied_children))]

    def _backup(self, leaf_node, leaf_value, selected_nodes_with_virtual_loss):
        selected_set = set(selected_nodes_with_virtual_loss)
        current = leaf_node
        value = float(leaf_value)

        while current is not None:
            # Virtual loss is removed during backprop before applying real stats.
            if current in selected_set:
                current.revert_virtual_loss(self.virtual_loss)

            current.apply_real_backup(value)
            value = -value
            current = current.parent