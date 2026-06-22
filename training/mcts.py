import math
import itertools
import logging
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
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


def _inference_autocast(device, precision):
    precision = str(precision or "fp32").strip().lower()
    if precision == "fp32":
        return nullcontext()
    if not str(device).startswith("cuda"):
        raise ValueError(f"Inference precision {precision!r} requires a CUDA device.")
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 inference was requested but is not supported by this CUDA device.")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError("inference_precision must be one of: fp32, fp16, bf16.")


def _validate_inference_precision(device, precision):
    precision = str(precision or "fp32").strip().lower()
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("inference_precision must be one of: fp32, fp16, bf16.")
    if precision != "fp32" and not str(device).startswith("cuda"):
        raise ValueError(f"Inference precision {precision!r} requires a CUDA device.")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA inference device {device} was requested, but CUDA is unavailable.")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 inference is not supported by the selected CUDA device.")
    return precision


def _encode_board_batch(boards):
    raw = np.stack(boards, axis=0)
    return np.stack((raw > 0, raw < 0), axis=1).astype(np.float32, copy=False)


def run_global_inference_server(
    model_state,
    model_config,
    worker_response_conns,
    request_queue,
    stop_event,
    device,
    batch_size,
    batch_timeout_s,
    inference_precision="fp32",
    stats_queue=None,
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

    request_count = 0
    batch_count = 0
    max_observed_batch = 0
    timeout_flushes = 0
    inference_time_s = 0.0
    try:
        shutdown_requested = False
        while not shutdown_requested:
            batch = []
            try:
                first_request = request_queue.get(timeout=batch_timeout_s)
                if first_request is None:
                    break
                batch.append(first_request)
            except queue.Empty:
                continue

            start = time.perf_counter()
            while len(batch) < batch_size:
                elapsed = time.perf_counter() - start
                if elapsed >= batch_timeout_s:
                    break
                try:
                    request = request_queue.get(timeout=max(0.0, batch_timeout_s - elapsed))
                    if request is None:
                        shutdown_requested = True
                        break
                    batch.append(request)
                except queue.Empty:
                    break

            if len(batch) < batch_size:
                timeout_flushes += 1
            request_count += len(batch)
            batch_count += 1
            max_observed_batch = max(max_observed_batch, len(batch))
            states = _encode_board_batch([req[2] for req in batch])
            tensor = torch.from_numpy(states).to(device)

            inference_start = time.perf_counter()
            with torch.inference_mode():
                with _inference_autocast(device, inference_precision):
                    log_pi, v = model(tensor)
            inference_time_s += time.perf_counter() - inference_start

            policies = torch.exp(log_pi.float()).cpu().numpy().astype(np.float32, copy=False)
            values = v.float().squeeze(1).cpu().numpy().astype(np.float32, copy=False)
            for idx, (worker_id, request_id, _) in enumerate(batch):
                worker_response_conns[int(worker_id)].send(
                    (int(request_id), policies[idx], float(values[idx]))
                )
    finally:
        if stats_queue is not None:
            stats_queue.put(
                {
                    "requests": int(request_count),
                    "batches": int(batch_count),
                    "average_batch_size": float(request_count / max(1, batch_count)),
                    "max_batch_size": int(max_observed_batch),
                    "batch_capacity": int(batch_size),
                    "batch_fill_ratio": float(request_count / max(1, batch_count * batch_size)),
                    "timeout_flushes": int(timeout_flushes),
                    "inference_time_s": float(inference_time_s),
                    "precision": str(inference_precision),
                }
            )
            stats_queue.close()
            stats_queue.join_thread()
        for conn in worker_response_conns.values():
            try:
                conn.close()
            except Exception:
                pass
        request_queue.close()
        request_queue.join_thread()


class GlobalInferenceServer:
    def __init__(
        self,
        model_state,
        worker_response_conns,
        model_config=None,
        device="cuda",
        batch_size=32,
        batch_timeout_s=0.003,
        inference_precision="fp32",
    ):
        inference_precision = _validate_inference_precision(device, inference_precision)
        self.request_queue = mp.Queue()
        self.stop_event = mp.Event()
        self.stats_queue = mp.Queue(maxsize=1)
        self.last_stats = {}
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
                str(inference_precision),
                self.stats_queue,
            ),
        )

    def start(self):
        self.process.start()

    def close(self):
        self.stop_event.set()
        self.request_queue.put(None)
        self.process.join(timeout=5.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2.0)
        try:
            self.last_stats = self.stats_queue.get(timeout=1.0)
        except queue.Empty:
            self.last_stats = {}
        if self.last_stats:
            logging.info("Inference server stats: %s", self.last_stats)
        self.request_queue.close()
        self.request_queue.join_thread()
        self.stats_queue.close()
        self.stats_queue.join_thread()
        return dict(self.last_stats)


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
        self.receiver_thread.join(timeout=1.0)
        try:
            self.response_conn.close()
        except Exception:
            pass
        self._fail_pending(RuntimeError("Remote inference client closed."))
        try:
            self.request_queue.close()
            self.request_queue.join_thread()
        except (AttributeError, OSError, ValueError):
            pass

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
        self.reuse_tree = bool(getattr(args, "reuse_mcts_tree", True))
        self.persistent_threads = bool(getattr(args, "persistent_mcts_threads", True))
        self.enable_search_stats = bool(getattr(args, "enable_mcts_search_stats", True))
        self.root = None
        self._closed = False
        self._executor = None
        if self.persistent_threads and self.num_worker_threads > 1:
            self._executor = ThreadPoolExecutor(
                max_workers=self.num_worker_threads,
                thread_name_prefix="mcts-search",
            )
        self.search_stats = {
            "root_builds": 0,
            "root_reuses": 0,
            "advance_reuses": 0,
            "advance_resets": 0,
            "retained_nodes": 0,
            "simulations": 0,
            "thread_tasks": 0,
            "reset_reasons": {},
        }

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
            self.close()
        except Exception:
            pass

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        if getattr(self, "owns_inference_manager", False) and hasattr(self, "inference_manager"):
            self.inference_manager.close()

    def get_search_stats(self):
        stats = dict(self.search_stats)
        stats["reset_reasons"] = dict(self.search_stats["reset_reasons"])
        return stats

    def _record_reset(self, reason):
        self.search_stats["advance_resets"] += 1
        reasons = self.search_stats["reset_reasons"]
        reasons[str(reason)] = int(reasons.get(str(reason), 0)) + 1

    def _new_root(self, canonical_board):
        self.search_stats["root_builds"] += 1
        return TreeNode(state=np.array(canonical_board, copy=True), prior_probability=1.0)

    def _prepare_root(self, canonical_board):
        canonical_board = np.asarray(canonical_board)
        if not self.reuse_tree:
            self.root = self._new_root(canonical_board)
            return self.root
        if self.root is not None and np.array_equal(self.root.state, canonical_board):
            self.root.parent = None
            self.search_stats["root_reuses"] += 1
            return self.root
        if self.root is not None:
            self._record_reset("root_state_mismatch")
        self.root = self._new_root(canonical_board)
        return self.root

    def _count_subtree_nodes(self, root):
        if not self.enable_search_stats or root is None:
            return 0
        count = 0
        stack = [root]
        while stack:
            node = stack.pop()
            count += 1
            with node.lock:
                stack.extend(node.children.values())
        return count

    def advance_to_action(self, action, next_canonical_board):
        """Advance a retained tree after a real move, or reset safely on a miss."""
        next_canonical_board = np.asarray(next_canonical_board)
        if not self.reuse_tree:
            self.root = self._new_root(next_canonical_board)
            self._record_reset("reuse_disabled")
            return False

        child = None
        if self.root is not None:
            with self.root.lock:
                child = self.root.children.get(int(action))
        if child is not None and np.array_equal(child.state, next_canonical_board):
            child.parent = None
            self.root = child
            self.search_stats["advance_reuses"] += 1
            self.search_stats["retained_nodes"] += self._count_subtree_nodes(child)
            return True

        reason = "missing_child" if child is None else "child_state_mismatch"
        self._record_reset(reason)
        self.root = self._new_root(next_canonical_board)
        return False

    def _normalize_probs(self, probs):
        probs = np.asarray(probs, dtype=np.float64)
        probs = np.clip(probs, 0.0, None)
        total = float(np.sum(probs))
        if not np.isfinite(total) or total <= 0.0:
            return None
        return (probs / total).astype(np.float64, copy=False)

    def get_action_prob(self, canonical_board, temp=1, training=False, dirichlet_alpha=None, dirichlet_epsilon=None):
        if self._closed:
            raise RuntimeError("MCTS is closed.")
        root = self._prepare_root(canonical_board)

        # Pre-expand the root node
        with root.lock:
            root_expanded = root.is_expanded
        if not root_expanded:
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

        total_workers = max(1, self.num_worker_threads)
        self.search_stats["thread_tasks"] += total_workers
        if self._executor is not None:
            futures = [
                self._executor.submit(self._worker_loop, root, sim_counter, counter_lock)
                for _ in range(total_workers)
            ]
            for future in futures:
                future.result()
        else:
            workers = []
            for _ in range(total_workers):
                worker = threading.Thread(target=self._worker_loop, args=(root, sim_counter, counter_lock), daemon=True)
                workers.append(worker)
                worker.start()
            for worker in workers:
                worker.join()
        self.search_stats["simulations"] += int(sim_counter["count"])

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
