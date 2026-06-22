import logging
import re
import queue
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

from model import Connect4Net, board_to_channels
from mcts import _inference_autocast, _validate_inference_precision


class LegacyResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return out


class LegacyConnect4Net(nn.Module):
    def __init__(self, board_layers, board_size, num_channels=128, residual_blocks=4, dropout=0.0):
        super().__init__()
        self.board_layers = int(board_layers)
        self.board_size = int(board_size)
        self.num_channels = int(num_channels)
        self.residual_blocks = int(residual_blocks)

        self.conv1 = nn.Conv3d(1, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(self.num_channels)

        for index in range(1, self.residual_blocks + 1):
            setattr(self, f"res{index}", LegacyResidualBlock(self.num_channels))

        self.prob_conv = nn.Conv3d(self.num_channels, 32, 1)
        self.prob_bn = nn.BatchNorm3d(32)
        self.prob_fc = nn.Linear(
            32 * self.board_layers * self.board_size * self.board_size,
            self.board_layers * self.board_size * self.board_size,
        )

        self.val_conv = nn.Conv3d(self.num_channels, 32, 1)
        self.val_bn = nn.BatchNorm3d(32)
        self.val_fc1 = nn.Linear(32 * self.board_layers * self.board_size * self.board_size, 64)
        self.val_fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, s):
        s = s.view(-1, 1, self.board_layers, self.board_size, self.board_size)
        x = F.relu(self.bn1(self.conv1(s)))
        for index in range(1, self.residual_blocks + 1):
            x = getattr(self, f"res{index}")(x)

        pi = F.relu(self.prob_bn(self.prob_conv(x)))
        pi = pi.view(-1, 32 * self.board_layers * self.board_size * self.board_size)
        pi = self.prob_fc(pi)

        v = F.relu(self.val_bn(self.val_conv(x)))
        v = v.view(-1, 32 * self.board_layers * self.board_size * self.board_size)
        v = F.relu(self.val_fc1(v))
        v = self.dropout(v)
        v = torch.tanh(self.val_fc2(v))
        return F.log_softmax(pi, dim=1), v


def load_checkpoint_payload(model_path):
    try:
        with torch.serialization.safe_globals([np._core.multiarray._reconstruct]):
            return torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception:
        try:
            with torch.serialization.safe_globals([np.core.multiarray._reconstruct]):
                return torch.load(model_path, map_location="cpu", weights_only=False)
        except Exception:
            return torch.load(model_path, map_location="cpu", weights_only=False)


def extract_state_dict_and_metadata(checkpoint):
    metadata = {}
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        metadata = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, (dict, OrderedDict)):
        raise TypeError("无法从模型文件中提取 state_dict。")
    return state_dict, metadata


def detect_architecture(state_dict):
    if any(key.startswith("res_blocks.") for key in state_dict):
        return "modern"
    if any(re.match(r"res\d+\.conv1\.weight", key) for key in state_dict):
        return "legacy-v21"
    raise ValueError("无法识别模型架构。")


def infer_res_block_count(state_dict):
    indices = []
    for key in state_dict:
        match = re.match(r"res_blocks\.(\d+)\.conv1\.weight", key)
        if match:
            indices.append(int(match.group(1)))
    if indices:
        return max(indices) + 1

    legacy_indices = []
    for key in state_dict:
        match = re.match(r"res(\d+)\.conv1\.weight", key)
        if match:
            legacy_indices.append(int(match.group(1)))
    return max(legacy_indices) if legacy_indices else 0


def infer_board_config_from_action_dim(action_dim):
    action_dim = int(action_dim)
    for board_size in range(4, 9):
        plane = board_size * board_size
        if action_dim % plane == 0:
            board_layers = action_dim // plane
            if board_layers > 0:
                return board_size, board_layers
    raise ValueError(f"无法根据动作维度 {action_dim} 推断棋盘配置。")


def infer_model_config(state_dict):
    action_dim = int(state_dict["prob_fc.bias"].shape[0])
    board_size, board_layers = infer_board_config_from_action_dim(action_dim)
    return {
        "board_size": board_size,
        "board_layers": board_layers,
        "num_channels": int(state_dict["conv1.weight"].shape[0]),
        "input_channels": int(state_dict["conv1.weight"].shape[1]),
        "num_res_blocks": infer_res_block_count(state_dict),
        "architecture": detect_architecture(state_dict),
    }


def load_compatible_model(model_path, device="cpu"):
    checkpoint = load_checkpoint_payload(model_path)
    state_dict, metadata = extract_state_dict_and_metadata(checkpoint)
    config = infer_model_config(state_dict)
    model = build_model_from_state_dict(state_dict, config, device=device)
    metadata.update(
        {
            "model_path": str(model_path),
            "architecture": config["architecture"],
            "board_size": config["board_size"],
            "board_layers": config["board_layers"],
            "input_channels": config["input_channels"],
        }
    )
    return model, config, metadata


def build_model_from_state_dict(state_dict, config, device="cpu"):
    if config["architecture"] == "legacy-v21":
        model = LegacyConnect4Net(
            board_layers=config["board_layers"],
            board_size=config["board_size"],
            num_channels=config["num_channels"],
            residual_blocks=config["num_res_blocks"],
        )
    else:
        model = Connect4Net(
            board_layers=config["board_layers"],
            board_size=config["board_size"],
            num_channels=config["num_channels"],
            num_res_blocks=config.get("num_res_blocks", 8),
        )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def adapt_board_for_model(board, config):
    board = np.asarray(board, dtype=np.int8)
    board_layers = int(config["board_layers"])
    board_size = int(config["board_size"])
    adapted = np.zeros((board_layers, board_size, board_size), dtype=np.int8)
    copy_layers = min(board.shape[0], board_layers)
    copy_rows = min(board.shape[1], board_size)
    copy_cols = min(board.shape[2], board_size)
    adapted[:copy_layers, :copy_rows, :copy_cols] = board[:copy_layers, :copy_rows, :copy_cols]
    return adapted


def encode_board_for_model(board, config):
    adapted = adapt_board_for_model(board, config)
    if int(config.get("input_channels", 2)) == 1:
        return adapted[np.newaxis, ...].astype(np.float32)
    return board_to_channels(adapted).astype(np.float32)


def encode_board_batch_for_model(boards, config):
    raw = np.stack(boards, axis=0).astype(np.int8, copy=False)
    board_layers = int(config["board_layers"])
    board_size = int(config["board_size"])
    adapted = np.zeros((raw.shape[0], board_layers, board_size, board_size), dtype=np.int8)
    copy_layers = min(raw.shape[1], board_layers)
    copy_rows = min(raw.shape[2], board_size)
    copy_cols = min(raw.shape[3], board_size)
    adapted[:, :copy_layers, :copy_rows, :copy_cols] = raw[:, :copy_layers, :copy_rows, :copy_cols]
    if int(config.get("input_channels", 2)) == 1:
        return adapted[:, np.newaxis, ...].astype(np.float32)
    return np.stack((adapted > 0, adapted < 0), axis=1).astype(np.float32, copy=False)


class CompatibleModelPredictor:
    def __init__(self, model, config, action_size):
        self.model = model
        self.config = config
        self.action_size = int(action_size)
        self.device = next(self.model.parameters()).device

    def predict(self, batch_states):
        encoded = encode_board_batch_for_model(batch_states, self.config)
        tensor = torch.from_numpy(encoded).to(self.device)
        with torch.inference_mode():
            log_pi, value = self.model(tensor)
        policy = torch.exp(log_pi).cpu().numpy().astype(np.float32, copy=False)
        policy = np.asarray([self._crop_and_normalize(row) for row in policy], dtype=np.float32)
        value = value.squeeze(1).cpu().numpy().astype(np.float32, copy=False)
        return policy, value

    def _crop_and_normalize(self, policy):
        cropped = np.asarray(policy[: self.action_size], dtype=np.float32)
        cropped = np.clip(cropped, 0.0, None)
        total = float(np.sum(cropped))
        if not np.isfinite(total) or total <= 0.0:
            return np.full(self.action_size, 1.0 / self.action_size, dtype=np.float32)
        return cropped / total


def run_compatible_inference_server(
    model_state,
    model_config,
    action_size,
    worker_response_conns,
    request_queue,
    stop_event,
    device,
    batch_size,
    batch_timeout_s,
    inference_precision="fp32",
    stats_queue=None,
):
    model = build_model_from_state_dict(model_state, model_config, device=device)

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
                first_req = request_queue.get(timeout=batch_timeout_s)
                if first_req is None:
                    break
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
                    request = request_queue.get(timeout=timeout_left)
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
            states = encode_board_batch_for_model([req[2] for req in batch], model_config)
            tensor = torch.from_numpy(states).to(device)

            inference_start = time.perf_counter()
            with torch.inference_mode():
                with _inference_autocast(device, inference_precision):
                    log_pi, value = model(tensor)
            inference_time_s += time.perf_counter() - inference_start

            policies = torch.exp(log_pi.float()).cpu().numpy().astype(np.float32, copy=False)
            values = value.float().squeeze(1).cpu().numpy().astype(np.float32, copy=False)

            for idx, (worker_id, request_id, _) in enumerate(batch):
                cropped = np.asarray(policies[idx][: int(action_size)], dtype=np.float32)
                cropped = np.clip(cropped, 0.0, None)
                total = float(np.sum(cropped))
                if not np.isfinite(total) or total <= 0.0:
                    cropped = np.full(int(action_size), 1.0 / int(action_size), dtype=np.float32)
                else:
                    cropped = cropped / total
                worker_response_conns[int(worker_id)].send(
                    (int(request_id), cropped, float(values[idx]))
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


class CompatibleGlobalInferenceServer:
    def __init__(
        self,
        model_state,
        model_config,
        action_size,
        worker_response_conns,
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
            target=run_compatible_inference_server,
            args=(
                model_state,
                model_config,
                int(action_size),
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
            logging.info("Compatible inference server stats: %s", self.last_stats)
        self.request_queue.close()
        self.request_queue.join_thread()
        self.stats_queue.close()
        self.stats_queue.join_thread()
        return dict(self.last_stats)
