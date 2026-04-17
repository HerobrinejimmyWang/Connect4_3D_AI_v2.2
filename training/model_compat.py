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


class CompatibleModelPredictor:
    def __init__(self, model, config, action_size):
        self.model = model
        self.config = config
        self.action_size = int(action_size)
        self.device = next(self.model.parameters()).device

    def predict(self, batch_states):
        encoded = np.stack(
            [encode_board_for_model(board, self.config) for board in batch_states],
            axis=0,
        ).astype(np.float32)
        tensor = torch.from_numpy(encoded).to(self.device)
        with torch.no_grad():
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
):
    model = build_model_from_state_dict(model_state, model_config, device=device)

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

        states = np.stack(
            [encode_board_for_model(req[2], model_config) for req in batch],
            axis=0,
        ).astype(np.float32)
        tensor = torch.from_numpy(states).to(device, non_blocking=True)

        with torch.no_grad():
            log_pi, value = model(tensor)

        policies = torch.exp(log_pi).cpu().numpy().astype(np.float32, copy=False)
        values = value.squeeze(1).cpu().numpy().astype(np.float32, copy=False)

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

    for conn in worker_response_conns.values():
        try:
            conn.close()
        except Exception:
            pass


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
    ):
        self.request_queue = mp.Queue()
        self.stop_event = mp.Event()
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