from __future__ import annotations

import re
import sys
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = Path(__file__).resolve().parent
TRAINING_DIR = CURRENT_DIR.parent / "training"
TRAIN_FEATURES_DIR = CURRENT_DIR.parent / "train_features"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))
if str(TRAIN_FEATURES_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_FEATURES_DIR))

from arena_game_rules import infer_board_size_from_action_dim
from mcts import AsyncBatchInferenceManager, MCTS
from model import Connect4Net, board_to_channels

try:
    from feature_extractor import CandidateFeatureExtractor
    from tiny_policy_model import TinyCandidatePolicyNet
except Exception:
    CandidateFeatureExtractor = None
    TinyCandidatePolicyNet = None


def _mask_policy_to_valid_moves(game, board, policy):
    valid_moves = game.get_valid_moves(board).astype(np.float64)
    valid_sum = float(np.sum(valid_moves))
    if valid_sum <= 0.0:
        return np.full(game.get_action_size(), 1.0 / game.get_action_size(), dtype=np.float64)

    masked_policy = np.asarray(policy, dtype=np.float64)
    masked_policy = np.clip(masked_policy, 0.0, None) * valid_moves
    masked_sum = float(np.sum(masked_policy))
    if not np.isfinite(masked_sum) or masked_sum <= 0.0:
        return valid_moves / valid_sum
    return masked_policy / masked_sum


class BaseAgent(ABC):
    def __init__(self, game, name: str):
        self.game = game
        self.name = name
        self.agent_type = "base"

    @abstractmethod
    def get_action(self, board, player, temp=0):
        raise NotImplementedError

    def close(self):
        return None


class MCTSAgent(BaseAgent):
    def __init__(
        self,
        game,
        model_path,
        name=None,
        device=None,
        model_config=None,
        num_mcts_sims=128,
        cpuct=1.0,
        num_mcts_threads=8,
        virtual_loss=1.0,
        inference_batch_size=32,
        inference_timeout_s=0.003,
    ):
        model_path = Path(model_path)
        resolved_device = self._resolve_device(device)
        model, loaded_config, metadata = load_model_checkpoint(
            model_path=model_path,
            requested_config=model_config,
            game=game,
            device=resolved_device,
        )

        display_name = name or model_path.stem
        super().__init__(game=game, name=display_name)
        self.agent_type = "mcts"
        self.model_path = model_path
        self.device = resolved_device
        self.model = model
        self.model_config = loaded_config
        self.metadata = metadata
        self.predictor = ArenaModelPredictor(
            model=self.model,
            game_layers=int(game.get_board_size()[0]),
            game_size=int(game.get_board_size()[1]),
            model_layers=int(self.model_config["board_layers"]),
            model_size=int(self.model_config["board_size"]),
            input_encoding="single-channel" if self.model_config.get("input_channels") == 1 else "two-channel",
        )
        self.inference_manager = AsyncBatchInferenceManager(
            self.predictor,
            batch_size=inference_batch_size,
            batch_timeout_s=inference_timeout_s,
        )
        self.mcts_args = SimpleNamespace(
            cpuct=float(cpuct),
            num_mcts_sims=int(num_mcts_sims),
            num_mcts_threads=int(num_mcts_threads),
            virtual_loss=float(virtual_loss),
            inference_batch_size=int(inference_batch_size),
            inference_timeout_s=float(inference_timeout_s),
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.0,
        )

        self._validate_game_compatibility()

    def get_action(self, board, player, temp=0):
        canonical_board = self.game.get_canonical_form(board, player)
        mcts = MCTS(self.game, self.inference_manager, self.mcts_args)
        probs = np.asarray(mcts.get_action_prob(canonical_board, temp=temp, training=False), dtype=np.float64)
        probs = _mask_policy_to_valid_moves(self.game, board, probs)

        if temp == 0:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(len(probs), p=probs))
        return action, {"policy": probs}

    def close(self):
        self.inference_manager.close()

    @staticmethod
    def _resolve_device(device):
        if device:
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _validate_game_compatibility(self):
        board_layers = int(self.model_config["board_layers"])
        board_size = int(self.model_config["board_size"])
        game_shape = self.game.get_board_size()
        if board_size != int(game_shape[1]):
            raise ValueError(
                f"Agent {self.name} expects board size {board_size}x{board_size}, but arena uses {game_shape}."
            )
        if board_layers < int(game_shape[0]):
            raise ValueError(
                f"Agent {self.name} expects {board_layers} layers, but arena uses {int(game_shape[0])} layers. "
                "仅支持旧模型层数大于等于竞技规则层数的兼容适配。"
            )


class RandomAgent(BaseAgent):
    def __init__(self, game, name):
        super().__init__(game=game, name=name)
        self.agent_type = "random"

    def get_action(self, board, player, temp=0):
        valid_moves = np.flatnonzero(self.game.get_valid_moves(board) > 0)
        if len(valid_moves) == 0:
            raise RuntimeError(f"{self.name} has no valid moves on a non-terminal board.")
        action = int(np.random.choice(valid_moves))
        return action, {}


class TinyPolicyAgent(BaseAgent):
    """Direct tiny policy agent that scores 25 candidates and maps back to legal 3D actions."""

    def __init__(self, game, model_path, name=None, device=None):
        if CandidateFeatureExtractor is None or TinyCandidatePolicyNet is None:
            raise RuntimeError("Tiny policy dependencies are unavailable in train_features module.")

        model_path = Path(model_path)
        resolved_device = self._resolve_device(device)
        payload = _load_checkpoint_payload(model_path)
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise ValueError(f"Tiny policy checkpoint format is invalid: {model_path}")

        model_cfg = dict(payload.get("model_config") or {})
        candidate_count = int(model_cfg.get("candidate_count", 25))
        board_shape = game.get_board_size()
        game_candidate_count = int(board_shape[1]) * int(board_shape[2])
        if candidate_count != game_candidate_count:
            raise ValueError(
                f"Tiny checkpoint candidate_count={candidate_count}, but game expects {game_candidate_count}."
            )

        self.model = TinyCandidatePolicyNet(
            global_dim=int(model_cfg.get("global_dim", 20)),
            candidate_dim=int(model_cfg.get("candidate_dim", 28)),
            global_hidden=int(model_cfg.get("global_hidden", 24)),
            candidate_hidden=int(model_cfg.get("candidate_hidden", 24)),
            fusion_hidden=int(model_cfg.get("fusion_hidden", 16)),
            dropout=float(model_cfg.get("dropout", 0.05)),
            value_hidden=int(model_cfg.get("value_hidden", 12)),
        )
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.model.to(resolved_device)
        self.model.eval()

        display_name = name or model_path.stem
        super().__init__(game=game, name=display_name)
        self.agent_type = "tiny"
        self.model_path = model_path
        self.device = resolved_device
        self.model_config = model_cfg
        self.metadata = {
            "model_path": str(model_path),
            "device": str(resolved_device),
            "architecture": str(model_cfg.get("architecture", "tiny-candidate-policy-v1")),
        }
        self.feature_extractor = CandidateFeatureExtractor(
            board_size=int(board_shape[1]),
            max_layers=int(board_shape[0]),
            connect_n=int(getattr(game, "connect_n", 4)),
        )

    def get_action(self, board, player, temp=0):
        feat = self.feature_extractor.extract(board, player)
        valid_mask = np.asarray(feat["valid_mask"], dtype=np.float64)
        action_map = np.asarray(feat["candidate_action_map"], dtype=np.int64)
        valid_indices = np.flatnonzero(valid_mask > 0.5)
        if len(valid_indices) == 0:
            raise RuntimeError(f"{self.name} has no valid candidate moves on a non-terminal board.")

        global_tensor = torch.from_numpy(feat["global"]).unsqueeze(0).to(self.device)
        candidate_tensor = torch.from_numpy(feat["candidate"]).unsqueeze(0).to(self.device)
        valid_tensor = torch.from_numpy(feat["valid_mask"]).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(global_tensor, candidate_tensor, valid_mask=valid_tensor, return_value=True)
            if isinstance(output, tuple):
                logits_tensor, value_tensor = output
                value_pred = float(value_tensor.squeeze(0).cpu().item())
            else:
                logits_tensor = output
                value_pred = None

        logits = logits_tensor.squeeze(0).cpu().numpy().astype(np.float64, copy=False)
        candidate_policy = _masked_candidate_softmax(logits, valid_mask, temperature=max(1e-6, float(temp) if temp > 0 else 1.0))

        if float(temp) <= 0.0:
            chosen_idx = int(valid_indices[np.argmax(logits[valid_indices])])
        else:
            chosen_idx = int(np.random.choice(len(candidate_policy), p=candidate_policy))
            if valid_mask[chosen_idx] <= 0.5:
                chosen_idx = int(np.random.choice(valid_indices))

        action = int(action_map[chosen_idx])
        if action < 0:
            action = int(action_map[int(np.random.choice(valid_indices))])

        action_policy = np.zeros((self.game.get_action_size(),), dtype=np.float64)
        for idx in valid_indices:
            mapped_action = int(action_map[idx])
            if 0 <= mapped_action < len(action_policy):
                action_policy[mapped_action] = float(candidate_policy[idx])
        action_policy = _mask_policy_to_valid_moves(self.game, board, action_policy)

        info = {
            "candidate_policy": candidate_policy,
            "policy": action_policy,
        }
        if value_pred is not None:
            info["value"] = value_pred
        return action, info

    @staticmethod
    def _resolve_device(device):
        if device:
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"


def is_tiny_policy_checkpoint(model_path) -> bool:
    try:
        payload = _load_checkpoint_payload(model_path)
    except Exception:
        return False

    if not isinstance(payload, dict):
        return False
    if "model_state_dict" not in payload:
        return False

    model_cfg = payload.get("model_config") or {}
    return (
        isinstance(model_cfg, dict)
        and "global_dim" in model_cfg
        and "candidate_dim" in model_cfg
        and "candidate_count" in model_cfg
    )


def _masked_candidate_softmax(logits, valid_mask, temperature=1.0):
    valid_mask = np.asarray(valid_mask, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)

    valid_indices = np.flatnonzero(valid_mask > 0.5)
    if len(valid_indices) == 0:
        return np.full((len(logits),), 1.0 / float(max(1, len(logits))), dtype=np.float64)

    scaled = np.full_like(logits, -1e9, dtype=np.float64)
    scaled[valid_indices] = logits[valid_indices] / float(max(1e-6, temperature))
    max_logit = float(np.max(scaled[valid_indices]))
    exp_v = np.zeros_like(logits, dtype=np.float64)
    exp_v[valid_indices] = np.exp(scaled[valid_indices] - max_logit)
    denom = float(np.sum(exp_v[valid_indices]))
    if not np.isfinite(denom) or denom <= 0.0:
        probs = np.zeros_like(logits, dtype=np.float64)
        probs[valid_indices] = 1.0 / float(len(valid_indices))
        return probs
    probs = exp_v / denom
    return probs


class ArenaModelPredictor:
    def __init__(self, model, game_layers, game_size, model_layers, model_size, input_encoding="two-channel"):
        self.model = model
        self.device = next(self.model.parameters()).device
        self.model_lock = threading.Lock()
        self.game_layers = int(game_layers)
        self.game_size = int(game_size)
        self.model_layers = int(model_layers)
        self.model_size = int(model_size)
        self.input_encoding = input_encoding

    def predict(self, batch_states):
        adapted_states = [self._adapt_state_for_model(state) for state in batch_states]
        tensor = torch.from_numpy(
            np.stack([self._encode_board(state) for state in adapted_states], axis=0).astype(np.float32)
        ).to(self.device)

        if self.model_lock is None:
            self.model.eval()
            with torch.no_grad():
                log_pi, values = self.model(tensor)
        else:
            with self.model_lock:
                self.model.eval()
                with torch.no_grad():
                    log_pi, values = self.model(tensor)

        policies = torch.exp(log_pi).cpu().numpy().astype(np.float32, copy=False)
        cropped_policies = np.asarray([self._adapt_policy_for_game(policy) for policy in policies], dtype=np.float32)
        value_array = values.squeeze(1).cpu().numpy().astype(np.float32, copy=False)
        return cropped_policies, value_array

    def _adapt_state_for_model(self, board):
        board = np.asarray(board, dtype=np.int8)
        if self.model_layers == self.game_layers:
            return board
        padded = np.zeros((self.model_layers, self.model_size, self.model_size), dtype=np.int8)
        padded[: self.game_layers, :, :] = board
        return padded

    def _adapt_policy_for_game(self, policy):
        policy = np.asarray(policy, dtype=np.float32)
        action_count = self.game_layers * self.game_size * self.game_size
        return policy[:action_count]

    def _encode_board(self, board):
        if self.input_encoding == "single-channel":
            return board[np.newaxis, ...].astype(np.float32)
        return board_to_channels(board)


def load_model_checkpoint(model_path, requested_config, game, device):
    checkpoint = _load_checkpoint_payload(model_path)
    state_dict, metadata = _extract_state_dict_and_metadata(checkpoint)
    inferred_config = infer_model_config(state_dict, requested_config=requested_config, game=game)
    architecture = inferred_config["architecture"]
    if architecture == "legacy-v21":
        model = LegacyConnect4Net(
            board_layers=inferred_config["board_layers"],
            board_size=inferred_config["board_size"],
            num_channels=inferred_config["num_channels"],
            residual_blocks=inferred_config["num_res_blocks"],
            dropout=float(inferred_config.get("dropout", 0.0) or 0.0),
        )
    else:
        model = Connect4Net(
            board_layers=inferred_config["board_layers"],
            board_size=inferred_config["board_size"],
            num_channels=inferred_config["num_channels"],
            num_res_blocks=inferred_config.get("num_res_blocks", 8),
            dropout=float(inferred_config.get("dropout", 0.0) or 0.0),
        )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    metadata.update(
        {
            "model_path": str(model_path),
            "device": str(device),
            "architecture": architecture,
        }
    )
    return model, inferred_config, metadata


def infer_model_config(state_dict, requested_config=None, game=None):
    requested_config = requested_config or {}
    action_dim = int(state_dict["prob_fc.bias"].shape[0])
    preferred_size = requested_config.get("board_size")
    if preferred_size is None and game is not None:
        preferred_size = int(game.get_board_size()[1])
    inferred_size, inferred_layers = infer_board_size_from_action_dim(action_dim, preferred_size=preferred_size)

    requested_board_size = requested_config.get("board_size")
    requested_board_layers = requested_config.get("board_layers")
    if requested_board_size is not None and int(requested_board_size) != int(inferred_size):
        raise ValueError(
            f"显式指定的棋盘边长 {requested_board_size} 与模型权重推断值 {inferred_size} 不一致。"
        )
    if requested_board_layers is not None and int(requested_board_layers) != int(inferred_layers):
        raise ValueError(
            f"显式指定的棋盘层数 {requested_board_layers} 与模型权重推断值 {inferred_layers} 不一致。"
        )

    architecture = _detect_architecture(state_dict)
    config = {
        "board_size": int(inferred_size),
        "board_layers": int(inferred_layers),
        "num_channels": int(state_dict["conv1.weight"].shape[0]),
        "dropout": float(requested_config.get("dropout") or 0.0),
        "num_res_blocks": _infer_res_block_count(state_dict),
        "policy_channels": int(state_dict["prob_conv.weight"].shape[0]),
        "value_channels": int(state_dict["val_conv.weight"].shape[0]),
        "value_hidden_dim": int(state_dict["val_fc1.weight"].shape[0]),
        "input_channels": int(state_dict["conv1.weight"].shape[1]),
        "architecture": architecture,
    }
    return config


def _infer_res_block_count(state_dict):
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


def _detect_architecture(state_dict):
    if any(key.startswith("res_blocks.") for key in state_dict):
        return "modern"
    if any(re.match(r"res\d+\.conv1\.weight", key) for key in state_dict):
        return "legacy-v21"
    raise ValueError("无法识别模型架构：既不是当前版 res_blocks.*，也不是旧版 res1..resN。")


def _load_checkpoint_payload(model_path):
    try:
        with torch.serialization.safe_globals([np._core.multiarray._reconstruct]):
            return torch.load(model_path, map_location="cpu")
    except Exception:
        try:
            with torch.serialization.safe_globals([np.core.multiarray._reconstruct]):
                return torch.load(model_path, map_location="cpu")
        except Exception:
            return torch.load(model_path, map_location="cpu", weights_only=False)


def _extract_state_dict_and_metadata(checkpoint):
    metadata = {}
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        metadata = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, (dict, OrderedDict)):
        raise TypeError("无法从模型文件中提取 state_dict。")
    return state_dict, metadata


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
        self.prob_fc = nn.Linear(32 * self.board_layers * self.board_size * self.board_size, self.board_layers * self.board_size * self.board_size)

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
