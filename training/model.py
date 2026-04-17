import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from game_rules import BOARD_SIZE, MAX_LAYERS


def board_to_channels(board):
    """
        Convert a canonical board of shape (MAX_LAYERS, BOARD_SIZE, BOARD_SIZE)
        with values 1/0/-1 to a 2-channel float32 array of shape
        (2, MAX_LAYERS, BOARD_SIZE, BOARD_SIZE).
      channel 0: current player's pieces (board > 0)
      channel 1: opponent's pieces       (board < 0)
    """
    ch0 = (board > 0).astype(np.float32)
    ch1 = (board < 0).astype(np.float32)
    return np.stack([ch0, ch1], axis=0)


# Number of input channels fed to Connect4Net (current player + opponent)
NUM_INPUT_CHANNELS = 2
DEFAULT_NUM_RES_BLOCKS = 8


class Connect4Net(nn.Module):
    def __init__(
        self,
        board_layers=MAX_LAYERS,
        board_size=BOARD_SIZE,
        num_channels=256,
        num_res_blocks=DEFAULT_NUM_RES_BLOCKS,
        dropout=0.3,
    ):
        super(Connect4Net, self).__init__()
        self.board_layers = board_layers
        self.board_size = board_size
        self.num_channels = int(num_channels)
        self.num_res_blocks = int(num_res_blocks)
        
        # Input: 2 channels (current player + opponent), Output: num_channels
        self.conv1 = nn.Conv3d(NUM_INPUT_CHANNELS, self.num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(self.num_channels)
        
        # Residual Blocks
        self.res_blocks = nn.ModuleList([ResidualBlock(self.num_channels) for _ in range(self.num_res_blocks)])
        
        # Policy Head
        self.prob_conv = nn.Conv3d(self.num_channels, 32, 1) # Reduce channels
        self.prob_bn = nn.BatchNorm3d(32)
        self.prob_fc = nn.Linear(32 * board_layers * board_size * board_size, 
                                 board_layers * board_size * board_size)
        
        # Value Head
        self.val_conv = nn.Conv3d(self.num_channels, 32, 1)
        self.val_bn = nn.BatchNorm3d(32)
        self.val_fc1 = nn.Linear(32 * board_layers * board_size * board_size, 64)
        self.val_fc2 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, s):
        # s: (batch, 2, board_layers, board_size, board_size) for Conv3d.
        s = s.view(-1, NUM_INPUT_CHANNELS, self.board_layers, self.board_size, self.board_size)
        
        x = F.relu(self.bn1(self.conv1(s)))
        for res in self.res_blocks:
            x = res(x)
        
        # Policy
        pi = F.relu(self.prob_bn(self.prob_conv(x)))
        pi = pi.view(-1, 32 * self.board_layers * self.board_size * self.board_size)
        pi = self.prob_fc(pi)
        
        # Value
        v = F.relu(self.val_bn(self.val_conv(x)))
        v = v.view(-1, 32 * self.board_layers * self.board_size * self.board_size)
        v = F.relu(self.val_fc1(v))
        v = self.dropout(v)
        v = torch.tanh(self.val_fc2(v))
        
        return F.log_softmax(pi, dim=1), v

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


def build_model_config(
    board_layers=MAX_LAYERS,
    board_size=BOARD_SIZE,
    num_channels=256,
    num_res_blocks=DEFAULT_NUM_RES_BLOCKS,
    input_channels=NUM_INPUT_CHANNELS,
):
    return {
        "board_layers": int(board_layers),
        "board_size": int(board_size),
        "num_channels": int(num_channels),
        "num_res_blocks": int(num_res_blocks),
        "input_channels": int(input_channels),
        "architecture": "modern",
    }


def extract_model_config(model):
    return build_model_config(
        board_layers=getattr(model, "board_layers", MAX_LAYERS),
        board_size=getattr(model, "board_size", BOARD_SIZE),
        num_channels=getattr(model, "num_channels", 256),
        num_res_blocks=getattr(model, "num_res_blocks", DEFAULT_NUM_RES_BLOCKS),
        input_channels=NUM_INPUT_CHANNELS,
    )