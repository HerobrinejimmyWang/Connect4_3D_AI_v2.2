import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def board_to_channels(board):
    """
    Convert a canonical board of shape (8, 5, 5) with values 1/0/-1
    to a 2-channel float32 array of shape (2, 8, 5, 5).
      channel 0: current player's pieces (board > 0)
      channel 1: opponent's pieces       (board < 0)
    """
    ch0 = (board > 0).astype(np.float32)
    ch1 = (board < 0).astype(np.float32)
    return np.stack([ch0, ch1], axis=0)


# Number of input channels fed to Connect4Net (current player + opponent)
NUM_INPUT_CHANNELS = 2


class Connect4Net(nn.Module):
    def __init__(self, board_layers=8, board_size=5, num_channels=128, dropout=0.3):
        super(Connect4Net, self).__init__()
        self.board_layers = board_layers
        self.board_size = board_size
        
        # Input: 2 channels (current player + opponent), Output: num_channels
        self.conv1 = nn.Conv3d(NUM_INPUT_CHANNELS, num_channels, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(num_channels)
        
        # Residual Blocks
        self.res1 = ResidualBlock(num_channels)
        self.res2 = ResidualBlock(num_channels)
        self.res3 = ResidualBlock(num_channels)
        self.res4 = ResidualBlock(num_channels)
        
        # Policy Head
        self.prob_conv = nn.Conv3d(num_channels, 32, 1) # Reduce channels
        self.prob_bn = nn.BatchNorm3d(32)
        self.prob_fc = nn.Linear(32 * board_layers * board_size * board_size, 
                                 board_layers * board_size * board_size)
        
        # Value Head
        self.val_conv = nn.Conv3d(num_channels, 32, 1)
        self.val_bn = nn.BatchNorm3d(32)
        self.val_fc1 = nn.Linear(32 * board_layers * board_size * board_size, 64)
        self.val_fc2 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, s):
        # s: (batch, 2, 8, 5, 5) for Conv3d (2 channels: current player + opponent)
        s = s.view(-1, NUM_INPUT_CHANNELS, self.board_layers, self.board_size, self.board_size)
        
        x = F.relu(self.bn1(self.conv1(s)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        
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