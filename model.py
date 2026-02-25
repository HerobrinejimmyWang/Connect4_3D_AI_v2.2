import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class Connect4Net(nn.Module):
    """
    AlphaZero-style network for 3-D Connect Four (8×5×5).
    Input : (batch, 8, 5, 5)
    Output: log-policy (batch, 200),  value (batch, 1)
    """

    def __init__(self, board_layers=8, board_size=5,
                 num_channels=128, num_res_blocks=4, dropout=0.3):
        super().__init__()
        self.board_layers = board_layers
        self.board_size = board_size
        action_size = board_layers * board_size * board_size
        flat_feat = 32 * board_layers * board_size * board_size

        # Stem
        self.conv1 = nn.Conv3d(1, num_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(num_channels)

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # Policy head
        self.pi_conv = nn.Conv3d(num_channels, 32, 1, bias=False)
        self.pi_bn = nn.BatchNorm3d(32)
        self.pi_fc = nn.Linear(flat_feat, action_size)

        # Value head
        self.v_conv = nn.Conv3d(num_channels, 32, 1, bias=False)
        self.v_bn = nn.BatchNorm3d(32)
        self.v_fc1 = nn.Linear(flat_feat, 64)
        self.v_fc2 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, s):
        # s: (batch, 8, 5, 5) → (batch, 1, 8, 5, 5)
        s = s.view(-1, 1, self.board_layers, self.board_size, self.board_size)

        x = F.relu(self.bn1(self.conv1(s)))
        x = self.res_blocks(x)

        # Policy
        pi = F.relu(self.pi_bn(self.pi_conv(x)))
        pi = pi.view(pi.size(0), -1)
        pi = self.pi_fc(pi)

        # Value
        v = F.relu(self.v_bn(self.v_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        v = self.dropout(v)
        v = torch.tanh(self.v_fc2(v))

        return F.log_softmax(pi, dim=1), v