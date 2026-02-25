import torch
import torch.nn as nn
import torch.nn.functional as F

class Connect4Net(nn.Module):
    def __init__(self, board_layers=8, board_size=5, num_channels=128, dropout=0.3):
        super(Connect4Net, self).__init__()
        self.board_layers = board_layers
        self.board_size = board_size
        
        # Input: 1 channel (board state), Output: num_channels
        self.conv1 = nn.Conv3d(1, num_channels, 3, stride=1, padding=1)
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
        # s: (batch, 8, 5, 5) -> needs to be (batch, 1, 8, 5, 5) for Conv3d
        s = s.view(-1, 1, self.board_layers, self.board_size, self.board_size)
        
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