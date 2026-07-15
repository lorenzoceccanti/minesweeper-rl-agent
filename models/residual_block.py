import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """ Conv -> ReLU -> Conv -> Skip connection -> ReLU.
    A residual block takes in input a number of channels
    and produces in output the same number of channels."""
    
    def __init__(self, 
                 channels: int,
                 kernel_size: tuple[int, int]):
        
        super().__init__()
        
        padding_size = tuple(k // 2 for k in kernel_size)
        
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels,
                    kernel_size=kernel_size, padding = padding_size, stride=1)
        self.norm1 = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels,
                    kernel_size=kernel_size, padding = padding_size, stride = 1)
        self.norm2 = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.relu2 = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out1 = self.conv1(x)
        # Group normalization used in place of the Layer Normalization
        # in order to continue to maintain the independence on
        # the size of the grid. Mathematically the operation is equivalent
        # to LN.
        out1 = self.norm1(out1)
        out1 = self.relu1(out1)
        out2 = self.conv2(out1)
        out2 = self.norm2(out2)
        out2 = identity + out2 # skip connection
        return self.relu2(out2)
    
