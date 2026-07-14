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
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels,
                    kernel_size=kernel_size, padding = padding_size, stride = 1)
        self.relu2 = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out1 = self.conv1(x)
        out1 = self.relu1(out1)
        out2 = self.conv2(out1)
        out2 = identity + out2 # skip connection
        return self.relu2(out2)
    
