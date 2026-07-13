import torch
import torch.nn as nn
class FullyConvQNetwork(nn.Module):
    def __init__(
            self,
            input_channels: int = 11,
            hidden_channels: int = 64,
            output_channels: int = 1,
            kernel_size: tuple[int, int] = (3, 3)
        ):
        
        super().__init__()
        # Making an half-padded convolution to not change
        # the orignal tensor shape
        # L'operatore // prende già la parte intera inferiore
        padding_size = tuple(k // 2 for k in kernel_size)


        self.feature_extractor = nn.Sequential(
            # 1)
            # # shape:
            # [B, 10+1, H, W] -> [B, 64, H, W]
            nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, 
                      kernel_size=kernel_size, padding=padding_size, stride=1),
            nn.ReLU(),
            # 2)
            # shape:
            # [B, 64, H, W] -> [B, 64, H, W]
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels,
                      kernel_size=kernel_size, padding=padding_size, stride=1),
            nn.ReLU(),
            # 3)
            # shape:
            # [B, 64, H, W] -> [B, 64, H, W]
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels,
                      kernel_size=kernel_size, padding=padding_size, stride=1),
            nn.ReLU(),
        )

        # produces a Q-value for each cell
        # shape: [B, 64, H, W] -> [B, 1, H, W]
        self.q_head = nn.Conv2d(in_channels=hidden_channels, out_channels=output_channels,
                                kernel_size=1, padding=0, stride=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: x is a one-hot embedding representation of the state
        features = self.feature_extractor(x)
        # The q_head produces a 4d tensor of shape [B,1,H,W]
        q_map = self.q_head(features)
        # The flatten applied on a tensor collapses the channel dimension
        # from dimension 1 onwards
        # [B,1, H, W] -> [B, 1*H*W] -> [B, H*W]
        q_values = q_map.flatten(start_dim=1)

        return q_values
    
