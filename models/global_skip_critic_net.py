import torch
import torch.nn as nn
from models import global_skip_backbone

class GlobalSkipCriticNetwork(nn.Module):
    
    def __init__(
            self,
            input_channels: int = 11, # C
            hidden_channels: int = 64, # F
            global_features_dim: int = 16, #G
            kernel_size: tuple[int, int] = (3,3),
            pooling_output: tuple[int, int] = (4,4),
            hidden_size: int = 256 # MLP of the value head
    ):
        super().__init__()
        padding_size = tuple(k // 2 for k in kernel_size)
        # total_channels = F + G
        total_channels = hidden_channels + global_features_dim
        pooled_features = (
            total_channels
            * pooling_output[0]
            * pooling_output[1]
        )

        self.backbone = global_skip_backbone.GlobalSkipBackbone(input_channels,
                            hidden_channels, global_features_dim, kernel_size)
        
        # for the critic net, we do exactly as the critic net using fully_convolution
        # we use an adaptive avg pool 2d to maintain a little bit of spatial information
        # critic-side.
        self.adaptiveAvgPooling = nn.AdaptiveAvgPool2d(pooling_output)
        self.value_head = nn.Sequential(
            # la value head del critic prende in ingresso ciò che produce l'adaptive pooling
            # che preserva una dimensione spaziale aggregata pari a pooling_output, nel caso 
            # default 4x4
            # [B, F+G, 4, 4] -> [B, (F+G)*4*4] -> caso default: [B, 1280]
            nn.Flatten(start_dim=1),
            # [B, (F+G)*16] -> params [(F+G)*16, 256] -> [B, 256]
            nn.Linear(pooled_features, hidden_size),
            nn.ReLU(),
            # [B, 256] -> [256, 1] -> [B, 1]
            nn.Linear(hidden_size, 1)
         )
    def forward(self, x):
        # [B, C, H, W] -> [B, F+G, H, W]
        features = self.backbone(x)
        # [B, F+G, H, W] -> [B, F+G, pooling_output.shape]
        # -> [B, 80, 4, 4] in the default case
        pooling = self.adaptiveAvgPooling(features)
        
        # [B, F+G, 4, 4] -> 
        #   -> [B, (F+G)*16] flatten
        #   -> [B, hidden_size] feed-forward
        #   -> [B, 1] linear
        # [B, 1]

        values = self.value_head(pooling)
        # we want B scalar values, as the elements of the mini-batch
        return values.squeeze(-1) # [B, 1] -> [B]
