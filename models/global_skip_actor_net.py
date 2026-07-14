import torch
import torch.nn as nn
from models import global_skip_backbone

class GlobalSkipActorNetwork(nn.Module):
    
    def __init__(
            self,
            input_channels: int = 11, # C
            hidden_channels: int = 64, # F
            global_features_dim: int = 16, #G
            output_channels: int = 1,
            kernel_size: tuple[int, int] = (3,3)
    ):
        super().__init__()
        # la backbone è tutto il pezzo frontale di CNN, esclusa la parte finale
        # della convoluzione 1x1
        self.backbone = global_skip_backbone.GlobalSkipBackbone(
            input_channels, hidden_channels, global_features_dim,
            kernel_size)
        # [B, (F+G), H, W] -> [B, 1, H, W]
        self.final_conv = nn.Conv2d(hidden_channels + global_features_dim, out_channels=output_channels,
                    kernel_size=1, padding=0, stride=1)
    
    def forward(self, x):
        # [B, C, H, W] -> [B, (F+G), H, W]
        combined_features = self.backbone(x)
        # [B, (F+G), H, W] -> [B, 1, H, W]
        logits = self.final_conv(combined_features)
        # [B, 1, H, W] -> [B, H*W]
        return logits.flatten(start_dim=1)