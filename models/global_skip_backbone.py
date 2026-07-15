import torch
import torch.nn as nn
from models import residual_block

class GlobalSkipBackbone(nn.Module):
    def __init__(
            self,
            input_channels: int, # C
            hidden_channels: int, # F
            global_features_dim: int, #G
            kernel_size: tuple[int, int]
    ):
        # nel costruttore ci mettiamo solo i blocchi principali
        # tutte le operazioni di flattening/reshaping/etc nella forward
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.global_features_dim = global_features_dim
        self.kernel_size = kernel_size

        padding_size = tuple(k // 2 for k in self.kernel_size)
        
        # prima convoluzione [B, C, H, W] -> [B, F, H, W]
        self.init_conv = nn.Conv2d(
            in_channels=self.input_channels,
            out_channels=self.hidden_channels,
            kernel_size = self.kernel_size,
            padding=padding_size,
            stride=1
        )
        self.init_norm = nn.GroupNorm(num_groups=1, num_channels=self.hidden_channels)
        self.init_relu = nn.ReLU()

        # feature locali: 2 residual blocks con skip connection
        # [B, F, H, W] -> [B, F, H, W]
        self.res_block1 = residual_block.ResidualBlock(self.hidden_channels, self.kernel_size)
        self.res_block2 = residual_block.ResidualBlock(self.hidden_channels, self.kernel_size)

        # ramo globale: qua nel costruttore si include solo il pezzo relativo
        # al ramo feed-forward
        # [B, F] -> [F, G] parametri -> [B, G]
        self.global_mlp = nn.Sequential(
            nn.Linear(self.hidden_channels, self.global_features_dim),
            # nel ramo globale è possibile utilizzare la classica LayerNorm e non la
            # GroupNorm perché G, global_features_dim, è una dimensione fissa
            # e valida per tutte le board (è deciso come hyperparametro a priori della fase
            # di train dell'agente)
            nn.LayerNorm(self.global_features_dim),
            nn.ReLU(),

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        # retrieving H and W
        _, _, H, W = x.shape

        # == local features == 
        # [B, C, H, W] -> [B, F, H, W]
        x0 = self.init_conv(x)
        x0 = self.init_relu(self.init_norm(x0))
        # [B, F, H, W] -> [B, F, H, W]
        x1 = self.res_block1(x0)
        # [B, F, H, W] -> [B, F, H, W]
        local_features = self.res_block2(x1)

        # == global features ==
        # [B, F, H, W] -> [B, F]
        global_pooled = torch.mean(local_features, dim=(2,3))
        # [B, F] -> [B, G]
        out_global_ff = self.global_mlp(global_pooled)
        # unsqueeze + expand
        # [B, G] -> [B, G, 1, 1]
        global_features = out_global_ff.unsqueeze(-1).unsqueeze(-1)
        # [B, G, 1, 1] -> [B, G, H, W]
        global_features = global_features.expand(-1, -1, H, W)

        # == concat local and global features
        # concat([B, F, H, W]; [B, G, H, W]) -> [B, (F+G), H, W]
        combined = torch.cat([local_features, global_features], dim=1)

        return combined