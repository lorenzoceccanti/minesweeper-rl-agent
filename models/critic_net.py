import torch
import torch.nn as nn
class CriticNetwork(nn.Module):
    """ The critic network is a network
    that has to produce just a single
    value for each state. V(s) is the
    approximation of the expectation of
    the return."""
    def __init__(
            self,
            input_channels: int = 10,
            hidden_channels: int = 64,
            kernel_size: tuple[int, int] = (3,3),
            pooling_output: tuple[int, int] = (4,4),
            hidden_size: int = 256 # MLP
    ):
        super().__init__()
        # Making an half-padded convolution to not change
        # the orignal tensor shape
        # L'operatore // prende già la parte intera inferiore
        padding_size = tuple(k // 2 for k in kernel_size)
        pooled_features = (
            hidden_channels
            * pooling_output[0]
            * pooling_output[1]
        )

        self.cnn = nn.Sequential(
            #1) [B, 10, H, W] -> [B, 64, H, W]
            nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, 
                      kernel_size=kernel_size, padding=padding_size, stride=1),
            nn.ReLU(),
            #2) [B, 64, H, W] -> [B, 64, H, W]
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels,
                      kernel_size=kernel_size, padding=padding_size, stride=1),
            nn.ReLU(),
            #3) [B, 64, H, W] -> [B, 64, H, W]
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels,
                      kernel_size=kernel_size, padding=padding_size, stride=1),
            nn.ReLU(),
        )

        # IMPLEMENTATION CHOICE: here we decide to use a variant of pooling
        # layer called adaptive average pooling. Differently from a conventional
        # pooling layer like the ones seen in CIDL/MIRCV, here the pooling operation
        # is applied according to a global rule. In this way, the output of the 
        # adaptive pooling layer is always fixed to the shape of pooling_output.
        # In this way, we continue to maintain a neural architecture whose weights
        # DO NOT depend on the shape of the board, despite losing some spatial
        # information. In any case, for each channel, the critic can distinguish
        # if a feature appears at the top/bottom, on the left/right or at the center/borders.
        # (i.e. we maintain 16 spatial features).
        # By doing this, we can have the critic head's weights which do not depend
        # on the size of the board.
        self.adaptiveAvgPooling = nn.AdaptiveAvgPool2d(pooling_output)
        # the value head is responsible for producing V(s)
        self.value_head = nn.Sequential(
            # [B, 64, 4, 4] -> [B, 64*4*4] = [B, 1024]
            nn.Flatten(start_dim=1),
            # [B, 1024] -> [1024, 256] -> [B, 256]
            nn.Linear(pooled_features, hidden_size),
            nn.ReLU(),
            # [B, 256] -> [256, 1] -> [B,1]
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] -> [B, hidden_channels, H, W]
        features = self.cnn(x)
        # [B, hidden_channles, H, W] -> [B, hidden_channels, pooling_output.shape]
        # -> [B, 64, 4, 4] in the default case
        pooling = self.adaptiveAvgPooling(features)
       
        # [B, 64, 4, 4] -> 
        #   -> [B, 1024] flatten
        #   -> [B, 256] feed-forward
        #   -> [B, 1] linear
        # [B, 1]
        values = self.value_head(pooling)
        # we want B scalar values, as the elements of the mini-batch
        return values.squeeze(-1) # [B, 1] -> [B]