import torch
import torch.nn as nn
class ActorNetwork(nn.Module):
    """ ActorNetwork is a policy-based network:
    when estimating the policy pi(a|s),
    produces a probability distribution over
    the action space A."""
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

        self.network = nn.Sequential(
            #1) [B, 10+1, H, W] -> [B, 64, H, W]
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
            # 4) Producing a value for each action
            # [B, 64, H, W] -> [B, 1, H, W]
            # These are interpreted as logits
            nn.Conv2d(in_channels=hidden_channels, out_channels=output_channels,
                    kernel_size=1, padding=0, stride=1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # the backbone network produces [B, 1, H, W]
        logits = self.network(state)
        # we collapse the 4D tensor into a 2D one
        # [B, 1, H, W] -> [B, H*W]
        logits = logits.flatten(start_dim = 1)
        # an actor nn approximates a policy: it's a probability distribution
        # over the action space. it's a long vector of size H*W.
        # for each element of the vector we expect >= 0
        # and their sum equal to 1. we need a softmax

        return logits