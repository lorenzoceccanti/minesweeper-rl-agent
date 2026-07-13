import torch.nn as nn
from models.fully_conv_qnet import FullyConvQNetwork
from models.actor_net import ActorNetwork
from models.critic_net import CriticNetwork

def get_q_network(
        architecture_name: str,
        input_channels: int = 11,
        hidden_channels: int = 64,
        output_channels: int = 1,
        kernel_size: tuple[int, int] = (3,3)
    ) -> nn.Module:
    match architecture_name:
        case "fully_conv_3layer_64ch_11in":
            return FullyConvQNetwork(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                kernel_size=kernel_size
            )
        case "global_skip_conv_3layer_64ch_11in":
            # TODO: return a GlobalSkipQNetwork object
            pass
        case _:
            raise ValueError(f"The architecture {architecture_name} is unsupported.")

def get_actor_network(
        architecture_name: str,
        input_channels: int = 11,
        hidden_channels: int = 64,
        output_channels: int = 1,
        kernel_size: tuple[int, int] = (3,3)
    ) -> nn.Module:
    match architecture_name:
        case "fully_conv_3layer_64ch_11in":
            return ActorNetwork(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                kernel_size=kernel_size
            )
        case "global_skip_conv_3layer_64ch_11in":
            # TODO: return a GlobalActorNetwork object
            pass
        case _:
            raise ValueError(f"The architecture {architecture_name} is unsupported.")

def get_critic_network(
        architecture_name: str,
        input_channels: int = 11,
        hidden_channels: int = 64,
        output_channels: int = 1,
        kernel_size: tuple[int, int] = (3,3)
    ) -> nn.Module:
    match architecture_name:
        case "fully_conv_3layer_64ch_11in":
            return CriticNetwork(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                kernel_size=kernel_size
            )
        case "global_skip_conv_3layer_64ch_11in":
            # TODO: return a GlobalCriticNetwork object
            pass
        case _:
            raise ValueError(f"The architecture {architecture_name} is unsupported.")