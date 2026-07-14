import torch.nn as nn
from models.fully_conv_qnet import FullyConvQNetwork
from models.actor_net import ActorNetwork
from models.critic_net import CriticNetwork

from models.global_skip_qnet import GlobalSkipQNetwork
from models.global_skip_actor_net import GlobalSkipActorNetwork
from models.global_skip_critic_net import GlobalSkipCriticNetwork

def get_q_network(
        architecture_name: str,
        input_channels: int = 11,
        hidden_channels: int = 64,
        global_features_dim: int | None = 16,
        output_channels: int = 1,
        kernel_size: tuple[int, int] = (3,3)
    ) -> nn.Module:

    # salvaguardia, nel caso ci si dimentichi di specificare
    # il global_features_dim passando dall'architettura solo locale
    # a quella globale
    if global_features_dim is None:
        global_features_dim = 16

    match architecture_name:
        case "fully_conv_3layer_64ch_11in":
            return FullyConvQNetwork(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                kernel_size=kernel_size
            )
        case "global_skip_conv_3layer_64ch_11in":
          return GlobalSkipQNetwork(
              input_channels=input_channels,
              hidden_channels=hidden_channels,
              global_features_dim=global_features_dim,
              output_channels=output_channels,
              kernel_size=kernel_size
          )
        case _:
            raise ValueError(f"The architecture {architecture_name} is unsupported.")

def get_actor_network(
        architecture_name: str,
        input_channels: int = 11,
        hidden_channels: int = 64,
        output_channels: int = 1,
        kernel_size: tuple[int, int] = (3,3),
        global_features_dim: int | None = 16
    ) -> nn.Module:

    # salvaguardia, nel caso ci si dimentichi di specificare
    # il global_features_dim passando dall'architettura solo locale
    # a quella globale
    if global_features_dim is None:
        global_features_dim = 16

    match architecture_name:
        case "fully_conv_3layer_64ch_11in":
            return ActorNetwork(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                kernel_size=kernel_size
            )
        case "global_skip_conv_3layer_64ch_11in":
            return GlobalSkipActorNetwork(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                global_features_dim=global_features_dim,
                output_channels=output_channels,
                kernel_size=kernel_size
            )
        case _:
            raise ValueError(f"The architecture {architecture_name} is unsupported.")

def get_critic_network(
        architecture_name: str,
        input_channels: int = 11,
        hidden_channels: int = 64,
        output_channels: int = 1,
        kernel_size: tuple[int, int] = (3,3),
        global_features_dim: int | None = 16,
        critic_hidden_size: int | None = 256
    ) -> nn.Module:

    # salvaguardia, nel caso ci si dimentichi di specificare
    # il global_features_dim passando dall'architettura solo locale
    # a quella globale. nel caso del critic, la salvaguardia si estende
    # anche al numero di colonne della matrice mlp critic head
    if global_features_dim is None:
        global_features_dim = 16
    if critic_hidden_size is None:
        critic_hidden_size = 256

    match architecture_name:
        case "fully_conv_3layer_64ch_11in":
            return CriticNetwork(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                hidden_size=critic_hidden_size
            )
        case "global_skip_conv_3layer_64ch_11in":
            return GlobalSkipCriticNetwork(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                global_features_dim=global_features_dim,
                kernel_size=kernel_size,
                hidden_size=critic_hidden_size
            )
        case _:
            raise ValueError(f"The architecture {architecture_name} is unsupported.")