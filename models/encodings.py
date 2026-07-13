import torch
import torch.nn.functional as F

def one_hot_encode_board(
        states: torch.Tensor,
        mine_density: float | torch.Tensor,
        num_classes: int = 10
) -> torch.Tensor:
    """ Converts a minesweeper board to one-hot embedding representation
    of the board
    Possible input shapes:
    - Input:
        [H,W] or [B, H, W]
    - mine_density:
        a scalar value, if all boards in the batch have the same density;
        a tensor of shape [B] if each board has a different density
    - Output:
        [B, num_classes+1, H, W]
    """

    # If the input tensor is specified without
    # the usage of batches, a new dimension for the batch_size equal 1
    # is added as first dimension.
    if states.ndim == 2:
        # In this case the tensor passes from [H,W] to [1,H,W] 
        states = states.unsqueeze(0)
    
    # Shape check: notice that the check on states.ndim != 3 is now
    # the only one required, because if the passed tensor shape was
    # 2, at this point a new dimension has already been added
    if states.ndim !=3:
        raise ValueError(
            f"Expected shape [H, W] or [B, H, W], "
            f"received {tuple(states.shape)}."
        )

    batch_size, height , width = states.shape

    # Mapping between what we see as number in the player_board
    # and the one hot channels
    # With this trick, we first convert -2 -> -2, 0 -> 0, .., 8->8
    # Then, since we want to associate to the unrevealed cell the 10th channel
    # (the one with id 9), we do in this way: "THE TRICK"
    
    # We force a cast to a integer on 64 bits because the PyTorch method
    # performing one-hot embedding expects an input tensor made of integers on 64 bits.
    # Also, we notice from the documentation
    # (https://docs.pytorch.org/docs/2.12/generated/torch.nn.functional.one_hot.html)
    # that the tensor returned has shape [*, num_classes]
    
    class_indices = states.to(dtype=torch.long).clone()
    class_indices[class_indices == -2] = 9 # "THE TRICK"
    
    # Producing the one-hot encoding of states
    # [B,H,W] -> [B,H,W,C = 10] 
    encoded_states = F.one_hot(class_indices, num_classes=num_classes)

    # as we are used to from other courses, 
    # we prefer to have the number of channel in front.
    # also, because the channels are the second dimension in the nn.Module
    # we have wrote.
    # [0, 1, 2, 3] -> [0, 3, 1, 2]
    # [B, H, W, C] -> [B,C,H,W]
    encoded_states = encoded_states.permute(0, 3, 1, 2)
    encoded_states = encoded_states.to(dtype=torch.float32)

    # converting the mine_density object to a tensor.
    # this is done to place it in the same device of the encoded_states tensor
    # and/or if the object is a constant.
    # viene forzato anche lo stesso tipo di encoded_states con il dtype
    densities = torch.as_tensor(mine_density, dtype=encoded_states.dtype, device=encoded_states.device)

    # caso in cui mine_density era uno scalare e tutte le board hanno
    # stessa densità di mine
    if densities.ndim == 0:
        # densities è ora un tensore di shape [B]
        densities = densities.expand(batch_size)
    
    # anche un tensore 1d con un solo elemento ha la stessa densità di mine
    elif densities.ndim == 1 and densities.numel() == 1:
        # anche in questo caso si espande a shape [B]
        densities = densities.expand(batch_size)
    
    elif densities.ndim != 1 or densities.numel() != batch_size:
        raise ValueError(
            "mine_density must be a scalar or a tensor with shape [B]. "
            f"Received shape {tuple(densities.shape)} for batch size "
            f"{batch_size}."
        )

    if not torch.all((densities >= 0.0) & (densities <= 1.0)):
        raise ValueError(
            "mine_density values must be between 0 and 1."
        )
    
    # produzione one-hot encoding per la densità
    # abbiamo bisogno di passare da [B] a [B, 1, H, W]
    # questo perché poi quando andremo a fare la concatenazione
    # la dimensione 0 deve essere B
    # le ultime due dimensioni devono essere H,W
    # per rendere compatibile l'operazione

    density_channel = (
        densities.reshape(batch_size,1, 1, 1) # da [B] a [B, 1, 1, 1]
        .expand(batch_size, 1, height, width) # da [B, 1, 1, 1] a [B, 1, H, W]
    )

    # si trasforma l'encoded_states tensore originale che contiene
    # i 10 canali dei numerini nella board e si aggancia il canale costante
    # della mine_density
    # concatenazione([B,10,H,W], [B, 1, H, W]) -> [B, 11, H, W]
    # dim = 1 indica che la concatenazione è fatta sulla dimensione di indice 1,
    # cioè la seconda dimensione
    encoded_states = torch.cat((encoded_states, density_channel), dim=1)

    # a questo punto encoded_states conterrà all'undicesimo canale, di indice 10,
    # la mine_density.

    # Conv2d requires floating-point inputs
    # By using contiguous we ensure that after the permutation the order
    # [B,C,H,W] is respected also in memory. If not used, there might be the risk
    # that only the indexing is modified, similarly to how happens when using view().
    return encoded_states.contiguous()