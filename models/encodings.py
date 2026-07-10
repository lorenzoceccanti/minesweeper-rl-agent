import torch
import torch.nn.functional as F

def one_hot_encode_board(
        states: torch.Tensor,
        num_classes: int = 10
) -> torch.Tensor:
    """ Converts a minesweeper board to one-hot embedding representation
    of the board
    Possible input shapes:
    - Input:
        [H,W] or [B, H, W]
    - Output:
        [B, num_classes, H, W]
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

    # Conv2d requires floating-point inputs
    # By using contiguous we ensure that after the permutation the order
    # [B,C,H,W] is respected also in memory. If not used, there might be the risk
    # that only the indexing is modified, similarly to how happens when using view().
    return encoded_states.to(dtype=torch.float32).contiguous()