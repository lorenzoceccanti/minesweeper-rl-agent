import gymnasium as gym
import models.actor_net as actor_net
import models.critic_net as critic_net
import buffers.replay_buffer as replay_buffer
import models.encodings as encodings
import numpy as np
import torch

class PPOAgent:
    def __init__(
            self,
            env: gym.Env,
            device: torch.device,
            seed,
            batch_size: int = 64,
            logger = None
        ):
        
        self.seed = seed
        self.logger = logger
        self.device = device

        # == ENVIRONMENT == 
        self.env = env
        if seed is not None:
            self.env.action_space.seed(seed)
        # ==================

        # == NEURAL NETWORKS == 
        self.actor = actor_net.ActorNetwork().to(self.device)
        self.critic = critic_net.CriticNetwork().to(self.device)
    
    def get_action(self, obs: np.ndarray) -> tuple[int, float]:
        """ Samples an action from the current policy
        Args:
            observation: An observation returned by the
            Gymnasium environment.
        Returns:
            action: An action to pass to env.step()
            log_prob: The log probability of the action required
            to compute the probability ratio
        """

        # Converting from NumPy ndarray to a PyTorch tensor
        # with shape [H,W]
        obs_tensor = torch.as_tensor(obs, dtype=torch.long, device=self.device)
        
        # Producing a one-hot encoding of the observation tensor
        # [H,W] -> [1, C, H, W]
        encoded_obs = encodings.one_hot_encode_board(obs_tensor)

        # The action mask is a reshaped version of the actions
        # having obs == -2 in their cell.
        # here we work with the orginal tensor, not with the one-hot
        # encoding. In fact, we begin from a representation [H,W] (i.e. the coordinates)
        # and inside there's the number of the cell.
        # final shape: [1, H*W]
        action_mask = (
            obs_tensor
            .eq(-2) # we look inside the tensor element, looking for unrevealed cells
            .flatten() # passing from [H,W] to [H*W]
            .unsqueeze(0) # we need to add a unitary dimension in front, as the actor nn returns
            # [B, H*W]. In this case, since we are working with 1 state, we have [1, H*W]
        )

        if not action_mask.any():
            raise RuntimeError("No valid actions are available.")
        
        # We're not training anything in this function. We're just using
        # the probability distribution produced by the actor network
        # to pick the next action. We can disable the construction of the
        # computational graph
        with torch.no_grad:
            # For how we designed the actor network, the network return
            # a probability distribution (we have a softmax at the end)
            # NN input: [1, C, H, W]
            # NN output: [1, H*W]
            #TODO: continue from here
            pass

    # TODO: training actor-critic NNs
    def optimize_model(self):
        """
        One PPO update step.
        """
        pass