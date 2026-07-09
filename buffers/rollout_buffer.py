from dataclasses import dataclass
import random
import numpy as np
@dataclass
class RolloutTransition:
    obs: np.ndarray
    action: int
    reward: float
    terminated: bool
    truncated: bool
    next_obs: np.ndarray
    # We require to also to store the old
    # logits. Otherwise we don't have it
    # when the actor weights are updating
    # and we would not be able to compute the
    # probability ratio
    old_log_prob: float

class RolloutBuffer:
    
    def __init__(self):
        # al contrario del replay buffer, è una semplice lista.
        # non c'è necessità di mantenere dati vecchi relativi
        # ad esperienze storiche. ci vengono parcheggiate temporanamente
        # transizioni della stessa policy pi(old)
        self.buffer = []
    
    def push(
            self,
            obs: np.ndarray,
            action: int,
            reward: float,
            terminated: bool,
            truncated: bool,
            next_obs: np.ndarray,
            old_log_prob: float
    ) -> None:
        # Nota: passaggio per copia degli stati, stesso motivo per cui
        # si era fatto anche nel replay buffer.
        self.buffer.append(
            RolloutTransition(
                obs = np.array(obs, copy=True),
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                next_obs=np.array(next_obs, copy=True),
                old_log_prob=old_log_prob
            )
        )
    
    def sample_minibatch(
            self,
            batch_size: int
    ):
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a rollout buffer "
                f"containing {len(self.buffer)} transitions."
            )
        
        minibatch = random.sample(
            self.buffer,
            batch_size,
        )

        obs_list = []
        actions_list = []
        rewards_list = []
        terminateds_list = []
        truncateds_list = []
        next_obs_list = []
        old_log_probs_list = []

        for transition in minibatch:
            obs_list.append(transition.obs)
            actions_list.append(transition.action)
            rewards_list.append(transition.reward)
            terminateds_list.append(transition.terminated)
            truncateds_list.append(transition.truncated)
            next_obs_list.append(transition.next_obs)
            old_log_probs_list.append(transition.old_log_prob)
        
        # obs_list is a list made of batch_size elements
        # [H,W]. By stacking batch_size observations we produce
        # [B, H, W] tensor.
        obs = np.stack(obs_list)
        next_obs = np.stack(next_obs_list)

        actions = np.asarray(actions_list, dtype=np.int64)
        rewards = np.asarray(rewards_list, dtype=np.float32)
        terminateds = np.asarray(terminateds_list, dtype=np.bool_)
        truncateds = np.asarray(truncateds_list, dtype=np.bool_)
        old_log_probs = np.asarray(
            old_log_probs_list,
            dtype=np.float32
        )
        return obs, actions, rewards, terminateds, truncateds, next_obs, old_log_probs
    
    def clear(self) -> None:
        self.buffer.clear()
    
    def __len__(self) -> int:
        return len(self.buffer)