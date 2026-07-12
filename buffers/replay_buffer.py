from collections import deque
from dataclasses import dataclass
import random
import numpy as np

# A Python dataclass is used to avoid writing explictly the constructor.
# The fields are automatically assigned. You just only require to define

@dataclass
class Transition:
    obs: np.ndarray # Remember: the observation space is a gym.spaces.Box, so a np.ndarray
    action: int # Una delle possibili caselle da cliccare, nello spazio delle azioni è un int
    reward: float # Le reward possono essere anche floating point
    terminated: bool
    next_obs: np.ndarray
    mine_density: float # viene salvata anche mine_density per future implementazioni
    # in cui il numero di mine della board può cambiare nel tempo

class ReplayBuffer:
    
    # Here we are using a deque data structure, which allow us to
    # define a circular stack efficently
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    # Attenzione a cosa viene fatto nella push, è di fondamentale importanza!
    # Le osservazioni devono essere passate per VALORE, non PER RIFERIMENTO.
    # Questo perché se la player_board venisse modificata in_place, ci sarebbe il rischio
    # di andare a cambiare una transizione nel replay buffer.
    # L'effetto sarebbe disastroso: una ripetizione di transizioni nel replay buffer
    # che puntano alla medesima player_board aggiornata.
    
    def push(
            self,
            obs: np.ndarray,
            action: int,
            reward: float,
            terminated: bool,
            next_obs: np.ndarray,
            mine_density: float
    ) -> None:
        self.buffer.append(
            Transition(
                obs = np.array(obs, copy=True),
                action=action,
                reward=reward,
                terminated=terminated,
                next_obs=np.array(next_obs, copy=True),
                mine_density=float(mine_density)
            )
        )

    def sample(self, batch_size: int):
         
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a buffer "
                f"containing {len(self.buffer)} transitions."
            )
        batch = random.sample(self.buffer, batch_size)

        # Inside the batch, we'll have multiple
        # actions, multiple states, etc.
        # for this reason we require to have those lists

        obs_list = []
        actions_list = []
        rewards_list = []
        terminateds_list = []
        next_obs_list = []
        mine_densities_list = []

        for t in batch:
            obs_list.append(t.obs)
            actions_list.append(t.action)
            rewards_list.append(t.reward)
            terminateds_list.append(t.terminated)
            next_obs_list.append(t.next_obs)
            mine_densities_list.append(t.mine_density)

        # obs_list is a list made of batch_size elements
        # [H,W]. By stacking batch_size observations we produce
        # [B, H, W] tensor.
        obs = np.stack(obs_list)
        next_obs = np.stack(next_obs_list)

        # These contain scalar values, so np.asarray produces shape [B].
        actions = np.asarray(actions_list, dtype=np.int64)
        rewards = np.asarray(rewards_list, dtype=np.float32)
        terminateds = np.asarray(terminateds_list, dtype=np.bool_)
        mine_densities = np.asarray(mine_densities_list, dtype=np.float32)

        return obs, actions, rewards, terminateds, next_obs, mine_densities

    def __len__(self):
        return len(self.buffer)