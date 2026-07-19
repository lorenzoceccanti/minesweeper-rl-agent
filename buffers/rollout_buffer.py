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
    mine_density: float
    # We require to store the old logits. Otherwise we don't have it
    # when the actor weights are updating and we would not be able 
    # to compute the probability ratio
    old_log_prob: float
    # We have to store also the advantage in the rollout transition, 
    # otherwise we would loose the association between a transaction 
    # and the advantage. The advantage is computed after collecting
    # the complete rollout. At the beginning the advantage has None 
    # as initial value, because is known only when T transactions 
    # are stored
    advantage: float | None = None
    # Also the TD value targets (the ones which are treated
    # as constants during the backpropagation) have to be saved
    # into the rollout. This because, PPO executes K epochs
    # on the same rollout. Without fixing the TD value target
    # into the buffer, during the update (even for the same rollout)
    # the value targets would continuosly change during the K epochs
    # the only values that have to change are the prediction, not
    # the part treated as "ground-thruth"!
    value_target: float | None = None

class RolloutBuffer:
    
    def __init__(self):
        # al contrario del replay buffer, è una semplice lista.
        # non c'è necessità di mantenere dati vecchi relativi
        # ad esperienze storiche. ci vengono parcheggiate temporanamente
        # transizioni della stessa policy
        self.buffer = []
    
    # metodo privato di utilità della classe RolloutBuffer
    # è utilizzato sia dalla sample_minibatch che dalla get_all
    def _transitions_to_arrays(
            self,
            transitions: list[RolloutTransition],
            include_training_values: bool = False
    ):
        """ Converts a list of RolloutTransitions into NumPy arrays."""
        if len(transitions) == 0:
            raise ValueError("Cannot convert an empty transition list.")
        
        obs_list = []
        actions_list = []
        rewards_list = []
        terminateds_list = []
        truncateds_list = []
        next_obs_list = []
        old_log_probs_list = []
        mine_densities_list = []
        advantage_list = []
        value_targets_list = []

        for transition in transitions:
            obs_list.append(transition.obs)
            actions_list.append(transition.action)
            rewards_list.append(transition.reward)
            terminateds_list.append(transition.terminated)
            truncateds_list.append(transition.truncated)
            next_obs_list.append(transition.next_obs)
            old_log_probs_list.append(transition.old_log_prob)
            mine_densities_list.append(transition.mine_density)

            if include_training_values:
                if (transition.advantage is None or transition.value_target is None):
                    raise RuntimeError(
                        "Advantages and value targets must be computed before sampling "
                        "PPO minibatches."
                    )
                advantage_list.append(transition.advantage)
                value_targets_list.append(transition.value_target)
        
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
        mine_densities = np.asarray(mine_densities_list, dtype=np.float32)

        if include_training_values:
            advantages = np.asarray(advantage_list, dtype=np.float32)
            value_targets = np.asarray(value_targets_list, dtype=np.float32)
            return obs, actions, rewards, terminateds, truncateds, next_obs, old_log_probs, mine_densities, advantages, value_targets
        else:
            return obs, actions, rewards, terminateds, truncateds, next_obs, old_log_probs, mine_densities

    def push(
            self,
            obs: np.ndarray,
            action: int,
            reward: float,
            terminated: bool,
            truncated: bool,
            next_obs: np.ndarray,
            old_log_prob: float,
            mine_density: float
    ) -> None:
        # Nota: passaggio per copia degli stati, stesso motivo per cui
        # si era fatto anche nel replay buffer.
        self.buffer.append(
            RolloutTransition(
                obs = np.array(obs, copy=True),
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                next_obs=np.array(next_obs, copy=True),
                old_log_prob=old_log_prob,
                mine_density=float(mine_density)
            )
        )
    
    def sample_minibatches(
            self,
            batch_size: int
    ):
        """Randomly shuffles the entire rollout and divides it into minibatches."""
        if batch_size <= 0:
            raise ValueError(
                "Batch size must be greater than zero."
            )

        if len(self.buffer) == 0:
            raise ValueError(
                "Cannot sample minibatches from an empty rollout buffer."
            )
        
        # qua ci sono tutte le transizioni del rollout buffer
        # mescolate
        shuffled_transitions = random.sample(
            self.buffer,
            k=len(self.buffer),
        )

        # qui ci verranno messe un numero di transazioni pari a batch_size
        # per ogni elemento di questa lista.
        # quindi, primo batch avrà [T3, T1, T4, T9] se per esempio batch_size = 4
        minibatches = []
        for start in range(
            0,
            len(shuffled_transitions),
            batch_size
        ):
            # da start a batch_size escluso, vengono fatti dei balzi di step batch_size
            # fino a che non arrivo alla dimensione del rollout buffer (in termini di numero totale
            # di transizioni)
            minibatches_transitions = shuffled_transitions[start:start+batch_size]
            minibatch_arrays = self._transitions_to_arrays(transitions=minibatches_transitions, include_training_values=True)

            minibatches.append(minibatch_arrays)
        
        return minibatches
    
    def clear(self) -> None:
        # svuota il rollout buffer, rimuovendo tutte le transizioni accumulate
        self.buffer.clear()
    
    def set_advantages(
        self,
        advantages: np.ndarray,
    ) -> None:
        # le advantage vengono calcolate dopo che il rollout buffer è stato riempito, quindi
        # le transizioni vengono aggiornate con le advantage calcolate
        advantages = np.asarray(
            advantages,
            dtype=np.float32,
        )

        if advantages.ndim != 1:
            raise ValueError(
                f"Expected a one-dimensional advantage array, "
                f"received shape {advantages.shape}."
            )

        if len(advantages) != len(self.buffer):
            raise ValueError(
                f"Received {len(advantages)} advantages for "
                f"{len(self.buffer)} transitions."
            )

        for transition, advantage in zip(
            self.buffer,
            advantages,
        ):
            transition.advantage = float(advantage)

    def set_value_targets(
        self,
        value_targets: np.ndarray,
    ) -> None:
        # i value targets vengono calcolati dopo che il rollout buffer è stato riempito, quindi
        # le transizioni vengono aggiornate con i value targets calcolati
        value_targets = np.asarray(
            value_targets,
            dtype=np.float32,
        )
    
        if value_targets.ndim != 1:
            raise ValueError(
                f"Expected a one-dimensional value target array, "
                f"received shape {value_targets.shape}."
            )
    
        if len(value_targets) != len(self.buffer):
            raise ValueError(
                f"Received {len(value_targets)} value targets for "
                f"{len(self.buffer)} transitions."
            )
    
        for transition, value_target in zip(
            self.buffer,
            value_targets,
        ):
            transition.value_target = float(value_target)

    def get_all(self):
        """ Returns all the transitions in the same order
        with which they are collected."""
        return self._transitions_to_arrays(transitions=self.buffer, include_training_values=False)
    
    def __len__(self) -> int:
        return len(self.buffer)