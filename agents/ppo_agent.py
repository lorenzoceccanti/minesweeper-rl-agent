import gymnasium as gym
import models.actor_net as actor_net
import models.critic_net as critic_net
import buffers.rollout_buffer as rollout_buffer
import models.encodings as encodings
from pathlib import Path
import numpy as np
import torch

class PPOAgent:
    def __init__(
            self,
            env: gym.Env,
            device: torch.device,
            seed: int | None,
            env_seed_start: int | None,
            rollout_steps: int = 2048,
            discount_factor: float = 0.95,
            batch_size: int = 64,
            logger = None
        ):
        
        self.seed = seed
        self.env_seed_start = env_seed_start
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

        # == ROLLOUT BUFFER ==
        self.rollout_buffer = rollout_buffer.RolloutBuffer()
        self.total_rollout_steps = rollout_steps

        # == OTHER PARAMETERS ==
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        # ======================
    
    def reset_episode(self, episode_index: int):
        """ Resets the episode by propering handling the seed
        (due to the misaligment between rollout buffer refreshes
        and episode ends)"""
        if self.env_seed_start is None:
            # il chiamante non ha specificato un seed
            episode_seed = None
        else:
            episode_seed = self.env_seed_start + episode_index

        return self.env.reset(seed=episode_seed)

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
        with torch.no_grad():
            # For how we designed the actor network, the network return
            # a probability distribution (we have a softmax at the end)
            # NN input: [1, C, H, W]
            # NN output: [1, H*W]
            probabilities = self.actor(encoded_obs)
            # applicazione della maschera action_mask
            # se l'i-esima azione è permessa, i-esimo elemento di action_mask è diverso
            # da 0 e la probabilità è mantenuta. altrimenti, si azzera.
            masked_probabilities = probabilities * action_mask
            
        # è necessario rinormalizzare: questo perché va garantito che sum_i (p_i) = 1,
        # per definizione di ddp. la normalizzazione viene fatta solo sulle azioni
        # valide, quelle con pi > 0

        # significato dei parametri di sum:
        # a) occorre dim = -1 poiché dobbiamo ricordarci che l'actor
        # restituisce un tensore di shape [B, H*W].
        # Noi la sommatoria delle probs vogliamo farla per a = 1, ... numero_azioni.
        # dim = -1 serve propria a dire questo: non si scorre sull'indice b che
        # identifica il batch b-esimo.
        # b) occorre keepdim = True per mantenere la dimensione delle azioni,
        # altrimenti il tensore avrebbe modificato la propria shape a [B], anzichè
        # [B, H*W]. Questo perché per ogni stato b dividiamo per un unico numero reale
        # a comune per tutte le azioni 
        # che è appunto la somma lungo tutte le a = 1,..., H*W. La somma è un unico
        # valore reale, e quindi la 2a dimensione [H*W] è persa con l'aggregazione sum().
        # Mantenere la dimensione significa poi dividere per questo reale per tutte
        # le a = 1, ..., H*W come si fa nel denominatore di softmax ad esempio.
            
        masked_probabilities = (
            masked_probabilities / masked_probabilities.sum(dim=-1, keepdim=True)
        )

        # We need a torch.distributions.Categorical object in order
        # to call the sample method, which is the one responsible for
        # sampling the action later
        distribution = torch.distributions.Categorical(
            probs = masked_probabilities
        )
            
        # Here we effectively sample an action from the policy, with action masking
        action = distribution.sample()
        # The logits are required for computing the probability ratio later
        log_prob = distribution.log_prob(action)
        # si passa da tensore a numero unico, dato che si estrae un'azione singola.
        return action.item(), log_prob.item()

    def compute_advantage(self):
        """ Computing an advantage estimate, according to
        the one step TD approximation."""

        # Prendiamo tutte le transactions nel rollout_buffer

        obs, _, rewards, terminateds, _, next_obs, _ = self.rollout_buffer.get_all()
        # Convertiamo in tensori pytorch: alcuni vanno passati alla rete neurale
        # critic che lavora su one-hot encoding.
        
        obs_tensor = torch.as_tensor(obs, dtype=torch.long, device=self.device)
        next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.long, device=self.device)
        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        terminateds_tensor = torch.as_tensor(terminateds, dtype=torch.float32, device=self.device)
        
        # Producing a one-hot encoding of the observation tensor
        # [H,W] -> [1, C, H, W]
        encoded_obs = encodings.one_hot_encode_board(obs_tensor)
        encoded_next_obs = encodings.one_hot_encode_board(next_obs_tensor)

        # using the critic in inference mode: no computation graph construction
        # required
        with torch.no_grad():
            
            value = self.critic(encoded_obs)
            next_value = self.critic(encoded_next_obs)

            # qua si definisce il pezzo del value target
            # nella formula intera dell'advantage.
            # viene gestito anche il raggiungimento del terminal state
            # in unica istruzione, come era stato fatto per DQN

            value_target = (
                rewards_tensor + self.discount_factor * (1.0 - terminateds_tensor) *
                next_value
            )

            advantage = value_target - value

        # dato che il rollout buffer ha tipi numpy e non pytorch,
        # bisogna ripassare dalla CPU, prima di riconvertire in oggetti numpy
        self.rollout_buffer.set_advantages(advantage.cpu().numpy())
        self.rollout_buffer.set_value_targets(value_target.cpu().numpy())
    
    def update(self):
        # fa il pezzo sotto del ppo, che campiona dal rollout buffer, etc.
        # questo metodo è chiamato dalla train

        # per K epoche:
        #     per ogni mini-batch casuale:
        #         estrai transizioni, advantage e value target
        # 
        #         ricalcola la policy mascherata
        #         calcola le nuove log-probabilità
        #         calcola il probability ratio
        #         calcola la clipped actor loss
        #         aggiorna l’actor
        # 
        #         calcola V(s)
        #         confrontalo con value_target
        #         aggiorna il critic
        pass

    def train(self, n_episodes: int) -> None:
        
        # In PPO the usage of a buffer is completely different
        # with respect to the replay buffer seen in DQN.
        # In PPO a ROLLOUT BUFFER is employed. Inside a rollout buffer
        # transactions are stored for T consecutive transactions,
        # independently from the fact that are transactions of different
        # episodes. When T transactions are stored, we interrupt
        # temporarly the interaction with the environment.
        # During the collection of the T transactions, all the transactions
        # are generated by the same policy pi(old).
        # When we encounter the end of the episode, we'll continue to insert
        # transactions into the rollout buffer unless we arrive at the maximum
        # number of transactions T or the total number of episodes specified.
        
        # Due to the di-alligment between rollout and episodes, we require
        # a particular attention to how the seed is generated.
        # The fact is that an episode could end during a rollout store, and
        # in that case the seed has to be handled; but, when a rollout ends
        # (T steps elapsed) in the middle of an episode the seed must not be
        # updated

        # This counter is useful both for keep track of the episodes
        # elapsed and also for the seed advancement
        completed_episodes = 0
        # At the beginning we reset the episode
        obs, info = self.reset_episode(episode_index = completed_episodes)

        while completed_episodes < n_episodes:
    
            rollout_steps = 0
            # in questo while interno ci sono due condizioni di uscita
            # una sul numero di rollout steps, una anche sul numero di
            # episodes totali perché potrebbe anche capitare
            # che si arrivi alla fine del numero totale di episodes prima
            # di raggiungere il numero totale di rollout steps

            while (rollout_steps < self.total_rollout_steps and
            completed_episodes < n_episodes):
                
                action, old_log_prob = self.get_action(obs)
                
                # Given an action randomly picked from the pi_old distribution
                # we observe the environment in order to collect a trajectory
                # to put into the ROLLOUT BUFFER.
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                self.rollout_buffer.push(obs, action, reward, terminated, truncated, next_obs, old_log_prob)

                obs = next_obs
                rollout_steps += 1

                # Checking if we arrive in a terminal state
                # in which an episode ends.
                if terminated or truncated:
                    completed_episodes += 1

                    # We require to reset and make advance the seed
                    # only at the end of the episode, but we continue to put
                    # transactions into the buffer (in fact we stay in the loop)

                    if completed_episodes < n_episodes:
                        obs, info = self.reset_episode(episode_index=completed_episodes)

            # COMPUTING THE ADVANTAGE, AFTER T rollout steps
            self.compute_advantage()
            
            # K EPOCHS OF PPO UPDATE
            self.update()
            
            # INVALIDATING THE ROLLOUT BUFFER AFTER PPO UPDATE
            self.rollout_buffer.clear()