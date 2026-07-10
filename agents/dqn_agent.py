import gymnasium as gym
import models.fully_conv_qnet as fully_conv_qnet
import models.encodings as encodings
import buffers.replay_buffer as replay_buffer
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

class DQNAgent:
    def __init__(
        self,
        env: gym.Env,
        device: torch.device,
        seed,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        replay_buffer_capacity: int = 50_000,
        batch_size: int = 64,
        target_update_frequency: int = 500,
        learning_starts: int = 1_000,
        train_frequency: int = 1,
        logger = None
    ):
        self.seed = seed
        self.logger = logger
        self.rng = np.random.default_rng(seed)
        self.device = device
        # == Environment ==
        self.env = env
        if seed is not None:
            self.env.action_space.seed(seed)
        # == Neural Networks ==

        # number of environment steps collected before training starts
        self.learning_starts = learning_starts
        self.train_frequency = train_frequency
        # defines the frequency with which target_network weights are refreshed
        self.target_update_frequency = target_update_frequency

        # online_network is trained, target_network is updated periodically
        self.online_network = fully_conv_qnet.FullyConvQNetwork().to(self.device)
        self.target_network = fully_conv_qnet.FullyConvQNetwork().to(self.device)

        # At the initialization we have that
        # Q(s,a,theta_minus) = Q(s,a,theta)
        # La riga sotto sostanzialmente copia i parametri della rete
        # online nella rete offline
        self.target_network.load_state_dict(self.online_network.state_dict())
        # Moreover, we put the target_network in inference mode to
        # ensure that mechanisms like Dropout do not intervene.
        # The target has to be as stable as possible.
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(),
            lr=learning_rate
        )

        self.loss_fn = nn.MSELoss()

        # == Replay Buffer ==

        self.replay_buffer = replay_buffer.ReplayBuffer(replay_buffer_capacity)
        self.batch_size = batch_size

        # == Parameters
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # == Evaluation
        self.training_error = []
        self.loss_history = []
        self.episode_rewards = [] # i-th element contains the return of the i-th episode
        self.episode_lengths = [] # i-th element contains the number of actions of the i-th episode
        self.episode_wins = [] # i-th element contains 1 if the i-th episode concluded with a win
        self.epsilon_history = [] # i-th element contains the eps used in the i-th episode
        self.last_checkpoint_path = None

        # This is the counter of timesteps elapsed
        self.global_step = 0

    def save_checkpoint(
            self,
            checkpoint_dir: str | Path = "checkpoints/dqn",
    ) -> Path:
        
        checkpoint_dir = Path(checkpoint_dir)
        # Se il percorso è relativo, viene risolto rispetto
        # alla root del progetto.
        if not checkpoint_dir.is_absolute():
            project_root = Path(__file__).resolve().parents[1]
            checkpoint_dir = project_root / checkpoint_dir

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = datetime.now().strftime(
            "%Y-%m-%d-%H-%M-%S"
        )

        checkpoint_path = (
            checkpoint_dir / f"{timestamp}.pt"
        )

        checkpoint = {
            "algorithm": "dqn",
            # Model
            "online_network_state_dict":
                self.online_network.state_dict(),

            "target_network_state_dict":
                self.target_network.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            # Training state
            "epsilon": self.epsilon,
            "global_step": self.global_step,
            "seed": self.seed,

            # Per-episode metrics
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_wins": self.episode_wins,
            "epsilon_history": self.epsilon_history,

            # Gradient update metrics
            "loss_history": self.loss_history,
            "training_error": self.training_error,
        }

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        return checkpoint_path

    def get_action(self, obs: np.ndarray) -> int:
        """
        Selects an action using an epsilon-greedy policy.
        This implementation performs action masking, in order
        to consider only currently unrevealed cells.
        Args:
            obs: Current player board with shape [H, W].
        Returns:
            Flattened index of the selected cell, in the range
            [0, H * W - 1].
        """

        # A cell is selectable only if it's still unrevealed.
        # Recall:
        # obs tensor has shape [H, W]
        # while, actions are represented as integers between 0 and (H*W)1
        # For this reason, when creating the boolean mask for action
        # masking, we have to reshape the observation tensor to have shape [H*W]
        # shape of valid_action_mask: [H*W]
        valid_action_mask = obs.reshape(-1) == -2
        # qui ci sono solo gli indici delle celle non ancora selezionate,
        # come numpy array
        valid_actions = np.flatnonzero(valid_action_mask)

        if valid_actions.size == 0:
            raise RuntimeError(
                "No valid actions are available in a non-terminal state."
            )

        # Exploration: choosing a random cell (with probability epsilon)
        if self.rng.random() < self.epsilon:
            return int(self.env.action_space.sample())

        # Converting the NumPy board into a PyTorch tensor
        # with shape [H,W]:
        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.long,
            device=self.device,
        )

        # One hot encoding of the observation space
        # [H, W] -> [1, C, H, W]
        encoded_obs = encodings.one_hot_encode_board(obs_tensor)

        # valid_action_mask: [H*W]
        # valid_action_mask_tensor: [1, H*W]
        # the unsqueeze is done in order to make coincide
        # the shape of the action mask with the shape of Q values
        valid_action_mask_tensor = torch.as_tensor(
            valid_action_mask,
            dtype=torch.bool,
            device=self.device,
        ).unsqueeze(0)


        # Importante: durante l'exploitation (action selection con p = 1-eps)
        # i q_values sono presi dalla rete con parametri theta, cioè dalla
        # online network. Però tale online network deve essere utilizzata
        # in fase di inferenza. Per questo si introduce torch.no_grad: si evita
        # in questo modo di costruire il grafo computazionale e si risparmia memoria.
        with torch.no_grad():
            # [1, 10, H, W] -> [1, H * W]
            q_values = self.online_network(encoded_obs)

            # THE TRICK: IN ACTION MASKING INVALID ACTIONS RECEIVE Q = -INFINITY.
            # IN THIS WAY, THEY CANNOT BE SELECTED BY THE ARGMAX
            masked_q_values = q_values.masked_fill(
                ~valid_action_mask_tensor,
                -torch.inf,
            )
            # [1, H * W] -> scalar action index
            action = torch.argmax(masked_q_values, dim=1).item()
        return int(action)

    def decay_epsilon(self):
        self.epsilon = max(
            self.final_epsilon,
            self.epsilon - self.epsilon_decay
        )

    def update_target_network(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def optimize_model(self):
        """
        One DQN update step.
        """

        # == Sample random mini-batch of transitions from the replay buffer ==
        
        # Case in which there's nothing to extract
        if len(self.replay_buffer) < self.batch_size:
            return None

        obs, actions, rewards, terminateds, next_obs = self.replay_buffer.sample(
            self.batch_size
        )

        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.long,
            device=self.device,
        )

        actions = torch.as_tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        )

        rewards = torch.as_tensor(
            rewards,
            dtype=torch.float32,
            device=self.device,
        )

        terminateds = torch.as_tensor(
            terminateds,
            dtype=torch.float32,
            device=self.device,
        )

        next_obs_tensor = torch.as_tensor(
            next_obs,
            dtype=torch.long,
            device=self.device,
        )

        # [batch_size, H, W] -> [batch_size, H * W]
        next_valid_action_mask = (
            next_obs_tensor
            .eq(-2)
            .flatten(start_dim=1)
        )

        obs = encodings.one_hot_encode_board(obs_tensor)
        next_obs = encodings.one_hot_encode_board(next_obs_tensor)

        # Prendo i Q-values dati dalla online_network, relativamente
        # ad ognuno degli stati presenti nel minibatch (sono dentro al tensore obs)
        q_values = self.online_network(obs)

        # Prendo solo i Q-values relativi alle azioni compiute nel mini-batch
        # La unsqueeze iniziale è necessaria dato che altrimenti la index
        # non funzionerebbe tra tensori di shape differenti.
        # Poi si fa squeeze per eliminare la seconda dimensione in eccesso
        # che non serve più a niente, dal momento che per ogni pattern del
        # minibatch sopravvive solo un unico scalare che corrisponde
        # proprio a quello che nelle slide del corso viene indicato come
        # Q(s_j, a_j, theta)

        q_values_for_taken_actions = q_values.gather(
            dim=1,
            index=actions.unsqueeze(1)
        ).squeeze(1)

        # In this part of the code we compute the Target, by using the target net
        with torch.no_grad():
            next_q_values = self.target_network(next_obs)
            # TRICK: we put for actions already selected a -inf value
            masked_next_q_values = next_q_values.masked_fill(
                ~next_valid_action_mask,
                -torch.inf,
            )
            has_valid_actions = next_valid_action_mask.any(dim=1)
            # max_a'Q(s_j,a',theta-), only on unmasked actions
            max_next_q_values = masked_next_q_values.max(dim=1).values

            # Avoid -inf values when no valid actions exist.
            # This is mainly relevant for terminal states.
            max_next_q_values = torch.where(
                has_valid_actions,
                max_next_q_values,
                torch.zeros_like(max_next_q_values),
            )

            # in one shot, we handle both the case for terminal
            # and not terminal states
            # if we are in a terminal state, terminateds = 1.0
            # in this way target = rewards
            targets = rewards + self.discount_factor * max_next_q_values * (1.0 - terminateds)

        loss = self.loss_fn(q_values_for_taken_actions, targets)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping: a technique to prevent exploding gradients
        # in backpropagation
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=10.0)

        self.optimizer.step()

        with torch.no_grad():
            td_error = targets - q_values_for_taken_actions
            mean_abs_td_error = torch.mean(torch.abs(td_error)).item()
       
        self.training_error.append(mean_abs_td_error)
        self.loss_history.append(loss.item())
        return loss.item()

    def train(self, n_episodes: int, save_checkpoint: bool = False,
              checkpoint_dir: str | Path = "checkpoints/dqn",
              env_seed_start: int | None = None):
        # Log output is an array of strings to be returned to stdout
        # if the log option is set to true
        for episode in tqdm(range(n_episodes)):
            # each episode will have its own seed, different
            # from the agent seed.
            if env_seed_start is None:
                episode_seed = None
            else:
                episode_seed = env_seed_start + episode
            obs, info = self.env.reset(seed=episode_seed)
            terminated = False
            truncated = False

            episode_reward = 0.0
            episode_steps = 0

            while not (terminated or truncated):
                # Choosing an action using eps-greedy
                action = self.get_action(obs)
                # Take action, observe reward and next state from the environment
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                # === LOGGING ===================================
                # Updating statistics of the episode
                episode_steps += 1
                episode_reward += float(reward)

                # Reconstruction of the coordinate of the clicked cell
                # starting from the obs object
                # action = row * board_width + column
                # same formula as the one used in the environment
                board_width = obs.shape[1]
                row = int(action) // board_width
                column = int(action) % board_width

                # Log of a single timestep of an episode
                if self.logger is not None:
                    self.logger.debug(
                        f"step={episode_steps:3d} | "
                        f"global_step={self.global_step + 1:6d} | "
                        f"action={int(action):3d} | "
                        f"cell=({row}, {column}) | "
                        f"reward={float(reward):6.1f} | "
                        f"status={info.get('status')}"
                    )
                
                # ===========================================

                # Storing a transition into the replay buffer
                self.replay_buffer.push(
                    obs=obs,
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    next_obs=next_obs
                )

                # Increment by one the counter of timesteps by one
                self.global_step += 1

                # This piece is used in place of updating the Q value function
                # We are adjusting the weights of the onlineNetwork
                # and periodically, after target_update_frequency timesteps
                # we proceed to copy the parameters of the onlineNetwork in the
                # target offline network

                if(
                    self.global_step >= self.learning_starts
                    and self.global_step % self.train_frequency == 0
                ):
                    self.optimize_model()
                
                if self.global_step % self.target_update_frequency == 0:
                    self.update_target_network()

                # Moving to the next state
                obs = next_obs

            # ===== LOGGING ====== 
            # Log at the end of the episode
            if terminated:
                end_reason = info.get("status", "terminated")
            else:
                end_reason = "truncated"

            self.episode_rewards.append(float(episode_reward))
            self.episode_lengths.append(int(episode_steps))
            self.episode_wins.append(int(end_reason == "won"))
            self.epsilon_history.append(float(self.epsilon))
            if self.logger is not None:
                self.logger.info(
                    f"Episode {episode + 1}/{n_episodes} completed: "
                    f"reason={end_reason}, "
                    f"steps={episode_steps}, "
                    f"total_reward={episode_reward:.1f}, "
                    f"epsilon={self.epsilon:.4f}"
                )

            # Reduce the exploration rate (the self becomes less random over time)
            self.decay_epsilon()
        if save_checkpoint:
            self.last_checkpoint_path = self.save_checkpoint(
                checkpoint_dir=checkpoint_dir,
            )

            print(
                f"Checkpoint saved to: "
                f"{self.last_checkpoint_path}"
            )