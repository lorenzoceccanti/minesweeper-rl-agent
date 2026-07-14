import copy
import gymnasium as gym
import models.fully_conv_qnet as fully_conv_qnet
import models.factory as model_factory
import models.encodings as encodings
import buffers.replay_buffer as replay_buffer
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from typing import Callable

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
        logger = None,
        validation_env: gym.Env | None = None,
        validation_episodes: int = 0,
        validation_seed_start: int = 500_000,
        validation_frequency: int = 100,
        architecture_name: str = "fully_conv_3layer_64ch_11in",
        checkpoint_dir: str | Path = "checkpoints/dqn",
        hidden_channels: int = 64, # F
        global_features_dim: int = 16, # G
        on_validation: Callable[[dict], None] | None = None,
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
        self.online_network = model_factory.get_q_network(architecture_name, hidden_channels=hidden_channels, global_features_dim=global_features_dim).to(self.device)
        self.target_network = model_factory.get_q_network(architecture_name, hidden_channels=hidden_channels, global_features_dim=global_features_dim).to(self.device)

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
        self.replay_buffer_capacity = replay_buffer_capacity
        self.batch_size = batch_size

        # == Parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.hidden_channels = hidden_channels # F
        self.global_features_dim = global_features_dim # G

        # == Evaluation
        self.training_error = []
        self.loss_history = []
        self.episode_returns = [] # i-th element contains the return of the i-th episode
        self.episode_lengths = [] # i-th element contains the number of actions of the i-th episode
        self.episode_wins = [] # i-th element contains 1 if the i-th episode concluded with a win
        self.epsilon_history = [] # i-th element contains the eps used in the i-th episode
        self.last_checkpoint_path = None

        # == Validation
        self.validation_env = validation_env
        self.validation_episodes = validation_episodes
        self.validation_seed_start = validation_seed_start
        self.validation_frequency = validation_frequency
        self.on_validation = on_validation

        self.checkpoint_dir = checkpoint_dir
        self.architecture_name = architecture_name
        self.validation_history = []
        self.best_validation_win_rate = -1.0
        self.checkpoint_path = None
        self.best_state_dict = None

        # This is the counter of timesteps elapsed
        self.global_step = 0

    def save_checkpoint(
            self,
            checkpoint_dir: str | Path = "checkpoints/dqn",
            filename: str | None = None,
            state_dict_overrides: dict | None = None,
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
        
        if filename is None:
            timestamp = datetime.now().strftime(
                "%Y-%m-%d-%H-%M-%S"
            )
            filename = f"{timestamp}.pt"

        checkpoint_path = (
            checkpoint_dir / filename
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
            "episode_returns": self.episode_returns,
            "episode_lengths": self.episode_lengths,
            "episode_wins": self.episode_wins,
            "epsilon_history": self.epsilon_history,

            # Gradient update metrics
            "loss_history": self.loss_history,
            "training_error": self.training_error,

            # Validation
            "validation_history": self.validation_history,
            "best_validation_win_rate": self.best_validation_win_rate,

            # Board configuration
            "board_config": {
                "board_height": self.env.unwrapped.board_height,
                "board_width": self.env.unwrapped.board_width,
                "n_mines": self.env.unwrapped.n_mines,
            },

            # Hyperparameters
            "hyperparameters": {
                "architecture_name": self.architecture_name,
                "hidden_channels": self.hidden_channels,
                "global_features_dim": self.global_features_dim,
                "learning_rate": self.learning_rate,
                "discount_factor": self.discount_factor,
                "initial_epsilon": self.initial_epsilon,
                "epsilon_decay": self.epsilon_decay,
                "final_epsilon": self.final_epsilon,
                "replay_buffer_capacity": self.replay_buffer_capacity,
                "batch_size": self.batch_size,
                "target_update_frequency": self.target_update_frequency,
                "learning_starts": self.learning_starts,
                "train_frequency": self.train_frequency,
            }
        }

        if state_dict_overrides:
            checkpoint.update(state_dict_overrides)

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        self.last_checkpoint_path = checkpoint_path
        return checkpoint_path

    def get_mine_density(self, env: gym.Env) -> float:
        """ Helper function used to compute the mine density
        from the environment"""
        # We'll call it after each env.reset() for support
        # at future implementations that may involve
        # a dynamically increasing number of mines in the environment

        # the unwrapped reference of the environment ensures
        # to expose correctly the constructor field of our MinesweeperEnv
        # this might be useful if the caller has used some wrappers
        # to block the timesteps at a maximum number of iterations

        base_env = env.unwrapped
        return float (
            base_env.n_mines / (
                base_env.board_height * base_env.board_width
            )
        )

    def get_greedy_action(self, obs: np.ndarray, mine_density: float) -> int:
        """ Selects the valid action with the highest Q-value.
        This implementation exploits action masking and the idea of mine density"""

        obs_tensor = torch.as_tensor(obs, dtype=torch.long, device=self.device)
        action_mask = (
            obs_tensor
            .eq(-2) # we look inside the tensor element, looking for unrevealed cells
            .flatten() # passing from [H,W] to [H*W]
            .unsqueeze(0) # we need to add a unitary dimension in front, for how the NN
            # works
        )

        if not action_mask.any():
            raise RuntimeError("No valid actions are available.")
        
        # One hot encoding of the observation space
        # [H, W] -> [1, C, H, W]
        encoded_obs = encodings.one_hot_encode_board(obs_tensor, mine_density)

        # Importante: durante l'exploitation (action selection con p = 1-eps)
        # i q_values sono presi dalla rete con parametri theta, cioè dalla
        # online network. Però tale online network deve essere utilizzata
        # in fase di inferenza. Per questo si introduce torch.no_grad: si evita
        # in questo modo di costruire il grafo computazionale e si risparmia memoria.
        with torch.no_grad():
            # [1, 11, H, W] -> [1, H * W]
            q_values = self.online_network(encoded_obs)

            # THE TRICK: IN ACTION MASKING INVALID ACTIONS RECEIVE Q = -INFINITY.
            # IN THIS WAY, THEY CANNOT BE SELECTED BY THE ARGMAX
            masked_q_values = q_values.masked_fill(
                ~action_mask,
                -torch.inf,
            )
            # [1, H * W] -> scalar action index
            action = torch.argmax(masked_q_values, dim=1).item()
        
        return int(action)


    def get_action(self, obs: np.ndarray, mine_density: float) -> int:
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
            return int(self.rng.choice(valid_actions))

        # With probability 1-eps, we perform greedy action selection (i.e. argmax of
        # q_values). We call the function get_greedy_action responsible to
        # do that part.

        return self.get_greedy_action(obs, mine_density)

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

        obs, actions, rewards, terminateds, next_obs, mine_densities = self.replay_buffer.sample(
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

        mine_densities_tensor = torch.as_tensor(
            mine_densities,
            dtype=torch.float32,
            device=self.device
        )

        # [batch_size, H, W] -> [batch_size, H * W]
        next_valid_action_mask = (
            next_obs_tensor
            .eq(-2)
            .flatten(start_dim=1)
        )

        obs = encodings.one_hot_encode_board(obs_tensor, mine_densities_tensor)
        next_obs = encodings.one_hot_encode_board(next_obs_tensor, mine_densities_tensor)

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
            mine_density = self.get_mine_density(self.env)
            terminated = False
            truncated = False

            episode_reward = 0.0
            episode_steps = 0

            while not (terminated or truncated):
                # Choosing an action using eps-greedy
                action = self.get_action(obs, mine_density)
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
                    next_obs=next_obs,
                    mine_density=mine_density
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

            self.episode_returns.append(float(episode_reward))
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

            # completed_episodes contiene il numero di episodes COMPLETATI
            # la validazione la vuoi effettuare al COMPLETAMENTO dell'episode
            # multiplo di validation_frequency. Questo spiega il perché sull'uso
            # di completed_episodes = episode + 1

            completed_episodes = episode + 1

            if (
                self.validation_env is not None
                and self.validation_episodes > 0
                and self.validation_frequency > 0
                and completed_episodes % self.validation_frequency == 0
            ):
                validation_win_rate = self.evaluate_greedy()

                self.validation_history.append({
                    "episode": completed_episodes,
                    "global_step": self.global_step,
                    "win_rate": validation_win_rate,
                })

                if self.on_validation is not None:
                    self.on_validation({
                        "episode": completed_episodes,
                        "global_step": self.global_step,
                        "win_rate": validation_win_rate,
                        "best_win_rate": max(self.best_validation_win_rate, validation_win_rate),
                    })

                # utilizzando il > anziché >=, in caso di parità
                # sul win rate si mantiene il primo checkpoint che ha raggiunto
                # il miglior checkpont.

                if validation_win_rate > self.best_validation_win_rate:
                    self.best_validation_win_rate = validation_win_rate

                    self.best_state_dict = {
                        "online_network_state_dict": copy.deepcopy(self.online_network.state_dict()),
                        "target_network_state_dict": copy.deepcopy(self.target_network.state_dict()),
                    }

        if self.best_state_dict is not None:
            timestamp = datetime.now().strftime(
                "%Y-%m-%d-%H-%M-%S"
            )

            self.checkpoint_path = self.save_checkpoint(
                checkpoint_dir=self.checkpoint_dir,
                filename=f"{timestamp}-best.pt",
                state_dict_overrides=self.best_state_dict,
            )

        if save_checkpoint:
            self.last_checkpoint_path = self.save_checkpoint(
                checkpoint_dir=checkpoint_dir,
            )

            print(
                f"Checkpoint saved to: "
                f"{self.last_checkpoint_path}"
            )
    
    def evaluate_greedy(self) -> float:
        """ Evaluates the online network deterministically on the fixed
        validation board and returns the validation win rate."""

        # verifica che l'ambiente di validazione sia configurato correttamente
        if self.validation_env is None or self.validation_episodes <= 0:
            raise RuntimeError("Validation is not configured.")

        # memorizziamo lo stato precedente della online network per ripristinarlo alla fine
        was_training = self.online_network.training
        # la rete viene messa in evaluation mode
        self.online_network.eval()
        wins = 0

        try:
            # loop di valutazione deterministica su un numero fisso di episodi
            for episode in range(self.validation_episodes):
                # seed sequenziale per riproducibilità tra i vari checkpoint di validazione
                obs, info = self.validation_env.reset(
                    seed=self.validation_seed_start + episode
                )
                mine_density = self.get_mine_density(self.validation_env)

                terminated = False
                truncated = False

                while not (terminated or truncated):
                    # la conversione one-hot viene fatta dentro la funzione get_greedy_action
                    # la funzione ritorna direttamente la prossima azione in maniera
                    # greedy, facendo argmax. inoltre, nella get_greedy_action
                    # già disabilitiamo la costruzione del computational graph,
                    # quindi non occorre rifarlo anche qui.
                    action = self.get_greedy_action(obs, mine_density)

                    obs, _, terminated, truncated, info = (
                        self.validation_env.step(action)
                    )

                wins += int(
                    terminated
                    and info.get("status") == "won"
                )

        finally:
            self.online_network.train(was_training)

        return wins / self.validation_episodes