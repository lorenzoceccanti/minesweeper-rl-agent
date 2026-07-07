import gymnasium as gym
import models.fully_conv_qnet as fully_conv_qnet
import models.encodings as encodings
import buffers.replay_buffer as replay_buffer
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

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
    ):
        self.seed = seed
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

        self.training_error = []
        self.loss_history = []

        # This is the counter of timesteps elapsed
        self.global_step = 0


    def get_action(self, obs: np.ndarray) -> int:
        """
        Selects an action using an epsilon-greedy policy.
        Args:
            obs: Current player board with shape [H, W].
        Returns:
            Flattened index of the selected cell, in the range
            [0, H * W - 1].
        """

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

        # Importante: durante l'exploitation (action selection con p = 1-eps)
        # i q_values sono presi dalla rete con parametri theta, cioè dalla
        # online network. Però tale online network deve essere utilizzata
        # in fase di inferenza. Per questo si introduce torch.no_grad: si evita
        # in questo modo di costruire il grafo computazionale e si risparmia memoria.
        with torch.no_grad():
            # [1, 10, H, W] -> [1, H * W]
            q_values = self.online_network(encoded_obs)
            # [1, H * W] -> scalar action index
            action = torch.argmax(q_values, dim=1).item()
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

        obs = torch.tensor(obs, dtype=torch.long, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        terminateds = torch.tensor(terminateds, dtype=torch.float32, device=self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.long, device=self.device)

        # One-hot encoding of the observations
        # We don't save directly the one hot encoding in the replay buffer to save
        # space.
        obs = encodings.one_hot_encode_board(obs)
        next_obs = encodings.one_hot_encode_board(next_obs)

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
            # max_a'Q(s_j,a',theta-)
            max_next_q_values = next_q_values.max(dim=1).values
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

    def train(self, n_episodes: int, log=False):
        # Log output is an array of strings to be returned to stdout
        # if the log option is set to true
        log_output = []
        for episode in tqdm(range(n_episodes)):
            # Il seed per la board lo imposto una volta sola, sennò
            # ad ogni episode avrei sempre la stessa board
            episode_seed = self.seed if episode == 0 else None
            obs, info = self.env.reset(seed=episode_seed)
            terminated = False
            truncated = False

            episode_reward = 0.0
            episode_steps = 0

            # == LOGGING ==
            if log:
                log_output.append(f"\n--- Episode {episode + 1}/{n_episodes} ---")
            # =============

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
                if log:
                    log_output.append(
                        f"step={episode_steps:3d} | "
                        f"global_step={self.global_step + 1:6d} | "
                        f"action={int(action):3d} | "
                        f"cell=({row}, {column}) | "
                        f"reward={float(reward):6.1f} | "
                        f"status={info.get('status')} | "
                        f"terminated={terminated} | "
                        f"truncated={truncated}"
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

            if log:
                log_output.append(
                    f"Episode completed: "
                    f"reason={end_reason}, "
                    f"steps={episode_steps}, "
                    f"total_reward={episode_reward:.1f}, "
                    f"epsilon={self.epsilon:.4f}"
                )

            # Reduce the exploration rate (the self becomes less random over time)
            self.decay_epsilon()
            if log:
                # Immediately print after the episode
                print("\n".join(log_output))
        # At the end of the episode, we return all the log history
        return log_output