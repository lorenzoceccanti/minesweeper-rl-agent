import gymnasium as gym
import models.actor_net as actor_net
import models.critic_net as critic_net
import buffers.rollout_buffer as rollout_buffer
import models.encodings as encodings
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

class PPOAgent:
    def __init__(
            self,
            env: gym.Env,
            device: torch.device,
            seed: int | None,
            env_seed_start: int | None,
            max_actor_grad_norm: float = 0.5,
            max_critic_grad_norm: float = 5.0,
            rollout_steps: int = 2048,
            discount_factor: float = 0.95,
            gae_lambda: float = 0.95,
            batch_size: int = 64,
            update_epochs: int = 10,
            clip_epsilon: float = 0.2,
            entropy_coefficient: float = 0.01,
            actor_learning_rate: float = 3e-4,
            critic_learning_rate: float = 3e-4,
            logger = None,
            validation_env: gym.Env | None = None,
            validation_episodes: int = 0,
            validation_seed_start: int = 500_000,
            validation_frequency: int = 5,
            best_checkpoint_dir: str | Path = "checkpoints/ppo/best",
        ):
        
        self.seed = seed
        self.env_seed_start = env_seed_start
        self.logger = logger
        self.device = device
        self.last_checkpoint_path = None
        self.best_checkpoint_path = None
        self.validation_env = validation_env
        self.validation_episodes = validation_episodes
        self.validation_seed_start = validation_seed_start
        self.validation_frequency = validation_frequency
        self.best_checkpoint_dir = best_checkpoint_dir

        # == ENVIRONMENT == 
        self.env = env
        if seed is not None:
            self.env.action_space.seed(seed)
        # ==================

        # == NEURAL NETWORKS == 
        self.actor = actor_net.ActorNetwork().to(self.device)
        self.critic = critic_net.CriticNetwork().to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_learning_rate
        )

        # == ROLLOUT BUFFER ==
        self.rollout_buffer = rollout_buffer.RolloutBuffer()
        self.total_rollout_steps = rollout_steps

        # == OTHER PARAMETERS ==
        self.max_actor_grad_norm = max_actor_grad_norm
        self.max_critic_grad_norm = max_critic_grad_norm
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.clip_epsilon = clip_epsilon
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.entropy_coefficient = entropy_coefficient
        # ======================

        # == TRAINING METRICS ==
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_wins = []

        self.actor_loss_history = []
        self.critic_loss_history = []
        self.training_error = []
        self.entropy_history = []
        self.validation_history = []
        self.best_validation_win_rate = -1.0
    
    def save_checkpoint(
            self,
            checkpoint_dir: str | Path = "checkpoints/ppo",
            filename: str | None = None,
    ) -> Path:

        checkpoint_dir = Path(checkpoint_dir)

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

        checkpoint_path = checkpoint_dir / filename

        checkpoint = {
            "algorithm":
                "ppo",

            "actor_state_dict":
                self.actor.state_dict(),

            "critic_state_dict":
                self.critic.state_dict(),

            "actor_optimizer_state_dict":
                self.actor_optimizer.state_dict(),

            "critic_optimizer_state_dict":
                self.critic_optimizer.state_dict(),

            "seed":
                self.seed,

            "env_seed_start":
                self.env_seed_start,

            "episode_rewards":
                self.episode_rewards,

            "episode_lengths":
                self.episode_lengths,

            "episode_wins":
                self.episode_wins,

            "actor_loss_history":
                self.actor_loss_history,

            "critic_loss_history":
                self.critic_loss_history,

            "training_error":
                self.training_error,

            "entropy_history":
                self.entropy_history,

            "validation_history":
                self.validation_history,

            "best_validation_win_rate":
                self.best_validation_win_rate,

            "hyperparameters": {
                "max_actor_grad_norm":
                    self.max_actor_grad_norm,

                "max_critic_grad_norm":
                    self.max_critic_grad_norm,

                "rollout_steps":
                    self.total_rollout_steps,

                "discount_factor":
                    self.discount_factor,

                "gae_lambda":
                    self.gae_lambda,

                "batch_size":
                    self.batch_size,

                "update_epochs":
                    self.update_epochs,

                "clip_epsilon":
                    self.clip_epsilon,

                "entropy_coefficient":
                    self.entropy_coefficient,

                "actor_learning_rate":
                    self.actor_learning_rate,

                "critic_learning_rate":
                    self.critic_learning_rate,
            },
        }

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

    def evaluate_greedy(self) -> float:
        # verifica che l'ambiente di validazione sia configurato correttamente
        if self.validation_env is None or self.validation_episodes <= 0:
            raise RuntimeError("Validation is not configured.")

        # memorizziamo lo stato precedente dell'attore (train/eval) per ripristinarlo alla fine,
        # impostando la rete in modalità .eval()
        was_training = self.actor.training
        self.actor.eval()
        wins = 0

        try:
            # loop di valutazione deterministica su un numero fisso di episodi
            for episode in range(self.validation_episodes):
                # seed sequenziale per riproducibilità tra i vari checkpoint di validazione
                obs, info = self.validation_env.reset(
                    seed=self.validation_seed_start + episode
                )
                validation_mine_density = self.get_mine_density(self.validation_env)
                terminated = False
                truncated = False

                while not (terminated or truncated):
                    # conversione in tensore e codifica one-hot dello stato del board
                    obs_tensor = torch.as_tensor(
                        obs,
                        dtype=torch.long,
                        device=self.device,
                    )
                    encoded_obs = encodings.one_hot_encode_board(obs_tensor, validation_mine_density)
                    # gestione action masking, analogo a quanto fatto nella get_action
                    action_mask = obs_tensor.eq(-2).flatten().unsqueeze(0)

                    with torch.no_grad():
                        logits = self.actor(encoded_obs)
                        action = logits.masked_fill(
                            ~action_mask,
                            -torch.inf,
                        ).argmax(dim=1).item()

                    obs, _, terminated, truncated, info = self.validation_env.step(
                        int(action)
                    )
                # incrementiamo il contatore se l'episodio è terminato con una vittoria
                wins += int(terminated and info.get("status") == "won")
        finally:
            # ripristino dello stato originale dell'actor
            self.actor.train(was_training)
            
        # restituisce il win-rate medio sugli episodes di validazione
        return wins / self.validation_episodes

    def get_action(self, obs: np.ndarray, mine_density: float) -> tuple[int, float]:
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
        encoded_obs = encodings.one_hot_encode_board(obs_tensor, mine_density)

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
        # the logits produced by the actor network
        # to pick the next action. We can disable the construction of the
        # computational graph
        with torch.no_grad():
            # For how we designed the actor network, the network return
            # directly the logits
            # NN input: [1, C, H, W]
            # NN output: [1, H*W]
            logits = self.actor(encoded_obs)
            # applicazione della maschera action_mask ai logits
            masked_logits = logits.masked_fill(
                ~action_mask,
                -torch.inf,
            )

        # We need a torch.distributions.Categorical object in order
        # to call the sample method, which is the one responsible for
        # sampling the action later
        distribution = torch.distributions.Categorical(
            logits=masked_logits
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

        obs, _, rewards, terminateds, truncateds, next_obs, _, mine_densities = self.rollout_buffer.get_all()
        
        # convertiamo in tensori pytorch: alcuni vanno passati alla rete neurale
        # critic che lavora su one-hot encoding.
        obs_tensor = torch.as_tensor(obs, dtype=torch.long, device=self.device)
        next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        terminateds = torch.as_tensor(terminateds, dtype=torch.float32, device=self.device)
        truncateds = torch.as_tensor(truncateds, dtype=torch.float32, device=self.device)
        mine_densities_tensor = torch.as_tensor(mine_densities, dtype=torch.float32, device=self.device)
        
        # Producing a one-hot encoding of the observation tensor
        # [H,W] -> [1, C, H, W]
        encoded_obs = encodings.one_hot_encode_board(obs_tensor, mine_densities_tensor)
        encoded_next_obs = encodings.one_hot_encode_board(next_obs_tensor, mine_densities_tensor)

        # using the critic in inference mode: no computation graph construction
        # required
        with torch.no_grad():
            
            values = self.critic(encoded_obs)
            next_values = self.critic(encoded_next_obs)

            # errore di previsione a un passo: δ_t = r_t + γ V(s_{t+1}) - V(s_t)
            # se l'episodio è realmente terminato, dopo non c'è valore da stimare
            deltas = (
                rewards
                + self.discount_factor * (1 - terminateds) * next_values
                - values
            )
            advantages = torch.zeros_like(deltas)
            gae = 0.0

            # si procede al contrario perché A_t dipende da A_{t+1}
            for t in reversed(range(len(deltas))):
                episode_finished = torch.maximum(terminateds[t], truncateds[t])

                # A_t = δ_t + γλ(1 - done) A_{t+1}
                gae = (
                    deltas[t]
                    + self.discount_factor
                    * self.gae_lambda
                    * (1 - episode_finished)
                    * gae
                )
                advantages[t] = gae

            # il critic deve approssimare il return: target = V(s_t) + A_t
            value_targets = values + advantages

            if advantages.numel() > 1:
                advantages = (
                    advantages - advantages.mean()
                ) / (advantages.std() + 1e-8)
            

        # dato che il rollout buffer ha tipi numpy e non pytorch,
        # bisogna ripassare dalla CPU, prima di riconvertire in oggetti numpy
        self.rollout_buffer.set_advantages(advantages.cpu().numpy())
        self.rollout_buffer.set_value_targets(value_targets.cpu().numpy())
    
    def update(self):
        # fa il pezzo sotto del ppo, che campiona dal rollout buffer, etc.
        # questo metodo è chiamato dalla train

        actor_losses = []
        critic_losses = []
        critic_errors = []  
        entropies = []

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
        #         confrontalo con value_target, con la MSE loss
        #         aggiorna il critic


        for epoch in range(self.update_epochs):

            # ALTRA PARTICOLARITA' di PPO NON VISTA NEL CORSO:
            # per ciascuna delle epoche quello che si fa in ppo è
            # - mescolare le transizioni del rollout; 
            # - dividere le transizioni
            #   in minibatch (possibilmente della stessa dimensione, fino ad esaurimento del rollout)
            # - eseguire un aggiornamento dei pesi SU OGNI MINIBATCH
            # - dopo K epoche, si svuota il buffer
    
            minibatches = self.rollout_buffer.sample_minibatches(
                batch_size=self.batch_size
            )

            for minibatch in minibatches:

                # === ACTOR === 

                # Ordine di ritorno dei parametri
                # obs, actions, rewards, terminated, truncated, next_obs, old_log_probs, advantages, value_targets
                obs, actions, _, _, _, _, old_log_probs, mine_densities, advantages, value_targets = minibatch
            
                # Ricordiamo che il formato di salvataggio nel buffer è NumPy, occorre una riconversione
                # in PyTorch
                
                obs_tensor = torch.as_tensor(obs, dtype=torch.long, device=self.device)
                actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device)
                mine_densities_tensor = torch.as_tensor(mine_densities, dtype=torch.float32, device=self.device)
                old_log_probs_tensor = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
                advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
                value_targets_tensor = torch.as_tensor(value_targets, dtype=torch.float32, device=self.device)

                # gli stati prima di essere dati in pasto alle reti neurali devono essere anche one-hot encodati
                encoded_obs = encodings.one_hot_encode_board(obs_tensor, mine_densities_tensor)

                # action masking, come è stato fatto nella get_action sopra
                action_mask = (
                    obs_tensor
                    .eq(-2)
                    .flatten(start_dim=1)
                )
                
                logits = self.actor(encoded_obs)
                # applicazione della maschera action_mask ai logits
                masked_logits = logits.masked_fill(
                    ~action_mask,
                    -torch.inf,
                )

                # ricaviamon la distribuzione di probabilità dai p_i
                distribution = torch.distributions.Categorical(logits=masked_logits)
                # Log-probabilities of the selected actions under the current policy
                new_log_probs = distribution.log_prob(actions_tensor)

                # Pertanto si possono sfruttare le proprietà dei logaritmi per calcolarci il prob. ratio
                # r = new_policy / old_policy
                # ma vale anche r = exp(log(new_policy/old_policy))
                # ma dato che valgono le proprietà dei logaritmi
                # r = exp(log(new_policy) - log(old_policy))
                # quindi siamo in grado di ricavarci il probability ratio come segue:

                probability_ratio = torch.exp(new_log_probs - old_log_probs_tensor)
                clipped = torch.clamp(probability_ratio, min=1-self.clip_epsilon, max=1+self.clip_epsilon)*advantages_tensor
                not_clipped = probability_ratio*advantages_tensor

                # Calcoliamo l'entropia della policy corrente: misura l'incertezza/casualità 
                # delle azioni.
                entropy = distribution.entropy().mean()
                entropies.append(entropy.item())

                # calcolo la clipped actor loss, il meno davanti perché
                # fare stiamo facendo una gradient descent su una NLL
                # che equivale a fare gradient ascent su actor loss
                policy_loss = -torch.min(
                    not_clipped,
                    clipped,
                ).mean()

                # Calcoliamo l'actor loss totale iniettando l'entropy bonus
                # Nota sul segno (-): dato che l'obiettivo globale è MINIMIZZARE l'actor_loss complessiva, sottrarre l'entropia equivale a MASSIMIZZARLA nel gradiente finale.
                # Questo spinge l'agente a esplorare maggiormente e previene il collasso prematuro 
                # della policy verso massimi locali deterministici (subottimali)
                actor_loss = (
                    policy_loss
                    - self.entropy_coefficient * entropy
                )

                actor_losses.append(
                    actor_loss.item()
                )

                self.actor_optimizer.zero_grad()
                actor_loss.backward()

                # gradient clipping
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(),
                            max_norm=self.max_actor_grad_norm)
            
                self.actor_optimizer.step()

                # === CRITIC === 
                # passo gli stati al critic, e il critic restituisce una stima della value function
                predicted_values = self.critic(encoded_obs)
                # value target tensor ricordiamoci che la TD-one step estimation del return,
                # pescata dal rollout buffer e sottoforma di tensore.
                critic_loss = torch.nn.functional.mse_loss(predicted_values, value_targets_tensor)
                
                critic_losses.append(
                    critic_loss.item()
                )

                with torch.no_grad():
                    critic_error = torch.mean(
                        torch.abs(
                            value_targets_tensor - predicted_values
                        )
                    ).item()
                critic_errors.append(critic_error)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    max_norm=self.max_critic_grad_norm
                )
                self.critic_optimizer.step()
        
        self.actor_loss_history.append(float(np.mean(actor_losses)))
        self.critic_loss_history.append(float(np.mean(critic_losses)))
        self.training_error.append(float(np.mean(critic_errors)))
        self.entropy_history.append(float(np.mean(entropies)))

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
        rollout_index = 0
        tqdm_bar = tqdm(total=n_episodes, desc="Training Progress", unit="episode")
        # At the beginning we reset the episode
        obs, info = self.reset_episode(episode_index = completed_episodes)
        current_mine_density = self.get_mine_density(self.env)
        current_episode_reward = 0.0
        current_episode_length = 0

        while completed_episodes < n_episodes:
    
            rollout_steps = 0
            # in questo while interno ci sono due condizioni di uscita
            # una sul numero di rollout steps, una anche sul numero di
            # episodes totali perché potrebbe anche capitare
            # che si arrivi alla fine del numero totale di episodes prima
            # di raggiungere il numero totale di rollout steps

            while (rollout_steps < self.total_rollout_steps and
            completed_episodes < n_episodes):
                
                action, old_log_prob = self.get_action(obs, current_mine_density)
                
                # Given an action randomly picked from the pi_old distribution
                # we observe the environment in order to collect a trajectory
                # to put into the ROLLOUT BUFFER.
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                self.rollout_buffer.push(obs, action, reward, terminated, truncated, next_obs, old_log_prob, current_mine_density)

                obs = next_obs
                rollout_steps += 1

                current_episode_reward += float(reward)
                current_episode_length += 1

                # Checking if we arrive in a terminal state
                # in which an episode ends.
                if terminated or truncated:
                    completed_episodes += 1
                    tqdm_bar.update(1)

                    if terminated:
                        end_reason = info.get("status", "terminated")
                    else:
                        end_reason = "truncated"

                    self.episode_rewards.append(
                        current_episode_reward
                    )

                    self.episode_lengths.append(
                        current_episode_length
                    )

                    self.episode_wins.append(
                        int(end_reason == "won")
                    )

                    current_episode_reward = 0.0
                    current_episode_length = 0

                    # We require to reset and make advance the seed
                    # only at the end of the episode, but we continue to put
                    # transactions into the buffer (in fact we stay in the loop)

                    if completed_episodes < n_episodes:
                        obs, info = self.reset_episode(episode_index=completed_episodes)
                        current_mine_density = self.get_mine_density(self.env)

            # COMPUTING THE ADVANTAGE, AFTER T rollout steps
            self.compute_advantage()
            
            # K EPOCHS OF PPO UPDATE
            self.update()

            rollout_index += 1
            # controllo periodico per la greedy evaluation del modello
            if (
                self.validation_env is not None
                and self.validation_episodes > 0
                and rollout_index % self.validation_frequency == 0
            ):
                validation_win_rate = self.evaluate_greedy()
                self.validation_history.append({
                    "rollout": rollout_index,
                    "win_rate": validation_win_rate,
                })

                # se il win rate corrente supera il massimo storico 
                # aggiorna il record e salva i pesi attuali come best model
                if validation_win_rate > self.best_validation_win_rate:
                    self.best_validation_win_rate = validation_win_rate
                    self.best_checkpoint_path = self.save_checkpoint(
                        checkpoint_dir=self.best_checkpoint_dir,
                        filename="best.pt",
                    )

            # INVALIDATING THE ROLLOUT BUFFER AFTER PPO UPDATE
            self.rollout_buffer.clear()