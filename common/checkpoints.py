import torch

from agents import dqn_agent as dqn
from agents import ppo_agent as ppo
from common.paths import resolve_project_path
from common.seeding import set_global_seed


def load_dqn_agent_from_checkpoint(checkpoint_path, env, device, fallback_seed) -> dqn.DQNAgent:
    checkpoint_path = resolve_project_path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    hyperparameters = checkpoint.get("hyperparameters", {})
    checkpoint_seed = checkpoint.get("seed")

    # older checkpoint versions might not have an architecture name:
    # in that case, it necessarly means that the architecture_name is the fully conv. one
    architecture_name = hyperparameters.get("architecture_name", "fully_conv_3layer_64ch_11in")

    if checkpoint_seed is None:
        checkpoint_seed = fallback_seed

    set_global_seed(checkpoint_seed)

    agent = dqn.DQNAgent(
        env=env,
        device=device,
        seed=checkpoint_seed,
        learning_rate=hyperparameters.get("learning_rate", 1e-3),
        initial_epsilon=0.0,
        epsilon_decay=0.0,
        final_epsilon=0.0,
        discount_factor=hyperparameters.get("discount_factor", 0.8),
        replay_buffer_capacity=1,
        batch_size=1,
        target_update_frequency=1,
        learning_starts=0,
        train_frequency=1,
        logger=None,
        architecture_name=architecture_name
    )

    agent.online_network.load_state_dict(checkpoint["online_network_state_dict"])
    agent.target_network.load_state_dict(
        checkpoint.get("target_network_state_dict", checkpoint["online_network_state_dict"])
    )
    agent.online_network.eval()
    agent.target_network.eval()

    return agent


def load_ppo_agent_from_checkpoint(checkpoint_path, env, device, fallback_seed) -> ppo.PPOAgent:
    checkpoint_path = resolve_project_path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    hyperparameters = checkpoint.get("hyperparameters", {})
    checkpoint_seed = checkpoint.get("seed")

    # older checkpoint versions might not have an architecture name:
    # in that case, it necessarly means that the architecture_name is the fully conv. one
    architecture_name = hyperparameters.get("architecture_name", "fully_conv_3layer_64ch_11in")

    if checkpoint_seed is None:
        checkpoint_seed = fallback_seed

    set_global_seed(checkpoint_seed)

    agent = ppo.PPOAgent(
        env=env,
        device=device,
        seed=checkpoint_seed,
        env_seed_start=None,
        max_actor_grad_norm=hyperparameters.get("max_actor_grad_norm", 0.5),
        max_critic_grad_norm=hyperparameters.get("max_critic_grad_norm", 5.0),
        rollout_steps=hyperparameters.get("rollout_steps", 2_048),
        discount_factor=hyperparameters.get("discount_factor", 0.95),
        gae_lambda=hyperparameters.get("gae_lambda", 0.95),
        batch_size=hyperparameters.get("batch_size", 64),
        update_epochs=hyperparameters.get("update_epochs", 10),
        clip_epsilon=hyperparameters.get("clip_epsilon", 0.2),
        entropy_coefficient=hyperparameters.get("entropy_coefficient", 0.01),
        actor_learning_rate=hyperparameters.get("actor_learning_rate", 3e-4),
        critic_learning_rate=hyperparameters.get("critic_learning_rate", 3e-4),
        logger=None,
        architecture_name=architecture_name
    )

    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.critic.load_state_dict(checkpoint["critic_state_dict"])
    agent.actor.eval()
    agent.critic.eval()

    return agent
