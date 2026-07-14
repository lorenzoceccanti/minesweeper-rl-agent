from agents import ppo_agent as ppo
from common.seeding import select_device, set_global_seed
from environment import minesweeper_env as mine
from plot import training_plots
from tracking import wandb_logger


def run(config: dict, on_validation=None) -> dict:
    set_global_seed(config["agent_seed"])

    device = select_device(config.get("device"))
    print(f"Using device: {device}")

    architecture_name = config["architecture_name"]

    env = mine.MinesweeperEnv(
        board_height=config["board_height"],
        board_width=config["board_width"],
        n_mines=config["n_mines"],
        render_mode=config["render_mode"],
    )
    validation_env = mine.MinesweeperEnv(
        board_height=config["board_height"],
        board_width=config["board_width"],
        n_mines=config["n_mines"],
        render_mode=None,
    )

    agent = ppo.PPOAgent(
        env=env,
        device=device,
        seed=config["agent_seed"],
        env_seed_start=config["train_env_seed_start"],
        max_actor_grad_norm=config["max_actor_grad_norm"],
        max_critic_grad_norm=config["max_critic_grad_norm"],
        rollout_steps=config["rollout_steps"],
        discount_factor=config["discount_factor"],
        gae_lambda=config["gae_lambda"],
        batch_size=config["batch_size"],
        update_epochs=config["update_epochs"],
        clip_epsilon=config["clip_epsilon"],
        entropy_coefficient=config["entropy_coefficient"],
        actor_learning_rate=config["actor_learning_rate"],
        critic_learning_rate=config["critic_learning_rate"],
        logger=None,
        validation_env=validation_env,
        validation_episodes=config["validation_episodes"],
        validation_seed_start=config["validation_seed_start"],
        validation_frequency=config["validation_frequency"],
        architecture_name=config["architecture_name"],
        checkpoint_dir=config["checkpoint_dir"],
        hidden_channels=config["hidden_channels"],
        global_features_dim=config["global_features_dim"],
        critic_hidden_size=config["critic_hidden_size"],
        on_validation=on_validation,
    )

    try:
        agent.train(n_episodes=config["n_episodes"])

        checkpoint_path = agent.save_checkpoint(
            checkpoint_dir=config["checkpoint_dir"],
        )

        print(f"Checkpoint saved to: {checkpoint_path}")
        print(f"Best validation checkpoint: {agent.checkpoint_path}")

    finally:
        plot_path = training_plots.plot_training_from_checkpoint(
            checkpoint_path=checkpoint_path,
            board_height=config["board_height"],
            board_width=config["board_width"],
            num_mines=config["n_mines"],
            output_dir="plots",
        )

        wandb_logger.log_run(
            algorithm="ppo",
            checkpoint_path=checkpoint_path,
            best_checkpoint_path=agent.checkpoint_path,
            plot_paths=[plot_path],
            project=config["wandb_project"],
            entity=config["wandb_entity"],
            architecture_name=config["architecture_name"],
        )

        env.close()
        validation_env.close()

    return {
        "final_checkpoint_path": checkpoint_path,
        "best_checkpoint_path": agent.checkpoint_path,
        "board_height": config["board_height"],
        "board_width": config["board_width"],
        "n_mines": config["n_mines"],
    }
