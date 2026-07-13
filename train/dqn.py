from agents import dqn_agent as dqn
from common.seeding import select_device, set_global_seed
from environment import minesweeper_env as mine
from plot import training_plots
from tracking import wandb_logger


def run(config: dict) -> dict:
    set_global_seed(config["agent_seed"])

    device = select_device()
    print(f"Using device: {device}")

    epsilon_decay = (
        config["start_epsilon"] - config["final_epsilon"]
    ) / (config["n_episodes"] / 2)

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

    agent = dqn.DQNAgent(
        env=env,
        device=device,
        seed=config["agent_seed"],
        learning_rate=config["learning_rate"],
        initial_epsilon=config["start_epsilon"],
        epsilon_decay=epsilon_decay,
        final_epsilon=config["final_epsilon"],
        discount_factor=config["discount_factor"],
        replay_buffer_capacity=config["replay_buffer_capacity"],
        batch_size=config["batch_size"],
        target_update_frequency=config["target_update_frequency"],
        learning_starts=config["learning_starts"],
        train_frequency=config["train_frequency"],
        logger=None,
        validation_env=validation_env,
        validation_episodes=config["validation_episodes"],
        validation_seed_start=config["validation_seed_start"],
        validation_frequency=config["validation_frequency"],
        checkpoint_dir=config["checkpoint_dir"],
    )

    try:
        agent.train(
            n_episodes=config["n_episodes"],
            save_checkpoint=False,
            env_seed_start=config["train_env_seed_start"],
        )

        final_checkpoint_path = agent.save_checkpoint(
            checkpoint_dir=config["checkpoint_dir"],
        )

        print(f"Final checkpoint saved to: {final_checkpoint_path}")
        print(
            "Best validation checkpoint: "
            f"{agent.checkpoint_path}"
        )

        plot_path = training_plots.plot_training_from_checkpoint(
            checkpoint_path=final_checkpoint_path,
            board_height=config["board_height"],
            board_width=config["board_width"],
            num_mines=config["n_mines"],
            output_dir="plots",
        )

        wandb_logger.log_run(
            algorithm="dqn",
            checkpoint_path=final_checkpoint_path,
            best_checkpoint_path=agent.checkpoint_path,
            plot_paths=[plot_path],
            project=config["wandb_project"],
            entity=config["wandb_entity"],
            architecture_name=config["architecture_name"],
        )

    finally:
        env.close()
        validation_env.close()

    return {
        "final_checkpoint_path": final_checkpoint_path,
        "best_checkpoint_path": agent.checkpoint_path,
        "board_height": config["board_height"],
        "board_width": config["board_width"],
        "n_mines": config["n_mines"],
    }
