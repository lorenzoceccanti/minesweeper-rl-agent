import numpy as np
from tqdm import tqdm

from common.checkpoints import load_dqn_agent_from_checkpoint
from common.paths import resolve_project_path
from common.seeding import select_device, set_global_seed
from environment import minesweeper_env as mine
from plot import test_plots
from tracking import wandb_logger

from pathlib import Path
import csv


def run(config: dict) -> dict:
    test_seeds = tuple(
        range(
            config["test_seed_start"],
            config["test_seed_start"] + config["test_seed_count"],
        )
    )

    if not test_seeds:
        raise ValueError("At least one test seed is required.")

    if len(set(test_seeds)) != len(test_seeds):
        raise ValueError("test_seeds contains duplicated seeds.")

    set_global_seed(config["agent_seed"])

    device = select_device(config.get("device"))
    print(f"Using device: {device}")

    env = mine.MinesweeperEnv(
        board_height=config["board_height"],
        board_width=config["board_width"],
        n_mines=config["n_mines"],
        render_mode=config["render_mode"],
    )

    agent = load_dqn_agent_from_checkpoint(
        checkpoint_path=config["checkpoint_path"],
        env=env,
        device=device,
        fallback_seed=config["agent_seed"],
    )

    episode_results = []
    if config.get("save_csv", True):
        checkpoint_name = Path(config["checkpoint_path"]).stem
        output_dir = Path(config.get("dir_csv", "csv/dqn"))
        output_dir.mkdir(parents=True, exist_ok=True)
        if config["name_csv"] is None:
            csv_path = output_dir / f"{checkpoint_name}_results.csv"
        else:
            csv_path = output_dir / f"{config["name_csv"]}_results.csv"
        # elimina il vecchio csv, se esiste
        if csv_path.exists():
            csv_path.unlink()

    try:
        with tqdm(enumerate(test_seeds, start=1), total=len(test_seeds), desc="Testing", unit="ep") as pbar:
            for episode_index, env_seed in pbar:
                observation, info = env.reset(seed=env_seed)
                mine_density = agent.get_mine_density(env)

                terminated = False
                truncated = False
                episode_return = 0.0
                episode_length = 0

                while not (terminated or truncated):
                    action = agent.get_greedy_action(observation, mine_density)

                    (
                        observation,
                        reward,
                        terminated,
                        truncated,
                        info,
                    ) = env.step(action)

                    episode_return += float(reward)
                    episode_length += 1

                status = (
                    info.get("status", "terminated") if terminated else "truncated"
                )
                won = status == "won"

                result = {
                    "episode": episode_index,
                    "seed": env_seed,
                    "status": status,
                    "won": won,
                    "return": episode_return,
                    "length": episode_length,
                    "mine_density": mine_density,
                }
                episode_results.append(result)

                pbar.set_postfix(status=status, steps=episode_length, ret=f"{episode_return:.1f}")

    finally:
        env.close()

    wins = sum(result["won"] for result in episode_results)

    returns = np.asarray(
        [result["return"] for result in episode_results],
        dtype=np.float64,
    )
    lengths = np.asarray(
        [result["length"] for result in episode_results],
        dtype=np.float64,
    )

    # sample std (Bessel's correction): questi episodi sono trattati come un
    # campione della vera distribuzione di performance dell'agente, utile per
    # eventuali confronti statistici tra checkpoint
    std_ddof = 1 if len(episode_results) > 1 else 0

    summary = {
        "algorithm": "dqn",
        "board_height": config["board_height"],
        "board_width": config["board_width"],
        "num_mines": config["n_mines"],
        "mine_density": config["n_mines"] / (config["board_height"] * config["board_width"]),
        "checkpoint_path": str(resolve_project_path(config["checkpoint_path"])),
        "n_episodes": len(episode_results),
        "wins": wins,
        "win_rate": wins / len(episode_results),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std(ddof=std_ddof)),
        "mean_length": float(lengths.mean()),
        "std_length": float(lengths.std(ddof=std_ddof)),
        "episodes": episode_results,
    }

    print("\n=== Test summary ===")
    print(f"Episodes:       {summary['n_episodes']}")
    print(f"Wins:           {summary['wins']}")
    print(f"Win rate:       {summary['win_rate']:.2%}")
    print(f"Mean return:    {summary['mean_return']:.3f}")
    print(f"Return std:     {summary['std_return']:.3f}")
    print(f"Mean length:    {summary['mean_length']:.3f}")
    print(f"Length std:     {summary['std_length']:.3f}")

    output_paths = test_plots.save_test_outputs(
        summary=summary,
        algorithm="dqn",
        output_dir=config["test_output_dir"],
    )

    print(f"Test results saved to: {output_paths['summary']}")

    if config.get("log_wandb", True):
        wandb_logger.log_test_run(
            algorithm="dqn",
            summary=summary,
            checkpoint_path=config["checkpoint_path"],
            project=config["wandb_project"],
            entity=config["wandb_entity"],
            architecture_name=config["architecture_name"],
            plot_paths=output_paths,
        )

    if config.get("save_csv", True):
        fieldsnames = ["episode", "seed", "won", "return", "length"]
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldsnames)
            writer.writeheader()
            for episode in episode_results:
                writer.writerow({k: episode[k] for k in fieldsnames})
        print(f"CSV file saved in: {csv_path}")
    return summary
