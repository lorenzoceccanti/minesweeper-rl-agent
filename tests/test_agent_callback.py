import torch

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from environment.minesweeper_env import MinesweeperEnv

DEVICE = torch.device("cpu")
BOARD = {"board_height": 3, "board_width": 3, "n_mines": 1}


def _make_env():
    return MinesweeperEnv(**BOARD, render_mode=None)


def test_dqn_on_validation_receives_expected_keys_and_no_wandb_dependency():
    env = _make_env()
    validation_env = _make_env()
    received = []

    agent = DQNAgent(
        env=env,
        device=DEVICE,
        seed=0,
        learning_rate=1e-3,
        initial_epsilon=1.0,
        epsilon_decay=0.1,
        final_epsilon=0.05,
        hidden_channels=8,
        global_features_dim=4,
        validation_env=validation_env,
        validation_episodes=2,
        validation_frequency=5,
        on_validation=received.append,
    )

    try:
        agent.train(n_episodes=10, env_seed_start=0)
    finally:
        env.close()
        validation_env.close()

    assert len(received) == 2  # validation triggered at episode 5 and 10
    for metrics in received:
        assert metrics.keys() == {"episode", "global_step", "win_rate", "best_win_rate"}
        assert 0.0 <= metrics["win_rate"] <= 1.0
        assert metrics["best_win_rate"] >= metrics["win_rate"] or metrics["best_win_rate"] == metrics["win_rate"]


def test_dqn_on_validation_defaults_to_none_and_is_optional():
    env = _make_env()
    agent = DQNAgent(
        env=env,
        device=DEVICE,
        seed=0,
        learning_rate=1e-3,
        initial_epsilon=1.0,
        epsilon_decay=0.1,
        final_epsilon=0.05,
        hidden_channels=8,
        global_features_dim=4,
    )
    try:
        agent.train(n_episodes=2, env_seed_start=0)
    finally:
        env.close()

    assert agent.on_validation is None


def test_ppo_on_validation_receives_expected_keys_and_no_wandb_dependency():
    env = _make_env()
    validation_env = _make_env()
    received = []

    agent = PPOAgent(
        env=env,
        device=DEVICE,
        seed=0,
        env_seed_start=0,
        rollout_steps=16,
        hidden_channels=8,
        global_features_dim=4,
        critic_hidden_size=16,
        validation_env=validation_env,
        validation_episodes=2,
        validation_frequency=1,
        on_validation=received.append,
    )

    try:
        agent.train(n_episodes=20)
    finally:
        env.close()
        validation_env.close()

    assert len(received) > 0
    for metrics in received:
        assert metrics.keys() == {"rollout", "win_rate", "best_win_rate"}
        assert 0.0 <= metrics["win_rate"] <= 1.0


def test_ppo_on_validation_defaults_to_none_and_is_optional():
    env = _make_env()
    agent = PPOAgent(
        env=env,
        device=DEVICE,
        seed=0,
        env_seed_start=0,
        rollout_steps=16,
        hidden_channels=8,
        global_features_dim=4,
        critic_hidden_size=16,
    )
    try:
        agent.train(n_episodes=16)
    finally:
        env.close()

    assert agent.on_validation is None
