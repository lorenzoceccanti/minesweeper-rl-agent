from sweeps.train_entrypoint import build_run_config, make_wandb_callback

BASE_CONFIG = {
    "main": {"device": "cuda"},
    "dqn": {
        "architecture_name": "global_skip_conv_3layer_64ch_11in",
        "hidden_channels": 128,
        "global_features_dim": 32,
        "train": {
            "agent_seed": 42,
            "learning_rate": 0.0003,
            "board_height": 6,
            "board_width": 6,
            "n_mines": 5,
            "n_episodes": 50000,
            "validation_frequency": 100,
        },
    },
}


def test_build_run_config_starts_from_base_train_defaults():
    run_config = build_run_config("dqn", BASE_CONFIG, sweep_config={})
    assert run_config["agent_seed"] == 42
    assert run_config["validation_frequency"] == 100


def test_build_run_config_injects_architecture_root_fields():
    run_config = build_run_config("dqn", BASE_CONFIG, sweep_config={})
    assert run_config["architecture_name"] == "global_skip_conv_3layer_64ch_11in"
    assert run_config["hidden_channels"] == 128
    assert run_config["global_features_dim"] == 32
    assert run_config["device"] == "cuda"


def test_build_run_config_sweep_values_override_base_defaults():
    sweep_config = {
        "architecture_name": "fully_conv_3layer_64ch_11in",
        "board_height": 9,
        "board_width": 9,
        "n_mines": 10,
        "n_episodes": 12_345,
        "learning_rate": 0.001,
    }
    run_config = build_run_config("dqn", BASE_CONFIG, sweep_config)

    assert run_config["architecture_name"] == "fully_conv_3layer_64ch_11in"
    assert run_config["board_height"] == 9
    assert run_config["n_episodes"] == 12_345
    assert run_config["learning_rate"] == 0.001
    # untouched by the sweep trial, still coming from the base config
    assert run_config["agent_seed"] == 42


def test_wandb_callback_logs_win_rate_as_search_objective(monkeypatch):
    logged = []
    monkeypatch.setattr("sweeps.train_entrypoint.wandb.log", logged.append)

    callback = make_wandb_callback()
    callback({"episode": 100, "win_rate": 0.42, "best_win_rate": 0.5})

    assert len(logged) == 1
    assert logged[0]["search/objective"] == 0.42
    assert logged[0]["win_rate"] == 0.42
    assert logged[0]["episode"] == 100
