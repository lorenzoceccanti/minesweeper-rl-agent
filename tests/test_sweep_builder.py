import pytest

from sweeps.sweep_builder import build_sweep_config, sweep_name

CAMPAIGN = {
    "campaign_name": "search_2026_07",
    "search_method": "random",
    "search_space": {
        "dqn": {
            "common": {"learning_rate": {"min": 1e-5, "max": 1e-2}},
            "fully_conv_3layer_64ch_11in": {"hidden_channels": {"values": [32, 64, 128]}},
            "global_skip_conv_3layer_64ch_11in": {
                "hidden_channels": {"values": [32, 64, 128]},
                "global_features_dim": {"values": [16, 32]},
            },
        },
    },
}

TASK = {"id": "board6x6_5mines", "board_height": 6, "board_width": 6, "n_mines": 5}


def test_sweep_name_encodes_the_full_triple():
    assert sweep_name("search_2026_07", "board6x6_5mines", "dqn", "fully_conv_3layer_64ch_11in") == (
        "search_2026_07-dqn-board6x6_5mines-fully_conv_3layer_64ch_11in"
    )


def test_build_sweep_config_has_expected_top_level_shape():
    config = build_sweep_config(CAMPAIGN, TASK, "dqn", "fully_conv_3layer_64ch_11in")

    assert config["name"] == "search_2026_07-dqn-board6x6_5mines-fully_conv_3layer_64ch_11in"
    assert config["method"] == "random"
    assert config["metric"] == {"name": "search/objective", "goal": "maximize"}
    assert config["early_terminate"]["type"] == "hyperband"


def test_build_sweep_config_fixes_board_and_architecture_as_values_not_search_dims():
    config = build_sweep_config(CAMPAIGN, TASK, "dqn", "fully_conv_3layer_64ch_11in")
    parameters = config["parameters"]

    assert parameters["architecture_name"] == {"value": "fully_conv_3layer_64ch_11in"}
    assert parameters["board_height"] == {"value": 6}
    assert parameters["board_width"] == {"value": 6}
    assert parameters["n_mines"] == {"value": 5}
    assert "n_episodes" in parameters and "value" in parameters["n_episodes"]
    assert "global_features_dim" not in parameters


def test_build_sweep_config_includes_architecture_specific_search_params():
    config = build_sweep_config(CAMPAIGN, TASK, "dqn", "global_skip_conv_3layer_64ch_11in")
    parameters = config["parameters"]

    assert parameters["global_features_dim"] == {"values": [16, 32]}
    assert "learning_rate" in parameters


def test_build_sweep_config_respects_episode_budget_override():
    task_with_override = {**TASK, "episode_budget_override": 12_345}
    config = build_sweep_config(CAMPAIGN, task_with_override, "dqn", "fully_conv_3layer_64ch_11in")
    assert config["parameters"]["n_episodes"] == {"value": 12_345}


def test_build_sweep_config_rejects_search_space_shadowing_fixed_params():
    misconfigured = {
        **CAMPAIGN,
        "search_space": {
            "dqn": {
                "common": {"board_height": {"values": [6, 9]}},
                "fully_conv_3layer_64ch_11in": {"hidden_channels": {"values": [64]}},
            }
        },
    }
    with pytest.raises(ValueError, match="board_height"):
        build_sweep_config(misconfigured, TASK, "dqn", "fully_conv_3layer_64ch_11in")


def test_build_sweep_config_defaults_search_method_to_random():
    campaign_without_method = {k: v for k, v in CAMPAIGN.items() if k != "search_method"}
    config = build_sweep_config(campaign_without_method, TASK, "dqn", "fully_conv_3layer_64ch_11in")
    assert config["method"] == "random"
