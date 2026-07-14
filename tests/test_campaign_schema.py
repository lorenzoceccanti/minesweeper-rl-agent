import copy

from sweeps.campaign_schema import validate_campaign

VALID_CAMPAIGN = {
    "campaign_name": "search_2026_07",
    "base_config": "config.yaml",
    "tasks": [
        {"id": "board6x6_5mines", "board_height": 6, "board_width": 6, "n_mines": 5},
    ],
    "architectures": {
        "dqn": ["fully_conv_3layer_64ch_11in", "global_skip_conv_3layer_64ch_11in"],
    },
    "search_method": "random",
    "search_space": {
        "dqn": {
            "common": {"learning_rate": {"min": 1e-5, "max": 1e-2}},
            "fully_conv_3layer_64ch_11in": {"hidden_channels": {"values": [32, 64]}},
            "global_skip_conv_3layer_64ch_11in": {
                "hidden_channels": {"values": [32, 64]},
                "global_features_dim": {"values": [16, 32]},
            },
        },
    },
    "worker_profiles": {
        "gpu_desktop": {"max_concurrent_runs": 1, "algorithms": ["dqn"]},
    },
    "promotion": {"finalists_per_sweep": 2, "confirm_episodes": 100_000, "confirm_seeds": [43, 44, 45]},
    "test": {"n_episodes_per_seed": 1000, "test_seed_start": 1_000_000},
}


def test_valid_campaign_has_no_errors():
    assert validate_campaign(copy.deepcopy(VALID_CAMPAIGN)) == []


def test_missing_top_level_key_is_reported():
    campaign = copy.deepcopy(VALID_CAMPAIGN)
    del campaign["promotion"]
    errors = validate_campaign(campaign)
    assert any("promotion" in error for error in errors)


def test_duplicate_task_id_is_reported():
    campaign = copy.deepcopy(VALID_CAMPAIGN)
    campaign["tasks"].append(dict(campaign["tasks"][0]))
    errors = validate_campaign(campaign)
    assert any("duplicate task id" in error for error in errors)


def test_n_mines_exceeding_board_cells_is_reported():
    campaign = copy.deepcopy(VALID_CAMPAIGN)
    campaign["tasks"][0]["n_mines"] = 36
    errors = validate_campaign(campaign)
    assert any("n_mines" in error for error in errors)


def test_unknown_architecture_is_reported():
    campaign = copy.deepcopy(VALID_CAMPAIGN)
    campaign["architectures"]["dqn"].append("not_a_real_architecture")
    errors = validate_campaign(campaign)
    assert any("not_a_real_architecture" in error for error in errors)


def test_unknown_algorithm_is_reported():
    campaign = copy.deepcopy(VALID_CAMPAIGN)
    campaign["architectures"]["not_an_algorithm"] = ["fully_conv_3layer_64ch_11in"]
    errors = validate_campaign(campaign)
    assert any("unknown algorithm" in error for error in errors)


def test_invalid_search_method_is_reported():
    campaign = copy.deepcopy(VALID_CAMPAIGN)
    campaign["search_method"] = "bayes"
    errors = validate_campaign(campaign)
    assert any("search_method" in error for error in errors)


def test_search_space_leaking_restricted_param_is_reported():
    campaign = copy.deepcopy(VALID_CAMPAIGN)
    campaign["search_space"]["dqn"]["common"]["global_features_dim"] = {"values": [16]}
    errors = validate_campaign(campaign)
    assert any("restricted to architecture" in error for error in errors)


def test_worker_profile_referencing_unknown_algorithm_is_reported():
    campaign = copy.deepcopy(VALID_CAMPAIGN)
    campaign["worker_profiles"]["gpu_desktop"]["algorithms"] = ["ppo"]
    errors = validate_campaign(campaign)
    assert any("ppo" in error for error in errors)


def test_promotion_missing_confirm_seeds_is_reported():
    campaign = copy.deepcopy(VALID_CAMPAIGN)
    campaign["promotion"]["confirm_seeds"] = []
    errors = validate_campaign(campaign)
    assert any("confirm_seeds" in error for error in errors)
