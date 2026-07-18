from common.config_merge import inject_algorithm_root_fields


def test_injects_architecture_name_and_device():
    config = {
        "main": {"device": "cpu"},
        "dqn": {"architecture_name": "fully_conv_3layer_64ch_11in", "hidden_channels": 64},
    }
    run_config = {"learning_rate": 1e-3}

    result = inject_algorithm_root_fields(run_config, config, "dqn")

    assert result is run_config
    assert result["architecture_name"] == "fully_conv_3layer_64ch_11in"
    assert result["hidden_channels"] == 64
    assert result["device"] == "cpu"
    assert "global_features_dim" not in result
    assert "critic_hidden_size" not in result


def test_includes_critic_hidden_size_when_present():
    config = {
        "main": {"device": "cuda"},
        "ppo": {
            "architecture_name": "global_skip_conv_3layer_64ch_11in",
            "hidden_channels": 128,
            "global_features_dim": 32,
            "critic_hidden_size": 512,
        },
    }
    run_config = {}

    result = inject_algorithm_root_fields(run_config, config, "ppo")

    assert result["hidden_channels"] == 128
    assert result["global_features_dim"] == 32
    assert result["critic_hidden_size"] == 512
    assert result["device"] == "cuda"


def test_device_defaults_to_none_when_main_section_missing():
    config = {"dqn": {"architecture_name": "fully_conv_3layer_64ch_11in"}}
    result = inject_algorithm_root_fields({}, config, "dqn")
    assert result["device"] is None
