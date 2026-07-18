import pytest

from sweeps.config_space import (
    ARCHITECTURE_RESTRICTED_PARAMS,
    build_parameter_space,
    iter_algorithm_architecture_pairs,
)


def test_iter_algorithm_architecture_pairs():
    architectures = {
        "dqn": ["fully_conv_3layer_64ch_11in", "global_skip_conv_3layer_64ch_11in"],
        "ppo": ["fully_conv_3layer_64ch_11in"],
    }
    pairs = list(iter_algorithm_architecture_pairs(architectures))
    assert pairs == [
        ("dqn", "fully_conv_3layer_64ch_11in"),
        ("dqn", "global_skip_conv_3layer_64ch_11in"),
        ("ppo", "fully_conv_3layer_64ch_11in"),
    ]


def test_iter_algorithm_architecture_pairs_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="Unknown architecture"):
        list(iter_algorithm_architecture_pairs({"dqn": ["not_a_real_architecture"]}))


SEARCH_SPACE = {
    "dqn": {
        "common": {"learning_rate": {"min": 1e-5, "max": 1e-2}, "discount_factor": {"values": [0.8, 0.9, 0.95]}},
        "fully_conv_3layer_64ch_11in": {"hidden_channels": {"values": [32, 64, 128]}},
        "global_skip_conv_3layer_64ch_11in": {
            "hidden_channels": {"values": [32, 64, 128]},
            "global_features_dim": {"values": [16, 32, 64]},
        },
    },
    "ppo": {
        "common": {"actor_learning_rate": {"min": 1e-5, "max": 1e-2}, "critic_hidden_size": {"values": [256, 512]}},
        "global_skip_conv_3layer_64ch_11in": {"global_features_dim": {"values": [16, 32]}},
    },
}


def test_build_parameter_space_merges_common_and_architecture_specific():
    space = build_parameter_space("dqn", "fully_conv_3layer_64ch_11in", SEARCH_SPACE)
    assert set(space) == {"learning_rate", "discount_factor", "hidden_channels"}
    assert "global_features_dim" not in space


def test_build_parameter_space_global_skip_includes_global_features_dim():
    space = build_parameter_space("dqn", "global_skip_conv_3layer_64ch_11in", SEARCH_SPACE)
    assert set(space) == {"learning_rate", "discount_factor", "hidden_channels", "global_features_dim"}


def test_build_parameter_space_ppo_critic_hidden_size_is_common_to_both_architectures():
    fully_conv_space = build_parameter_space("ppo", "fully_conv_3layer_64ch_11in", SEARCH_SPACE)
    global_skip_space = build_parameter_space("ppo", "global_skip_conv_3layer_64ch_11in", SEARCH_SPACE)
    assert "critic_hidden_size" in fully_conv_space
    assert "critic_hidden_size" in global_skip_space
    assert "global_features_dim" not in fully_conv_space
    assert "global_features_dim" in global_skip_space


def test_build_parameter_space_rejects_restricted_param_under_common():
    misconfigured = {
        "dqn": {
            "common": {"global_features_dim": {"values": [16, 32]}},  # wrong: belongs under global_skip only
            "fully_conv_3layer_64ch_11in": {"hidden_channels": {"values": [64]}},
        }
    }
    with pytest.raises(ValueError, match="restricted to architecture"):
        build_parameter_space("dqn", "fully_conv_3layer_64ch_11in", misconfigured)


def test_build_parameter_space_rejects_duplicate_key_in_common_and_arch_specific():
    misconfigured = {
        "dqn": {
            "common": {"hidden_channels": {"values": [64]}},
            "fully_conv_3layer_64ch_11in": {"hidden_channels": {"values": [128]}},
        }
    }
    with pytest.raises(ValueError, match="declared in both"):
        build_parameter_space("dqn", "fully_conv_3layer_64ch_11in", misconfigured)


def test_build_parameter_space_rejects_empty_search_space():
    with pytest.raises(ValueError, match="empty search space"):
        build_parameter_space("dqn", "fully_conv_3layer_64ch_11in", {"dqn": {}})


def test_architecture_restricted_params_only_lists_global_skip_features():
    # hidden_channels and critic_hidden_size apply to every known architecture
    # (see models/factory.py), so they must never show up as restricted.
    all_restricted = frozenset().union(*ARCHITECTURE_RESTRICTED_PARAMS.values())
    assert "hidden_channels" not in all_restricted
    assert "critic_hidden_size" not in all_restricted
