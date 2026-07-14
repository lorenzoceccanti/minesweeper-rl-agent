from sweeps.scoring import aggregate_confirmation, config_id, score, select_finalists


def test_config_id_stable_regardless_of_key_order():
    a = {"learning_rate": 1e-3, "architecture_name": "fully_conv_3layer_64ch_11in", "hidden_channels": 64}
    b = {"hidden_channels": 64, "architecture_name": "fully_conv_3layer_64ch_11in", "learning_rate": 1e-3}
    assert config_id(a) == config_id(b)


def test_config_id_distinguishes_architecture_at_equal_hyperparameters():
    base = {"learning_rate": 1e-3, "discount_factor": 0.95}
    fully_conv = {**base, "architecture_name": "fully_conv_3layer_64ch_11in", "hidden_channels": 64}
    global_skip = {**base, "architecture_name": "global_skip_conv_3layer_64ch_11in", "hidden_channels": 64, "global_features_dim": 16}
    assert config_id(fully_conv) != config_id(global_skip)


def test_config_id_distinguishes_network_size():
    base = {"learning_rate": 1e-3, "architecture_name": "fully_conv_3layer_64ch_11in"}
    small = {**base, "hidden_channels": 32}
    large = {**base, "hidden_channels": 128}
    assert config_id(small) != config_id(large)


def test_score_orders_by_win_rate_then_return():
    better = {"validation_win_rate": 0.6, "validation_mean_return": 1.0}
    worse = {"validation_win_rate": 0.5, "validation_mean_return": 5.0}
    assert score(better) > score(worse)


def _screen_record(config_id_value, win_rate, mean_return, algorithm="dqn", status="completed"):
    return {
        "stage": "screen",
        "algorithm": algorithm,
        "status": status,
        "config_id": config_id_value,
        "validation_win_rate": win_rate,
        "validation_mean_return": mean_return,
        "hyperparameters": {"config_id": config_id_value},
    }


def test_select_finalists_dedupes_by_config_id_keeping_best():
    records = [
        _screen_record("cfg-a", win_rate=0.4, mean_return=1.0),
        _screen_record("cfg-a", win_rate=0.7, mean_return=2.0),  # re-run of same config, better score
        _screen_record("cfg-b", win_rate=0.6, mean_return=1.5),
        _screen_record("cfg-c", win_rate=0.5, mean_return=1.5, status="failed"),  # excluded
        _screen_record("cfg-d", win_rate=0.9, mean_return=1.0, algorithm="ppo"),  # different algorithm
    ]
    finalists = select_finalists(records, algorithm="dqn", count=2)
    assert [f["config_id"] for f in finalists] == ["cfg-a", "cfg-b"]
    assert finalists[0]["validation_win_rate"] == 0.7  # kept the better of the two cfg-a runs


def test_select_finalists_respects_count():
    records = [_screen_record(f"cfg-{i}", win_rate=i / 10, mean_return=0.0) for i in range(5)]
    finalists = select_finalists(records, algorithm="dqn", count=2)
    assert len(finalists) == 2
    assert finalists[0]["config_id"] == "cfg-4"
    assert finalists[1]["config_id"] == "cfg-3"


def _confirm_record(config_id_value, win_rate, mean_return, algorithm="dqn", status="completed"):
    return {
        "stage": "confirm",
        "algorithm": algorithm,
        "status": status,
        "config_id": config_id_value,
        "validation_win_rate": win_rate,
        "validation_mean_return": mean_return,
        "hyperparameters": {"config_id": config_id_value},
    }


def test_aggregate_confirmation_computes_mean_and_ci():
    records = [
        _confirm_record("cfg-a", 0.5, 1.0),
        _confirm_record("cfg-a", 0.7, 1.5),
        _confirm_record("cfg-a", 0.6, 1.2),
        _confirm_record("cfg-b", 0.9, 2.0),
        _confirm_record("cfg-c", 0.9, 2.0, status="failed"),
        _confirm_record("cfg-d", 0.9, 2.0, algorithm="ppo"),
    ]
    aggregates = aggregate_confirmation(records, algorithm="dqn")
    ids = [a["config_id"] for a in aggregates]
    assert ids == ["cfg-b", "cfg-a"]  # cfg-b has higher mean win rate, ranked first

    cfg_a = next(a for a in aggregates if a["config_id"] == "cfg-a")
    assert cfg_a["mean_win_rate"] == (0.5 + 0.7 + 0.6) / 3
    assert cfg_a["std_win_rate"] > 0.0
    assert cfg_a["ci_low"] < cfg_a["mean_win_rate"] < cfg_a["ci_high"]
    assert len(cfg_a["records"]) == 3

    cfg_b = next(a for a in aggregates if a["config_id"] == "cfg-b")
    assert cfg_b["std_win_rate"] == 0.0  # single seed -> no spread
    assert cfg_b["ci_low"] == cfg_b["ci_high"] == cfg_b["mean_win_rate"]


def test_aggregate_confirmation_caller_can_detect_incomplete_configs():
    records = [
        _confirm_record("cfg-a", 0.5, 1.0),
        _confirm_record("cfg-a", 0.7, 1.5),
    ]
    aggregates = aggregate_confirmation(records, algorithm="dqn")
    expected_seed_count = 3
    incomplete = [a for a in aggregates if len(a["records"]) < expected_seed_count]
    assert len(incomplete) == 1
    assert incomplete[0]["config_id"] == "cfg-a"
