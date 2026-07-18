from sweeps.promote import (
    PromotionStore,
    _disjoint_env_seed_start,
    fetch_finished_screening_records,
    format_report,
    run_confirmation_trial,
    run_held_out_test,
    select_winner,
)

BASE_CONFIG = {
    "main": {"device": "cuda"},
    "dqn": {
        "architecture_name": "fully_conv_3layer_64ch_11in",
        "hidden_channels": 64,
        "train": {
            "agent_seed": 42,
            "train_env_seed_start": 50_000,
            "learning_rate": 0.0003,
            "board_height": 6,
            "board_width": 6,
            "n_mines": 5,
            "n_episodes": 200,
            "validation_frequency": 10,
        },
        "test": {
            "checkpoint_path": "checkpoints/dqn/best/best.pt",
            "board_height": 6,
            "board_width": 6,
            "n_mines": 5,
            "test_seed_start": 1_000_000,
            "test_seed_count": 1000,
            "render_mode": None,
            "agent_seed": 42,
            "moving_average_window": 50,
            "test_output_dir": "plots/test",
            "wandb_project": "minesweeper-rl",
            "wandb_entity": None,
        },
    },
}


class FakeSummary(dict):
    def get(self, key, default=None):
        return dict.get(self, key, default)


class FakeRun:
    def __init__(self, state, config, best_win_rate=None, run_id="run-1", summary_key="best_win_rate"):
        self.state = state
        self.config = config
        self.summary = FakeSummary({} if best_win_rate is None else {summary_key: best_win_rate})
        self.id = run_id


class FakeSweep:
    def __init__(self, runs):
        self.runs = runs


class FakeApi:
    def __init__(self, runs):
        self._runs = runs
        self.default_entity = "fake-entity"

    def sweep(self, path):
        assert path == "fake-entity/minesweeper-rl/sweep-1"
        return FakeSweep(self._runs)


def test_disjoint_env_seed_start_never_overlaps_across_seed_indices():
    confirm_episodes = 1000
    starts = [_disjoint_env_seed_start(50_000, confirm_episodes, i) for i in range(3)]
    assert starts == [50_000, 51_000, 52_000]
    assert starts[0] + confirm_episodes <= starts[1]
    assert starts[1] + confirm_episodes <= starts[2]


def test_fetch_finished_screening_records_filters_unfinished_and_unreported():
    runs = [
        FakeRun("finished", {"learning_rate": 0.001, "architecture_name": "fully_conv_3layer_64ch_11in"}, 0.7),
        FakeRun("running", {"learning_rate": 0.002, "architecture_name": "fully_conv_3layer_64ch_11in"}, 0.9),
        FakeRun("finished", {"learning_rate": 0.003, "architecture_name": "fully_conv_3layer_64ch_11in"}, None),
        # Hyperband-pruned but reported real progress before being killed:
        # rankable, via best_win_rate rather than best_validation_win_rate.
        FakeRun("failed", {"learning_rate": 0.004, "architecture_name": "fully_conv_3layer_64ch_11in"}, 0.64),
        # Pruned before its first validation report: nothing to rank.
        FakeRun("failed", {"learning_rate": 0.005, "architecture_name": "fully_conv_3layer_64ch_11in"}, None),
        # Crashed (e.g. the matplotlib background-thread bug) but still
        # reported progress before the process died: rankable too.
        FakeRun("crashed", {"learning_rate": 0.006, "architecture_name": "fully_conv_3layer_64ch_11in"}, 0.68),
        # Legacy run predating best_win_rate: falls back to
        # best_validation_win_rate, which only "finished" runs ever set.
        FakeRun(
            "finished",
            {"learning_rate": 0.007, "architecture_name": "fully_conv_3layer_64ch_11in"},
            0.81,
            summary_key="best_validation_win_rate",
        ),
    ]
    records = fetch_finished_screening_records("sweep-1", "dqn", "minesweeper-rl", api=FakeApi(runs))

    win_rates = {record["validation_win_rate"] for record in records}
    assert win_rates == {0.7, 0.64, 0.68, 0.81}
    for record in records:
        assert record["validation_mean_return"] == 0.0
        assert record["stage"] == "screen"
        assert record["status"] == "completed"


def test_run_confirmation_trial_uses_finalists_config_id_not_a_recomputed_one(monkeypatch):
    finalist = {
        "config_id": "abc123",
        "hyperparameters": {
            "architecture_name": "fully_conv_3layer_64ch_11in",
            "hidden_channels": 64,
            "learning_rate": 0.0005,
            "board_height": 6,
            "board_width": 6,
            "n_mines": 5,
            "n_episodes": 50,
        },
    }

    captured_run_config = {}

    def fake_train_run(run_config, on_validation=None):
        captured_run_config.update(run_config)
        return {
            "best_checkpoint_path": "checkpoints/dqn/2026-01-01-best.pt",
            "board_height": run_config["board_height"],
            "board_width": run_config["board_width"],
            "n_mines": run_config["n_mines"],
        }

    class FakeWandbRun:
        summary = FakeSummary({"best_validation_win_rate": 0.81})

    monkeypatch.setattr("sweeps.promote.TRAIN_MODULES", {"dqn": type("M", (), {"run": staticmethod(fake_train_run)})})
    monkeypatch.setattr("sweeps.promote.wandb.init", lambda **kwargs: None)
    monkeypatch.setattr("sweeps.promote.wandb.finish", lambda: None)
    monkeypatch.setattr("sweeps.promote.wandb.run", FakeWandbRun())

    record = run_confirmation_trial(
        "dqn", finalist, seed=43, seed_index=0, confirm_episodes=100_000,
        project="minesweeper-rl", entity=None, base_config=BASE_CONFIG,
    )

    assert record["config_id"] == "abc123"
    assert record["agent_seed"] == 43
    assert record["validation_win_rate"] == 0.81
    assert captured_run_config["n_episodes"] == 100_000
    assert captured_run_config["agent_seed"] == 43


def test_select_winner_excludes_configs_missing_a_required_seed():
    records = [
        {"stage": "confirm", "algorithm": "dqn", "status": "completed", "config_id": "complete",
         "hyperparameters": {}, "validation_win_rate": 0.6, "validation_mean_return": 0.0},
        {"stage": "confirm", "algorithm": "dqn", "status": "completed", "config_id": "complete",
         "hyperparameters": {}, "validation_win_rate": 0.62, "validation_mean_return": 0.0},
        {"stage": "confirm", "algorithm": "dqn", "status": "completed", "config_id": "complete",
         "hyperparameters": {}, "validation_win_rate": 0.58, "validation_mean_return": 0.0},
        {"stage": "confirm", "algorithm": "dqn", "status": "completed", "config_id": "incomplete",
         "hyperparameters": {}, "validation_win_rate": 0.99, "validation_mean_return": 0.0},
    ]

    winner = select_winner(records, "dqn", required_seed_count=3)

    assert winner is not None
    assert winner["config_id"] == "complete"
    assert len(winner["records"]) == 3


def test_select_winner_returns_none_when_nothing_completes_all_seeds():
    records = [
        {"stage": "confirm", "algorithm": "dqn", "status": "completed", "config_id": "only-one-seed",
         "hyperparameters": {}, "validation_win_rate": 0.9, "validation_mean_return": 0.0},
    ]

    assert select_winner(records, "dqn", required_seed_count=3) is None


def test_run_held_out_test_evaluates_best_replica_checkpoint(monkeypatch):
    winner = {
        "config_id": "cfg-1",
        "hyperparameters": {"architecture_name": "fully_conv_3layer_64ch_11in"},
        "records": [
            {"validation_win_rate": 0.7, "checkpoint": "checkpoints/dqn/worse.pt",
             "board_height": 6, "board_width": 6, "n_mines": 5},
            {"validation_win_rate": 0.9, "checkpoint": "checkpoints/dqn/best.pt",
             "board_height": 6, "board_width": 6, "n_mines": 5},
        ],
    }

    captured_config = {}

    def fake_evaluate_run(config):
        captured_config.update(config)
        return {"wins": 900, "episodes": 1000, "win_rate": 0.9, "mean_return": 1.23}

    monkeypatch.setattr(
        "sweeps.promote.EVALUATE_MODULES", {"dqn": type("M", (), {"run": staticmethod(fake_evaluate_run)})}
    )

    summary = run_held_out_test(winner, "dqn", test_seed_start=1_000_000, test_episode_count=1000, base_config=BASE_CONFIG)

    assert summary["win_rate"] == 0.9
    assert captured_config["checkpoint_path"] == "checkpoints/dqn/best.pt"
    assert captured_config["test_seed_start"] == 1_000_000
    assert captured_config["test_seed_count"] == 1000
    assert captured_config["architecture_name"] == "fully_conv_3layer_64ch_11in"


def test_promotion_store_round_trip(tmp_path):
    path = tmp_path / "campaign.promotion.json"
    store = PromotionStore(path)
    store.set_result("sweep-1", {"algorithm": "dqn", "winner": None})
    store.save()

    reloaded = PromotionStore(path)
    assert reloaded.get("sweep-1") == {"algorithm": "dqn", "winner": None}


def test_format_report_handles_missing_winner():
    results = {
        "sweep-1": {
            "algorithm": "dqn", "task_id": "board6x6_5mines", "architecture_name": "fully_conv_3layer_64ch_11in",
            "finalists": [], "confirm_records": [], "winner": None, "test_summary": None,
        },
    }
    report = format_report("smoke", results)
    assert "no winner" in report
    assert "sweep-1" in report


def test_format_report_includes_winner_and_test_summary():
    results = {
        "sweep-1": {
            "algorithm": "dqn", "task_id": "board6x6_5mines", "architecture_name": "fully_conv_3layer_64ch_11in",
            "finalists": [{"config_id": "cfg-1"}],
            "confirm_records": [],
            "winner": {
                "config_id": "cfg-1", "mean_win_rate": 0.75, "ci_low": 0.7, "ci_high": 0.8,
                "records": [1, 2, 3],
            },
            "test_summary": {"wins": 750, "n_episodes": 1000, "win_rate": 0.75, "mean_return": 1.1},
        },
    }
    report = format_report("smoke", results)
    assert "cfg-1" in report
    assert "750/1000" in report
