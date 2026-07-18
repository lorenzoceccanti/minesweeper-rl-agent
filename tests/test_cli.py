import argparse
import copy

import pytest
import yaml

from sweeps.cli import (
    cmd_promote,
    cmd_report,
    cmd_validate,
    expand_triples,
    grid_trial_count,
    load_campaign,
    registry_path_for,
)
from sweeps.registry import Registry

CAMPAIGN = {
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


def test_registry_path_for_uses_campaign_name():
    path = registry_path_for("search_2026_07")
    assert path.name == "search_2026_07.json"
    assert "sweeps/registry" in str(path)


def test_load_campaign_reads_yaml(tmp_path):
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(yaml.safe_dump(CAMPAIGN))
    assert load_campaign(campaign_path) == CAMPAIGN


def test_expand_triples_covers_every_task_algorithm_architecture_combo():
    triples = expand_triples(CAMPAIGN)
    assert len(triples) == 2
    architectures = {architecture_name for _, _, architecture_name in triples}
    assert architectures == {"fully_conv_3layer_64ch_11in", "global_skip_conv_3layer_64ch_11in"}
    assert all(algorithm == "dqn" for _, algorithm, _ in triples)


def test_grid_trial_count_is_none_for_random_method():
    sweep_config = {"method": "random", "parameters": {"learning_rate": {"values": [1e-3, 1e-4]}}}
    assert grid_trial_count(sweep_config) is None


def test_grid_trial_count_multiplies_values_and_ignores_fixed():
    sweep_config = {
        "method": "grid",
        "parameters": {
            "learning_rate": {"values": [1e-3, 1e-4, 1e-5]},
            "hidden_channels": {"values": [32, 64]},
            "architecture_name": {"value": "fully_conv_3layer_64ch_11in"},
        },
    }
    assert grid_trial_count(sweep_config) == 3 * 2


def test_cmd_validate_succeeds_on_valid_campaign(tmp_path, capsys):
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(yaml.safe_dump(CAMPAIGN))
    args = argparse.Namespace(campaign=campaign_path, allow_large_grid=False)

    exit_code = cmd_validate(args)

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "2 sweep(s) will be created" in output


def test_cmd_promote_saves_result_per_sweep(tmp_path, monkeypatch, capsys):
    registry_path = tmp_path / "search_2026_07.json"
    promotion_path = tmp_path / "search_2026_07.promotion.json"
    monkeypatch.setattr("sweeps.cli.registry_path_for", lambda name: registry_path)
    monkeypatch.setattr("sweeps.cli.promotion_path_for", lambda name: promotion_path)

    registry = Registry(registry_path)
    registry.register(
        "sweep-1", task_id="board6x6_5mines", algorithm="dqn",
        architecture_name="fully_conv_3layer_64ch_11in", code_state={"git_commit": "x", "git_dirty": False, "source_hashes": {}},
    )
    registry.save()

    captured_calls = []

    def fake_promote_sweep(sweep_id, algorithm, task_id, architecture_name, promotion, test, project, entity, **kwargs):
        captured_calls.append(sweep_id)
        return {
            "sweep_id": sweep_id, "algorithm": algorithm, "task_id": task_id, "architecture_name": architecture_name,
            "finalists": [], "confirm_records": [], "winner": None, "test_summary": None,
        }

    monkeypatch.setattr("sweeps.cli.promote_sweep", fake_promote_sweep)

    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(yaml.safe_dump(CAMPAIGN))
    args = argparse.Namespace(campaign=campaign_path, sweep_id=None, project="minesweeper-rl", entity=None)

    exit_code = cmd_promote(args)

    assert exit_code == 0
    assert captured_calls == ["sweep-1"]
    assert promotion_path.exists()
    assert "no winner" in capsys.readouterr().out


def test_cmd_report_requires_prior_promotion(tmp_path, monkeypatch, capsys):
    promotion_path = tmp_path / "search_2026_07.promotion.json"
    monkeypatch.setattr("sweeps.cli.promotion_path_for", lambda name: promotion_path)

    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(yaml.safe_dump(CAMPAIGN))
    args = argparse.Namespace(campaign=campaign_path)

    exit_code = cmd_report(args)

    assert exit_code == 1
    assert "promote" in capsys.readouterr().err


def test_cmd_report_writes_markdown_from_saved_results(tmp_path, monkeypatch):
    promotion_path = tmp_path / "search_2026_07.promotion.json"
    monkeypatch.setattr("sweeps.cli.promotion_path_for", lambda name: promotion_path)
    reports_dir = tmp_path / "reports"
    monkeypatch.setattr(
        "sweeps.cli.resolve_project_path",
        lambda relative: reports_dir / "search_2026_07.md" if "reports" in relative else tmp_path / relative,
    )

    from sweeps.promote import PromotionStore

    store = PromotionStore(promotion_path)
    store.set_result("sweep-1", {
        "algorithm": "dqn", "task_id": "board6x6_5mines", "architecture_name": "fully_conv_3layer_64ch_11in",
        "finalists": [], "confirm_records": [], "winner": None, "test_summary": None,
    })
    store.save()

    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(yaml.safe_dump(CAMPAIGN))
    args = argparse.Namespace(campaign=campaign_path)

    exit_code = cmd_report(args)

    assert exit_code == 0
    report_path = reports_dir / "search_2026_07.md"
    assert report_path.exists()
    assert "sweep-1" in report_path.read_text()


def test_cmd_validate_fails_on_invalid_campaign(tmp_path, capsys):
    invalid_campaign = copy.deepcopy(CAMPAIGN)
    del invalid_campaign["promotion"]
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(yaml.safe_dump(invalid_campaign))
    args = argparse.Namespace(campaign=campaign_path, allow_large_grid=False)

    exit_code = cmd_validate(args)

    assert exit_code == 1
    assert "invalid" in capsys.readouterr().err


def test_cmd_validate_blocks_large_grid_without_flag(tmp_path, capsys):
    large_grid_campaign = copy.deepcopy(CAMPAIGN)
    large_grid_campaign["search_method"] = "grid"
    large_grid_campaign["search_space"]["dqn"]["common"]["learning_rate"] = {
        "values": [1e-3, 1e-4, 1e-5, 1e-2, 3e-4, 3e-3, 3e-5, 1e-1, 5e-3, 5e-4, 5e-5]
    }
    large_grid_campaign["search_space"]["dqn"]["fully_conv_3layer_64ch_11in"] = {
        "hidden_channels": {"values": [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]}
    }
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(yaml.safe_dump(large_grid_campaign))
    args = argparse.Namespace(campaign=campaign_path, allow_large_grid=False)

    exit_code = cmd_validate(args)

    assert exit_code == 1
    assert "allow-large-grid" in capsys.readouterr().err


def test_cmd_validate_allows_large_grid_with_flag(tmp_path, capsys):
    large_grid_campaign = copy.deepcopy(CAMPAIGN)
    large_grid_campaign["search_method"] = "grid"
    large_grid_campaign["search_space"]["dqn"]["common"]["learning_rate"] = {
        "values": [1e-3, 1e-4, 1e-5, 1e-2, 3e-4, 3e-3, 3e-5, 1e-1, 5e-3, 5e-4, 5e-5]
    }
    large_grid_campaign["search_space"]["dqn"]["fully_conv_3layer_64ch_11in"] = {
        "hidden_channels": {"values": [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]}
    }
    campaign_path = tmp_path / "campaign.yaml"
    campaign_path.write_text(yaml.safe_dump(large_grid_campaign))
    args = argparse.Namespace(campaign=campaign_path, allow_large_grid=True)

    exit_code = cmd_validate(args)

    assert exit_code == 0
