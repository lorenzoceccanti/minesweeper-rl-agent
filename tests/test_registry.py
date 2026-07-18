from sweeps.registry import Registry, build_code_state, code_state_mismatches

RECORDED = {
    "git_commit": "abc123",
    "git_dirty": False,
    "source_hashes": {"agents/dqn_agent.py": "hash1", "agents/ppo_agent.py": "hash2"},
}


def test_code_state_mismatches_empty_when_identical():
    assert code_state_mismatches(RECORDED, dict(RECORDED)) == []


def test_code_state_mismatches_detects_commit_drift():
    current = {**RECORDED, "git_commit": "def456"}
    mismatches = code_state_mismatches(RECORDED, current)
    assert any("git_commit" in mismatch for mismatch in mismatches)


def test_code_state_mismatches_detects_dirty_drift():
    current = {**RECORDED, "git_dirty": True}
    mismatches = code_state_mismatches(RECORDED, current)
    assert any("git_dirty" in mismatch for mismatch in mismatches)


def test_code_state_mismatches_detects_changed_source_file():
    current = {
        **RECORDED,
        "source_hashes": {**RECORDED["source_hashes"], "agents/dqn_agent.py": "different-hash"},
    }
    mismatches = code_state_mismatches(RECORDED, current)
    assert any("agents/dqn_agent.py" in mismatch for mismatch in mismatches)


def test_code_state_mismatches_detects_new_or_removed_tracked_file():
    current = {**RECORDED, "source_hashes": {"agents/dqn_agent.py": "hash1"}}
    mismatches = code_state_mismatches(RECORDED, current)
    assert any("agents/ppo_agent.py" in mismatch for mismatch in mismatches)


def test_build_code_state_returns_expected_keys():
    state = build_code_state()
    assert set(state) == {"git_commit", "git_dirty", "source_hashes"}
    assert isinstance(state["git_commit"], str) and len(state["git_commit"]) > 0
    assert isinstance(state["git_dirty"], bool)
    assert "agents/dqn_agent.py" in state["source_hashes"]


def test_registry_register_and_save_load_round_trip(tmp_path):
    registry_path = tmp_path / "campaign.json"
    registry = Registry(registry_path)
    code_state = {"git_commit": "abc123", "git_dirty": False, "source_hashes": {}}

    registry.register(
        "sweep-1",
        task_id="board6x6_5mines",
        algorithm="dqn",
        architecture_name="fully_conv_3layer_64ch_11in",
        code_state=code_state,
    )
    registry.save()

    reloaded = Registry(registry_path)
    entry = reloaded.get("sweep-1")
    assert entry["task_id"] == "board6x6_5mines"
    assert entry["algorithm"] == "dqn"
    assert entry["architecture"] == "fully_conv_3layer_64ch_11in"
    assert entry["git_commit"] == "abc123"
    assert "created_at" in entry


def test_registry_entries_for_algorithms_filters(tmp_path):
    registry = Registry(tmp_path / "campaign.json")
    code_state = {"git_commit": "abc123", "git_dirty": False, "source_hashes": {}}
    registry.register("sweep-dqn", task_id="t1", algorithm="dqn", architecture_name="a", code_state=code_state)
    registry.register("sweep-ppo", task_id="t1", algorithm="ppo", architecture_name="a", code_state=code_state)

    filtered = registry.entries_for_algorithms(["dqn"])
    assert set(filtered) == {"sweep-dqn"}


def test_registry_starts_empty_when_file_does_not_exist(tmp_path):
    registry = Registry(tmp_path / "does_not_exist.json")
    assert registry.sweep_ids() == []
