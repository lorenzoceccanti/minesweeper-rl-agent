import pytest

from sweeps.budget import free_cells, resolve_episode_budget, scale_episode_budget


def test_free_cells():
    assert free_cells(6, 6, 5) == 31
    assert free_cells(9, 9, 10) == 71


def test_free_cells_rejects_invalid_mine_count():
    with pytest.raises(ValueError):
        free_cells(6, 6, 0)
    with pytest.raises(ValueError):
        free_cells(6, 6, 36)


def test_reference_board_returns_reference_episodes():
    assert scale_episode_budget(6, 6, 5) == 20_000


def test_larger_board_scales_up():
    budget_6x6 = scale_episode_budget(6, 6, 5)
    budget_9x9 = scale_episode_budget(9, 9, 10)
    assert budget_9x9 > budget_6x6
    # proportional to free cells: 71 free cells vs 31 free cells
    assert budget_9x9 == round(20_000 * (71 / 31))


def test_smaller_board_scales_down():
    budget_6x6 = scale_episode_budget(6, 6, 5)
    budget_4x4 = scale_episode_budget(4, 4, 2)
    assert budget_4x4 < budget_6x6


def test_budget_never_drops_below_minimum():
    budget = scale_episode_budget(2, 2, 3, minimum_episodes=500)
    assert budget >= 500


def test_resolve_episode_budget_prefers_explicit_override():
    assert resolve_episode_budget(9, 9, 10, override=12_345) == 12_345


def test_resolve_episode_budget_falls_back_to_scaling():
    assert resolve_episode_budget(9, 9, 10, override=None) == scale_episode_budget(9, 9, 10)
