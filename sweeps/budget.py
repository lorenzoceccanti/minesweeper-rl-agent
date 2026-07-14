"""Episode-budget scaling heuristic based on board size/difficulty.

The board configuration is a fixed input to a search task, never a sweep
parameter: it is not something the search optimizes over (see the plan --
for now a campaign targets one fixed board at a time, e.g. 6x6 or 9x9, to
find the best reachable performance on that configuration). This module only
decides how many screening episodes a fixed board deserves, so a bigger/
harder board isn't silently under-trained by a budget tuned for 6x6.
"""

from __future__ import annotations

REFERENCE_BOARD = {"board_height": 6, "board_width": 6, "n_mines": 5}
REFERENCE_EPISODES = 20_000
MINIMUM_EPISODES = 1_000


def free_cells(board_height: int, board_width: int, n_mines: int) -> int:
    cells = board_height * board_width
    if not (0 < n_mines < cells):
        raise ValueError(f"n_mines ({n_mines}) must be between 1 and board cells-1 ({cells - 1})")
    return cells - n_mines


def scale_episode_budget(
    board_height: int,
    board_width: int,
    n_mines: int,
    *,
    reference_episodes: int = REFERENCE_EPISODES,
    reference_board: dict[str, int] = REFERENCE_BOARD,
    minimum_episodes: int = MINIMUM_EPISODES,
) -> int:
    """Scale the screening episode budget proportionally to free cells.

    A board with more free cells than the reference gets proportionally more
    episodes; a smaller board gets proportionally fewer, floored at
    `minimum_episodes` so a tiny board never rounds down to near-zero.
    """
    reference_cells = free_cells(**reference_board)
    target_cells = free_cells(board_height, board_width, n_mines)
    scaled = round(reference_episodes * (target_cells / reference_cells))
    return max(minimum_episodes, scaled)


def resolve_episode_budget(
    board_height: int,
    board_width: int,
    n_mines: int,
    override: int | None = None,
    **kwargs,
) -> int:
    """Return `override` if the campaign YAML set one explicitly, else the scaled default."""
    if override is not None:
        return override
    return scale_episode_budget(board_height, board_width, n_mines, **kwargs)
