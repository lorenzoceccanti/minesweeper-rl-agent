"""quanti episodi di screening servono per una board dipende da quanto e' grande

la board non e' un parametro su cui si cerca, e' fissa per ogni task, quindi qui
si scala il numero di episodi in modo lineare rispetto a una board di riferimento
(6x6, 5 mine): una board piu' grande/difficile ha piu' celle da imparare quindi
si merita piu' episodi
"""

from __future__ import annotations

# board di riferimento presa come "unita' di misura": 6x6 con 5 mine
REFERENCE_BOARD = {"board_height": 6, "board_width": 6, "n_mines": 5}
# quanti episodi si usano per fare screening sulla board di riferimento
REFERENCE_EPISODES = 20_000
# soglia minima, non si scende mai sotto questo numero di episodi
MINIMUM_EPISODES = 1_000


# conta le celle che non sono mine, cioe' quelle che il modello deve imparare a trovare
def free_cells(board_height: int, board_width: int, n_mines: int) -> int:
    cells = board_height * board_width
    if not (0 < n_mines < cells):
        raise ValueError(f"n_mines ({n_mines}) must be between 1 and board cells-1 ({cells - 1})")
    return cells - n_mines


# function that actually does the linear scaling from the reference board
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
    # semplice regola del tre: piu' celle libere ha la board, piu' episodi servono
    reference_cells = free_cells(**reference_board)
    target_cells = free_cells(board_height, board_width, n_mines)
    scaled = round(reference_episodes * (target_cells / reference_cells))
    return max(minimum_episodes, scaled)


# entry point called by sweep_builder.py, wraps scale_episode_budget with a manual override escape hatch
def resolve_episode_budget(
    board_height: int,
    board_width: int,
    n_mines: int,
    override: int | None = None,
    **kwargs,
) -> int:
    """Return `override` if the campaign YAML set one explicitly, else the scaled default."""
    # se nello yaml qualcuno ha gia' scritto un numero a mano, usiamo quello e basta
    if override is not None:
        return override
    return scale_episode_budget(board_height, board_width, n_mines, **kwargs)
