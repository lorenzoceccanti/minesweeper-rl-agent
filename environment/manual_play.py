"""Manual test harness for MinesweeperEnv.

Controls:
    left click  -> reveal cell
    r           -> reset (new episode, same board config)
    esc / close -> quit
"""

import pygame

from environment.minesweeper_env import MinesweeperEnv


def cell_from_mouse_pos(env, pos):
    """Convert a pixel position into (row, col) board coordinates, or None if outside the grid."""
    x, y = pos
    j = (x - env.padding) // env.cell_size
    i = (y - env.padding) // env.cell_size

    if 0 <= i < env.board_height and 0 <= j < env.board_width:
        return int(i), int(j)
    return None


def play(config: dict) -> None:
    width = config["width"]
    height = config["height"]
    mines = config["mines"]
    seed = config["seed"]

    env = MinesweeperEnv(
        board_width=width,
        board_height=height,
        n_mines=mines,
        render_mode="human",
    )

    observation, info = env.reset(seed=seed)
    print(f"New episode -- board {width}x{height}, {mines} mines")

    running = True
    done = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    observation, info = env.reset(seed=seed)
                    done = False
                    print(f"New episode -- board {width}x{height}, {mines} mines")

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not done:
                cell = cell_from_mouse_pos(env, event.pos)
                if cell is not None:
                    i, j = cell
                    action = i * env.board_width + j
                    observation, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    print(f"click ({i},{j}) -> reward={reward:+.2f} status={info.get('status')}")
                    if done:
                        outcome = "WON" if info.get("status") == "won" else "LOST"
                        print(f"Episode finished: {outcome}. Press 'r' to play again, or close the window to quit.")

        env.render()

    env.close()
    print("Stats:", env.stats)
