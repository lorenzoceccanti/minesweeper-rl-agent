"""Constants used by the Pygame human renderer."""

GUI_CELL_SIZE = 48
GUI_PADDING = 12
GUI_FPS = 4

GUI_WINDOW_TITLE = "Minesweeper RL Agent"

GUI_BACKGROUND_COLOR = (35, 35, 35)

GUI_HIDDEN_CELL_COLOR = (150, 150, 150)
GUI_HIDDEN_CELL_HIGHLIGHT_COLOR = (195, 195, 195)
GUI_HIDDEN_CELL_SHADOW_COLOR = (95, 95, 95)

GUI_REVEALED_CELL_COLOR = (225, 225, 225)
GUI_MINE_CELL_COLOR = (220, 100, 100)

GUI_GRID_COLOR = (60, 60, 60)
GUI_MINE_COLOR = (20, 20, 20)

# Evidenzia l'ultima cella selezionata dall'agente.
GUI_LAST_ACTION_COLOR = (255, 170, 0)

GUI_NUMBER_COLORS = {
    1: (25, 80, 200),
    2: (20, 125, 45),
    3: (210, 40, 40),
    4: (85, 35, 145),
    5: (135, 45, 30),
    6: (25, 135, 140),
    7: (20, 20, 20),
    8: (100, 100, 100),
}