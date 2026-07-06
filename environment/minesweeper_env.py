# We're creating a customized environment
# To do so, according to the Gymnasium documentation, we need to extend the class
# gym.Env
# Every Gymnasium environment must define the following methods (https://gymnasium.farama.org/api/env/):
# - reset: Resets the environment to an initial state, required before calling step,
# and between two different episodes
# - step: Updates an environment with actions returining the next agent observation,
# the reward for taking that actions and whether the environment has terminated or truncated
# due to the latest actions, and information from the environment about the step
# - render: Renders the environment to help visualize what the agent see
# - close: Closes the environment

import numpy as np
import gymnasium as gym
from .game import Game
from .pygame_constants import *
import pygame

class MinesweeperEnv(gym.Env):
    def __init__(
            self,
            board_width: int,
            board_height: int,
            n_mines: int,
            render_mode = None
        ):

        """Constructs a customized environment for the Minesweeper Game"""
        super().__init__()
        
        self.metadata = {
            "render_modes": ["human"],
            "render_fps": GUI_FPS,
        }

        self.board_width = board_width
        self.board_height = board_height
        self.n_mines = n_mines
        self.render_mode = render_mode
    
        # Gymnasium has the Space module, which implements various spaces
        # Every Gym environment must have the attributes action_space and observation_space
        # There are some possible specializations of the general Space class
        # Among those we have:
        # - MultiDiscrete, which supports multiple discrete values with multiple axes
        # - Box, represents an interval

        # Let's define what actions are available action space.
        # The action space for the Minesweeper game depends on the width
        # and on the height of the board.
        # For example, if we have a board 6x6 with 5 mines the agent
        # will select a number between 0 and 35 and then we'll have
        # to define in the step() method the mapping between the number
        # and a coordinate in the board.

        # We could have been used also MultiDiscrete to directly represent
        # the action space as a 1D tensor containing [x y] coordinates,
        # but this kind of decision taken adapts more to the nature
        # of DQN, that produces as much as Q(s,a) values as many the
        # as the actions and has size board_height x board_width

        self.action_space = gym.spaces.Discrete(
            self.board_height * self.board_width
        )
        
        # Let's define the observation space
        # The observation for this game consists in the state of the board
        # A cell in the board could be in the following states
        # -2: unrevealed cell
        # -1: IS SKIPPED.
        # 0..8: a reveladed safe cell, with the number of mines around

        self.observation_space = gym.spaces.Box(
                low=-2,
                high=8,
                shape=(self.board_height, self.board_width),
                dtype=np.int8
        )

        # Definition of the rewards
        # according to Wenbo Wang et. al, 2025
        self.R_win = self.board_width * self.board_height # win the game
        self.R_lose = -(self.board_width * self.board_height) # find a mine
        self.R_progress = 1 # uncovering a safe cell
        self.R_guess = -0.5 # random guessing: the cell uncovered was safe, but all neighb. cells are undisclosed yet
        self.R_already_open = -0.5 # choosing an already revealed cell

        # We have a dictionary containing the stats about
        # the game played

        # TODO: pensare a come salvare i casi sfortunati,
        # e a come rappresentare il numero di timesteps per episode
        self.stats = {
            "n_win": 0,
            "n_lose": 0,
            "n_episodes": 0,
            "num_no_progress": 0,
            "num_guess": 0,
            "num_progress": 0
        }
        # GUI stuff
        self.cell_size = GUI_CELL_SIZE
        self.padding = GUI_PADDING
        self.screen = None
        self.window_width = (
            self.board_width * self.cell_size
            + 2 * self.padding
        )

        self.window_height = (
            self.board_height * self.cell_size
            + 2 * self.padding
        )

        self.game = Game(board_width=self.board_width, board_height=self.board_height, num_mines=self.n_mines)
    
        # Pygame resources are created only when human rendering is used.
        
        self.window = None
        # What are these clock and font objects?
        # The clock object is crucial to limit the speed with which the action
        # taken are rendered at screen
        self.clock = None
        # The font object is crucial to understand the numbers on the board
        self.font = None
        # Coordinates of the last cell selected by the agent.
        self.last_action = None
        # Prevents the window from reopening after the user closes it.
        self.window_closed = False

    def reset(self, *, seed=None, options=None):
        """ Reset the environment to an initial state
        Returns:
        - state: the initial state of the player's board after reset
        - info: a dictionary about game information
        """
        super().reset(seed=seed)
        self.board, self.player_board = self.game.reset(self.np_random)
        # done is a flag that indicates whether the game has ended
        self.done = False
        # first_move is another flag that is used to indicate whether
        # the agent is performing the first move or not.
        # It's particularly useful to avoid that the agent is penalized
        # when founds a mine just only after one action
        # In this way, the training is potentially more stable
        self.first_move = True
        # At the beginning the dictionary is empty
        self.info = dict()

        # Since player_board is a list of lists (i.e. a matrix), but
        # the observation space is of type Box, Gymnasium expects
        # NumPy array with coherent shape and type

        self.last_action = None
        observation = np.asarray(
            self.player_board,
            dtype=np.int8
        )

        if self.render_mode == "human":
            self.render()

        return observation, self.info

    def step(self, action):
        """ Take an action in the game environment"""

        if self.done:
            raise RuntimeError(
                "step() called after the episode ended; call reset() first"
            )

        self.info = {}

        # For how we've defined the action space, an action is a cell id
        # We convert a cell id into grid coordinates
        i = action // self.board_width
        j = action % self.board_width

        # Updating last action with the action just done
        self.last_action = (i,j)

        # If it's the first action that the agent performs, we
        # avoid penalizing the agent
        # If the selected cell contains a mine, regenerate the board
        # using the same RNG without reinitializing it.
        # The RNG advances and produces a different but reproducible board.

        # In pratica: continuiamo a ri-generare un piazzamento random
        # delle mine nella board per tutto il tempo in cui
        # continuo a rimanere nel caso molto sfortunato che in posizione
        # i, j c'è la mina.
        if self.first_move:
            while self.board[i][j] == -1:
                self.board, self.player_board = self.game.reset(self.np_random)

            self.first_move = False
        
        # Here we are differencing the game situations
        # Situation 1: the agent selects an already revealed safe cell
        if self.game.is_safe_cell_discovered(i, j):
            reward = self.R_already_open
            self.info["status"] = "no_progress"
            self.stats["num_no_progress"] += 1
        else:
            # We have to discriminate if we have
            # won, if we lose or if we have to continue to play
            status = self.game.check_game_status(i, j)

            # Situation 2: the agent selects a mine
            if status == -1:

                reward = self.R_lose
                self.done = True
                self.info["status"] = "lost"

                self.stats["n_lose"] += 1
                self.stats["n_episodes"] += 1
            else:
                # Not a mine, game not over
                # We have to distinguish between a random guess
                # or not.
                was_guess = self.game.is_random_guess(i, j)
                # Reveal the selected safe cell and its connected empty region
                self.game.uncover_cell(i, j)
                # Victory must be checked after uncovering the cells
                status = self.game.check_game_status(i, j)
                # Situation 3: all the safe cells have now been revealed: victory!
                if status == 1:
                    reward = self.R_win
                    self.done = True
                    self.info["status"] = "won"
                    self.stats["n_win"] += 1
                    self.stats["n_episodes"] += 1
                # Situation 4: safe cell guessed by chance
                elif was_guess:
                    reward = self.R_guess
                    self.info["status"] = "guess"
                    self.stats["num_guess"] += 1
                # Situation 5: safe cell selected near numbered cell: intelligent progression
                else:
                    reward = self.R_progress
                    self.info["status"] = "progress"
                    self.stats["num_progress"] += 1

        # Preparing the next state and the reward in the format expected by Gymnasium
        observation = np.asarray(self.player_board, dtype=np.int8)
        terminated = self.done
        # Truncated will become True automatically, if a gym.wrapper.TimeLimit
        # is used when creating the environment: for example, when you want
        # to early terminate the episode after a certain number of iterations
        truncated = False

        if self.render_mode == "human":
            self.render()

        return observation, float(reward), terminated, truncated, self.info
    
    def _initialize_pygame(self):
        """ Initialize Pygame window and its resources."""
        # The window is already instantiated
        if self.window is not None:
            return
        # Initializing pygame resources
        pygame.init()
        pygame.display.init()
        pygame.font.init()

        # Setting the window size
        self.window = pygame.display.set_mode(
            (self.window_width, self.window_height)
        )
        self.clock = pygame.time.Clock()
        # Setting the window title
        pygame.display.set_caption(GUI_WINDOW_TITLE)
        # Setting the font size, at least 18 pt.
        font_size = max(18, int(self.cell_size*0.65))
        self.font = pygame.font.Font(
            None, font_size
        )

    def _process_pygame_events(self):
        """ Process Pygame events.
        Returns False when the user closes the window"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False
        return True
    
    def _draw_hidden_cell(self, cell_rect):
        """Draw a covered cell with a simple raised appearance."""

        pygame.draw.rect(
            self.window,
            GUI_HIDDEN_CELL_COLOR,
            cell_rect,
        )

        # Bordo chiaro superiore
        pygame.draw.line(
            self.window,
            GUI_HIDDEN_CELL_HIGHLIGHT_COLOR,
            cell_rect.topleft,
            cell_rect.topright,
            width=3,
        )

        # Bordo chiaro sinistro
        pygame.draw.line(
            self.window,
            GUI_HIDDEN_CELL_HIGHLIGHT_COLOR,
            cell_rect.topleft,
            cell_rect.bottomleft,
            width=3,
        )

        # Bordo scuro inferiore.
        pygame.draw.line(
            self.window,
            GUI_HIDDEN_CELL_SHADOW_COLOR,
            cell_rect.bottomleft,
            cell_rect.bottomright,
            width=3,
        )

        # Bordo scuro destro
        pygame.draw.line(
            self.window,
            GUI_HIDDEN_CELL_SHADOW_COLOR,
            cell_rect.topright,
            cell_rect.bottomright,
            width=3,
        )

    def _draw_mine(self, cell_rect):
        """Draw a mine as an asterisk centered on a red cell."""

        mine_surface = self.font.render(
            "*",
            True,
            GUI_MINE_COLOR,
        )

        mine_rect = mine_surface.get_rect(
            center=cell_rect.center
        )

        self.window.blit(
            mine_surface,
            mine_rect,
        )

    def _draw_number(self, cell_rect, number):
        """Draw a centered Minesweeper number."""

        number_color = GUI_NUMBER_COLORS.get(
            number,
            GUI_GRID_COLOR,
        )

        text_surface = self.font.render(
            str(number),
            True,
            number_color,
        )

        text_rect = text_surface.get_rect(
            center=cell_rect.center
        )

        self.window.blit(
            text_surface,
            text_rect,
        )

    def _draw_cell(self, i, j, cell_value):
        """Draw one cell of the player board."""

        # x e y sono le coordinate della cella sulla griglia di pygame.
        # secondo la logica di pygame, il punto (0,0) è l'angolo in alto a sinistra
        # x è la coordinata che indica lo spostamento nella direzione orizzontale,
        # cosa che viene fatta dall'indice j del nostro environment.
        # di conseguenza, y detta lo spostamento nella direzione verticale, ed è
        # l'indice i di riga che determina la posizione verticale. 
        # il padding serve ad evitare che si disegni la board esattamente
        # da dove inizia il bordo.
        x = self.padding + j * self.cell_size
        y = self.padding + i * self.cell_size

        cell_rect = pygame.Rect(
            x,
            y,
            self.cell_size,
            self.cell_size,
        )

        # Cella ignota
        if cell_value == -2:
            self._draw_hidden_cell(cell_rect)

        # Cella nota, ed è una mina
        elif cell_value == -1:
            pygame.draw.rect(
                self.window,
                GUI_MINE_CELL_COLOR,
                cell_rect,
            )

            self._draw_mine(cell_rect)
        # Cella nota, ma non è una mina
        else:
            pygame.draw.rect(
                self.window,
                GUI_REVEALED_CELL_COLOR,
                cell_rect,
            )

            if cell_value > 0:
                self._draw_number(
                    cell_rect,
                    int(cell_value),
                )

        # Bordo ordinario della cella.
        pygame.draw.rect(
            self.window,
            GUI_GRID_COLOR,
            cell_rect,
            width=1,
        )

        # Evidenzia la cella scelta nell'ultima azione.
        if self.last_action == (i, j):
            pygame.draw.rect(
                self.window,
                GUI_LAST_ACTION_COLOR,
                cell_rect,
                width=4,
            )

    def _draw_board(self):
        """ Draws the game visibile board"""
        for i in range(self.board_height):
            for j in range(self.board_width):
                cell_value = self.player_board[i][j]

                # The triggered mine is shown only by the renderer.
                # It is not inserted into the observation seen by the agent.
                # We do in this way, because this would introduce a mis-match
                # in the number of channels to be used in the tensor to be
                # passed to the q-net.

                # se done è True e se lo stato è lost e l'ultima azione fatta
                # dall'agente è proprio la cella (i,j) che sto renderizzando
                # e la cella che sto renderizzando è una mina
                is_triggered_mine = (
                    self.done
                    and self.info.get("status") == "lost"
                    and self.last_action == (i, j)
                    and self.board[i][j] == -1
                )

                if is_triggered_mine:
                    cell_value = -1

                # cell_value does not represent a value present in the player_board:
                # it's only a temporary value used during the rendering process.
                self._draw_cell(
                i=i,
                j=j,
                cell_value=cell_value,
            )

    def _render_frame(self):
        """Draw and display one frame of the current game state."""
        if self.window_closed:
            return
        self._initialize_pygame()
        # se la finestra è chiusa
        if not self._process_pygame_events():
            return
        self.window.fill(GUI_BACKGROUND_COLOR)
        self._draw_board()
        pygame.display.flip()
        self.clock.tick(
            self.metadata["render_fps"]
        )

    def render(self):
        """Render the current player board in human mode."""

        if self.render_mode is None:
            return None

        if self.render_mode == "human":
            self._render_frame()
            return None

        raise NotImplementedError(
            f"Render mode '{self.render_mode}' is not implemented"
        )
    
    def close(self):
        """Close the Pygame window and release rendering resources."""

        self.window_closed = True

        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

        self.window = None
        self.clock = None
        self.font = None