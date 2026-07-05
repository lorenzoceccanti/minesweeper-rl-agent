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
        # -1: revealed mined cell
        # 0..8: a safe cell, with the number of mines around

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

        self.game = Game(board_width=self.board_width, board_height=self.board_height, num_mines=self.n_mines)
    
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

        return np.asarray(self.player_board, dtype=np.int8), self.info

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
                self.player_board[i][j] = -1

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
        return observation, float(reward), terminated, truncated, self.info