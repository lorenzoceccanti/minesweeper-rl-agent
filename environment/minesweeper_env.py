# We're creating a customized environment
# To do so, according to the Gymnasium documentation, we need to extend the class
# gym.Env

import gymnasium as gym

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

        self.action_space = gym.space.Discrete(
            self.board_height * self.board_width
        )

        self.action_space = gym.spaces.Discrete([self.board_height, self.board_width])

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
                dtype=int
        )

        # TODO: remember to define the reward function