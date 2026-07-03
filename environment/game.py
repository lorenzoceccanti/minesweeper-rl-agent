import random

class Game:
    def __init__(
            self,
            board_width: int,
            board_height: int,
            num_mines: int,
            seed: int = None):
        
        self.board_width = board_width
        self.board_height = board_height
        self.num_mines = num_mines
        self.seed = seed

        # As field of the class, we'll have also
        # a board object, storing the state
        # of the cells during a game session
        # in particular, if the cell is unrevealed
        # or if it's revealed the number of mines around

        self.board = []

        self.reset()

    def reset(self):
        """ Resets the game session to a fresh state"""
        # Setting the seed, if specified
        if self.seed is not None:
            random.seed(self.seed)
        # Allocating the space for the board
        # At reset, all the cells are unrevealed -> 0

        # Attenzione, height ci da il numero di righe
        # mentre width ci da il numero di colonne
        for _ in range(self.board_height):
            row = []
            for _ in range(self.board_width):
                row.append(0)
            self.board.append(row)

        # Placing the mines
        self.place_mines()

        # Computing the numbers of mines around
        self.compute_mines_around()