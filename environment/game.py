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
        # Empyting the board first
        self.board = []
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
    
    def place_mines(self):
        """" Place mines in the board
        A mine is identified into the board with -1
        """
        
        # We have to randomly generate a number of mines
        # equal to num_mines between 0 and (board_width x board_height) - 
        number_of_cells = self.board_width * self.board_height
        # range generates cell ids between 0 and no_cells - 1
        generated_mines = random.sample(range(number_of_cells), self.num_mines)

        # Place the mines into the board
        # Note: if you have a matrix MxN and ids ranging from 0 to (MXN)-1
        # to convert ids into coordinates you do
        # i coordinate <- id // N
        # j coordinate <- id % N
        
        for mine_id in generated_mines:
            # converting cell id into a coordinate
            i = mine_id % self.board_width
            j = mine_id // self.board_width

            self.board[i][j] = -1
    
    def compute_mines_around(self):
        # TODO: implement the function which counts for how many mines
        # are around a cell
        pass
    
    def print_board(self):
        """ Utility function to print the board status."""
        print(f"--- Board {self.board_width}x{self.board_height} ({self.num_mines} mines) ---")
        for row in self.board:
            # Priting 'X' for mine (-1) and '.' for empty cells (0)
            rendered_row = [ "X" if cell == -1 else "." for cell in row ]
            print(" ".join(rendered_row))
        print("-" * 40)