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
            i = mine_id // self.board_width
            j = mine_id % self.board_width

            self.board[i][j] = -1
    
    def increment_neighbors(self, neigh_i, neigh_j):
        """ Utility function used by compute_mines_around
        to increment the counter of neighboring mines"""
        # The position of the neighbor is given
        # We check if it's a mine
        if neigh_i < 0 or neigh_i >= self.board_height:
            return
        if neigh_j < 0 or neigh_j >= self.board_width:
            return
        
        if self.board[neigh_i][neigh_j] == -1:
            return
        else:
            self.board[neigh_i][neigh_j] += 1

    def compute_mines_around(self):
        # Identify a mine in the board
        # Once the mine is identified, we increment the number by one
        # in all the cell around, exception made for the mines
        for i in range(self.board_height):
            for j in range(self.board_width):
                if self.board[i][j] == -1:
                    for ni in [-1, 0, 1]:
                        for nj in [-1, 0, 1]:
                            # ni and nj range in the discrete interval [-1, 1]
                            # this is a trick to avoid calling increment_neighbors 8 times
                            if ni == 0 and nj == 0:
                                continue # we obviously don't involve the mine itself in the increment
                            self.increment_neighbors(i + ni, j + nj)
    
    def print_board(self):
        """ Utility function to print the board status."""
        print(f"--- Board {self.board_width}x{self.board_height} ({self.num_mines} mines) ---")
        for row in self.board:
            # Priting 'X' for mine (-1) and '.' for empty cells (0)
            rendered_row = [ "X" if cell == -1 else str(cell) for cell in row ]
            print(" ".join(rendered_row))
        print("-" * 40)