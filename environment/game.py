class Game:
    def __init__(
            self,
            board_width: int,
            board_height: int,
            num_mines: int,
        ):
        
        self.board_width = board_width
        self.board_height = board_height
        self.num_mines = num_mines

        # Come campi della classe abbiamo:
        # - board che corrisponde alla soluzione "a carte scoperte"
        # della partita, cosa che l'agent non vede
        # - player_board, è lo stato della partita "a carte coperte"
        # cosa che invece l'agent vede

        # la board conterrà:
        # 0 cella sicura, priva di mine intorno
        # 1..8 cella sicura, con indicato quante mine intorno ad una cella sicura
        # -1 mina
        self.board = []
        # la player board rappresenta l'informazione delle celle non ancora
        # rivelate con -2
        self.player_board = []


    def reset(self, rng):
        """ Resets the game session to a fresh state"""
    
        # Empyting the board first
        self.board = []
        # At reset, all the cells are unrevealed -> 0
        self.player_board = []

        # Attenzione, height ci da il numero di righe
        # mentre width ci da il numero di colonne
        for _ in range(self.board_height):
            row = []
            player_row = []
            for _ in range(self.board_width):
                row.append(0)
                player_row.append(-2)
            self.board.append(row)
            self.player_board.append(player_row)

        # Placing the mines
        self.place_mines(rng)

        # Computing the numbers of mines around
        self.compute_mines_around()
        
        return self.board, self.player_board
    
    def place_mines(self, rng):
        """" Place mines in the board
        A mine is identified into the board with -1
        """
        
        # We have to randomly generate a number of mines
        # equal to num_mines between 0 and (board_width x board_height) - 
        number_of_cells = self.board_width * self.board_height
        # range generates cell ids between 0 and no_cells - 1
        # according to a rng strategy
        generated_mines = rng.choice(number_of_cells, size=self.num_mines, replace=False)

        # Place the mines into the board
        # Note: if you have a matrix MxN and ids ranging from 0 to (MXN)-1
        # to convert ids into coordinates you do
        # i coordinate <- id // N
        # j coordinate <- id % N
        
        for mine_id in generated_mines:
            # converting cell id into a coordinate
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
    
    def is_safe_cell_discovered(self, i, j):
        """ Returns True if the cell, identified by the coordinates (i,j)
        is already known not to be a mine. If the cell is to discover yet,
        the method returns False."""
        if self.player_board[i][j] >= 0:
            return True
        else:
            return False

    def print_board(self):
        """ Utility function to print the board status."""
        print(f"--- Board {self.board_width}x{self.board_height} ({self.num_mines} mines) ---")
        for row in self.board:
            # Priting 'X' for mine (-1) and '.' for empty cells (0)
            rendered_row = [ "X" if cell == -1 else str(cell) for cell in row ]
            print(" ".join(rendered_row))
        print("-" * 40)
    
    def print_player_board(self):
        print(f"--- PLAYER Board {self.board_width}x{self.board_height} ({self.num_mines} mines) ---")
        for row in self.player_board:
            rendered_row = [
                "X" if cell == -1 else ("?" if cell == -2 else str(cell)) 
                for cell in row
            ]
            print(" ".join(rendered_row))
        print("-" * 40)