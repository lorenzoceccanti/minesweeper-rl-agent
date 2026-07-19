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
        # The player board contains only observable information:
        # -2 for unrevealed cells and 0..8 for revealed safe cells.
        # Mines are never stored in the player board.
        self.player_board = []


        # Il gioco mantiene un contatore del numero delle celle scoperte
        self.opened_cells = 0


    def reset(self, rng):
        """ Resets the game session to a fresh state"""
    
        # Empyting the board first
        self.board = []
        # At reset, all the cells are unrevealed -> -2
        self.player_board = []
        self.opened_cells = 0

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
        """ Place mines in the board
        A mine is identified into the board with -1
        """
        
        # we have to randomly generate a number of mine positions
        # equal to num_mines, in the interval between 0 and 
        # (board_width x board_height) - 1
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
    
    def check_boundaries(self, i, j):
        """ Return True if coordinates (i, j) are inside the board."""
        return (
            0 <= i < self.board_height
            and 0 <= j < self.board_width
        )

    def increment_neighbors(self, neigh_i, neigh_j):
        """ Utility function used by compute_mines_around
        to increment the counter of neighboring mines"""
        # The position of the neighbor is given
        # We check if it's a mine
        if not self.check_boundaries(neigh_i, neigh_j):
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
    
    def is_random_guess(self, i, j):
        # This method has to access player_board only. The reward will
        # depend on the board status available for the agent
        """ Return True if all valid neighboring cells are still hidden (i.e. the
        agent makes a random guess)."""

        if not self.check_boundaries(i, j):
            raise IndexError("Cell coordinates are outside the board")

        for off_i in (-1, 0, 1):
            for off_j in (-1, 0, 1):
                if off_i == 0 and off_j == 0:
                    continue

                neighbor_i = i + off_i
                neighbor_j = j + off_j

                if not self.check_boundaries(neighbor_i, neighbor_j):
                    continue
                # Arrivati a questo punto, se trovo un vicino numerato
                # mi fermo subito: sicuramente l'azione non è stato
                # frutto di un random guess
                if self.player_board[neighbor_i][neighbor_j] >= 0:
                    return False
        return True
            
    
    def check_game_status(self, i, j):
        """ Given an action as coordinate of a cell, the method determines
        whether the game is ended, is won or still has to continue.
        Returns:
        -1:  for game lost
        1:  for game won
        0: for game to still continue"""

        num_safe_cells = (self.board_height * self.board_width) - self.num_mines
        # Losing scenario: finding a mine
        if self.board[i][j] == -1:
            return -1
        # Winning scenario
        if self.opened_cells == num_safe_cells:
            return 1
        else:
            return 0
        
    def uncover_cell(self, i, j) -> int:
        """
        Reveal a safe cell and, if it is empty, its connected empty region (Flood Fill). 
        The method returns the number of newly revealed cells.
        """
        if not self.check_boundaries(i, j):
            raise IndexError("Cell coordinates are outside the board")

        if self.board[i][j] == -1:
            raise ValueError("uncover_cell must be called only on safe cells")
        
        # number of cells revealed during the current action.
        # this will correspond to delta_n_t in the reward function.
        newly_opened_cells = 0

        # Important: the temptation to write this function using a recursive approach
        # was very high. So, we reasoned about the usage of a stack, in which
        # the next neighbouring coordinate cells to visit are stored

        # At the beginning, only the current cell has to be visited.
        cells_to_visit = [(i, j)]

        while cells_to_visit:
            # Pick the next neighboring cell from the stack
            current_i, current_j = cells_to_visit.pop()

            # The cell has already been revelead
            if self.player_board[current_i][current_j] != -2:
                continue

            # Ensuring not to reveal a mine during the uncovering process
            if self.board[current_i][current_j] == -1:
                continue
                
            # Revealing the current safe cell (the current safe cell for
            # how the code is implemented could be both a clicked cell or
            # a neighbor of a 0 cell)
            self.player_board[current_i][current_j] = self.board[current_i][current_j]
            # Total number of cells revealed during the entire game
            self.opened_cells += 1
            # Number of cells revealed during the current action
            newly_opened_cells += 1

            # If the just revealed cell has a number > 0 (mines around), stop
            # current the iteration here. We do not propagate the lookup, and consequently
            # nothing is added to the stack in this iteration.
            if self.board[current_i][current_j] > 0:
                continue

            # If we arrive at this point, the current cell is 0.
            # We inspect all the neighbors, and we'll have something to add in
            # the stack
            for off_i in (-1, 0, 1):
                for off_j in (-1, 0, 1):
                    # Skipping the cell itself
                    if off_i == 0 and off_j == 0:
                        continue
                    neighbor_i = current_i + off_i
                    neighbor_j = current_j + off_j

                    # Same checks made on the start
                    if not self.check_boundaries(neighbor_i, neighbor_j):
                        continue

                    if self.player_board[neighbor_i][neighbor_j] != -2:
                        continue

                    if self.board[neighbor_i][neighbor_j] == -1:
                        continue
                    
                    # Adding to the stack a neighbor
                    cells_to_visit.append((neighbor_i, neighbor_j))
        return newly_opened_cells
    

    # Debug methods to print the board in cli
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
                "?" if cell == -2 else str(cell)
                for cell in row
            ]
            print(" ".join(rendered_row))
        print("-" * 40)