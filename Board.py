import numpy as np

class Board:
    def __init__(self, parent=None, rows=6, cols=7):
        if parent:
            # Copy the state of the parent board
            self.current_state = np.copy(parent.current_state)
            self.rows, self.cols = parent.rows, parent.cols
            self.available_places = parent.available_places
        else:
            # Initialize a new board with zeros (empty spaces)
            self.rows, self.cols = rows, cols
            self.current_state = np.zeros((rows, cols), dtype=int)
            self.available_places = rows * cols
        self.f = 0  # Heuristic score initialization

    def load(self, state):
        if state is None:
            raise ValueError("State cannot be None.")
        state = np.array(state, dtype=int)  # Ensure it's a NumPy array
        self.current_state = state
        self.available_places = np.count_nonzero(self.current_state == 0)  # Count available places

    def __hash__(self):
        return hash(self.current_state.tobytes())  # Hash based on the state of the board

    def __eq__(self, other):
        return self.f == other.f

    def __lt__(self, other):
        return self.f < other.f

    def __str__(self):
        rows_str = "\n".join(
            "| " + " | ".join(map(str, row)) + " |" for row in self.current_state
        )
        divider = "--" * (self.cols * 2)
        indices = "   " + "     ".join(map(str, range(1, self.cols + 1))) + "  "
        return f"{rows_str}\n{divider}\n{indices}"

    def evaluate(self, window, piece):
        score = 0
        opp_piece = 1 if piece == 2 else 2  # Identify opponent's piece

        scoring = {
            (4, 0): 1000,  # Four in a row
            (3, 1): 10,    # Three in a row with one empty
            (2, 2): 3      # Two in a row with two empty
        }

        # Score for the current player's patterns
        piece_count = window.count(piece)
        empty_count = window.count(0)
        score += scoring.get((piece_count, empty_count), 0)

        # Penalize for opponent's potential patterns
        opp_count = window.count(opp_piece)
        score -= scoring.get((opp_count, empty_count), 0)

        if opp_count == 4:
            score -= 1000

        return score

    def calculate_score(self, piece):
        score = 0
        center_col = self.current_state[:, self.cols // 2]  # check center
        score += np.sum(center_col == piece) * 3

        # Directions to evaluate (horizontal, vertical, diagonal)
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            for r in range(self.rows):
                for c in range(self.cols):
                    if 0 <= r + 3 * dr < self.rows and 0 <= c + 3 * dc < self.cols:
                        window = [
                            self.current_state[r + i * dr, c + i * dc]
                            for i in range(4)
                        ]
                        score += self.evaluate(window, piece)  # Evaluate the window for scoring

        self.f = score  # Update heuristic score
        return score

    def empty_tile(self, col):
        empty_rows = np.where(self.current_state[:, col] == 0)[0]
        return empty_rows[-1] if empty_rows.size > 0 else None  # Return the lowest empty row

    def add_piece(self, col, value):
        row = self.empty_tile(col)
        if row is not None:
            self.current_state[row, col] = value  # Place the piece
            self.available_places -= 1
            return True
        return False

    def remove_piece(self, col):
        filled_rows = np.where(self.current_state[:, col] != 0)[0]
        if filled_rows.size > 0:
            self.current_state[filled_rows[0], col] = 0  # Remove the piece
            self.available_places += 1

    def is_terminal(self): #Check if the board is in a terminal state (no available places left).
        return self.available_places == 0
