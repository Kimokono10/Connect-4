from functools import lru_cache
import numpy as np
import math
import matplotlib.pyplot as plt
import networkx as nx
from Board import *
import matplotlib
matplotlib.use('agg')
import random

AI_PIECE = 1
PLAYER_PIECE = 2
ALGORITHMS = ["minimax", "α-β pruning", "expectiminimax"]

class TreeNode:
    def __init__(self, value=0, node_type="Max"):
        self.value = value
        self.node_type = node_type
        self.children = []
        self.parent = None

    def add_child(self, child_node):
        child_node.parent = self  #Add a child node to the current node.
        self.children.append(child_node)

    def __repr__(self, level=0):
        return f"{self.indent(level)}{self.node_type} Node(Value: {self.value})\n" + ''.join(child.__repr__(level + 1) for child in self.children)

    def indent(self, level):
        return "\t" * level

    def build_graph(self, graph, positions=None, x=0, y=0, depth=1, horizontal_spacing=1.0, max_depth=float('inf')):
        positions = positions or {}
        node_id = id(self)
        positions[node_id] = (x, y)

        # Add the current node to the graph
        graph.add_node(node_id, label=f"{self.node_type}\n{self.value}", color=self.get_color())

        # Add edges to child nodes if within depth limits
        if depth < max_depth:
            next_x = x - horizontal_spacing * (len(self.children) - 1) / 2
            for child in self.children:
                graph.add_edge(node_id, id(child))
                child.build_graph(graph, positions, next_x, y - 1, depth + 1, horizontal_spacing / 2, max_depth)
                next_x += horizontal_spacing

        return positions

    def get_color(self):
        color_mapping = {
            "Max": "blue",
            "Min": "red",
            "Chance": "yellow"
        }
        return color_mapping.get(self.node_type, "gray")

    def draw(self, filename="tree.svg", max_depth=3):
        graph = nx.DiGraph()
        pos = self.build_graph(graph, max_depth=max_depth)

        colors = [graph.nodes[node]['color'] for node in graph.nodes()]
        labels = {node: graph.nodes[node]['label'] for node in graph.nodes()}

        plt.figure(figsize=(12, 12))
        nx.draw(graph, pos, labels=labels, node_color=colors, node_size=1500, font_size=5, font_color="white", font_weight='bold', edge_color='gray', linewidths=1.5)

        plt.title("Game Tree Visualization", fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.savefig(filename, format="svg", bbox_inches='tight')
        plt.close()

        return filename

class Solver:
    def __init__(self, depth=4, ai_piece=AI_PIECE, player_piece=PLAYER_PIECE, algorithm="α-β pruning", draw_tree=False):
        self.max_depth = depth
        self.ai_piece = ai_piece
        self.player_piece = player_piece
        self.algorithm = algorithm
        self.draw_tree = draw_tree
        self.tree = TreeNode(value=0, node_type="Max") if draw_tree else None
        self.node_counter = 0

    def search(self, board):

        col = None
        value = None
        self.node_counter = 0

        # Initialize the tree root if drawing is enabled
        if self.tree is None and self.draw_tree:
            self.tree = TreeNode(0, "Max")
        root_node = self.tree if self.draw_tree else None

        if self.algorithm.lower() not in ALGORITHMS:
            raise ValueError(f"Invalid algorithm: {self.algorithm}")

        # choose the appropriate algorithm
        if self.algorithm.lower() == "minimax":
            col, value = self.MiniMax(board, 0, False, root_node)
        elif self.algorithm.lower() == "α-β pruning":
            col, value = self.MiniMax_alpha_beta_pruning(board, 0, -math.inf, math.inf, False, root_node)
        elif self.algorithm.lower() == "expectiminimax":
            col, value = self.ExpectiMiniMax(board, 0, False, root_node)

        print("Num Nodes = ", self.node_counter)
        return col, value

    def add_node(self, parent_node, child_value, child_type="Max"):
        new_node = TreeNode(child_value, child_type) #Add a child node to the tree if tree drawing is enabled.
        parent_node.add_child(new_node)
        return new_node

    def update_node_value(self, node, value):
        if self.draw_tree and node:
            node.value = value

    @lru_cache(maxsize=None)  # Cache the results to avoid redundant calculations and improve performance
    def MiniMax_alpha_beta_pruning(self, board, current_depth, alpha, beta, maximizer_player=False, node=None):
        self.node_counter += 1  #track how many nodes are evaluated

        # Base case
        if current_depth >= self.max_depth or board.available_places == 0:
            evaluation_value = self.eval_board(board)[1]
            self.update_node_value(node, evaluation_value)  # Update the value of the current node
            return None, evaluation_value

        valid_columns = self.find_neighbours(board)  # Get the column where a piece can be placed

        if maximizer_player:
            #Try to maximize the score
            best_col, best_value = None, -math.inf  # Initialize best column and value
            for col in valid_columns:
                board.add_piece(col, self.ai_piece)  # Simulate placing AI's piece in the column
                child_node = self.add_node(node, 0,
                                           "Min") if self.draw_tree else None
                _, current_score = self.MiniMax_alpha_beta_pruning(board, current_depth + 1, alpha, beta, False,child_node)
                board.remove_piece(col)  # Undo the move
                if current_score > best_value:
                    best_value, best_col = current_score, col  # Update if the new score is better
                alpha = max(alpha, best_value)
                if alpha >= beta:
                    break  # Prune the branch if alpha is greater than or equal to beta (Beta cutoff)

            self.update_node_value(node, best_value)
            return best_col, best_value

        else:
            #Try to minimize the score
            best_col, best_value = None, math.inf
            for col in valid_columns:
                board.add_piece(col, self.player_piece)
                child_node = self.add_node(node, 0,
                                           "Max") if self.draw_tree else None
                _, current_score = self.MiniMax_alpha_beta_pruning(board, current_depth + 1, alpha, beta, True,child_node)
                board.remove_piece(col)  # Undo the move
                if current_score < best_value:
                    best_value, best_col = current_score, col
                beta = min(beta, best_value)
                if alpha >= beta:
                    break  # Prune the branch if alpha is greater than or equal to beta (Alpha cutoff)

            self.update_node_value(node, best_value)
            return best_col, best_value

    @lru_cache(maxsize=None)  # Cache the results to avoid redundant calculations and improve performance
    def MiniMax(self, board, current_depth, maximizer_player=False, node=None):
        self.node_counter += 1  # track how many nodes are evaluated

        # Base case
        if current_depth >= self.max_depth or board.available_places == 0:
            evaluation_value = self.eval_board(board)[1]
            self.update_node_value(node, evaluation_value)
            return None, evaluation_value

        valid_columns = self.find_neighbours(board)

        if maximizer_player:
            #Try to maximize the score
            best_col, best_value = None, -math.inf
            for col in valid_columns:
                board.add_piece(col, self.ai_piece)
                child_node = self.add_node(node, 0, "Min") if self.draw_tree else None
                _, current_score = self.MiniMax(board, current_depth + 1, False,child_node)
                board.remove_piece(col)  # Undo the move
                if current_score > best_value:
                    best_value, best_col = current_score, col

            self.update_node_value(node, best_value)  # Update the value of the current node with the best value
            return best_col, best_value

        else:
            #  Try to minimize the score
            best_col, best_value = None, math.inf
            for col in valid_columns:
                board.add_piece(col, self.player_piece)
                child_node = self.add_node(node, 0,"Max") if self.draw_tree else None
                _, current_score = self.MiniMax(board, current_depth + 1, True,child_node)
                board.remove_piece(col)
                if current_score < best_value:
                    best_value, best_col = current_score, col

            self.update_node_value(node, best_value)
            return best_col, best_value

    @lru_cache(maxsize=None)
    def ExpectiMiniMax(self, board, current_depth, maximizer_player=False, node=None):
        self.node_counter += 1  # Increment the node counter for each function call

        # Base case
        if current_depth >= self.max_depth or board.available_places == 0:
            evaluation_value = self.eval_board(board)[1]
            self.update_node_value(node, evaluation_value)
            return None, evaluation_value

        cols = self.find_neighbours(board)

        if maximizer_player:
            #  Try to maximize the expected value
            max_value, best_col = -math.inf, None
            for col in cols:
                expected_value = 0  # Initialize expected value for the current move
                columns, probabilities = self.find_columns(board, col)  # Get adjacent columns and probabilities
                child_node = self.add_node(node, 0,
                                           "Chance") if self.draw_tree else None

                for idx, col_to_simulate in enumerate(columns):
                    board.add_piece(col_to_simulate, self.ai_piece)
                    _, value = self.ExpectiMiniMax(board, current_depth + 1, False,child_node)
                    board.remove_piece(col_to_simulate)
                    expected_value += probabilities[idx] * value  # Update expected value using probability of the move

                if expected_value > max_value:
                    max_value, best_col = expected_value, col

            self.update_node_value(node, max_value)
            return best_col, max_value

        else:
            #  Try to minimize the expected value
            min_value, best_col = math.inf, None
            for col in cols:
                expected_value = 0
                columns, probabilities = self.find_columns(board, col)
                child_node = self.add_node(node, 0,"Chance") if self.draw_tree else None

                for idx, col_to_simulate in enumerate(columns):
                    board.add_piece(col_to_simulate, self.player_piece)
                    _, value = self.ExpectiMiniMax(board, current_depth + 1, True, child_node)
                    board.remove_piece(col_to_simulate)
                    expected_value += probabilities[idx] * value

                if expected_value < min_value:
                    min_value, best_col = expected_value, col

            self.update_node_value(node, min_value)
            return best_col, min_value

    def eval_board(self, board):
        # Evaluate the board's state and return a score for AI or Player
        if board.available_places == 0:
            ai_wins = self.count_wins(board.current_state, self.ai_piece)  # Count AI's winning sequences
            player_wins = self.count_wins(board.current_state, self.player_piece)  # Count player's winning sequences

            if ai_wins > player_wins:
                return None, math.inf  # AI wins
            elif player_wins > ai_wins:
                return None, -math.inf  # Player wins
            return None, 0  # Draw

        return None, board.calculate_score(self.ai_piece)

    def find_neighbours(self, board):
        # Find valid columns where a piece can be placed and sort them based on potential move evaluation
        valid_columns = [c for c in range(board.cols) if board.empty_tile(c) is not None]  # Check columns with empty spaces
        return sorted(valid_columns, key=lambda col: self.eval_move(board, col),reverse=True)  # Sort columns by evaluation

    def eval_move(self, board, col):
        # Evaluate the potential score for a move in a specific column
        simulated_board = Board(board)  # Create a copy of the board to simulate the move
        simulated_board.add_piece(col, self.ai_piece)
        _, potential_score = self.eval_board(simulated_board)  # Evaluate the board after the move
        return potential_score

    def find_columns(self, board, col):
        # Find adjacent columns to simulate possible moves based on current position
        adjacent_columns = [col + offset for offset in (-1, 0, 1)if 0 <= col + offset < board.cols and board.empty_tile(col + offset) is not None]
        probabilities = {
            3: [0.2, 0.6, 0.2],  # Probabilities for 3 adjacent columns
            2: [0.6, 0.4] if adjacent_columns[0] == col else [0.4, 0.6],  # Probabilities for 2 adjacent columns
            1: [1.0]  # Probability for 1 adjacent column
        }
        return adjacent_columns, probabilities[
            len(adjacent_columns)]



    def count_wins(self, board, piece):
        # Count the number of winning sequences for the given piece (AI or Player)
        rows, cols = board.shape
        sequence_count = 0
        length = 4

        # Check for horizontal and vertical sequences
        for row in range(rows):
            for col in range(cols):
                if col <= cols - length:
                    sequence_count += all(board[row, col + i] == piece for i in range(length))  # Horizontal check
                if row <= rows - length:
                    sequence_count += all(board[row + i, col] == piece for i in range(length))  # Vertical check

        # Check for diagonal (\ and /) sequences
        for row in range(rows - length + 1):
            for col in range(cols - length + 1):
                sequence_count += all(board[row + i, col + i] == piece for i in range(length))  # Diagonal (\) check
            for col in range(length - 1, cols):
                sequence_count += all(board[row + i, col - i] == piece for i in range(length))  # Diagonal (/) check

        return sequence_count