import pygame
import sys
from Board import Board
from Solver import Solver
import time
import random


SCREEN_WIDTH = 740
SCREEN_HEIGHT = 735
CIRCLE_RADIUS = 50
MARGIN = 5
BACKGROUND_COLOR = (30, 30, 60)
GRID_COLOR = (20, 20, 50)
EMPTY_COLOR = (0, 0, 0)
PLAYER_COLOR = (220, 20, 60)
AI_COLOR = (255, 215, 0)
FONT_COLOR = (255, 255, 255)
BUTTON_COLOR = (50, 50, 100)
HOVER_COLOR = (70, 70, 120)
FPS = 60

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Connect Four")
font = pygame.font.SysFont("Arial", 24)
large_font = pygame.font.SysFont("Arial", 50, bold=True)
clock = pygame.time.Clock()


def draw_grid(board):
    screen.fill(BACKGROUND_COLOR)

    for row in range(board.rows):
        for col in range(board.cols):
            pygame.draw.rect(
                screen,
                GRID_COLOR,
                (col * (CIRCLE_RADIUS * 2 + MARGIN) + MARGIN,
                 row * (CIRCLE_RADIUS * 2 + MARGIN) + MARGIN + CIRCLE_RADIUS * 2,
                 CIRCLE_RADIUS * 2, CIRCLE_RADIUS * 2),
            )
            color = EMPTY_COLOR
            if board.current_state[row, col] == 1:
                color = PLAYER_COLOR
            elif board.current_state[row, col] == 2:
                color = AI_COLOR

            pygame.draw.circle(
                screen,
                color,
                (
                    col * (CIRCLE_RADIUS * 2 + MARGIN) + CIRCLE_RADIUS + MARGIN,
                    row * (CIRCLE_RADIUS * 2 + MARGIN) + CIRCLE_RADIUS + MARGIN + CIRCLE_RADIUS * 2,
                ),
                CIRCLE_RADIUS,
            )
    pygame.display.update()


def draw_header(message="Your Turn!"):
    header_rect = pygame.Rect(0, 0, SCREEN_WIDTH, CIRCLE_RADIUS * 2)
    pygame.draw.rect(screen, BACKGROUND_COLOR, header_rect)
    text = large_font.render(message, True, FONT_COLOR)
    text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, CIRCLE_RADIUS))
    screen.blit(text, text_rect)
    pygame.display.update()


def get_column_from_mouse(pos_x):
    return pos_x // (CIRCLE_RADIUS * 2 + MARGIN)


def display_message(message):
    overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    overlay.set_alpha(200)
    overlay.fill((0, 0, 0))
    screen.blit(overlay, (0, 0))
    text = large_font.render(message, True, FONT_COLOR)
    text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
    screen.blit(text, text_rect)
    pygame.display.update()
    pygame.time.wait(3000)


def draw_button(x, y, width, height, text, is_hovered):
    color = HOVER_COLOR if is_hovered else BUTTON_COLOR
    pygame.draw.rect(screen, color, (x, y, width, height))
    text_surface = font.render(text, True, FONT_COLOR)
    text_rect = text_surface.get_rect(center=(x + width // 2, y + height // 2))
    screen.blit(text_surface, text_rect)


def choose_algorithm():
    algorithms = ["Minimax", "Minimax with Pruning", "Expected Minimax"]
    button_width = 200
    button_height = 50
    padding = 20
    selected_algorithm = None

    while selected_algorithm is None:
        screen.fill(BACKGROUND_COLOR)

        for i, algo in enumerate(algorithms):
            button_x = (SCREEN_WIDTH - button_width) // 2
            button_y = (SCREEN_HEIGHT - button_height) // 2 + (i * (button_height + padding))
            is_hovered = pygame.mouse.get_pos()[0] in range(button_x, button_x + button_width) and pygame.mouse.get_pos()[1] in range(button_y, button_y + button_height)
            draw_button(button_x, button_y, button_width, button_height, algo, is_hovered)

        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                for i, algo in enumerate(algorithms):
                    button_x = (SCREEN_WIDTH - button_width) // 2
                    button_y = (SCREEN_HEIGHT - button_height) // 2 + (i * (button_height + padding))
                    if button_x <= mouse_x <= button_x + button_width and button_y <= mouse_y <= button_y + button_height:
                        selected_algorithm = algo
                        break

    return selected_algorithm


def probabilty_expectiminimax(col, num_columns=7):

    if col == 0:  # Left edge
        probability_distribution = [0.6, 0.4]
        columns = [col, col + 1]  # Can only go to the right
    elif col == num_columns - 1:  # Right edge
        probability_distribution = [0.4, 0.6]
        columns = [col - 1, col]  # Can only go to the left
    else:  # Middle columns
        probability_distribution = [0.2, 0.6, 0.2]
        columns = [col - 1, col, col + 1]  # Can go left, stay, or go right

    selected_col = random.choices(columns, probability_distribution)[0]
    print(f"Probabilistic selection: From column {col}, selected column {selected_col}")
    return selected_col

def main():
    algorithm = choose_algorithm()
    if algorithm == "Minimax":
        solver = Solver(depth=6, algorithm="minimax", draw_tree=False)
    elif algorithm == "Minimax with Pruning":
        solver = Solver(depth=8, algorithm="α-β pruning", draw_tree=False)
    else:
        solver = Solver(depth=8, algorithm="expectiminimax", draw_tree=False)

    board = Board()
    running = True
    turn = 1

    draw_grid(board)
    draw_header("Your Turn!")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                sys.exit()


            if turn % 2 != 0 and event.type == pygame.MOUSEBUTTONDOWN:  # Player's turn
                pos_x = event.pos[0]
                col = get_column_from_mouse(pos_x)

                if algorithm == "Expected Minimax":
                    col = probabilty_expectiminimax(col)

                if board.empty_tile(col) is not None:
                    board.add_piece(col, 1)
                    draw_grid(board)

                    if board.is_terminal():
                        running = False
                        break

                    turn += 1

        if turn % 2 == 0 and running:  # AI's turn
            draw_header("AI's Turn...")
            pygame.time.wait(1000)

            start_time = time.time()
            col, val = solver.search(board)
            end_time = time.time()
            move_time = end_time - start_time

            if col is not None and board.empty_tile(col) is not None:
                board.add_piece(col, 2)
                draw_grid(board)
                if solver.draw_tree:
                    filename = f"tree_{turn}.svg"
                    solver.tree.draw(filename)

                print("AI's Move:")
                print(board)
                print(f"Time taken: {move_time:.2f} seconds")

                if board.is_terminal():
                    running = False
                    break

                turn += 1
                draw_header("Your Turn!")

    player_fours = solver.count_wins(board.current_state, 1.0)
    ai_fours = solver.count_wins(board.current_state, 2.0)

    if player_fours > ai_fours:
        display_message(f"Player Wins! Fours: {player_fours} vs {ai_fours}")
    elif ai_fours > player_fours:
        display_message(f"AI Wins! Fours: {ai_fours} vs {player_fours}")
    else:
        display_message(f"It's a Tie! Fours: {player_fours} vs {ai_fours}")
    pygame.time.wait(10000)
    pygame.quit()


if __name__ == "__main__":
    main()
