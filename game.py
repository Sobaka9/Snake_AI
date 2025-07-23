from enums import Direction
from snake import Snake, Cell
from itertools import product
import random
from gui import GUI

class Game:
    def __init__(self, root, board_size: int = 4, framerate: int = 20):
        self.board_size: int = board_size
        mid_pos: tuple[int, int] = (board_size/2, board_size/2)
        self.framerate = framerate
        self.pause = True
        self.snake: Snake = Snake(mid_pos, board_size)
        self.apple: Cell = self._spawn_apple()
        self.root = root
        self.gui: GUI = GUI(root, self)

        self._game_loop()
        self.root.mainloop()

    def handle_key_press(self, direction: Direction):
        """
        """
        if self.pause : return
        self.snake.current_dir = direction

    def _game_loop(self):
        # Calls itself FRAME_RATE times per second
        self.root.after(1000 // self.framerate, self._game_loop)

        if not self.pause:
            self._update(self.snake.current_dir)
        self.gui.draw_board()

    def _update(self, direction: Direction):
        """
        Handles the Game loop :
        - Moves the Snake
        - Spawn the apple when necessary
        - Checks for collision with the borders
        """
        if self.snake.move(
            direction,
            (self.apple.pos_x, self.apple.pos_y)
        ) :
            self.apple = self._spawn_apple()

        # Check collisions with borders
        if self.snake.head.pos_x < 0 or self.snake.head.pos_x > self.board_size - 1 \
            or self.snake.head.pos_y < 0 or self.snake.head.pos_y > self.board_size - 1:
            exit(0)
    
    def _spawn_apple(self) -> list[int, int]:
        """
        """
        # List of all cell positions in the grid
        grid_cells: list = list(product(range(self.board_size), repeat=2))
        snake_cells: list = [(cell.pos_x, cell.pos_y) for cell in self.snake.cells]
        # Substract the Snake cells from the grid cells
        available_cells: list = list(set(grid_cells) - set(snake_cells))

        # Choose a random cell from the available ones
        random_cell: tuple = random.choice(available_cells)

        return Cell(random_cell[0], random_cell[1], 1)

    def update_framerate(self, new_framerate):
        self.framerate = int(new_framerate)

    def reset(self, new_size=None):
        if new_size is not None and new_size > 0:
            self.board_size = new_size
        self.snake = Snake((new_size // 2, new_size // 2), new_size)
        self.apple = self._spawn_apple()
