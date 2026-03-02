from enum import Enum
from games.game import Game

import random
import numpy as np

class Direction(Enum):
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

class Cell:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

class SnakeGame(Game):
    """
    Snake game environment for reinforcement learning.

    State representation:
    - Danger straight (1 or 0)
    - Danger right (1 or 0)
    - Danger left (1 or 0)
    - Move direction (one-hot: right, down, left, up)
    - Food location (one-hot: food left, food right, food up, food down

    Action space:
    - 0: Straight
    - 1: Turn right
    - 2: Turn left

    Rewards:
    - +10 for eating food
    - -10 for collision (game over)
    - +0.1 for each step taken (to encourage shorter paths to food)
    """
    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height
        self.reset()

    # --- API ---

    def reset(self):
        """Reset the game to its initial state."""
        cx, cy = self.width // 2, self.height // 2
        self.direction: Direction = Direction.RIGHT

        self.head: Cell = Cell(cx, cy)
        self.snake: list[Cell] = [self.head, Cell(cx - 1, cy), Cell(cx - 2, cy)]

        self.score: int = 0
        self.steps: int = 0
        self.steps_since_food: int = 0
        
        self._place_food()
        return self._get_state()

    def step(self, action: Direction):
        """Apply the given action

        Args:
            action (Direction)

        Returns:
            (state, reward, done, score)
        """
        
        self.steps += 1
        self.steps_since_food += 1

        reward: float = 0
        done: bool = False

        head: Cell = self.head

        # Update direction based on action
        if action == Direction.RIGHT.value and self.direction != Direction.LEFT.value:
            self.direction = Direction.RIGHT
        elif action == Direction.DOWN.value and self.direction != Direction.UP.value:
            self.direction = Direction.DOWN
        elif action == Direction.LEFT.value and self.direction != Direction.RIGHT.value:
            self.direction = Direction.LEFT
        elif action == Direction.UP.value and self.direction != Direction.DOWN.value:
            self.direction = Direction.UP

        # Move the snake
        x, y = head.x, head.y
        if self.direction == Direction.RIGHT:
            x += 1
        elif self.direction == Direction.DOWN:
            y += 1
        elif self.direction == Direction.LEFT:
            x -= 1
        elif self.direction == Direction.UP:
            y -= 1
        
        new_head: Cell = Cell(x, y)

        # Check for collisions
        if self._is_collision(new_head):
            reward = -10
            done = True
            return self._get_state(), reward, done, {"score": self.score} # Game over

        # Move snake
        self.snake.insert(0, new_head)
        self.head = new_head

        # Check for food consumption
        if new_head.x == self.food.x and new_head.y == self.food.y:
            reward = 10
            self.score += 1
            self.steps_since_food = 0
            self._place_food()
        else:
            self.snake.pop()

        if self._distance(head, self.food) > self._distance(self.head, self.food):
            reward += 0.1  # Encourage moving towards food

        return self._get_state(), reward, done, {"score": self.score}

    def get_render_data(self):
        """Return data necessary for rendering the game."""
        return {
            'snake': [cell for cell in self.snake],
            'food': self.food,
            'score': self.score,
            'direction': self.direction,
            'width': self.width,
            'height': self.height,
        }

    # --- Internal methods ---

    def _place_food(self):
        """Place food in a random cell that is not occupied by the snake."""
        free_cells: list[Cell] = \
        [Cell(x, y) for x in range(self.width) for y in range(self.height) if Cell(x, y) not in self.snake]
        
        self.food: Cell = random.choice(free_cells)

    def _is_collision(self, cell: Cell) -> bool:
        """Check if the given cell collides with the snake or the walls."""
        x, y = cell.x, cell.y
        return (
            x < 0 or x >= self.width or
            y < 0 or y >= self.height or
            cell in self.snake
        )

    def _get_state(self):
        """Return the current state of the game."""

        head: Cell = self.head
        cell_left: Cell = Cell(head.x - 1, head.y)
        cell_right: Cell = Cell(head.x + 1, head.y)
        cell_up: Cell = Cell(head.x, head.y - 1)
        cell_down: Cell = Cell(head.x, head.y + 1)

        direction_left = self.direction == Direction.LEFT
        direction_right = self.direction == Direction.RIGHT
        direction_up = self.direction == Direction.UP
        direction_down = self.direction == Direction.DOWN

        state = [
            # Danger straight
            (direction_right and self._is_collision(cell_right)) or
            (direction_down and self._is_collision(cell_down)) or
            (direction_left and self._is_collision(cell_left)) or
            (direction_up and self._is_collision(cell_up)),

            # Danger right
            (direction_right and self._is_collision(cell_down)) or
            (direction_down and self._is_collision(cell_left)) or
            (direction_left and self._is_collision(cell_up)) or
            (direction_up and self._is_collision(cell_right)),

            # Danger left
            (direction_right and self._is_collision(cell_up)) or
            (direction_down and self._is_collision(cell_right)) or
            (direction_left and self._is_collision(cell_down)) or
            (direction_up and self._is_collision(cell_left)),

            # Move direction
            direction_right,
            direction_down,
            direction_left,
            direction_up,

            # Food location
            self.food.x < head.x,  # food left
            self.food.x > head.x,  # food right
            self.food.y < head.y,  # food up
            self.food.y > head.y   # food down
        ]

        return np.array(state, dtype=np.float32)

    def _distance(self, cell1: Cell, cell2: Cell) -> float:
        """Calculate Manhattan distance between two cells."""
        return abs(cell1.x - cell2.x) + abs(cell1.y - cell2.y)

    @property
    def state_size(self):
        """Return the size of the game state."""
        return 11  # 3 danger + 4 direction + 4 food location
    
    @property
    def action_size(self):
        """Return the size of the action space."""
        return 4  # [straight, right, left, up]
