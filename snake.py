from enums import Direction

class Cell:
    def __init__(self, pos_x, pos_y, value = 0):
        # 0 --> EMPTY | 1 --> APPLE | 2 --> HEAD
        # board_size^2 --> TAIL
        # The Snake's cells have a descending value 3 --> board_size^2 - 1
        self.pos_x: int = pos_x
        self.pos_y: int = pos_y
        self.value: int = value

class Snake:
    def __init__(self, position: tuple[int, int], board_size: int):
        self.cells: list[Cell] = []
        self.head: Cell = None
        self.tail: Cell = None
        self.current_dir: Direction = Direction.LEFT
        self.length: int = 0
        self.head_value: int = 2
        self.tail_value: int = board_size ** 2

        self._initialize_snake(position)

    def _initialize_snake(self, position: tuple[int, int]):
        pos_x: int = position[0]
        pos_y: int = position[1]
        
        head: Cell = Cell(pos_x, pos_y, self.head_value)
        self.head = head

        self.cells.append(head)
        self.length = 1

    def handle_key_press(self, direction: Direction):
        """Tells the game logic to move the player in a given direction."""
        self.snake.current_dir = direction

    def move(self, direction: Direction, apple_position: tuple[int, int]) -> bool:
        has_eaten_apple: bool = False
        dx: int = 0
        dy: int = 0

        match direction:
            case Direction.LEFT:
                dx = -1
                dy = 0
            case Direction.RIGHT:
                dx = 1
                dy = 0
            case Direction.DOWN:
                dx = 0
                dy = 1
            case Direction.UP:
                dx = 0
                dy = -1
            case _:
                print("Unknown Direction")
                exit(1)

        # The head's new position
        new_pos_x: int = self.head.pos_x + dx
        new_pos_y: int = self.head.pos_y + dy
        
        # Self collision ?
        snake_cells: list = [(cell.pos_x, cell.pos_y) for cell in self.cells]
        if (new_pos_x, new_pos_y) in snake_cells:
            exit(0)

        new_head: Cell = Cell (
            new_pos_x,
            new_pos_y,
            self.head_value
        )
        self.cells.append(new_head)
        self.head = new_head
        
        # Apple collision ?
        if apple_position == (new_pos_x, new_pos_y):
            self.length += 1
            has_eaten_apple = True
            
        # Remove the tail of the Snake unless the apple was eaten
        if self.length < len(self.cells):
            self.cells.pop(0)
            self.tail = self.cells[0]

        # Update the body's cell value
        for i, cell in enumerate(self.cells[-2:0:-1]):
            cell.value = 3 + i
        self.cells[0].value = self.tail_value

        return has_eaten_apple
