from enums import Direction
import tkinter as tk

class GUI:
    """
    Manages the Tkinter GUI for the game.
    """
    WINDOW_SIZE = 800

    def __init__(self, root, game):
        self.root = root
        self.game = game
        self.root.title("Ai Snake")
        self.root.geometry('%dx%d+%d+%d' % (
            self.WINDOW_SIZE + 20,
            self.WINDOW_SIZE + 150,
            (1920 - self.WINDOW_SIZE) / 2,
            (1080 - (self.WINDOW_SIZE + 150 + 40)) / 2)
        )
        self.root.resizable(False, False)
        self.grid_visible = tk.BooleanVar(value=False)
        self.cell_size = self.WINDOW_SIZE // self.game.board_size

        self.canvas = tk.Canvas(
            self.root,
            width=self.game.board_size * self.cell_size,
            height=self.game.board_size * self.cell_size,
            bg='black',
            highlightthickness=0
        )
        self.canvas.pack(pady=10, padx=10)

        controls_frame = tk.Frame(self.root, padx=10, pady=10, bg='#f0f0f0')
        controls_frame.pack(fill='x', padx=10, pady=(0, 10))
        self._create_controls(controls_frame)

        self.draw_board()

        # Bind keyboard events
        self.root.bind('<Left>', lambda event: self.game.handle_key_press(Direction.LEFT))
        self.root.bind('<Right>', lambda event: self.game.handle_key_press(Direction.RIGHT))
        self.root.bind('<Up>', lambda event: self.game.handle_key_press(Direction.UP))
        self.root.bind('<Down>', lambda event: self.game.handle_key_press(Direction.DOWN))
        self.root.bind('<Escape>', lambda event: setattr(self.game, 'pause', not self.game.pause))

    def _create_controls(self, parent_frame):
        """Creates all the GUI widgets in the control panel."""
        parent_frame.columnconfigure(1, weight=1)
        tk.Label(parent_frame, text="Board Size:", bg='#f0f0f0').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        size_frame = tk.Frame(parent_frame, bg='#f0f0f0')
        size_frame.grid(row=0, column=1, sticky='ew')

        self.board_size_var = tk.StringVar(value=str(self.game.board_size))
        width_entry = tk.Entry(size_frame, textvariable=self.board_size_var, width=5)
        width_entry.pack(side='left')
        apply_size_button = tk.Button(size_frame, text="Apply", command=self.update_board_size)
        apply_size_button.pack(side='left', padx=5)

        self.game_framerate = tk.StringVar(value=str(self.game.framerate))
        tk.Label(parent_frame, text="Speed:", bg='#f0f0f0').grid(row=1, column=0, sticky='w', padx=5, pady=2)
        speed_scale = tk.Scale(parent_frame, from_=4, to=200, orient='horizontal', bg='#f0f0f0', highlightthickness=0, command=self.game.update_framerate)
        speed_scale.set(self.game.framerate)
        speed_scale.grid(row=1, column=1, sticky='ew')

        tk.Label(parent_frame, text="Draw Grid:", bg='#f0f0f0').grid(row=2, column=0, sticky='w', padx=5, pady=2)
        grid_check = tk.Checkbutton(parent_frame, variable=self.grid_visible, command=self.draw_board, bg='#f0f0f0')
        grid_check.grid(row=2, column=1, sticky='w')

    def update_board_size(self):
        try:
            new_size = int(self.board_size_var.get())
            if new_size > 0:
                self.game.reset(new_size)
                self.cell_size = self.WINDOW_SIZE // new_size
                new_canvas_dim = self.game.board_size * self.cell_size
                self.canvas.config(width=new_canvas_dim, height=new_canvas_dim)
                self.draw_board()
        except ValueError:
            print("Invalid board size. Please enter a whole number.")

    def draw_board(self):
        self.canvas.delete("all")

        # Draw Grid
        if self.grid_visible.get():
            for i in range(1, self.game.board_size):
                self.canvas.create_line(i * self.cell_size, 0, i * self.cell_size, self.game.board_size * self.cell_size, fill='gray20')
                self.canvas.create_line(0, i * self.cell_size, self.game.board_size * self.cell_size, i * self.cell_size, fill='gray20')
        
        # Draw Apple
        self.draw_cell(self.game.apple.pos_x, self.game.apple.pos_y, 'red')

        # Draw Snake
        self.draw_snake()

    def draw_snake(self):
        base_color: tuple[int, int, int] = (128, 0, 255)
        color_step = 1 / len(self.game.snake.cells)
        for i, cell in enumerate(self.game.snake.cells[::-1]) :
            new_color = (
                int(base_color[0]),
                int(base_color[1] + (255 * i * color_step)),
                int(base_color[2])
            )
            # Converts the int tuple into a hex string
            new_color = '#%02x%02x%02x' % new_color
            self.draw_cell(cell.pos_x, cell.pos_y, new_color)

    def draw_cell(self, col, row, color):
        x1 = col * self.cell_size
        y1 = row * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color)
