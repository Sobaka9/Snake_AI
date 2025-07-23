import tkinter as tk
from game import Game

def main():
    root = tk.Tk()
    Game(root, board_size=30, framerate=5)

if __name__ == '__main__':
    main()
