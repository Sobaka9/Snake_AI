import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import time
import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from games import GAME_REGISTRY
from games.game import Game
from agents import AGENT_REGISTRY
from agents.agent import Agent

# Palette
BG = "#0d0b14"
PANEL_BG = "#12101c"
ACCENT = "#1e1a30"
HIGHLIGHT = "#ff2079"
TEXT = "#e8e3ff"
TEXT_DIM = "#5e5280"
SNAKE_HEAD = "#ff2079"
SNAKE_BODY = "#7b2fff"
FOOD_COLOR = "#00f5d4"
GRID_LINE = "#1a1528"

FONT_TITLE = ("JetBrainsMono Nerd Font", 12, "bold")
FONT_LABEL = ("JetBrainsMono Nerd Font", 10)
FONT_VALUE = ("JetBrainsMono Nerd Font", 10, "bold")
FONT_MONO = ("JetBrainsMono Nerd Font", 9)
FONT_BTN = ("JetBrainsMono Nerd Font", 11, "bold")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Snake AI Trainer")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(1100, 700)

        # State
        self.training_thread = None
        self.stop_event = threading.Event()
        self.ui_queue = queue.Queue() # thread -> GUI updates
        self.game: Game = None
        self.agent: Agent = None
        self.is_training = False
        self.is_paused = False
        self._model_path = None

        self._build_ui()
        self._apply_style()
        # Style ttk combobox popdown listbox
        self.option_add("*TCombobox*Listbox.background", ACCENT)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", HIGHLIGHT)
        self.option_add("*TCombobox*Listbox.selectForeground", TEXT)
        self.option_add("*TCombobox*Listbox.relief", "flat")
        self._process_queue()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # --- UI Construction ---

    def _build_ui(self):

        # --- Top bar ---

        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=12, pady=(12, 0))

        tk.Label(top, text="GAME", bg=BG, fg=TEXT_DIM, font=FONT_LABEL).pack(side="left")
        self.game_var = tk.StringVar(value=list(GAME_REGISTRY.keys())[0]) # Snake by default
        game_menu = ttk.Combobox(
            top, textvariable=self.game_var,
            values=list(GAME_REGISTRY.keys()),
            state="readonly", width=14, font=FONT_LABEL
        )
        game_menu.pack(side="left", padx=(6, 24))

        tk.Label(top, text="SPEED", bg=BG, fg=TEXT_DIM, font=FONT_LABEL).pack(side="left")
        self.speed_var = tk.IntVar(value=30)
        speed_slider = ttk.Scale(
            top, from_=1, to=200, orient="horizontal",
            variable=self.speed_var, length=160
        )
        speed_slider.pack(side="left", padx=(6, 4))
        self.speed_label = tk.Label(top, text="30 fps", bg=BG, fg=TEXT, font=FONT_LABEL, width=7)
        self.speed_label.pack(side="left")
        self.speed_var.trace_add("write", lambda *_: self.speed_label.config(
            text=f"{self.speed_var.get()} fps"))

        # --- Three columns ---

        cols = tk.Frame(self, bg=BG)
        cols.pack(fill="both", expand=True, padx=12, pady=8)
        cols.columnconfigure(0, weight=0, minsize=230)
        cols.columnconfigure(1, weight=1, minsize=380)
        cols.columnconfigure(2, weight=0, minsize=230)
        cols.rowconfigure(0, weight=1)

        self._build_left(cols)
        self._build_center(cols)
        self._build_right(cols)
        self._build_bottom()

    # --- Left panel---

    def _build_left(self, parent):
        frame = tk.Frame(parent, bg=PANEL_BG, bd=1, relief="solid")
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        tk.Label(frame, text="AGENT", bg=PANEL_BG, fg=HIGHLIGHT,
                 font=FONT_TITLE).pack(anchor="w", padx=12, pady=(12, 2))

        # Agent selector
        row = tk.Frame(frame, bg=PANEL_BG)
        row.pack(fill="x", padx=12, pady=(0, 10))
        self.agent_var = tk.StringVar(value=list(AGENT_REGISTRY.keys())[0])
        agent_menu = ttk.Combobox(
            row, textvariable=self.agent_var,
            values=list(AGENT_REGISTRY.keys()),
            state="readonly", width=10, font=FONT_LABEL
        )
        agent_menu.pack(side="left")
        self.agent_var.trace_add("write", self._on_agent_change)

        ttk.Separator(frame, orient="horizontal").pack(fill="x", padx=8, pady=4)
        tk.Label(frame, text="HYPERPARAMETERS", bg=PANEL_BG, fg=TEXT_DIM,
                 font=FONT_MONO).pack(anchor="w", padx=12, pady=(4, 6))

        # Scrollable param area
        canvas = tk.Canvas(frame, bg=PANEL_BG, highlightthickness=0)
        scrollbar = tk.Scrollbar(
            frame, orient="vertical", command=canvas.yview,
            bg=ACCENT, troughcolor=PANEL_BG,
            activebackground=HIGHLIGHT,
            relief="flat", bd=0, width=8
        )
        self.param_frame = tk.Frame(canvas, bg=PANEL_BG)
        self.param_frame.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.param_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True, padx=(8, 0))
        scrollbar.pack(side="right", fill="y")

        self.param_widgets = {}
        self._populate_params()

    def _populate_params(self):
        for w in self.param_frame.winfo_children():
            w.destroy()
        self.param_widgets = {}

        agent_cls = AGENT_REGISTRY[self.agent_var.get()]
        for name, (dtype, default, lo, hi) in agent_cls.HYPERPARAMS.items():
            row = tk.Frame(self.param_frame, bg=PANEL_BG)
            row.pack(fill="x", padx=4, pady=3)

            label_text = name.replace("_", " ")
            tk.Label(row, text=label_text, bg=PANEL_BG, fg=TEXT_DIM,
                     font=FONT_MONO, width=15, anchor="w").pack(side="left")

            var = tk.StringVar(value=str(default))
            entry = tk.Entry(row, textvariable=var, font=FONT_MONO,
                             bg=ACCENT, fg=TEXT, insertbackground=TEXT,
                             relief="flat", width=10, bd=4)
            entry.pack(side="right")
            self.param_widgets[name] = (var, dtype)

    def _on_agent_change(self, *_):
        self._populate_params()

    def _get_hyperparams(self):
        params = {}
        for name, (var, dtype) in self.param_widgets.items():
            raw = var.get().strip()
            try:
                if dtype == "float":
                    params[name] = float(raw)
                elif dtype == "int":
                    params[name] = int(raw)
                else:
                    params[name] = raw   # str (e.g. hidden_size)
            except ValueError:
                pass
        return params

    # --- Center panel ---

    def _build_center(self, parent):
        frame = tk.Frame(parent, bg=PANEL_BG, bd=1, relief="solid")
        frame.grid(row=0, column=1, sticky="nsew", padx=6)

        self.canvas = tk.Canvas(frame, bg="#08070f", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Overlay labels
        self.canvas_score_label = tk.Label(frame, text="score: 0",
            bg=PANEL_BG, fg=HIGHLIGHT, font=FONT_LABEL)
        self.canvas_score_label.place(relx=0.0, rely=0.0, anchor="nw", x=8, y=8)

        self._cell = 20   # will update on resize
        self._offset_x = 0
        self._offset_y = 0

    def _on_canvas_resize(self, event):
        # Recalculate cell size to fit game grid
        if self.game:
            gw, gh = self.game.width, self.game.height
            cw, ch = event.width - 8, event.height - 8
            self._cell = max(4, min(cw // gw, ch // gh))
            self._offset_x = (cw - self._cell * gw) // 2 + 4
            self._offset_y = (ch - self._cell * gh) // 2 + 4
            self._draw_idle()

    def _init_canvas_for_game(self):
        if not self.game:
            return
        cw = self.canvas.winfo_width() or 400
        ch = self.canvas.winfo_height() or 400
        gw, gh = self.game.width, self.game.height
        self._cell = max(4, min((cw - 8) // gw, (ch - 8) // gh))
        self._offset_x = (cw - self._cell * gw) // 2 + 4
        self._offset_y = (ch - self._cell * gh) // 2 + 4

    def _draw_idle(self):
        """Draw the welcome screen before training starts."""
        c = self.canvas
        c.delete("all")
        w = c.winfo_width() or 400
        h = c.winfo_height() or 400
        c.create_text(w // 2, h // 2, text="Press  START TRAINING\n or  LOAD MODEL",
                      fill=TEXT_DIM, font=("Courier New", 12), justify="center")

    def render_frame(self, data):
        """Called from main thread via queue."""
        c = self.canvas
        c.delete("all")
        cell = self._cell
        ox, oy = self._offset_x, self._offset_y
        gw, gh = data["width"], data["height"]

        # Grid
        for gx in range(gw + 1):
            x = ox + gx * cell
            c.create_line(x, oy, x, oy + gh * cell, fill=GRID_LINE, width=1)
        for gy in range(gh + 1):
            y = oy + gy * cell
            c.create_line(ox, y, ox + gw * cell, y, fill=GRID_LINE, width=1)

        # Food
        fx, fy = data["food"].x, data["food"].y
        c.create_oval(ox + fx * cell + 2, oy + fy * cell + 2,
                      ox + (fx + 1) * cell - 2, oy + (fy + 1) * cell - 2,
                      fill=FOOD_COLOR, outline="")

        # Snake
        for i, pt in enumerate(data["snake"]):
            color = SNAKE_HEAD if i == 0 else SNAKE_BODY
            c.create_rectangle(ox + pt.x * cell + 1, oy + pt.y * cell + 1,
                                ox + (pt.x + 1) * cell - 1, oy + (pt.y + 1) * cell - 1,
                                fill=color, outline="")

        self.canvas_score_label.config(text=f"score: {data['score']}")

    # ── Right panel ──────────────────────────────────────────────────

    def _build_right(self, parent):
        frame = tk.Frame(parent, bg=PANEL_BG, bd=1, relief="solid")
        frame.grid(row=0, column=2, sticky="nsew", padx=(6, 0))

        tk.Label(frame, text="TRAINING DETAILS", bg=PANEL_BG,
                 fg=HIGHLIGHT, font=FONT_TITLE).pack(anchor="w", padx=12, pady=(12, 6))

        # Stats grid
        stats_frame = tk.Frame(frame, bg=PANEL_BG)
        stats_frame.pack(fill="x", padx=12, pady=4)

        self.stat_vars = {}
        stats = [
            ("episode",  "0"),
            ("score",    "0"),
            ("best",     "0"),
            ("ε (eps)",  "1.000"),
            ("avg score","0.00"),
            ("avg loss", "—"),
        ]
        for i, (label, init) in enumerate(stats):
            tk.Label(stats_frame, text=label, bg=PANEL_BG, fg=TEXT_DIM,
                     font=FONT_MONO, anchor="w", width=10).grid(
                row=i, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=init)
            tk.Label(stats_frame, textvariable=var, bg=PANEL_BG, fg=TEXT,
                     font=FONT_VALUE, anchor="e", width=8).grid(
                row=i, column=1, sticky="e", pady=2)
            self.stat_vars[label] = var

        ttk.Separator(frame, orient="horizontal").pack(fill="x", padx=8, pady=8)

        # Loss chart
        tk.Label(frame, text="LOSS CURVE", bg=PANEL_BG, fg=TEXT_DIM,
                 font=FONT_MONO).pack(anchor="w", padx=12)

        fig = Figure(figsize=(2.8, 1.9), dpi=90, facecolor=PANEL_BG)
        self.ax = fig.add_subplot(111)
        self.ax.set_facecolor("#08070f")
        self.ax.tick_params(colors=TEXT_DIM, labelsize=6)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(ACCENT)
        self.loss_line, = self.ax.plot([], [], color=HIGHLIGHT, linewidth=1.2)
        self.ax.set_xlabel("episode", color=TEXT_DIM, fontsize=6)
        self.ax.set_ylabel("loss", color=TEXT_DIM, fontsize=6)
        fig.tight_layout(pad=0.8)

        self.fig_canvas = FigureCanvasTkAgg(fig, master=frame)
        self.fig_canvas.get_tk_widget().pack(fill="x", padx=8, pady=4)

        # Score chart
        ttk.Separator(frame, orient="horizontal").pack(fill="x", padx=8, pady=4)
        tk.Label(frame, text="SCORE CURVE", bg=PANEL_BG, fg=TEXT_DIM,
                 font=FONT_MONO).pack(anchor="w", padx=12)

        fig2 = Figure(figsize=(2.8, 1.9), dpi=90, facecolor=PANEL_BG)
        self.ax2 = fig2.add_subplot(111)
        self.ax2.set_facecolor("#08070f")
        self.ax2.tick_params(colors=TEXT_DIM, labelsize=6)
        for spine in self.ax2.spines.values():
            spine.set_edgecolor(ACCENT)
        self.score_line, = self.ax2.plot([], [], color=FOOD_COLOR, linewidth=1.2)
        self.ax2.set_xlabel("episode", color=TEXT_DIM, fontsize=6)
        self.ax2.set_ylabel("score", color=TEXT_DIM, fontsize=6)
        fig2.tight_layout(pad=0.8)

        self.fig_canvas2 = FigureCanvasTkAgg(fig2, master=frame)
        self.fig_canvas2.get_tk_widget().pack(fill="x", padx=8, pady=(0, 8))

    # --- Bottom buttons ---

    def _build_bottom(self):
        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=12, pady=(0, 12))

        btn_cfg = dict(font=FONT_BTN, relief="flat", bd=0, padx=18, pady=8, cursor="hand2")

        self.btn_start = tk.Button(bar, text="START TRAINING",
            bg=HIGHLIGHT, fg="white", command=self._start_training, **btn_cfg)
        self.btn_start.pack(side="left", padx=(0, 6))

        self.btn_pause = tk.Button(bar, text="PAUSE",
            bg=ACCENT, fg=TEXT, command=self._toggle_pause, **btn_cfg,
            state="disabled")
        self.btn_pause.pack(side="left", padx=6)

        self.btn_stop = tk.Button(bar, text="STOP",
            bg="#2a1f3d", fg=TEXT, command=self._stop_training, **btn_cfg,
            state="disabled")
        self.btn_stop.pack(side="left", padx=6)

        tk.Frame(bar, bg=BG, width=30).pack(side="left")

        self.btn_load = tk.Button(bar, text="LOAD MODEL",
            bg=ACCENT, fg=TEXT, command=self._load_model, **btn_cfg)
        self.btn_load.pack(side="left", padx=6)

        self.btn_save = tk.Button(bar, text="SAVE MODEL",
            bg=ACCENT, fg=TEXT, command=self._save_model, **btn_cfg,
            state="disabled")
        self.btn_save.pack(side="left", padx=6)

        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(bar, textvariable=self.status_var, bg=BG, fg=TEXT_DIM,
                 font=FONT_MONO).pack(side="right", padx=8)


    #  --- Style ---

    def _apply_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=ACCENT, background=ACCENT,
                        foreground=TEXT, selectbackground=ACCENT,
                        selectforeground=TEXT, arrowcolor=TEXT,
                        bordercolor=GRID_LINE, lightcolor=ACCENT, darkcolor=ACCENT,
                        insertcolor=TEXT, padding=4)
        style.map("TCombobox",
                  fieldbackground=[("readonly", ACCENT), ("disabled", PANEL_BG)],
                  background=[("readonly", ACCENT), ("active", ACCENT)],
                  foreground=[("readonly", TEXT), ("disabled", TEXT_DIM)],
                  bordercolor=[("focus", HIGHLIGHT), ("!focus", GRID_LINE)],
                  arrowcolor=[("readonly", TEXT), ("active", HIGHLIGHT)])
        style.configure("TSeparator", background=GRID_LINE)
        style.configure("TScale", background=BG, troughcolor=ACCENT,
                        sliderthickness=12, bordercolor=ACCENT)

    #  ---Training Control ---

    def _start_training(self):
        if self.is_training:
            return

        game_name = self.game_var.get()
        agent_name = self.agent_var.get()
        params = self._get_hyperparams()

        self.game = GAME_REGISTRY[game_name]()
        agent_cls = AGENT_REGISTRY[agent_name]
        self.agent = agent_cls(
            self.game.state_size, self.game.action_size, **params
        )

        self._init_canvas_for_game()

        self.stop_event.clear()
        self.is_training = True
        self.is_paused   = False

        self.btn_start.config(state="disabled")
        self.btn_pause.config(state="normal")
        self.btn_stop.config(state="normal")
        self.btn_load.config(state="disabled")
        self.btn_save.config(state="normal")
        self.status_var.set(f"Training {agent_name} on {game_name}…")

        self.training_thread = threading.Thread(
            target=self._training_loop, daemon=True)
        self.training_thread.start()

    def _training_loop(self):
        scores, losses = [], []
        episode = 0

        while not self.stop_event.is_set():
            # Pause support
            while self.is_paused and not self.stop_event.is_set():
                time.sleep(0.05)
            if self.stop_event.is_set():
                break

            state = self.game.reset()
            done  = False
            ep_losses = []

            while not done and not self.stop_event.is_set():
                while self.is_paused and not self.stop_event.is_set():
                    time.sleep(0.05)

                action = self.agent.select_action(state, training=True)
                next_state, reward, done, info = self.game.step(action)
                
                # TODO: consider moving remember() inside the agent's logic
                self.agent.remember(state, action, reward, next_state, done)
                
                loss = self.agent.train_step()
                if loss is not None:
                    ep_losses.append(loss)
                    
                state = next_state

                # Send frame to GUI
                render_data = self.game.get_render_data()
                self.ui_queue.put(("frame", render_data))

                # Throttle to speed setting
                fps = self.speed_var.get()
                time.sleep(1 / fps)

            episode += 1
            self.agent.end_episode()

            scores.append(info["score"])
            avg_loss = float(np.mean(ep_losses)) if ep_losses else None
            avg_score = float(np.mean(scores[-50:]))

            self.ui_queue.put(("stats", {
                "episode":   episode,
                "score":     info["score"],
                "best":      max(scores),
                "epsilon":   self.agent.epsilon,
                "avg_score": avg_score,
                "avg_loss":  avg_loss,
                "all_scores": scores.copy(),
                "all_losses": [l for l in
                               [float(np.mean(ep_losses)) if ep_losses else None]
                               if l is not None],
                "loss_history": getattr(self, "_loss_history", []) +
                                ([avg_loss] if avg_loss is not None else []),
            }))
            if avg_loss is not None:
                if not hasattr(self, "_loss_history"):
                    self._loss_history = []
                self._loss_history.append(avg_loss)

        self.ui_queue.put(("stopped", None))

    def _toggle_pause(self):
        self.is_paused = not self.is_paused
        self.btn_pause.config(
            text="RESUME" if self.is_paused else "PAUSE")
        self.status_var.set("Paused." if self.is_paused else "Resumed.")

    def _stop_training(self):
        self.stop_event.set()
        self.is_paused = False

    def _load_model(self):
        path = filedialog.askopenfilename(
            title="Load Model", filetypes=[("Model files", "*.pkl"), ("All", "*.*")])
        if not path:
            return
        game_name  = self.game_var.get()
        agent_name = self.agent_var.get()
        params = self._get_hyperparams()

        game_cls = GAME_REGISTRY[game_name]
        self.game = game_cls()
        agent_cls = AGENT_REGISTRY[agent_name]
        self.agent = agent_cls(
            self.game.state_size, self.game.action_size, **params)
        try:
            self.agent.load(path)
            self._model_path = path
            self.status_var.set(f"Loaded: {os.path.basename(path)}")
            messagebox.showinfo("Model Loaded",
                f"Model loaded successfully.\nPath: {path}")
            self.btn_save.config(state="normal")
            self.btn_start.config(state="normal")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))

    def _save_model(self):
        if not self.agent:
            return
        path = filedialog.asksaveasfilename(
            title="Save Model", defaultextension=".pkl",
            filetypes=[("Model files", "*.pkl"), ("All", "*.*")])
        if not path:
            return
        try:
            self.agent.save(path)
            self.status_var.set(f"Saved: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))


    #  --- Queue Processing (thread-safe GUI updates) ---

    def _process_queue(self):
        try:
            while True:
                msg, data = self.ui_queue.get_nowait()

                if msg == "frame":
                    self.render_frame(data)

                elif msg == "stats":
                    ep      = data["episode"]
                    score   = data["score"]
                    best    = data["best"]
                    eps     = data["epsilon"]
                    avg_s   = data["avg_score"]
                    avg_l   = data["avg_loss"]
                    scores  = data["all_scores"]

                    self.stat_vars["episode"].set(str(ep))
                    self.stat_vars["score"].set(str(score))
                    self.stat_vars["best"].set(str(best))
                    self.stat_vars["ε (eps)"].set(f"{eps:.4f}")
                    self.stat_vars["avg score"].set(f"{avg_s:.2f}")
                    self.stat_vars["avg loss"].set(
                        f"{avg_l:.4f}" if avg_l is not None else "—")

                    # Update loss chart
                    loss_hist = getattr(self, "_loss_history", [])
                    if loss_hist:
                        xs = list(range(1, len(loss_hist) + 1))
                        self.loss_line.set_data(xs, loss_hist)
                        self.ax.relim(); self.ax.autoscale_view()
                        self.fig_canvas.draw_idle()

                    # Update score chart
                    if scores:
                        xs = list(range(1, len(scores) + 1))
                        self.score_line.set_data(xs, scores)
                        self.ax2.relim(); self.ax2.autoscale_view()
                        self.fig_canvas2.draw_idle()

                elif msg == "stopped":
                    self.is_training = False
                    self.btn_start.config(state="normal")
                    self.btn_pause.config(state="disabled",
                                          text="PAUSE")
                    self.btn_stop.config(state="disabled")
                    self.btn_load.config(state="normal")
                    self.status_var.set("Training stopped.")

        except queue.Empty:
            pass

        self.after(16, self._process_queue) # ~60 Hz UI refresh


    #  --- Cleanup ---

    def _on_close(self):
        self.stop_event.set()
        self.destroy()
