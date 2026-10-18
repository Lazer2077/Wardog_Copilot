"""Shared dark UI styling and adjustable whole-window opacity."""
import math
import tkinter as tk
from tkinter import ttk

BG = '#111923'
SURFACE = '#192532'
CARD = '#243445'
BORDER = '#354b60'
TEXT = '#e8f1f8'
MUTED = '#9cafc0'
ACCENT = '#60dfca'
GREEN = '#267e6b'
BLUE = '#315f98'
RED = '#b94152'
PURPLE = '#665191'
AMBER = '#805c31'
KEY_OFF = '#263747'
FONT = ('Microsoft YaHei UI', 10)


def opacity_value(value):
    value = float(value)
    return min(1., max(.55, value)) if math.isfinite(value) else .9


def window_theme(window, opacity):
    window.configure(bg=BG)
    window.attributes('-alpha', opacity_value(opacity))
    style = ttk.Style(window)
    if style.theme_use() != 'clam':
        style.theme_use('clam')
    for name in ('Wardog.TCombobox', 'Wardog.TSpinbox'):
        style.configure(name, fieldbackground=BG, background=CARD, foreground=TEXT,
                        bordercolor=BORDER, arrowcolor=ACCENT, lightcolor=BORDER, darkcolor=BORDER,
                        padding=5, font=FONT, insertcolor=TEXT)
        style.map(name, fieldbackground=[('readonly', BG)], foreground=[('readonly', TEXT)],
                  selectbackground=[('readonly', CARD)], selectforeground=[('readonly', TEXT)],
                  bordercolor=[('focus', ACCENT)])
    window.option_add('*TCombobox*Listbox.background', BG)
    window.option_add('*TCombobox*Listbox.foreground', TEXT)
    window.option_add('*TCombobox*Listbox.selectBackground', BLUE)
    window.option_add('*TCombobox*Listbox.selectForeground', TEXT)
    style.configure('Wardog.Horizontal.TScale', background=ACCENT, troughcolor=BG,
                    bordercolor=BG, lightcolor=ACCENT, darkcolor=ACCENT)
    style.configure('Wardog.Vertical.TScrollbar', background=BORDER, troughcolor=BG,
                    bordercolor=BG, arrowcolor=ACCENT, lightcolor=BORDER, darkcolor=BORDER,
                    arrowsize=11)


def dark_titlebar(window):
    """Use the OS dark caption for this Tk window, when supported by Windows."""
    import sys
    if sys.platform != 'win32':
        return
    import ctypes
    try:
        window.update_idletasks()
        user = ctypes.windll.user32
        user.GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        user.GetAncestor.restype = ctypes.c_void_p
        hwnd = user.GetAncestor(window.winfo_id(), 2)
        enabled = ctypes.c_int(1)
        api = ctypes.windll.dwmapi.DwmSetWindowAttribute
        api.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]
        api(hwnd, 20, ctypes.byref(enabled), ctypes.sizeof(enabled))
    except (OSError, AttributeError):
        pass


def style_widgets(parent):
    """Restyle native controls while preserving commands, layout and semantic colors."""
    for widget in parent.winfo_children():
        if getattr(widget, '_keep_theme', False):
            continue
        if isinstance(widget, ttk.Combobox):
            widget.configure(style='Wardog.TCombobox')
        elif isinstance(widget, ttk.Spinbox):
            widget.configure(style='Wardog.TSpinbox')
        elif isinstance(widget, tk.Button):
            color = {'#2e7d32': GREEN, '#1565c0': BLUE, '#c62828': RED,
                     '#6a1b9a': PURPLE, '#ef6c00': AMBER}.get(widget.cget('bg'), CARD)
            widget.configure(bg=color, fg=TEXT, activebackground=BORDER, activeforeground=TEXT,
                             relief='flat', bd=0, highlightthickness=1, highlightbackground=BORDER,
                             highlightcolor=ACCENT, padx=10, pady=6, font=FONT, cursor='hand2')
        elif isinstance(widget, tk.LabelFrame):
            widget.configure(bg=SURFACE, fg=ACCENT, relief='flat', bd=0,
                             highlightthickness=1, highlightbackground=BORDER)
        elif isinstance(widget, tk.Frame):
            widget.configure(bg=SURFACE)
        elif isinstance(widget, tk.Label):
            old = widget.cget('fg')
            color = {'#666': MUTED, '#777': MUTED, '#795548': '#e4b97d', '#c62828': '#ff8895'}.get(old, TEXT)
            widget.configure(bg=SURFACE, fg=color)
        elif isinstance(widget, (tk.Entry, tk.Listbox)):
            widget.configure(bg=BG, fg=TEXT, selectbackground=BLUE, selectforeground=TEXT,
                             relief='flat', bd=0, highlightthickness=1, highlightbackground=BORDER,
                             highlightcolor=ACCENT, font=FONT)
            if isinstance(widget, tk.Entry):
                widget.configure(insertbackground=ACCENT)
        elif isinstance(widget, tk.Menubutton):
            widget.configure(bg=CARD, fg=TEXT, activebackground=BORDER, activeforeground=TEXT,
                             relief='flat', bd=0, font=FONT)
            menu = widget.nametowidget(widget.cget('menu'))
            menu.configure(bg=SURFACE, fg=TEXT, activebackground=BLUE, activeforeground=TEXT, bd=0)
        style_widgets(widget)


def save_opacity(cfg, name, value, *, remember=None, path=None):
    value = opacity_value(value)
    def change(c):
        setattr(c.ui, name, value)
    if remember is not None:
        remember(change)
    else:
        change(cfg)
        if path:
            from .config import Config
            saved = Config.load(path)
            change(saved)
            saved.save(path)


class OpacityControl(tk.Frame):
    def __init__(self, master, window, value, *, on_save=None):
        super().__init__(master, bg=SURFACE, padx=10, pady=5)
        self._keep_theme = True
        self.window, self.on_save = window, on_save
        self.value = tk.DoubleVar(value=round(opacity_value(value) * 100))
        tk.Label(self, text='不透明度', bg=SURFACE, fg=MUTED,
                 font=('Microsoft YaHei UI', 9)).pack(side='left')
        self.percent = tk.Label(self, width=5, bg=SURFACE, fg=ACCENT, font=('Consolas', 10))
        self.percent.pack(side='right')
        self.scale = ttk.Scale(self, from_=55, to=100, orient='horizontal', variable=self.value,
                               command=self._apply, style='Wardog.Horizontal.TScale', takefocus=True)
        self.scale.pack(side='left', fill='x', expand=True, padx=8)
        self.scale.bind('<ButtonRelease-1>', self._save)
        self.scale.bind('<KeyRelease>', self._save)
        self._apply()

    def _apply(self, value=None):
        opacity = opacity_value(round(self.value.get()) / 100)
        self.window.attributes('-alpha', opacity)
        self.percent.configure(text=f'{opacity:.0%}')

    def _save(self, event=None):
        self._apply()
        if self.on_save is not None:
            self.on_save(opacity_value(round(self.value.get()) / 100))


class ScrollPanel(tk.Frame):
    """A control rail whose wheel scrolling never changes combo/spinbox values."""
    def __init__(self, master, *, height=720, width=350):
        super().__init__(master, bg=SURFACE)
        self.canvas = tk.Canvas(self, bg=SURFACE, height=height, width=width, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview,
                                       style='Wardog.Vertical.TScrollbar')
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side='right', fill='y')
        self.canvas.pack(side='left', fill='both', expand=True)
        self.body = tk.Frame(self.canvas, bg=SURFACE, padx=12, pady=12)
        self.item = self.canvas.create_window(0, 0, window=self.body, anchor='nw')
        self.body.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.bind('<Configure>', lambda e: self.canvas.itemconfigure(self.item, width=e.width))

    def bind_wheel(self):
        def wheel(event):
            if event.delta:
                self.canvas.yview_scroll(-max(1, abs(event.delta) // 120) if event.delta > 0
                                         else max(1, abs(event.delta) // 120), 'units')
            return 'break'
        def visit(widget):
            widget.bind('<MouseWheel>', wheel)
            for child in widget.winfo_children():
                visit(child)
        visit(self)
