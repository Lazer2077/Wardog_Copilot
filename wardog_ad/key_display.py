"""Shared read-only WASD display. OS-observed states can include injected input."""
import tkinter as tk

from .input_win import key_is_down
from .ui_theme import SURFACE, MUTED, KEY_OFF, BORDER


class KeyStateDisplay(tk.Frame):
    def __init__(self, master, *, sent=None, observed=None):
        super().__init__(master, bg=SURFACE, padx=4, pady=4)
        self._keep_theme = True
        self.sent = sent
        self.observed = observed or (lambda: {k for k in 'WASD' if key_is_down(k)})
        self.labels = {}
        for column, (source, title) in enumerate([('observed', '系统观测')] +
                                                 ([('sent', '程序发送')] if sent else [])):
            self.columnconfigure(column, weight=1)
            frame = tk.Frame(self, padx=12, pady=5, bg=SURFACE)
            frame.grid(row=0, column=column, sticky='ew')
            for col in range(3):
                frame.columnconfigure(col, weight=1)
            tk.Label(frame, text=title, bg=SURFACE, fg=MUTED,
                     font=('Microsoft YaHei UI', 9)).grid(row=0, column=0, columnspan=3, pady=(0, 4))
            for key, row, col in [('W', 1, 1), ('A', 2, 0), ('S', 2, 1), ('D', 2, 2)]:
                label = tk.Label(frame, text=key, width=3, relief='flat', font=('Consolas', 12, 'bold'),
                                 highlightthickness=1, highlightbackground=BORDER, pady=2)
                label.grid(row=row, column=col, padx=1, pady=1)
                self.labels[source, key] = label
        self._refresh()

    def _refresh(self):
        observed = set(self.observed())
        sent = set(self.sent()) if self.sent else set()
        for (source, key), label in self.labels.items():
            down = key in (observed if source == 'observed' else sent)
            label.config(bg=('#2e7d32' if source == 'observed' else '#1565c0') if down else KEY_OFF,
                         fg='white' if down else MUTED)
        self.after(40, self._refresh)
