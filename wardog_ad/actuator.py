"""Turn continuous commands in [-1, 1] into binary key presses with PWM."""
from __future__ import annotations

from dataclasses import dataclass

from .input_win import Keyboard


@dataclass
class Axis:
    negative: str          # key held for values < 0
    positive: str          # key held for values > 0
    period_s: float = 0.2
    deadband: float = 0.06
    full: float = 0.94     # above this, hold continuously


class PWMActuator:
    """Each axis is pulse-width modulated over ``period_s``; phases are staggered."""

    def __init__(self, keyboard: Keyboard, axes: dict[str, Axis]):
        self.keyboard = keyboard
        self.axes = axes
        self.values = {name: 0.0 for name in axes}
        self.enabled = False

    def command(self, **values: float):
        for name, value in values.items():
            if name in self.axes:
                self.values[name] = max(-1.0, min(1.0, float(value)))

    def neutral(self):
        for name in self.values:
            self.values[name] = 0.0

    def update(self, now: float):
        """Call frequently (≥ 30 Hz) to refresh key states."""
        wanted: dict[str, bool] = {}
        for i, (name, axis) in enumerate(self.axes.items()):
            value = self.values[name] if self.enabled else 0.0
            duty = abs(value)
            if duty < axis.deadband:
                on = False
            elif duty >= axis.full:
                on = True
            else:
                phase = ((now + i * axis.period_s / len(self.axes)) % axis.period_s) / axis.period_s
                on = phase < duty
            active = axis.positive if value > 0 else axis.negative
            idle = axis.negative if value > 0 else axis.positive
            wanted[active] = wanted.get(active, False) or on
            wanted.setdefault(idle, False)
        for key, on in wanted.items():
            self.keyboard.set(key, on)

    def release(self):
        self.enabled = False
        self.neutral()
        self.keyboard.release_all()
