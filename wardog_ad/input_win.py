"""Keyboard output and hotkey polling.

Games reading DirectInput/raw input ignore virtual-key events, so keys are sent
as hardware scan codes through ``SendInput``. Capture gating additionally queries
the foreground executable name with limited read-only process identity access;
there is no game-memory access or hook.
"""
from __future__ import annotations

import ctypes
import os
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path

SCANCODES = {
    'ESC': 0x01, '1': 0x02, '2': 0x03, '3': 0x04, '4': 0x05, '5': 0x06, '6': 0x07, '7': 0x08,
    '8': 0x09, '9': 0x0A, '0': 0x0B, 'Q': 0x10, 'W': 0x11, 'E': 0x12, 'R': 0x13, 'T': 0x14,
    'Y': 0x15, 'U': 0x16, 'I': 0x17, 'O': 0x18, 'P': 0x19, 'ENTER': 0x1C, 'LCTRL': 0x1D,
    'A': 0x1E, 'S': 0x1F, 'D': 0x20, 'F': 0x21, 'G': 0x22, 'H': 0x23, 'J': 0x24, 'K': 0x25,
    'L': 0x26, 'LSHIFT': 0x2A, 'Z': 0x2C, 'X': 0x2D, 'C': 0x2E, 'V': 0x2F, 'B': 0x30,
    'N': 0x31, 'M': 0x32, 'RSHIFT': 0x36, 'LALT': 0x38, 'SPACE': 0x39, 'CAPSLOCK': 0x3A,
    'F1': 0x3B, 'F2': 0x3C, 'F3': 0x3D, 'F4': 0x3E, 'F5': 0x3F, 'F6': 0x40, 'F7': 0x41,
    'F8': 0x42, 'F9': 0x43, 'F10': 0x44, 'F11': 0x57, 'F12': 0x58,
    'UP': 0xE048, 'LEFT': 0xE04B, 'RIGHT': 0xE04D, 'DOWN': 0xE050,
}

VIRTUAL_KEYS = {**{f'F{i}': 0x6F + i for i in range(1, 13)}, 'ESC': 0x1B, 'PAUSE': 0x13,
                'SPACE': 0x20, 'TAB': 0x09, 'LSHIFT': 0xA0, 'LCTRL': 0xA2, 'ALT': 0x12,
                **{c: ord(c) for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'}}


class Keyboard:
    """Interface shared by the real keyboard and the simulator."""

    def __init__(self):
        self.down: set[str] = set()

    def press(self, key: str):
        if key not in self.down:
            self.down.add(key)
            self._send(key, True)

    def release(self, key: str):
        if key in self.down:
            self.down.discard(key)
            self._send(key, False)

    def set(self, key: str, pressed: bool):
        self.press(key) if pressed else self.release(key)

    def tap(self, key: str, hold_s: float = 0.05):
        self.press(key)
        time.sleep(hold_s)
        self.release(key)

    def release_all(self):
        for key in list(self.down):
            self.release(key)

    def _send(self, key: str, pressed: bool):
        raise NotImplementedError


class FakeKeyboard(Keyboard):
    def __init__(self):
        super().__init__()
        self.log: list[tuple[str, bool]] = []
        self.taps: list[str] = []

    def tap(self, key: str, hold_s: float = 0.0):
        self.taps.append(key)
        self.log.append((key, True))
        self.log.append((key, False))

    def _send(self, key, pressed):
        self.log.append((key, pressed))


if sys.platform == 'win32':
    from ctypes import wintypes

    _user32 = ctypes.WinDLL('user32', use_last_error=True)
    ULONG_PTR = ctypes.c_size_t
    _user32.GetForegroundWindow.restype = wintypes.HWND
    _user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    _user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    _user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    _user32.IsIconic.argtypes = [wintypes.HWND]
    _user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    _kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    _kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    _kernel32.CloseHandle.argtypes = [wintypes.HANDLE]

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [('wVk', wintypes.WORD), ('wScan', wintypes.WORD), ('dwFlags', wintypes.DWORD),
                    ('time', wintypes.DWORD), ('dwExtraInfo', ULONG_PTR)]

    class MOUSEINPUT(ctypes.Structure):
        _fields_ = [('dx', wintypes.LONG), ('dy', wintypes.LONG), ('mouseData', wintypes.DWORD),
                    ('dwFlags', wintypes.DWORD), ('time', wintypes.DWORD), ('dwExtraInfo', ULONG_PTR)]

    class _INPUTUNION(ctypes.Union):
        _fields_ = [('ki', KEYBDINPUT), ('mi', MOUSEINPUT)]

    class INPUT(ctypes.Structure):
        _anonymous_ = ('u',)
        _fields_ = [('type', wintypes.DWORD), ('u', _INPUTUNION)]

    _KEYEVENTF_EXTENDEDKEY, _KEYEVENTF_KEYUP, _KEYEVENTF_SCANCODE = 0x1, 0x2, 0x8


class WindowsKeyboard(Keyboard):
    def __init__(self):
        if sys.platform != 'win32':
            raise RuntimeError('WindowsKeyboard requires Windows')
        super().__init__()
        self.lock = threading.Lock()

    def _send(self, key: str, pressed: bool):
        code = SCANCODES[key.upper()]
        flags = _KEYEVENTF_SCANCODE | (0 if pressed else _KEYEVENTF_KEYUP)
        if code > 0xFF:
            flags |= _KEYEVENTF_EXTENDEDKEY
            code &= 0xFF
        event = INPUT(type=1, ki=KEYBDINPUT(0, code, flags, 0, 0))
        with self.lock:
            if _user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT)) != 1:
                raise ctypes.WinError(ctypes.get_last_error())


def key_is_down(name: str) -> bool:
    if sys.platform != 'win32':
        return False
    return bool(_user32.GetAsyncKeyState(VIRTUAL_KEYS[name.upper()]) & 0x8000)


class EdgeDetector:
    """Report a hotkey once per physical press."""

    def __init__(self, name: str):
        self.name, self.was = name, False

    def pressed(self) -> bool:
        now = key_is_down(self.name)
        edge = now and not self.was
        self.was = now
        return edge


def foreground_title() -> str:
    if sys.platform != 'win32':
        return ''
    hwnd = _user32.GetForegroundWindow()
    return _window_title(hwnd)


def _other_process(hwnd) -> bool:
    pid = wintypes.DWORD()
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value != os.getpid()


def foreground_is(title: str) -> bool:
    """The game is in front: title matches and it is not one of our own windows (the
    control panel is called 'WARDOGS ...' too, and keys sent there go nowhere)."""
    if sys.platform != 'win32':
        return True
    hwnd = _user32.GetForegroundWindow()
    return bool(hwnd) and title.lower() in _window_title(hwnd).lower() and _other_process(hwnd)


def focus_window(title: str) -> bool:
    """Give the keyboard focus back to the game after a click on our panel."""
    if sys.platform != 'win32':
        return False
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _):
        if (_user32.IsWindowVisible(hwnd) and title.lower() in _window_title(hwnd).lower()
                and _other_process(hwnd)):
            found.append(hwnd)
            return False
        return True

    _user32.EnumWindows(visit, 0)
    return bool(found) and bool(_user32.SetForegroundWindow(found[0]))


def _window_title(hwnd):
    length = _user32.GetWindowTextLengthW(hwnd)
    buffer = ctypes.create_unicode_buffer(length + 1)
    _user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value


@lru_cache(maxsize=64)
def _process_name(pid):
    handle = _kernel32.OpenProcess(0x1000, False, pid)  # query identity only; no memory access
    if not handle:
        return ''
    try:
        buffer, length = ctypes.create_unicode_buffer(32768), wintypes.DWORD(32768)
        if not _kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(length)):
            return ''
        return Path(buffer.value).name
    finally:
        _kernel32.CloseHandle(handle)


def foreground_client(expected_process=None):
    """Read the foreground client rectangle; never activates or changes a window."""
    if sys.platform != 'win32':
        return None
    from .config import Region
    hwnd = _user32.GetForegroundWindow()
    if not hwnd or _user32.IsIconic(hwnd):
        return None
    if expected_process:
        pid = wintypes.DWORD()
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if _process_name(pid.value).casefold() != expected_process.casefold():
            return None
    rect, point = wintypes.RECT(), wintypes.POINT(0, 0)
    if not _user32.GetClientRect(hwnd, ctypes.byref(rect)) or not _user32.ClientToScreen(hwnd, ctypes.byref(point)):
        return None
    region = Region(point.x, point.y, rect.right - rect.left, rect.bottom - rect.top)
    return (int(hwnd), _window_title(hwnd), region) if region.valid else None


def set_dpi_aware():
    """Make screen coordinates physical pixels so they match mss captures."""
    if sys.platform == 'win32':
        try:
            ctypes.WinDLL('shcore').SetProcessDpiAwareness(2)
        except (OSError, AttributeError):
            _user32.SetProcessDPIAware()
