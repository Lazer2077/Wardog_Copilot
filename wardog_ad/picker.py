"""Map window to choose the destination / return point, manage favourites and the
helicopter cruise height.

Mouse: wheel = zoom, left-drag = pan, left-click = destination, right-click = return point.
"""
from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
import tkinter as tk
from tkinter import simpledialog, ttk

import cv2
from PIL import Image, ImageTk

from .geometry import heading_vector
from .ui_theme import (ACCENT, OpacityControl, ScrollPanel, window_theme,
                       style_widgets, dark_titlebar, save_opacity)


@dataclass(frozen=True)
class MapPosition:
    point: tuple[float, float]
    heading: float | None
    timestamp: float
    estimated: bool = False
    manual: bool = False



class MapPicker:
    def __init__(self, master, pilot, *, embedded=False, go_command=None):
        self.pilot = pilot
        self.embedded = embedded
        self.assets = pilot.assets
        self._last_position = None
        self._position_context = (self.assets.map_id, pilot.cfg.profile)
        self._auto_center_pending = True
        self._position_action = None
        self._team_bases = []
        self._map_changing = False
        self._map_change_result = queue.SimpleQueue()
        rgb = cv2.cvtColor(self.assets.load_image(), cv2.COLOR_BGR2RGB)
        self.image = Image.fromarray(rgb)
        self.win = tk.Frame(master) if embedded else tk.Toplevel(master)
        if embedded:
            self.win.pack(fill='both',expand=True)
        else:
            self.win.title(f'选择目的地 - {self.assets.name}')
            self.win.attributes('-topmost', True)
            self.win.protocol('WM_DELETE_WINDOW', self.close)
            window_theme(self.win, pilot.cfg.ui.map_opacity)
        font = ('Microsoft YaHei UI', 10)

        self.cw = min(600, max(280, self.win.winfo_screenwidth() - 1100)) if embedded else min(1000, max(640, self.win.winfo_screenwidth() - 460))
        self.ch = min(680, max(320, self.win.winfo_screenheight() - 200)) if embedded else min(760, max(480, self.win.winfo_screenheight() - 180))
        self.canvas = tk.Canvas(self.win, width=self.cw, height=self.ch, bg='#202020', highlightthickness=0,
                                cursor='crosshair')
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.win.rowconfigure(0, weight=1)
        self.win.columnconfigure(0, weight=1)
        self.canvas.bind('<Configure>', self._resized)
        self.sidebar = ScrollPanel(self.win, height=self.ch)
        self.sidebar.grid(row=0, column=1, sticky='ns')
        side = self.sidebar.body
        zoom_bar = tk.Frame(side)
        zoom_bar.pack(fill='x', pady=(0, 6))
        tk.Button(zoom_bar, text='＋', command=lambda: self.zoom_by(1.25)).pack(side='left')
        tk.Button(zoom_bar, text='−', command=lambda: self.zoom_by(.8)).pack(side='left', padx=4)
        tk.Button(zoom_bar, text='全图', command=self.fit_map).pack(side='left')
        brand = tk.Label(side, text='WARDOGS  /  战术地图', font=('Microsoft YaHei UI', 13, 'bold'))
        brand.pack(anchor='w', pady=(0, 10))

        from .assets import DEFAULT_ASSETS, available_maps
        try:
            maps = available_maps(getattr(self.assets, 'root', DEFAULT_ASSETS))
        except (OSError, ValueError):
            maps = [self.assets.map_id]
        self._map_choices = {name.upper(): name for name in maps}
        tk.Label(side, text='游戏地图', font=font).pack(anchor='w')
        self.map_choice = tk.StringVar(value='自动识别' if pilot.cfg.auto_map.enabled else self.assets.map_id.upper())
        self.map_combo = ttk.Combobox(side, state='readonly', textvariable=self.map_choice,
                                      values=['自动识别', *self._map_choices], width=26)
        self.map_combo.pack(fill='x', pady=2)
        self.map_combo.bind('<<ComboboxSelected>>', self._change_map)
        self.map_error = tk.Label(side, font=('Microsoft YaHei UI', 9), fg='#c62828', wraplength=260)
        self.map_error.pack(fill='x')
        self.opacity = None
        if not embedded:
            self.opacity = OpacityControl(side, self.win, pilot.cfg.ui.map_opacity,
                on_save=lambda value: save_opacity(pilot.cfg, 'map_opacity', value, remember=pilot._remember))
            self.opacity.pack(fill='x', pady=(2, 8))
        self.info = tk.Label(side, font=font, justify='left', anchor='w', wraplength=300, pady=8)
        self.info.pack(fill='x')
        tk.Label(side, text='左键：设为目的地    右键：设为返航点\n滚轮：缩放    拖动：平移', font=('Microsoft YaHei UI', 9),
                 fg='#666', justify='left').pack(fill='x', pady=(2, 8))

        tk.Label(side, text='游戏坐标（大地图十字坐标）', font=font).pack(anchor='w')
        row = tk.Frame(side)
        row.pack(fill='x')
        self.coord = tk.Entry(row, font=font, width=14)
        self.coord.pack(side='left')
        if embedded:
            self.coord.bind('<Button-1>',lambda event:event.widget.focus_force(),add='+')
        self.coord.bind('<Return>', lambda e: self._coord_dest())
        tk.Button(row, text='设为目的地', font=font, command=self._coord_dest).pack(side='left', padx=4)
        self.coord_error = tk.Label(side, font=('Microsoft YaHei UI', 9), fg='#c62828')
        self.coord_error.pack(anchor='w')

        bases = tk.LabelFrame(side, text='三队初始基地 / 手动起点', font=font, padx=4, pady=3)
        bases.pack(fill='x', pady=(6, 0))
        self.base_choice = tk.StringVar()
        self.base_combo = ttk.Combobox(bases, state='readonly', textvariable=self.base_choice, width=26)
        self.base_combo.pack(fill='x')
        self.base_combo.bind('<<ComboboxSelected>>', self._base_preview)
        self.base_auto = tk.BooleanVar(value=True)
        tk.Checkbutton(bases, text='从小地图自动识别队伍', variable=self.base_auto,
                       command=self._sync_team).pack(anchor='w')
        self.team_hint = tk.Label(bases, text='队伍：识别中', anchor='w')
        self.team_hint.pack(fill='x')
        row = tk.Frame(bases)
        row.pack(fill='x', pady=2)
        self.use_base_button = tk.Button(row, text='我在这个基地', font=font, command=self._use_base)
        self.use_base_button.pack(side='left')
        tk.Button(row, text='点选当前位置', font=font,
                  command=lambda: self._start_position_pick('initial')).pack(side='left', padx=3)
        row = tk.Frame(bases)
        row.pack(fill='x')
        tk.Button(row, text='修正基地位置', font=font, command=self._edit_base).pack(side='left')
        self.base_hint = tk.Label(bases, font=('Microsoft YaHei UI', 9), fg='#795548',
                                  wraplength=260, justify='left')
        self.base_hint.pack(fill='x')
        self._load_bases()
        self._escape_owner = self.win.winfo_toplevel()
        self._escape_binding = self._escape_owner.bind('<Escape>', lambda e: self._cancel_position_pick(),add='+')

        tk.Label(side, text='收藏', font=font).pack(anchor='w', pady=(8, 0))
        self.favs = tk.Listbox(side, font=font, height=5)
        self.favs.pack(fill='x')
        self.favs.bind('<Double-Button-1>', lambda e: self._fav_go())
        bar = tk.Frame(side)
        bar.pack(fill='x', pady=2)
        tk.Button(bar, text='收藏目的地', font=font, command=self._fav_add).pack(side='left')
        tk.Button(bar, text='设为目的地', font=font, command=self._fav_go).pack(side='left', padx=3)
        tk.Button(bar, text='删除', font=font, command=self._fav_delete).pack(side='left')

        self.cruise = tk.IntVar(value=int(pilot.cfg.aircraft.cruise_agl_m))
        frame = tk.Frame(side)
        frame.pack(fill='x', pady=(12, 0))
        tk.Label(frame, text='直升机巡航离地高度 (m)', font=font).pack(side='left')
        spin = ttk.Spinbox(frame, from_=30, to=500, increment=10, width=6, textvariable=self.cruise,
                           command=self._cruise_changed)
        spin.pack(side='left', padx=4)
        spin.bind('<Return>', lambda e: self._cruise_changed())
        spin.bind('<FocusOut>', lambda e: self._cruise_changed())
        if embedded:
            spin.bind('<Button-1>',lambda event:event.widget.focus_force(),add='+')
        self.cruise_speed = tk.IntVar(value=int(pilot.cfg.aircraft.cruise_kmh))
        frame = tk.Frame(side)
        frame.pack(fill='x', pady=(4, 0))
        tk.Label(frame, text='直升机巡航速度 (km/h)', font=font).pack(side='left')
        spin = ttk.Spinbox(frame, from_=60, to=350, increment=10, width=6, textvariable=self.cruise_speed,
                           command=self._speed_changed)
        spin.pack(side='left', padx=4)
        spin.bind('<Return>', lambda e: self._speed_changed())
        spin.bind('<FocusOut>', lambda e: self._speed_changed())
        if embedded:
            spin.bind('<Button-1>',lambda event:event.widget.focus_force(),add='+')

        tk.Button(side, text='定位到我', font=font, command=self.center_on_vehicle).pack(fill='x', pady=(12, 2))
        tk.Button(side, text=f'出发 ({pilot.cfg.keys.toggle})', font=font, bg='#2e7d32', fg='white',
                  command=go_command or pilot.go).pack(fill='x', pady=2)
        if not embedded:
            tk.Button(side, text='关闭', font=font, command=self.close).pack(fill='x', pady=2)
        style_widgets(self.win)
        brand.configure(fg=ACCENT)
        self.sidebar.bind_wheel()
        if not embedded:
            dark_titlebar(self.win)

        # view: map pixel at the canvas centre, screen px per map px
        self.zoom = min(self.cw / self.assets.width, self.ch / self.assets.height)
        self.cx, self.cy = self.assets.width / 2, self.assets.height / 2
        self.photo = None
        self.drag = None
        self.view_dirty = True
        self.canvas.bind('<MouseWheel>', self._wheel)
        self.canvas.bind('<ButtonPress-1>', self._press)
        self.canvas.bind('<B1-Motion>', self._motion_drag)
        self.canvas.bind('<ButtonRelease-1>', self._release)
        self.canvas.bind('<Button-3>', self._right)
        self.canvas.bind('<Motion>', self._hover)
        self.hover_text = ''
        self._load_favs()
        self.center_on_vehicle(zoom=2.0)
        self.alive = True
        self._tick()

    def _resized(self, event):
        if (event.width, event.height) != (self.cw, self.ch) and event.width > 50 and event.height > 50:
            self.cw, self.ch = event.width, event.height
            self.view_dirty = True

    # ------------------------------------------------------------ geometry
    def to_screen(self, p):
        return (p[0] - self.cx) * self.zoom + self.cw / 2, (p[1] - self.cy) * self.zoom + self.ch / 2

    def to_map(self, sx, sy):
        return (sx - self.cw / 2) / self.zoom + self.cx, (sy - self.ch / 2) / self.zoom + self.cy

    def center_on_vehicle(self, zoom=None):
        position = self._vehicle_position()
        self._auto_center_pending = position is None
        if position is not None:
            self.cx, self.cy = position.point
            if zoom:
                self.zoom = zoom
        self.view_dirty = True

    def _vehicle_position(self):
        """A map fix does not require a heading; keep stale fixes visibly distinct."""
        p, now = self.pilot, time.monotonic()
        context = (p.assets.map_id, p.cfg.profile)
        if context != self._position_context:
            self._last_position = None
            self._position_context = context
        if self.assets.map_id != p.assets.map_id:
            return None

        def valid(point, timestamp):
            return (timestamp is not None and all(math.isfinite(v) for v in (*point, timestamp))
                    and 0 <= point[0] < self.assets.width and 0 <= point[1] < self.assets.height)

        candidates = []
        st, fix = p.telemetry.state, p.telemetry.fix
        manual = getattr(p, 'manual_position', None)
        is_manual = manual is not None and manual.map_id == self.assets.map_id
        if st is not None and valid(st.position, st.timestamp):
            estimated = st.heading_source not in ('arrow', 'motion', 'hold', 'fix')
            candidates.append(MapPosition(st.position, st.heading, st.timestamp, estimated, is_manual))
        if is_manual and valid(manual.point, manual.timestamp):
            candidates.append(MapPosition(manual.point, None, manual.timestamp, manual=True))
        if not is_manual and fix is not None and fix.valid and valid(fix.position, fix.timestamp):
            candidates.append(MapPosition(fix.position, fix.heading, fix.timestamp))
        # Opening the window during a dropout should still show the last accepted fix,
        # even when the current telemetry has already discarded its state and fix.
        est = getattr(p, 'estimator', None)
        if est is not None and est.pos is not None and valid(est.pos, est.t):
            candidates.append(MapPosition(tuple(est.pos), est.heading, est.t))
        fresh = next((v for v in candidates[:2] if now - v.timestamp <= 2.), None)
        if fresh is not None:
            self._last_position = fresh
        elif candidates:
            newest = max(candidates, key=lambda v: v.timestamp)
            if self._last_position is None or newest.timestamp > self._last_position.timestamp:
                self._last_position = newest
        return self._last_position

    def _position_style(self, position):
        age = max(0., time.monotonic() - position.timestamp)
        if position.manual:
            if age > 2.:
                return '#ffb300', f'手动初始位置（{age:.0f} 秒前）'
            return '#b388ff', '手动起点推算' if position.estimated else '手动初始位置'
        if age > 2.:
            return '#ffb300', f'最后定位（{age:.0f} 秒前）'
        if position.estimated:
            return '#40c4ff', '推算位置'
        return '#00e676', '当前位置'

    # ------------------------------------------------------------ drawing
    def _render_map(self):
        x0, y0 = self.to_map(0, 0)
        x1, y1 = self.to_map(self.cw, self.ch)
        crop = self.image.crop((int(math.floor(x0)), int(math.floor(y0)), int(math.ceil(x1)), int(math.ceil(y1))))
        # crop() pads outside the image with black; scale exactly onto the canvas
        ox, oy = self.to_screen((math.floor(x0), math.floor(y0)))
        size = (max(1, int(crop.width * self.zoom)), max(1, int(crop.height * self.zoom)))
        resample = Image.NEAREST if self.zoom > 3 else Image.BILINEAR
        self.photo = ImageTk.PhotoImage(crop.resize(size, resample))
        self.canvas.delete('map')
        self.canvas.create_image(ox, oy, image=self.photo, anchor='nw', tags='map')
        self.canvas.tag_lower('map')
        self.view_dirty = False

    def _draw_overlay(self):
        c, p = self.canvas, self.pilot
        c.delete('ovl')
        route = p.telemetry.route
        if len(route) >= 2:
            pts = [v for q in route for v in self.to_screen(q)]
            c.create_line(*pts, fill='#ffb300', width=3, tags='ovl')
        for base in self._team_bases:
            x, y = self.to_screen(base.point)
            c.create_rectangle(x - 8, y - 8, x + 8, y + 8, fill=base.color, outline='white', width=2,
                               tags=('ovl', 'team_base'))
            c.create_text(x, y + 13, text=base.name + '基地', fill=base.color, anchor='n',
                          font=('Microsoft YaHei UI', 9, 'bold'), tags=('ovl', 'team_base_label'))
        for fav in self._favs():
            x, y = self.to_screen(fav['point'])
            c.create_text(x, y, text='★', fill='#fff176', font=('Segoe UI Symbol', 13), tags='ovl')
            c.create_text(x + 10, y + 10, text=fav['name'], fill='#fff176', anchor='w',
                          font=('Microsoft YaHei UI', 9), tags='ovl')
        if p.home:
            x, y = self.to_screen(p.home)
            c.create_oval(x - 8, y - 8, x + 8, y + 8, outline='white', fill='#1565c0', width=2, tags='ovl')
            c.create_text(x, y - 16, text='返航点', fill='white', font=('Microsoft YaHei UI', 9), tags='ovl')
        if p.destination:
            x, y = self.to_screen(p.destination)
            c.create_oval(x - 9, y - 9, x + 9, y + 9, outline='white', fill='#c62828', width=2, tags='ovl')
            c.create_text(x, y - 18, text='目的地', fill='white', font=('Microsoft YaHei UI', 9), tags='ovl')
        position = self._vehicle_position()
        if position is not None:
            x, y = self.to_screen(position.point)
            color, label = self._position_style(position)
            tags = ('ovl', 'vehicle')
            c.create_oval(x - 17, y - 17, x + 17, y + 17, outline='white', width=3, tags=tags)
            if position.heading is not None and math.isfinite(position.heading):
                f = heading_vector(position.heading)
                r = heading_vector(position.heading + 90)
                tri = [x + f[0] * 14, y + f[1] * 14, x - f[0] * 8 + r[0] * 8, y - f[1] * 8 + r[1] * 8,
                       x - f[0] * 8 - r[0] * 8, y - f[1] * 8 - r[1] * 8]
                c.create_polygon(*tri, fill=color, outline='black', tags=tags)
            else:
                c.create_oval(x - 8, y - 8, x + 8, y + 8, fill=color, outline='black', tags=tags)
            c.create_text(x + 23, y, text=label, fill=color, anchor='w',
                          font=('Microsoft YaHei UI', 11, 'bold'), tags=('ovl', 'vehicle_label'))
        if self.hover_text:
            c.create_text(8, self.ch - 8, text=self.hover_text, fill='white', anchor='sw',
                          font=('Consolas', 10), tags='ovl')

    def _tick(self):
        if not self.alive:
            return
        self._finish_map_change()
        if not self._map_changing:
            self.map_choice.set('自动识别' if self.pilot.cfg.auto_map.enabled else self.pilot.assets.map_id.upper())
        if self.assets.map_id != self.pilot.assets.map_id:
            self.assets = self.pilot.assets
            self.image = Image.fromarray(cv2.cvtColor(self.assets.load_image(), cv2.COLOR_BGR2RGB))
            if not self.embedded:
                self.win.title(f'选择目的地 - {self.assets.name}')
            self.zoom = min(self.cw / self.assets.width, self.ch / self.assets.height)
            self.cx, self.cy = self.assets.width / 2, self.assets.height / 2
            self.drag, self.hover_text = None, ''
            self.coord.delete(0, 'end')
            self._load_favs()
            self._load_bases()
            self.view_dirty = True
            self._auto_center_pending = True
        if self._auto_center_pending and self._vehicle_position() is not None:
            self.center_on_vehicle(zoom=2.)
        self._sync_team()
        if self.view_dirty:
            self._render_map()
        self._draw_overlay()
        self._update_info()
        self._tick_timer = self.win.after(300, self._tick)

    def _update_info(self):
        p = self.pilot

        def game(pt):
            if pt is None:
                return '未设置'
            try:
                gx, gy = self.assets.map_to_game(pt)
                return f'x{gx:.2f}, y{gy:.2f}'
            except ValueError:
                return f'地图像素 {pt[0]:.0f}, {pt[1]:.0f}'

        vehicle = '卡车' if p.cfg.profile == 'truck' else '直升机'
        mode = '自动识别' if p.cfg.auto_map.enabled else '手动固定'
        lines = [f'载具：{vehicle}', f'地图：{self.assets.name}（{mode}）',
                 f'目的地：{game(p.destination)}', f'返航点：{game(p.home)}']
        position = self._vehicle_position()
        if position is None:
            lines.insert(1, '我的位置：等待小地图定位')
        else:
            _, label = self._position_style(position)
            lines.insert(1, f'{label}：{game(position.point)}')
            if position.heading is None:
                lines.insert(2, '朝向：尚未识别（圆点仍表示位置）')
        if position is not None and p.destination:
            d = math.hypot(position.point[0] - p.destination[0], position.point[1] - p.destination[1]) * p.mpp
            lines.append(f'直线距离：{d:.0f} m')
        self.info.config(text='\n'.join(lines))

    # ------------------------------------------------------------ mouse
    def _wheel(self, event):
        if event.delta:
            self.zoom_by(1.25 if event.delta > 0 else .8, event.x, event.y)
        return 'break'

    def fit_map(self):
        self._auto_center_pending = False
        self.zoom = min(self.cw / self.assets.width, self.ch / self.assets.height)
        self.cx, self.cy = self.assets.width / 2, self.assets.height / 2
        self._render_map()
        self._draw_overlay()

    def zoom_by(self, factor, x=None, y=None):
        self._auto_center_pending = False
        x = self.cw / 2 if x is None else x
        y = self.ch / 2 if y is None else y
        before = self.to_map(x, y)
        minimum = min(self.cw / self.assets.width, self.ch / self.assets.height) * .5
        self.zoom = min(12.0, max(minimum, self.zoom * factor))
        after = self.to_map(x, y)
        self.cx += before[0] - after[0]
        self.cy += before[1] - after[1]
        self.view_dirty = True
        self._render_map()
        self._draw_overlay()

    def _press(self, event):
        self._auto_center_pending = False
        self.drag = (event.x, event.y, self.cx, self.cy, False)

    def _motion_drag(self, event):
        if self.drag is None:
            return
        x, y, cx, cy, moved = self.drag
        if moved or abs(event.x - x) + abs(event.y - y) > 4:
            self.cx = cx - (event.x - x) / self.zoom
            self.cy = cy - (event.y - y) / self.zoom
            self.drag = (x, y, cx, cy, True)
            self._render_map()
            self._draw_overlay()

    def _release(self, event):
        if self.drag is not None and not self.drag[4]:
            point = self.to_map(event.x, event.y)
            if self._position_action is not None:
                self._finish_position_pick(point)
            else:
                self._set_destination(point)
        self.drag = None

    def _right(self, event):
        if self._position_action is not None:
            self._cancel_position_pick()
            return
        point = self.to_map(event.x, event.y)
        with self.pilot.lock:
            if self.assets.map_id == self.pilot.cfg.map_id and self.assets.contains(point):
                self.pilot.set_home(point)
                self._draw_overlay()

    def _hover(self, event):
        mx, my = self.to_map(event.x, event.y)
        try:
            gx, gy = self.assets.map_to_game((mx, my))
            self.hover_text = f'游戏坐标 x{gx:.2f}, y{gy:.2f}'
        except ValueError:
            self.hover_text = f'地图像素 {mx:.0f}, {my:.0f}'
        self._draw_overlay()

    # ------------------------------------------------------------ actions
    def _load_bases(self):
        from .bases import team_bases
        self._team_bases = team_bases(self.assets, self.pilot.cfg)
        self._base_choices = {b.name + ('（已修正）' if b.custom else ''): b for b in self._team_bases}
        self.base_combo.configure(values=list(self._base_choices))
        self.base_choice.set('')
        self._sync_team()
        self._cancel_position_pick()

    def _sync_team(self):
        from .team_detection import TEAM_NAMES
        if self.base_auto.get():
            reading = getattr(getattr(self.pilot,'sensors',None),'team_reading',None)
            faction = reading.faction_id if reading and time.monotonic()-reading.timestamp <= 3 else None
            label = next((label for label,base in self._base_choices.items() if base.id == faction),'')
            self.base_choice.set(label)
            self.team_hint.config(text=f'队伍：{TEAM_NAMES[faction]}（小地图识别）' if label else '队伍：等待稳定友军标记，可手动选择')
        else:
            self.team_hint.config(text='队伍：手动选择')
        self.use_base_button.config(state='normal' if self.base_choice.get() in self._base_choices else 'disabled')

    def _base_preview(self, event=None):
        self.base_auto.set(False)
        self._sync_team()
        base = self._base_choices.get(self.base_choice.get())
        if base is not None:
            self.cx, self.cy = base.point
            self.zoom = 2.
            self.view_dirty, self._auto_center_pending = True, False

    def _use_base(self):
        base = self._base_choices.get(self.base_choice.get())
        if base is not None:
            self._apply_initial_position(base.point, f'{base.name}基地', base.sigma_m)

    def _apply_initial_position(self, point, source, sigma_m=200.):
        try:
            self.pilot.set_initial_position(point, source=source, map_id=self.assets.map_id, sigma_m=sigma_m)
        except (ValueError, OSError) as error:
            self.base_hint.config(text=f'设置失败：{error}')
            return False
        self._last_position = None
        self.cx, self.cy = point
        self.view_dirty, self._auto_center_pending = True, False
        self._draw_overlay()
        self._update_info()
        self.base_hint.config(text='已提供手动初始位置；视觉定位成功后校正。')
        return True

    def _start_position_pick(self, action):
        self._position_action = action
        self.base_hint.config(text='请在地图点击实际位置；拖动平移，Esc / 右键取消。')

    def _cancel_position_pick(self):
        self._position_action = None
        self.base_hint.config(text='基地为部署终端附近参考点，请确认自己确实在基地内。')

    def _edit_base(self):
        base = self._base_choices.get(self.base_choice.get())
        if base is not None:
            self._start_position_pick(('base', base.id))

    def _finish_position_pick(self, point):
        if self._position_action == 'initial':
            if self._apply_initial_position(point, 'map click'):
                self._position_action = None
            return
        _, faction_id = self._position_action
        try:
            self.pilot.set_base_position(faction_id, point, map_id=self.assets.map_id)
        except (ValueError, OSError) as error:
            self.base_hint.config(text=f'修正失败：{error}')
            return
        self._load_bases()
        self._draw_overlay()
        self.base_hint.config(text='基地位置已保存；不会更改当前车辆位置。')

    def _change_map(self, event=None):
        if self._map_changing:
            return
        choice = self.map_choice.get()
        map_id = None if choice == '自动识别' else self._map_choices.get(choice)
        if choice != '自动识别' and map_id is None:
            return
        self._map_changing = True
        self.map_combo.configure(state='disabled')
        self.map_error.config(text='正在加载地图，驾驶将暂停…')

        def load():
            # Never touch Tk from the worker. Errors and completion are consumed
            # by the regular UI timer, even if this window closes during loading.
            try:
                self.pilot.set_map_mode(map_id)
                error = None
            except Exception as exc:
                error = str(exc)
            self._map_change_result.put(error)

        self._map_change_thread = threading.Thread(target=load,name='manual-map-load',daemon=True)
        self._map_change_thread.start()

    def _finish_map_change(self):
        if not self._map_changing:
            return
        try:
            error = self._map_change_result.get_nowait()
        except queue.Empty:
            return
        self._map_changing = False
        self.map_combo.configure(state='readonly')
        self.map_error.config(text=f'切图失败：{error}' if error is not None else '')
        self.map_choice.set('自动识别' if self.pilot.cfg.auto_map.enabled else self.pilot.assets.map_id.upper())

    def _set_destination(self, point):
        with self.pilot.lock:
            if self.assets.map_id != self.pilot.cfg.map_id or not self.assets.contains(point):
                return
            self.pilot.set_destination(point)
        self._draw_overlay()
        self._update_info()

    def _coord_dest(self):
        from .__main__ import _point
        try:
            point = self.assets.game_to_map(_point(self.coord.get()))
        except Exception as error:
            self.coord_error.config(text=f'坐标无效：{error}')
            return
        self.coord_error.config(text='')
        self._set_destination(point)
        self._auto_center_pending = False
        self.cx, self.cy = point
        self.view_dirty = True

    def _favs(self):
        return self.pilot.cfg.favorites.get(self.pilot.cfg.map_id, [])

    def _load_favs(self):
        self.favs.delete(0, 'end')
        for fav in self._favs():
            self.favs.insert('end', fav['name'])

    def _fav_add(self):
        p = self.pilot
        if p.destination is None:
            return
        name = simpledialog.askstring('收藏', '名称：', parent=self.win)
        if not name:
            return
        p.add_favorite(name.strip(), p.destination)
        self._load_favs()

    def _selected(self):
        sel = self.favs.curselection()
        favs = self._favs()
        return favs[sel[0]] if sel and sel[0] < len(favs) else None

    def _fav_go(self):
        fav = self._selected()
        if fav:
            self._auto_center_pending = False
            self._set_destination(tuple(fav['point']))
            self.cx, self.cy = fav['point']
            self.view_dirty = True

    def _fav_delete(self):
        fav = self._selected()
        if fav:
            self.pilot.remove_favorite(fav['name'])
            self._load_favs()

    def _cruise_changed(self):
        try:
            value = float(self.cruise.get())
        except (tk.TclError, ValueError):
            return
        self.pilot.set_cruise_height(max(30.0, min(500.0, value)))

    def _speed_changed(self):
        try:
            value = float(self.cruise_speed.get())
        except (tk.TclError, ValueError):
            return
        self.pilot.set_cruise_speed(max(60.0, min(350.0, value)))

    def close(self):
        if not self.alive:
            return
        self.alive = False
        timer = getattr(self,'_tick_timer',None)
        if timer:
            self.win.after_cancel(timer)
        if self._escape_binding:
            self._escape_owner.unbind('<Escape>',self._escape_binding)
            self._escape_binding = None
        self.win.destroy()
