"""Small always-on-top control panel: 出发/暂停, 返航, 降落, 设为返航点 + live status (急停: F9 key).

On Windows the window is created with WS_EX_NOACTIVATE, so clicking a button does
not take keyboard focus away from the game (inputs stay enabled).  It is visible
over borderless/windowed games; in exclusive fullscreen use the hotkeys.
"""
from __future__ import annotations

import sys
import tkinter as tk
from .ui_theme import (SURFACE, TEXT, MUTED, ACCENT, GREEN, PURPLE, AMBER, OpacityControl,
                       window_theme, style_widgets, dark_titlebar, save_opacity, ScrollPanel)

STATUS_CN = {'map search': '识别地图中', 'standby': '待机', 'searching': '定位中', 'lost': '定位丢失', 'no goal': '无目标', 'drive': '行驶',
             'paused': '已暂停',
             'fly': '飞行', 'arrived': '已到达', 'kturn': '掉头', 'reverse': '倒车脱困', 'bypass': '绕行',
             'blocked': '道路受阻', 'replan': '重新规划', 'route error': '规划失败', 'takeoff': '起飞爬升',
             'cruise': '巡航', 'approach': '减速进近', 'hover': '悬停', 'pull up': '拉高避地', 'hold': '保持',
             'hud lost': '读不到仪表，悬停等待', 'ground': '地面', 'climb': '垂直爬升',
             'terrain': '地形升高，先爬高', 'no position': '无位置，悬停', 'need position': '等待起点定位',
             'need heading': '等待罗盘读数'}
MISSION_CN = {'idle': '空闲', 'outbound': '前往目的地', 'return': '返航'}
LOCK_CN = {'none': '无', 'acquiring': '正在被锁定', 'locked': '已被锁定！'}


def _no_activate(root: tk.Tk):
    if sys.platform != 'win32':
        return
    try:
        import ctypes
        user32 = ctypes.windll.user32
        hwnd = user32.GetParent(root.winfo_id()) or root.winfo_id()
        GWL_EXSTYLE, WS_EX_NOACTIVATE, WS_EX_TOPMOST, WS_EX_TOOLWINDOW = -20, 0x08000000, 0x8, 0x80
        style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_NOACTIVATE | WS_EX_TOPMOST | WS_EX_TOOLWINDOW)
    except Exception:
        pass


class ControlPanel:
    def __init__(self, pilot, stop_event, *, flares_only: bool = False):
        self.pilot, self.stop_event, self.flares_only = pilot, stop_event, flares_only
        keys = pilot.cfg.keys
        self.root = root = tk.Tk()
        root.title('WARDOGS 自动驾驶')
        root.attributes('-topmost', True)
        root.resizable(True, True)
        root.protocol('WM_DELETE_WINDOW', self.close)
        window_theme(root, pilot.cfg.ui.panel_opacity)
        body = tk.Frame(root)
        body.pack(fill='both',expand=True)
        self.map_host = tk.Frame(body)
        if not flares_only:
            self.map_host.pack(side='left',fill='both',expand=True)
        rail = ScrollPanel(body, height=580)
        rail.pack(side='right',fill='y')
        controls = rail.body
        font = ('Microsoft YaHei UI', 10)
        header = tk.Frame(controls, padx=14, pady=10)
        header.pack(fill='x', padx=8, pady=(8, 1))
        self.brand = tk.Label(header, text='WARDOGS', font=('Segoe UI', 16, 'bold'))
        self.brand.pack(side='left')
        self.mode_badge = tk.Label(header, text='MANUAL', font=('Consolas', 10, 'bold'), padx=10, pady=4)
        self.mode_badge.pack(side='right')
        bar = tk.Frame(controls, padx=8, pady=10)
        bar.pack(fill='x', padx=8)
        if not flares_only:
            self.go_button = tk.Button(bar, text=f'出发 ({keys.toggle})', width=12, font=font, bg='#2e7d32', fg='white',
                                       command=self._then_game(pilot.toggle))
            self.go_button.grid(row=0, column=0, padx=3)
            tk.Button(bar, text=f'返航 ({keys.return_home})', width=12, font=font, bg='#1565c0', fg='white',
                      command=self._then_game(pilot.return_home)).grid(row=0, column=1, padx=3)
            self.land_button = tk.Button(bar, text=f'降落 ({keys.land})', width=12, font=font, bg='#ef6c00',
                                         fg='white', command=self._then_game(pilot.land_here))
            self.land_button.grid(row=0, column=2, padx=3)
        if not flares_only:
            self.switch_button = tk.Button(bar, width=12, font=font, command=pilot.toggle_profile)
            self.switch_button.grid(row=1, column=0, padx=3, pady=(6, 0))
            tk.Button(bar, text='当前位置设为返航点', font=font, command=pilot.set_home_here).grid(
                row=1, column=1, columnspan=2, sticky='we', padx=3, pady=(6, 0))
        self.picker = None
        from .key_display import KeyStateDisplay
        self.key_display = KeyStateDisplay(controls, sent=lambda: set(pilot.keyboard.down))
        self.key_display.pack(fill='x', padx=8, pady=(1, 6))
        self.status = tk.Label(controls, font=('Microsoft YaHei UI', 10), justify='left', anchor='w',
                               padx=12, pady=10, wraplength=500)
        self.status.pack(fill='x', padx=8)
        self.alert = tk.Label(controls, font=('Microsoft YaHei UI', 11, 'bold'), padx=8, pady=2)
        self.alert.pack(fill='x', padx=8, pady=3)
        self.opacity = OpacityControl(controls, root, pilot.cfg.ui.panel_opacity,
            on_save=lambda value: save_opacity(pilot.cfg, 'panel_opacity', value, remember=pilot._remember))
        self.opacity.pack(fill='x', padx=8, pady=(2, 8))
        if not flares_only:
            self.open_picker()
        style_widgets(root)
        self.brand.configure(fg=ACCENT)
        dark_titlebar(root)
        root.update_idletasks()
        rail.canvas.configure(width=controls.winfo_reqwidth())
        rail.bind_wheel()
        root.minsize(controls.winfo_reqwidth() + (560 if not flares_only else 30), 420)
        _no_activate(root)
        root.geometry(f'+{max(0,root.winfo_screenwidth() - root.winfo_reqwidth() - 30)}+40')

    def refresh(self):
        if self.stop_event.is_set():
            self.close()
            return
        p, tel = self.pilot, self.pilot.telemetry
        pending = getattr(p,'pending_go',False)
        self.mode_badge.configure(text='WAIT  /  等待地图' if pending else 'AUTO  /  自动' if p.engaged else 'MANUAL  /  手动',
                                   bg=GREEN if p.engaged else SURFACE, fg=TEXT if p.engaged else MUTED)
        if not self.flares_only:
            if pending:
                text = '取消等待'
            elif p.engaged:
                text = '暂停'
            elif p.mission != 'idle' and not p.arrived:
                text = '继续'
            else:
                text = '出发'
            self.go_button.config(text=f'{text} ({p.cfg.keys.toggle})')
            heli = p.cfg.profile == 'aircraft'
            self.switch_button.config(text=f'{"直升机" if heli else "卡车"}  ·  {p.cfg.keys.switch_vehicle} 切换',
                                      bg=PURPLE if heli else AMBER, fg=TEXT)
        lines = []
        if not self.flares_only:
            lines.append(f'地图: {p.assets.name}' + (f'  {tel.map_status}' if tel.map_status else ''))
            st = tel.state
            lines.append(f'任务: {MISSION_CN.get(p.mission, p.mission)}   状态: {STATUS_CN.get(tel.status, tel.status)}'
                         f'   {"自动驾驶中" if p.engaged else "手动"}')
            hud = tel.hud
            if hud is not None:
                fmt = lambda v: '--' if v is None else f'{v:.0f}'
                lines.append(f'仪表  速度: {fmt(hud.speed_kmh)} km/h   离地: {fmt(hud.agl_m)} m   '
                             f'海拔: {fmt(hud.asl_m)} m   航向: {fmt(hud.heading)}°')
                if st is not None and tel.nav_sigma is not None:
                    src = {'fix': '小地图修正', 'dead reckoning': '罗盘+空速推算', 'return point': '起点=返航点',
                           'fix (reset)': '小地图重置'}.get(st.heading_source, st.heading_source)
                    lines.append(f'位置来源: {src}   估计误差 ±{tel.nav_sigma:.0f} m')
            elif st is not None:
                lines.append(f'速度: {st.speed_ms * 3.6:5.0f} km/h   航向: {st.heading:5.1f}°')
            if tel.remaining_m is not None and p.mission != 'idle':
                lines.append(f'剩余: {tel.remaining_m:6.0f} m')
            lines.append(f'返航点: {"已记录" if p.home else "未记录（首次出发时记录）"}   '
                         f'目的地: {"已设置" if p.destination else "未设置"}')
            if not p.foreground_ok:
                lines.append('游戏窗口不在前台，按键已暂停')
            elif pending:
                lines.append('等待地图确认；确认后开始，再按出发可取消')
            elif not p.engaged:
                lines.append('按键输出：未启用（出发 / 继续后才给油门）')
            elif tel.status in ('need heading','need position','map search','lost','route error','no goal'):
                lines.append(f'按键输出受限：{STATUS_CN.get(tel.status,tel.status)}')
            if tel.message:
                lines.append(f'消息: {tel.message}')
        if p.flares is not None and p.cfg.profile == 'aircraft':
            lines.append(f'导弹告警: {LOCK_CN.get(tel.lock_state, tel.lock_state)}   热诱弹已放: {tel.flares}'
                         f'   自动诱饵弹: {"开" if p.countermeasures_on else "关"}')
        self.status.config(text='\n'.join(lines))
        if tel.lock_state == 'locked':
            self.alert.config(text='⚠ 已被锁定', fg='white', bg='#c62828')
        elif tel.lock_state == 'acquiring':
            self.alert.config(text='⚠ 正在被锁定', fg='black', bg='#ffb300')
        else:
            self.alert.config(text='', bg=self.root.cget('bg'))
        self.root.after(200, self.refresh)

    def _then_game(self, command):
        """Run a button command, then hand the keyboard back to the game: while our panel
        has the focus no key reaches the helicopter (it coasts with everything released)."""
        def run():
            command()
            from .input_win import focus_window
            self.root.after(50, lambda: focus_window(self.pilot.cfg.window_title))
        return run

    def open_picker(self):
        from .picker import MapPicker
        if self.picker is not None and self.picker.alive:
            return
        self.picker = MapPicker(self.map_host, self.pilot, embedded=True, go_command=self._then_game(self.pilot.go))

    def close(self):
        self.stop_event.set()
        if self.picker is not None and self.picker.alive:
            self.picker.close()
        self.root.destroy()

    def run(self):
        self.root.after(200, self.refresh)
        self.root.mainloop()
