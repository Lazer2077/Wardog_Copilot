"""Destination picker for the read-only collector. Does not create Autopilot."""
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from .assets import available_maps, load_map
from .ui_theme import window_theme, style_widgets, dark_titlebar


class CollectionMapPicker:
    def __init__(self, master, collector, *, embedded=False):
        self.c = collector
        self.embedded = embedded
        self.win = tk.Frame(master) if embedded else tk.Toplevel(master)
        if embedded:
            self.win.pack(fill='both',expand=True)
        else:
            self.win.title('采集导航目的地 · 不控制车辆')
            self.win.attributes('-topmost',True)
            window_theme(self.win,collector.cfg.ui.map_opacity)
        self.choice = tk.StringVar(value=(collector.latest_nav or {}).get('map_id') or collector.cfg.map_id)
        self.pin = tk.BooleanVar(value=not collector.cfg.auto_map.enabled)
        self.follow = tk.BooleanVar(value=True)
        tk.Label(self.win,text='战术地图',font=('Microsoft YaHei UI',12,'bold')).pack(anchor='w',padx=10,pady=(10,0))
        bar = tk.Frame(self.win)
        bar.pack(fill='x',padx=10,pady=8)
        self.combo = ttk.Combobox(bar,state='readonly',textvariable=self.choice,
                                  values=available_maps(collector.assets.root),width=18)
        self.combo.pack(side='left')
        self.combo.bind('<<ComboboxSelected>>',self.select_map)
        tk.Checkbutton(bar,text='跟随当前地图',variable=self.follow).pack(side='left',padx=4)
        tk.Checkbutton(self.win,text='设置目标时固定此图（关闭自动切图）',variable=self.pin).pack(anchor='w',padx=10)
        zoom_bar = tk.Frame(self.win)
        zoom_bar.pack(fill='x',padx=10)
        tk.Button(zoom_bar,text='＋',command=lambda: self.zoom_by(1.25)).pack(side='left')
        tk.Button(zoom_bar,text='−',command=lambda: self.zoom_by(.8)).pack(side='left',padx=4)
        tk.Button(zoom_bar,text='全图',command=self.fit_map).pack(side='left')
        tk.Label(zoom_bar,text='滚轮缩放 · 左键拖动平移').pack(side='left',padx=6)
        self.canvas = tk.Canvas(self.win,bg='#111923',highlightthickness=0)
        self.canvas.pack(fill='both',expand=True,padx=10,pady=4)
        self.canvas.bind('<Configure>',self._resized)
        self.canvas.bind('<MouseWheel>',self._wheel)
        self.canvas.bind('<ButtonPress-1>',self._press)
        self.canvas.bind('<B1-Motion>',self._drag)
        self.canvas.bind('<ButtonRelease-1>',self._release)
        self.status = tk.Label(self.win,text='绿点：当前位置 · 橙点：目的地\n蓝线：参考路线；需要改目的地时左键点选。',wraplength=400 if embedded else 720)
        self.status.pack(fill='x',padx=10,pady=5)
        tk.Button(self.win,text='清除本图目的地',command=self.clear).pack(side='left',padx=10,pady=8)
        if not embedded:
            tk.Button(self.win,text='关闭，回到游戏按 F10 开始',command=self.win.destroy).pack(side='right',padx=10,pady=8)
        style_widgets(self.win)
        if not embedded:
            dark_titlebar(self.win)
        self.load()
        self.refresh()

    def load(self):
        self.assets = load_map(self.choice.get(),self.c.assets.root)
        with Image.open(self.assets.image_path) as raw:
            self.image = raw.convert('RGB')
            self.scale = min((400 if self.embedded else 720)/raw.width,(430 if self.embedded else 650)/raw.height,
                             (self.win.winfo_screenwidth()-180)/raw.width,
                             (self.win.winfo_screenheight()-240)/raw.height)
            self.size = int(raw.width*self.scale),int(raw.height*self.scale)
        self.canvas.config(width=self.size[0],height=self.size[1])
        self.zoom = 1.
        self.cx, self.cy = self.image.width/2, self.image.height/2
        self.drag = None
        self._render(max(self.canvas.winfo_width(), self.size[0]),
                     max(self.canvas.winfo_height(), self.size[1]))

    def _resized(self,event):
        if hasattr(self,'image'):
            self._render(event.width,event.height)

    def _render(self,width,height):
        self.cw, self.ch = max(1,width),max(1,height)
        self.scale = min(self.cw/self.image.width,self.ch/self.image.height)*self.zoom
        x0,y0 = self.to_map(0,0)
        viewport = self.image.transform((self.cw,self.ch),Image.Transform.AFFINE,
            (1/self.scale,0,x0,0,1/self.scale,y0),Image.Resampling.BILINEAR)
        self.photo = ImageTk.PhotoImage(viewport)
        self.canvas.delete('map')
        self.canvas.create_image(0,0,image=self.photo,anchor='nw',tags='map')
        self.canvas.tag_lower('map')
        self._draw_overlay()

    def to_map(self,x,y):
        return (x-self.cw/2)/self.scale+self.cx,(y-self.ch/2)/self.scale+self.cy

    def to_screen(self,point):
        return (point[0]-self.cx)*self.scale+self.cw/2,(point[1]-self.cy)*self.scale+self.ch/2

    def fit_map(self):
        self.zoom = 1.
        self.cx,self.cy = self.image.width/2,self.image.height/2
        self._render(self.cw,self.ch)

    def zoom_by(self,factor,x=None,y=None):
        x,y = (self.cw/2 if x is None else x),(self.ch/2 if y is None else y)
        before = self.to_map(x,y)
        self.zoom = min(32.,max(.5,self.zoom*factor))
        self.scale = min(self.cw/self.image.width,self.ch/self.image.height)*self.zoom
        after = self.to_map(x,y)
        self.cx += before[0]-after[0]
        self.cy += before[1]-after[1]
        self._render(self.cw,self.ch)

    def _wheel(self,event):
        if event.delta:
            self.zoom_by(1.25 if event.delta>0 else .8,event.x,event.y)
        return 'break'

    def _press(self,event):
        self.drag = (event.x,event.y,self.cx,self.cy,False)

    def _drag(self,event):
        if self.drag is None:
            return
        x,y,cx,cy,moved = self.drag
        if moved or abs(event.x-x)+abs(event.y-y)>4:
            self.cx,self.cy = cx-(event.x-x)/self.scale,cy-(event.y-y)/self.scale
            self.drag = (x,y,cx,cy,True)
            self._render(self.cw,self.ch)

    def _release(self,event):
        if self.drag is not None and not self.drag[4]:
            self.choose(event)
        self.drag = None

    def select_map(self,event=None):
        self.follow.set(False)  # allow manual inspection without automatic snap-back
        self.load()

    def pause_for_edit(self):
        if not self.c.paused:
            self.c.toggle_pause()

    def choose(self,event):
        point = list(self.to_map(event.x,event.y))
        if not (0 <= point[0] < self.image.width and 0 <= point[1] < self.image.height):
            return  # Letterboxed space is outside the map.
        try:
            self.pause_for_edit()
            self.c.set_destination(self.assets.map_id,point,pin_map=self.pin.get())
            self.status.config(text=f'已保存 {self.assets.map_id} 目的地 ({point[0]:.1f}, {point[1]:.1f})。回到游戏后按 F10 恢复采集。')
        except (ValueError,OSError) as exc:
            self.status.config(text=str(exc))

    def clear(self):
        self.pause_for_edit()
        self.c.set_destination(self.assets.map_id,None,pin_map=self.pin.get())
        self.status.config(text='已清除本图目的地；此时采集将缺少导航条件。')

    def refresh(self):
        if not self.win.winfo_exists():
            return
        nav = self.c.latest_nav or {}
        fresh = (nav.get('context_id') == self.c.context_id and 0 <= self.c.now()-nav.get('t',-100) <= 1.)
        follow_map = nav.get('map_id') if fresh else None
        if not self.c.cfg.auto_map.enabled:
            follow_map = self.c.cfg.map_id
        if self.follow.get() and follow_map and follow_map != self.assets.map_id:
            self.choice.set(follow_map)
            self.load()
        self._draw_overlay()
        self.win.after(250,self.refresh)

    def _draw_overlay(self):
        nav = self.c.latest_nav or {}
        self.canvas.delete('overlay')
        goal = self.c.cfg.destinations.get(self.assets.map_id)
        if (nav.get('map_id') == self.assets.map_id and nav.get('context_id') == self.c.context_id
                and 0 <= self.c.now()-nav.get('t',-100) <= 1.):
            state = nav.get('state')
            fix = nav.get('fix')
            position = state or (fix if fix and fix.get('valid') else None)
            route = getattr(self.c,'reference_routes',{}).get(nav.get('route_id')) or {}
            if route.get('map_id') == self.assets.map_id and len(route.get('points',[]))>1:
                self.canvas.create_line(*[v for p in route['points'] for v in self.to_screen(p)],fill='#4977ac',width=2,tags='overlay')
            points = (nav.get('local_plan') or {}).get('reference_map_px',[])
            if len(points)>1:
                self.canvas.create_line(*[v for p in points for v in self.to_screen(p)],fill='#64b5f6',width=3,tags='overlay')
            if position:
                self.dot([position['x'],position['y']],'#60dfca',5)
        if goal:
            self.dot(goal,'#ffbd69',7)

    def dot(self,point,color,radius):
        x,y = self.to_screen(point)
        self.canvas.create_oval(x-radius,y-radius,x+radius,y+radius,fill=color,outline='white',tags='overlay')
