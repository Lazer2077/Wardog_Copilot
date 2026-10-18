# Initial setup / 初始配置

The example contains no usable screen calibration: zero-sized rectangles disable
those inputs. New interactive calibrations save a `reference` monitor rectangle
(`[left, top, width, height]`). Captures scale automatically for resolution changes
on the same display with the same aspect ratio, including collector navigation crops.
Window position, HUD layout, aspect-ratio changes or ambiguous monitor changes still
require recalibration. Old regions without `reference` remain absolute pixels.
示例没有可用标定；宽高为零表示禁用。新标定自动保存显示器位置与分辨率 `reference`，
同屏幕、同宽高比的分辨率变化会自动换算截屏区域，采集器导航裁剪也使用相同换算。
窗口位置、HUD 布局、宽高比改变或多屏归属不明确时仍需重新标定。
旧配置没有 `reference` 时仍按绝对像素读取，越界会提示错误，不会自动猜测旧分辨率。

```powershell
if (-not (Test-Path config.json)) { Copy-Item config.example.json config.json }
python -m wardog_ad calibrate --profile aircraft
python -m wardog_ad calibrate --profile aircraft --anchor-only
```

The UI selects the minimap, optional central motion area and helicopter warning
region. `--anchor-only` selects your own position marker, not another vehicle or
the arrow's tip. 标定自己的车辆中心，不是友军、导航点或箭头尖端。

The map panel defaults to automatic faction selection from several small coloured
friendly markers in the minimap, confirmed across multiple frames. Large zone
tints, score bars and the CONTROL line are excluded. This is a heuristic for the
default faction colours; custom colours, missing markers or conflicting markers
can leave it unconfirmed. Manual selection remains available. Selecting a faction
does **not** establish your position at that faction's HQ; only use “我在这个基地”
when you are actually there. Both recorder navigation metadata and flight logs
include the inferred faction, confidence and observation time.

地图面板默认自动识别队伍并同步基地选项，以多个小型友军标记的颜色及连续多帧一致性判断。
不把大面积区域着色、比分条或“控制”占领方当作所属队伍。该启发式方法适用于默认阵营颜色；
自定义颜色、标记不足或冲突时可能无法确认，仍可手动修正。
**识别队伍不代表已经定位到基地**；只有确实在基地内，才使用“我在这个基地”提供初始位置。
采集器导航数据与飞行日志会记录队伍、置信度和观测时间。

## Truck recovery / 卡车脱困

Automatic stuck recovery requires a calibrated main-view crop, finite low visual
motion, sustained throttle, window-estimated speed no higher than 0.75 m/s and insufficient movement.
Missing visual input disables stuck-triggered reversing. Defaults: 6 s startup
grace, 8 s cooldown after each manoeuvre, and at most 2 reverse/K-turn attempts
within a 30 m area. The vehicle stops and requests manual control at the limit.
Pause, localization loss and replanning preserve the attempt budget; driving more
than 30 m away restores it. These settings are under `truck` in the local config:
`startup_grace_s`, `recovery_cooldown_s`, `max_reverse_attempts`, `recovery_reset_m`.

卡车倒车脱困需要已标定的前方画面、有效的低视觉运动、持续给油、时间窗口内拟合车速不超过 0.75 m/s，
以及位移不足同时成立；没有视觉输入时不会因“被困”自动倒车。
默认起步宽限 6 秒，每次机动结束后冷却 8 秒，同一区域最多倒车 / K-turn 2 次，
达到上限后停车交还人工。暂停、定位丢失和重规划不会刷新次数，驶离原脱困区域 30 米后恢复。
对应配置为 `truck.startup_grace_s`、`recovery_cooldown_s`、
`max_reverse_attempts`、`recovery_reset_m`。

Flight logs now include per-map candidate confidence, inliers, residual and score
under `map_candidates`. A local match or an old `map_id` label alone is not proof
of the current map's identity. 飞行日志的 `map_candidates` 记录各地图候选的置信度、
内点数、误差及得分，用于核对错图与歧义；旧的地图名称标签本身不能作为识别成功的证据。

## HUD and compass / 仪表与罗盘

Edit the generated local `config.json` using **absolute physical screen pixels**:

- `hud`: helicopter instrument cluster containing SPD / ALT AGL / ALT ASL. Its
  internal layout must match the reader in `wardog_ad/hud.py`.
- `compass`: the numeric heading box at the top, excluding neighboring labels as
  far as possible. Trucks can use this separately from the helicopter HUD.

填写物理屏幕像素的 left/top/width/height。当前 CLI 不会自动标定这两个矩形；
小地图标定不等于仪表标定。先确认读数，再确认地图、位置、航向和目的地后允许驾驶。
For manually entered HUD/compass regions, also set `reference` to the display
rectangle used for those coordinates (for example `[0, 0, 3840, 2160]` for a 4K
primary display). 手填 HUD / 罗盘区域时也应填写这些坐标所属显示器的 `reference`。

```powershell
python -m wardog_ad hud-test
python -m wardog_ad drive --profile aircraft --panel --no-flares
```

## Audio and flares / 声音与热诱弹

```powershell
python -m wardog_ad audio-record --kind background --seconds 8
python -m wardog_ad audio-record --kind acquire --seconds 8
python -m wardog_ad audio-record --kind locked --seconds 8
# Optional / 可选
python -m wardog_ad audio-record --kind missile --seconds 8
python -m wardog_ad audio-fit
python -m wardog_ad audio-test
```

Record each corresponding sound when it is actually present. `audio-fit` saves a
local profile and path. Validate detection, then set `aircraft.lock_source` to
`audio`, or `both` with calibrated visual warning input. The example defaults to
`none`. Default flare key is V; match actual game bindings.

录制与标签对应的实际声音；验证后再把 `aircraft.lock_source` 改为 `audio`，视觉告警也
标定好后可选 `both`。默认热诱弹键 V 须匹配实际设置。音频示范、拟合参数和本机配置
均不上传仓库。
