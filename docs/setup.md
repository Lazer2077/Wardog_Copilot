# Initial setup / 初始配置

The example contains no usable screen calibration: zero-sized rectangles disable
those inputs. Repeat calibration after resolution, scaling, window or HUD changes.
示例没有可用标定；宽高为零表示禁用。分辨率、缩放、窗口位置或 HUD 改变后须重新标定。

```powershell
if (-not (Test-Path config.json)) { Copy-Item config.example.json config.json }
python -m wardog_ad calibrate --profile aircraft
python -m wardog_ad calibrate --profile aircraft --anchor-only
```

The UI selects the minimap, optional central motion area and helicopter warning
region. `--anchor-only` selects your own position marker, not another vehicle or
the arrow's tip. 标定自己的车辆中心，不是友军、导航点或箭头尖端。

## HUD and compass / 仪表与罗盘

Edit the generated local `config.json` using **absolute physical screen pixels**:

- `hud`: helicopter instrument cluster containing SPD / ALT AGL / ALT ASL. Its
  internal layout must match the reader in `wardog_ad/hud.py`.
- `compass`: the numeric heading box at the top, excluding neighboring labels as
  far as possible. Trucks can use this separately from the helicopter HUD.

填写物理屏幕像素的 left/top/width/height。当前 CLI 不会自动标定这两个矩形；
小地图标定不等于仪表标定。先确认读数，再确认地图、位置、航向和目的地后允许驾驶。

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
