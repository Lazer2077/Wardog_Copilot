# WARDOGS Copilot v0.1

[English](README.md) | **简体中文**

**当前仅为 v0.1 早期实验版本。** 功能与接口仍在持续开发，实际游戏中的可靠性尚需进一步验证。

基于 **游戏画面、HUD 与扬声器回采声音** 的载具导航及控制研究原型。当前主要研究直升机导航、自动热诱弹与 J-hook 机动，同时开发地面车辆的分层自动驾驶和人工示范采集流程。

**仅供非商业科研、技术交流与个人学习，严禁商用。** 这是实验性研究代码，不是成熟的自动驾驶产品，也不保证任务完成、避障、着陆或导弹规避成功。使用条款见 [LICENSE](LICENSE)，第三方声明见 [NOTICE](NOTICE)。

## 致谢

特别感谢 **[greatredo 及 wardogs-navigator 的贡献者](https://github.com/greatredo/wardogs-navigator)**。该项目为本项目的地图导航提供了重要基础：地图图像、预计算 SIFT 特征、道路网络和坐标转换数据均从单独下载的 wardogs-navigator 本地仓库加载；本项目的小地图定位代码也参考并改编了其 SIFT 匹配与箭头朝向识别方法。

基于这些成果开展后续工作时，请保留对上游项目的署名。上游代码继续遵守 Apache-2.0 许可，地图素材保留各自权利；详细署名与保留的许可声明见 [NOTICE](NOTICE) 和 [LICENSES](LICENSES)。

## 程序如何获取信息

- 输入：外部屏幕截图、小地图、HUD 数字与罗盘 OCR、扬声器声音回采，以及系统报告的按键状态。
- 输出：普通 Windows 模拟按键；连续控制量通过按键 PWM 执行。
- **不读取或写入游戏进程内存，不提取游戏内部状态，不做 DLL 注入，不 Hook 游戏，不绕过反作弊。**
- Windows 进程查询只用于验证前台窗口对应的可执行文件身份；代码使用受限查询权限，不申请游戏内存读写权限。

不读取内存不代表游戏允许自动化。请仅在获得许可的受控研究环境中使用，自行遵守游戏规则。本项目不提供反检测功能，与游戏开发商、发行商无隶属关系。

## 已实现的研究功能

| 模块 | 功能与边界 |
|---|---|
| 视觉定位 | SIFT 特征、匹配筛选、RANSAC 仿射估计及有条件的地形精细对齐；位置和朝向分别估计 |
| 地图管理 | OZETI / BAKURANI / ZESTAFONA 自动识别与手动固定；换图清除旧图任务和坐标 |
| 状态估计 | 小地图位置融合、速度/航向/横摆估计；直升机结合 HUD 与航位推算，补偿延迟定位 |
| 战术地图 | 主面板自动内嵌地图，显示位置、路线、目的地和返航点；支持缩放、平移、基地参考点、手动起点和收藏 |
| 任务规划 | 卡车道路图 A* 与路径重采样；直升机航点、目的地、返航和就地降落任务 |
| 仿真与记录 | 简化闭环模拟器、回归测试、控制日志和离线回放；仿真不等于真实游戏物理 |

### 直升机自动驾驶

实现了起飞、巡航、航点跟踪、返航、进近、悬停与着陆等状态控制，以及高度/速度调节、地形净空和仪表丢失处理。控制依赖 HUD 标定、定位质量与具体机型，参数需要实验调整。

当前默认使用预测式速度矢量制导，根据期望地速与实测/推算地速的差异计算加速度需求，再协调偏航与俯仰，处理机头方向和运动方向不一致的侧滑情况。它依赖近似动力学模型，仍需真实游戏验证。

**自动热诱弹**使用本地音频示范学习锁定告警与来袭声音特征，可结合画面告警进行判断。策略支持告警阶段、触发延迟、连发间隔、冷却和可选数量限制；未采集来袭声音时可退回锁定音触发。检测与投放逻辑也可单独运行。它不读取导弹实体数据，不能保证拦截或规避成功。

**J-hook 有两类实验逻辑：**

- 高速改航/返航：保留偏航与速度方向调节的 J-hook 逻辑；默认开启速度矢量制导时，由制导器接管这部分改航，不使用旧的独立高速掉头模式。
- J-hook 进近着陆：从着陆点侧方进近，根据估算速度、剩余距离和响应延迟安排制动、偏航与下降，再转入悬停/垂直着陆阶段。

这些是针对游戏的控制启发式，不是经过飞行安全验证的真实航空算法。

### 地面车辆：正在开发

目前已有地图路线规划、Pure Pursuit/横摆反馈、速度控制、停车与部分脱困状态机，另有局部规划接口、只读人工采集器和离线模仿学习基线。**可靠的视觉障碍感知、绕障规划以及跨车辆泛化尚未完成。** 现有按键预测网络没有接管在线驾驶，不应视为已具备避障能力。

计划采用以下分层结构：

```text
地图定位 + 目的地 → 全局参考路线
                            ↓
画面历史 + 速度 + 局部路线 → 视觉感知 / 学习型局部规划
                            ↓
                  局部轨迹 + 目标速度 / 停车意图
                            ↓
                   规则跟踪控制器 → WASD
```

学习模型主要处理局部可通行性、绕障和路径调整；规则层负责执行与约束。当前局部规划基线只采样地图参考线，明确标记尚未做视觉障碍检查。

研发参考如下，**参考设计不等于已经移植、复现或获得相同效果**：

- [EuroPilot](https://github.com/marsauto/europilot)：游戏画面与人工控制标签的配对采集、行为克隆。
- [TMRL](https://github.com/trackmania-rl/tmrl)：低分辨率时序输入与实时交互中的动作延迟处理；其强化学习框架不是本项目当前的训练实现。
- [CARLA Garage](https://github.com/autonomousvision/carla_garage)：专家数据、训练和闭环评估的工程流程；其传感器条件与本项目不同。
- [TCP](https://github.com/OpenDriveLab/TCP)：轨迹与控制预测思路；本项目保留离线探测适配器，不附带权重，不默认在线执行。
- [DAgger](https://arxiv.org/abs/1011.0686)：后续计划使用人工纠正片段与数据聚合，缓解闭环状态分布偏移。

## 安装与运行

实时采集/控制需要 Windows、可访问的桌面、Python 3.10+ 和图形界面。推荐先使用仿真或界面预览。

```powershell
git clone https://github.com/Lazer2077/Wardog_Copilot.git
cd Wardog_Copilot
python -m pip install -r requirements.txt
git clone --depth 1 https://github.com/greatredo/wardogs-navigator third_party/wardogs-navigator
python -m wardog_ad --help
```

第三方地图、SIFT 特征与路网需要单独取得，**不包含在本仓库中**。遵守上游许可证与地图素材权利声明。

在相应载具中标定小地图与 HUD；改变分辨率或界面布局后须重新标定：

```powershell
python -m wardog_ad calibrate --profile aircraft
python -m wardog_ad calibrate --profile aircraft --anchor-only
python -m wardog_ad calibrate --profile truck
python -m wardog_ad --help
```

按[初始配置说明](docs/setup.md)在本机 `config.json` 中设置 HUD 与罗盘的截图范围。音频流程见 `audio-record --help`、`audio-fit --help`，仪表检查使用 `hud-test`。

```powershell
python -m wardog_ad drive --profile aircraft --panel
# 或双击 Wardog_launcher.vbs（无终端窗口）；默认不直接出发
python tools/ui_preview.py --mode control --seconds 30
```

默认热键：F8 出发/暂停/恢复，F7 返航，F6 切换载具，F9 急停，F10 直升机就地降落。以配置与游戏实际键位为准。地图中设置目的地后再开始；不要把热诱弹模式的启动等同于导航模式启动。

## 人工驾驶数据与学习

采集器与自动驾驶程序应分别运行，避免自动注入的按键混入人工示范。

```powershell
python -m wardog_ad collect --vehicle KODIAK --camera-view unknown --fps 15 --max-width 512
```

默认暂停启动，确认第一/第三视角并在内嵌地图设置目的地后开始。记录画面、实际按键、速度与来源、完整参考路线、局部目标、规则控制建议及纠偏事件。采集器**不发送按键**；规则建议与人工控制分开保存。详见 [分层导航采集](docs/分层导航采集.md)。

采集模式热键不同：F10 开始/暂停，F11 视角标签，F12 保存退出，F8 纠偏段起止，F7 障碍标记，F9 碰撞标记。全程手动驾驶下的纠偏标记不是实际自动/人工控制权交接。

离线训练使用低分辨率 CNN＋GRU 按键预测基线，按时间留出验证/测试，并与多数动作及人工历史按键基线比较。它是工程起点，后续局部轨迹学习尚待完善。使用方法见 [训练流程](docs/training.md)。

## 仓库内容、测试与许可

仅发布源码、测试、文档、必要的基地坐标元数据和第三方署名。**不发布录制数据、日志、模型权重、截图、音频样本、个人配置或服务器信息。**

```powershell
python -m pytest tests/test_core.py tests/test_truck_control.py tests/test_truck_startup.py -q
```

更完整的地图/仿真测试依赖外部地图素材，UI 测试依赖图形桌面，训练测试依赖 PyTorch。通过单测不代表真实游戏中可可靠自动驾驶。

本项目原创部分采用自定义[非商业科研许可](LICENSE)，不是无条件允许商用的许可证。第三方部分继续遵守各自许可证，不能对它们已有的授权追加本项目的限制。地图定位参考 [wardogs-navigator](https://github.com/greatredo/wardogs-navigator)，基地坐标参考 [wardogs-calculator](https://github.com/apollyon-sys/wardogs-calculator)，具体声明与许可证见 [NOTICE](NOTICE)。
