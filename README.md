# WARDOGS Copilot v0.1

**English** | [简体中文](README.zh-CN.md)

**Current version: v0.1 — an early experimental prototype.** Features and interfaces are still evolving, and real-game reliability requires further validation.

A **screen- and audio-based research prototype** for vehicle navigation and control in WARDOGS. The current focus is helicopter navigation, automatic flares and J-hook maneuvers. Ground-vehicle autonomy and human-demonstration collection are under active development.

**For non-commercial scientific research, technical exchange and personal learning only. Commercial use is strictly prohibited.** This is experimental code, not a finished autopilot. Mission completion, obstacle avoidance, landing and missile evasion are not guaranteed. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Acknowledgments

Special thanks to **[greatredo and the wardogs-navigator contributors](https://github.com/greatredo/wardogs-navigator)**. Their work provides an important foundation for this project's map navigation: map imagery, precomputed SIFT features, road networks and coordinate transforms are loaded from a separate local checkout of wardogs-navigator. Our minimap localization code adapts its SIFT matching and arrow-heading approach.

Please credit the upstream project when building on these contributions. Upstream code retains its Apache-2.0 license; map assets retain their respective rights. Attribution and retained license notices are in [NOTICE](NOTICE) and [LICENSES](LICENSES).

## Observation and control boundaries

- Inputs: external screenshots, minimap imagery, HUD/compass OCR, speaker loopback audio and key states reported by Windows.
- Outputs: ordinary simulated keyboard input; continuous commands become key pulses through PWM.
- **No game-process memory reading/writing, internal game-state extraction, DLL injection, game hooks or anti-cheat bypass.**
- Limited Windows process queries only verify the executable identity of the foreground window; they do not request game-memory read/write permissions.

External observation does not imply that the game permits automation. Use only in authorized, controlled research environments and follow applicable game rules. This project provides no anti-detection functionality and is not affiliated with the game's developers or publisher.

## Implemented research features

| Component | Capabilities and limitations |
|---|---|
| Localization | SIFT matching, RANSAC affine estimation and conditional terrain alignment refinement; position and heading are separate observations |
| Map management | Automatic identification or manual pinning of OZETI, BAKURANI and ZESTAFONA; discard old-map missions and coordinates on changes |
| State estimation | Position fusion, speed, heading and yaw rate; helicopter HUD/dead reckoning with delayed-fix compensation |
| Tactical UI | Embedded live map, position, route, destination and home; zoom, pan, team-base references, manual initial position and favorites |
| Planning | Road-graph A* and path resampling for trucks; helicopter waypoint, destination, return-home and land-here missions |
| Evaluation | Simplified closed-loop simulator, regression tests, control logs and offline replay; simulation is not the game's actual physics |

### Helicopter navigation and automatic flares

The controller implements takeoff, cruise, waypoint tracking, return, approach, hover and landing states, with speed/altitude regulation, terrain-clearance logic and HUD-loss handling. Behavior depends on calibration, localization quality and aircraft parameters.

The current default is predictive velocity-vector guidance: desired versus measured/estimated ground velocity determines acceleration demand, coordinated through yaw and pitch to account for sideslip. It relies on approximate dynamics and still needs real-game validation.

**Automatic flares** use locally recorded examples of lock-warning and incoming-missile audio, optionally combined with a visual warning. The policy supports warning stages, trigger delays, bursts, shot intervals, cooldowns and an optional capacity limit. Without a missile-sound example, it can fall back to lock-tone triggering. Detection and countermeasures can also run independently of navigation. No missile entity data is read, and successful evasion is not guaranteed.

**Two experimental J-hook behaviors:**

- **High-speed redirection/return:** legacy J-hook yaw/velocity-direction logic remains available; default velocity-vector guidance handles redirection instead of invoking that separate legacy mode.
- **J-hook landing:** approach beside the landing point; schedule braking, yaw and descent from estimated speed, distance and response lag, then transition toward hover/vertical touchdown.

These are game-specific control heuristics, not flight-safety-validated aviation algorithms.

### Ground vehicles: work in progress

Available components include map routing, Pure Pursuit/yaw feedback, speed control, stopping and basic recovery, a local-planning interface, a read-only human-driving recorder and an offline imitation-learning baseline. **Reliable visual obstacle perception, avoidance and cross-vehicle generalization are not complete.** The key-prediction model is not connected to online driving.

Planned architecture:

```text
Map localization + destination -> global reference route
                                          |
Image history + speed + local route -> perception / learned local planning
                                          |
                         local trajectory + target speed / stop intent
                                          |
                              rule-based tracking controller -> WASD
```

Learning is intended to handle local traversability, avoidance and path adjustment. Rule-based control executes trajectories and constraints. The current nominal local planner only samples the map reference and explicitly marks visual obstacle checking as unavailable.

Design references, **not claims of reproduction, weight transfer or equivalent results**:

- [EuroPilot](https://github.com/marsauto/europilot): game screenshots paired with human controls for behavior cloning.
- [TMRL](https://github.com/trackmania-rl/tmrl): low-resolution temporal input and real-time action-delay handling; its RL framework is not the current training implementation here.
- [CARLA Garage](https://github.com/autonomousvision/carla_garage): separate expert collection, training and closed-loop evaluation; sensor assumptions differ.
- [TCP](https://github.com/OpenDriveLab/TCP): trajectory/control prediction ideas. An offline probing adapter is included, without checkpoints or default online execution.
- [DAgger](https://arxiv.org/abs/1011.0686): planned corrective demonstrations/data aggregation to address closed-loop distribution shift.

## Installation and launch

Live capture/control requires Windows, an accessible graphical desktop and Python 3.10+. Start with simulation or UI preview.

```powershell
git clone https://github.com/Lazer2077/Wardog_Copilot.git
cd Wardog_Copilot
python -m pip install -r requirements.txt
git clone --depth 1 https://github.com/greatredo/wardogs-navigator third_party/wardogs-navigator
python -m wardog_ad --help
```

Map images, SIFT features and road graphs are obtained separately and **are not bundled**. Follow upstream licenses and asset-rights notices.

Calibrate each vehicle; repeat after resolution or HUD-layout changes:

```powershell
python -m wardog_ad calibrate --profile aircraft
python -m wardog_ad calibrate --profile aircraft --anchor-only
python -m wardog_ad calibrate --profile truck
```

Helicopter HUD/compass regions and audio examples also need configuration. See the [setup guide](docs/setup.md), `audio-record --help`, `audio-fit --help` and `hud-test`.

```powershell
python -m wardog_ad drive --profile aircraft --panel
# Or double-click Wardog_launcher.vbs (no terminal); navigation does not depart immediately.
python tools/ui_preview.py --mode control --seconds 30
```

Navigation defaults: F8 go/pause/resume, F7 return home, F6 switch vehicle, F9 emergency stop, F10 helicopter land-here. Verify configuration and game bindings. Select a destination before departing. Standalone countermeasure mode has different activation semantics from navigation.

## Human demonstrations and learning

Run the recorder separately from the autopilot to avoid contaminating human demonstrations with injected actions.

```powershell
python -m wardog_ad collect --vehicle KODIAK --camera-view unknown --fps 15 --max-width 512
```

Recording starts paused. Confirm first-/third-person view and a destination on the embedded map. The recorder stores images, observed keys, speed/source, reference routes, local targets, rule advice and correction events. It **does not send keys**; advice and human actions remain separate. See the [collection guide in Chinese](docs/分层导航采集.md).

Recorder hotkeys differ: F10 record/pause, F11 view label, F12 save/exit, F8 correction episode, F7 obstacle marker, F9 collision marker. Fully manual correction annotations are not actual autonomous-to-human control transfers.

The offline baseline uses low-resolution CNN + GRU key prediction, temporal holdouts and majority-action/human-history baselines. Learned local-trajectory planning remains future work. See [training workflow](docs/training.md).

## Contents, testing and license

Only source, tests, documentation, required team-base coordinate metadata and attribution are distributed. **No recorded datasets, logs, weights, screenshots, audio examples, personal configuration or server details.**

```powershell
python -m pytest tests/test_core.py tests/test_truck_control.py tests/test_truck_startup.py -q
```

Map/simulation tests need external assets; UI tests need a graphical desktop; training tests need PyTorch. Passing tests does not demonstrate reliable real-game autonomous driving.

Original portions use a custom [Non-Commercial Research License](LICENSE). This is source-available research software, not an unrestricted commercial-use license. Third-party portions retain their original licenses; project restrictions do not override rights already granted by those licenses.

Localization references [wardogs-navigator](https://github.com/greatredo/wardogs-navigator); team-base coordinates reference [wardogs-calculator](https://github.com/apollyon-sys/wardogs-calculator). See [NOTICE](NOTICE), [LICENSES](LICENSES) and component notices.
