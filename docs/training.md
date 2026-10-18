# Offline imitation learning / 离线模仿学习

This workflow predicts observed keys from image history and available navigation
features. It never sends game input. It is **not** the planned learned local
trajectory planner and has not demonstrated reliable closed-loop avoidance.

此流程是离线按键预测基线，不发送按键，不等同于已完成的局部轨迹规划或避障模型。

## Prepare, train, evaluate / 准备、训练与评估

```powershell
python -m pip install -r requirements-training.txt
python -m wardog_ad dataset-check data/driving/YOUR_SESSION
python -m wardog_ad.imitation prepare data/driving/YOUR_SESSION --vehicle KODIAK --output data/imitation/prepared
python -m wardog_ad.imitation train data/imitation/prepared --output models/imitation/run01 --epochs 12 --device cpu
python -m wardog_ad.imitation predict data/imitation/prepared --checkpoint models/imitation/run01/best.pt --output models/imitation/run01/replay.jsonl
```

Replace `YOUR_SESSION` and the vehicle with the actual recording and confirmed
vehicle. Output directories must be new. Use `--device cuda` with compatible
GPU/PyTorch installations. Review unknown views, menus, collisions and bad fixes.

请使用真实会话和已确认车型；输出目录须为新目录。具备兼容 GPU/PyTorch 环境时可指定
`--device cuda`。视角、失焦、菜单、碰撞、时间与定位质量都需要审核。

Images default to 160×90 RGB with four timestamp-selected history frames roughly
0.2 seconds apart. Excluded frames, long gaps and view changes break sequences.
Missing/stale navigation is masked. Future data and current human actions are
not model inputs.

The tiny CNN + GRU uses weighted binary classification for W/A/S/D/SPACE. Metrics
include per-key precision/recall/F1, supported-class macro F1, exact match and
conflicting axes, plus majority-action and prior-human-action baselines. Select
checkpoints using validation F1, not test F1.

Single-session splits use contiguous time holdouts and two-second boundary guards.
For meaningful generalization, collect multiple independent sessions and hold out
entire routes/maps/sessions. Evaluate first-/third-person views separately.

单会话时间留出不能证明泛化。应增加路口选择、停车、绕障和纠偏示范，并分别评估第一、
第三视角；相邻序列高度重叠，单纯增加相似直行画面帮助有限。

## Hierarchical data / 分层数据

The v2 recorder stores reference routes, local goals, nominal local plans,
counterfactual rule advice and explicit human correction annotations. Nominal
`trajectory_*` fields are map-reference samples, **not** human avoidance paths.
Build trajectory supervision from verified actual position histories with correct
timestamps, coordinate frames and quality masks. Future trajectories are targets
only. Never derive an input command from the action being predicted.

v2 中的规则建议不是专家真值，地图参考线也不是人工绕障轨迹。实际定位轨迹需要处理
延迟、缺失和置信度；未来信息不能进入推理输入。人工纠偏标记不等于真实接管事件，
碰撞和失败纠偏不能自动当作成功专家示范。离线指标不能替代影子预测与闭环测试。

## Optional remote packaging / 可选远程训练

```powershell
python tools/bundle_training.py --dataset data/imitation/prepared --output data/training_bundle.tar.gz
```

Transfer the reviewed low-resolution bundle to a server you control, unpack into
a new directory and use the same training CLI. `tools/run_training_job.py --help`
provides a single-GPU wrapper with persistent status. No server addresses,
credentials or datasets are distributed here.
