# Lab 1 可执行基线方案

## 目标与交付物

本方案让模拟器和真实 Sphero 在 100 个、每个 0.1 s 的控制周期中分别运行同一个 PD 反馈策略。控制器根据各自状态计算动作，目标位置均为相对起点 `(0.5 m, 0.5 m)`。真实运行结束后生成自动评分器要求的：

```text
<student-id>_lab1.csv
```

文件严格包含 `sim_x, sim_y, real_x, real_y` 四列和 100 行数据。

## 控制方案

外环采用距离 PD 控制：

```text
distance = ||target - position||
speed = Kp * distance + Kd * d(distance)/dt
heading = atan2(error_x, error_y)
```

这里航向角 `0 rad` 对应 `+y`，所以参数顺序不是常见的 `atan2(y, x)`。速度限制为 `0.15 m/s`；进入目标中心半径 `0.025 m` 后速度置零。微分项经过低通滤波，以降低真实里程计量化和噪声造成的速度抖动。

## 模拟器模型

`dynamics.py` 使用：

- 一阶速度响应；
- 加速和减速率限制；
- 最大转向率限制；
- 速度命令死区；
- 中点积分计算位置。

这比原始代码中不断累加速度命令更符合“输入为期望速度”的接口定义。当前参数已使用 BP-2E84 的 100 步真实日志标定：相同命令离线重放时，轨迹 RMSE 为 `0.0149 m`，模拟最终目标误差为 `0.0237 m`。

## 先做无硬件验证

在仓库根目录运行：

```powershell
.\.venv\Scripts\Activate.ps1
python labs\lab1\lab1.py --sim
```

如果不需要动画窗口：

```powershell
python labs\lab1\lab1.py --sim --no-render
```

模拟结果写入 `labs/lab1/logs/`，它不是最终提交文件。

## 实验室真实运行

1. 准备至少 `1 m x 1 m` 的平整区域。
2. 唤醒 Sphero，放在起点并保持静止。
3. 将下面的 `12345678` 换成真实学号：

```powershell
python labs\lab1\lab1.py --student-id 12345678
```

BP-2E84 已完成低速标定：原始速度 `15/255` 在 1 秒内移动 `0.140 m`。正式运行把该值固定为硬件上限，并从低速逐步爬升，不会在第一步直接发送满功率命令。

不需要动画时运行：

```powershell
python labs\lab1\lab1.py --student-id 12345678 --no-render
```

程序会扫描并要求确认蓝牙机器人。运行结束后，立即检查：

如果 Windows BLE 单次连接超时，程序会自动重试最多 3 次，每次间隔 3 秒。重试期间保持机器人唤醒并靠近电脑；连接完成前程序不会发送运动命令。

```powershell
python labs\lab1\analyze_lab1.py labs\lab1\12345678_lab1.csv
```

该命令验证列名、100 行数据和有限数值，并计算：

- 真实机器人最终目标距离，要求不超过 `0.10 m`；
- 模拟器最终目标距离，要求不超过 `0.10 m`；
- 模拟与真实轨迹 RMSE，要求不超过 `0.20 m`。

同时会生成轨迹和逐步误差图。

## 用真实日志标定

只改变 `dynamics.py` 顶部的 `MODEL_CONFIG`，每次只调一到两个参数：

| 现象 | 优先调整 |
| --- | --- |
| 模拟全程比真实走得远或近 | `speed_gain` |
| 起步阶段模拟过快或过慢 | `speed_time_constant_s`、`max_acceleration_m_s2` |
| 停车后模拟滑行距离不符 | `max_deceleration_m_s2` |
| 转向阶段轨迹弯曲程度不符 | `max_turn_rate_rad_s` |
| 小速度命令下真实机器人不动 | `command_deadband_m_s` |

如果真实机器人最终误差较大、但模拟器已达标，再调整 `lab1.py` 中的 `ControllerConfig`。先调 `kp`，再少量调 `kd`；不要用模拟器参数掩盖真实控制器误差。

## 重要边界

当前已经证明的是离线模拟器和 CSV 生成逻辑。真实定位精度、实际速度响应以及 sim-to-real RMSE 必须在实验室连接指定 Sphero 后验证，不能由无硬件模拟结果代替。
