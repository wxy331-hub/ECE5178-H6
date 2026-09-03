# Lab 1 代码实现说明

## 1. 任务目标

程序控制仿真机器人和真实 Sphero，在 100 个控制周期内移动到相对起点 `(0.5 m, 0.5 m)`。每个周期为 `0.1 s`，总运行时间约为 10 秒。

正式运行结束后，程序生成自动评分器要求的文件：

```text
<student-id>_lab1.csv
```

CSV 包含 100 行数据和四列：`sim_x`、`sim_y`、`real_x`、`real_y`。

整体流程为：

```text
读取参数
  -> 创建仿真环境
  -> 连接真实 Sphero（正式运行）
  -> 分别读取仿真和实机状态
  -> 分别运行相同的 PD 控制策略
  -> 执行并记录 100 步轨迹
  -> 生成 CSV
  -> 计算最终误差和 sim-to-real RMSE
```

## 2. 基本参数

主要常量定义在 [`lab1.py`](./lab1.py)：

```python
DT = 0.1
N_STEPS = 100
TARGET = np.array([0.5, 0.5], dtype=np.float32)
RAW_SPEED_LIMIT = 15
```

- `DT`：控制周期，单位为秒；
- `N_STEPS`：控制步数；
- `TARGET`：相对起点的目标位置；
- `RAW_SPEED_LIMIT`：真实 Sphero 的底层速度上限 `15/255`。

BP-2E84 的低速测试结果是在原始速度 `15/255` 下，一秒约移动 `0.140 m`。该限制用于避免机器人突然高速移动。

## 3. PD 位置控制器

控制器首先计算目标位置误差和距离：

```python
error = TARGET - observation[:2]
distance = float(np.linalg.norm(error))
```

对应公式：

```text
error = target - position
distance = sqrt(error_x^2 + error_y^2)
```

速度命令使用距离 PD 控制：

```text
speed = Kp * distance + Kd * d(distance)/dt
```

当前控制参数为：

```python
kp = 0.80
kd = 0.08
derivative_filter = 0.70
max_speed = 0.15
min_speed = 0.025
stop_tolerance = 0.025
max_speed_increase_per_step = 0.01
```

P 项使机器人在距离较远时移动较快，接近目标时逐渐减速。机器人接近目标时距离变化率为负，因此 D 项会提前降低速度，减少越过目标和往复振荡。

距离变化率经过低通滤波：

```python
filtered_rate = alpha * previous_filtered_rate + (1 - alpha) * distance_rate
```

这可以降低真实里程计噪声和量化误差造成的速度抖动。本方案没有积分项，因此严格来说是 PD 控制器，而不是完整 PID 控制器。

## 4. 速度限制和停车

PD 输出被限制在允许的速度范围内：

```python
speed = np.clip(speed, min_speed, max_speed)
```

速度每个周期最多增加 `0.01 m/s`：

```python
speed = min(speed, previous_speed + max_speed_increase_per_step)
```

这使机器人从低速逐渐启动，而不是第一步直接发送最大命令。真实机器人还受到 `raw_speed_limit=15` 的底层限制。

当机器人与目标的距离不超过 `0.025 m` 时：

```python
if distance <= stop_tolerance:
    return np.array([0.0, observation[2]], dtype=np.float32)
```

速度被设为零，航向保持不变。因此机器人到达目标附近后停止是预期行为，不是程序卡住。

## 5. 航向角

实验环境使用以下坐标约定：

- 航向 `0 rad` 指向 `+y`；
- 航向 `pi/2 rad` 指向 `+x`。

因此航向计算为：

```python
heading = wrap_angle(np.arctan2(error_x, error_y))
```

这里的参数顺序与常见的 `atan2(y, x)` 不同。`wrap_angle()` 将结果限制到 `[-pi, pi)`，保证机器人选择合理的转向角度。

## 6. 仿真动力学模型

[`dynamics.py`](./dynamics.py) 的状态和动作分别为：

```text
state  = [x, y, heading, speed]
action = [desired_speed, desired_heading]
```

模型包括：

- 最大转向率限制；
- 速度命令死区；
- 一阶速度响应；
- 加速和减速限制；
- 中点位置积分。

真实机器人无法瞬间转到目标角度，所以每一步的航向变化被限制为：

```python
max_heading_step = max_turn_rate * dt
heading_step = np.clip(heading_error, -max_heading_step, max_heading_step)
```

速度死区模拟小命令下机器人只晃动、不明显前进的现象：

```python
effective_command = max(abs(speed_command) - command_deadband, 0.0)
```

一阶速度响应模拟电机达到目标速度所需的时间：

```python
new_target = speed + (1 - exp(-dt / tau)) * (desired_speed - speed)
```

最后使用平均速度和中间航向更新位置：

```python
x_new = x + speed_mid * sin(heading_mid) * dt
y_new = y + speed_mid * cos(heading_mid) * dt
```

中点积分在同时转向和加速时比只使用旧状态或新状态更稳定。

## 7. 相对起点坐标

目标是相对机器人起点的 `(0.5 m, 0.5 m)`，而不是地图中的绝对位置。环境重置后，程序分别保存真实机器人和仿真器的起点：

```python
real_origin = real_info["state_odom"][:2].copy()
sim_origin = sim_info["state_odom"][:2].copy()
```

随后所有观测位置都减去自己的起点：

```python
relative_position = position - origin
```

因此无论机器人最初放在哪里，控制器都要求它相对移动到 `(0.5 m, 0.5 m)`。

## 8. 仿真和实机独立闭环

正式运行时创建两个控制器：

```python
real_controller = PositionPDController()
sim_controller = PositionPDController()
```

每一步分别根据各自状态计算动作：

```text
真实状态 -> real_controller -> 真实动作
仿真状态 -> sim_controller  -> 仿真动作
```

两者使用相同算法和参数，但不共享控制器内部状态。这一点非常重要：如果实机先到目标并停止，仿真器仍然会根据自己的剩余距离继续移动。它解决了早期版本中“实机到达目标后，仿真也提前停下”的问题。

## 9. 100 步控制循环

在每个控制周期中，程序：

1. 计算真实机器人的动作；
2. 计算仿真机器人的动作；
3. 执行两个动作；
4. 保存真实里程计位置和仿真真实状态；
5. 按需刷新动画；
6. 等待下一个 `0.1 s` 控制时刻。

时间同步使用单调时钟：

```python
next_tick += DT
time.sleep(max(0.0, next_tick - time.monotonic()))
```

这种方式可以减少普通 `sleep(0.1)` 产生的累计时间漂移。

## 10. CSV 输出

学号只用于自动评分文件名。例如：

```powershell
python labs\lab1\lab1.py --student-id 33377006
```

生成：

```text
labs/lab1/33377006_lab1.csv
```

文件标题严格为：

```text
sim_x,sim_y,real_x,real_y
```

程序使用 `zip(..., strict=True)` 同时写入两条轨迹，保证仿真和真实轨迹长度一致。

## 11. 指标验证

最终目标误差为：

```text
final_distance = ||final_position - target||
```

仿真与真实轨迹 RMSE 为：

```text
RMSE = sqrt(mean((sim_x-real_x)^2 + (sim_y-real_y)^2))
```

[`analyze_lab1.py`](./analyze_lab1.py) 还会检查：

- 是否包含四个要求的列；
- 是否正好有 100 行；
- 是否存在 `NaN` 或无穷值；
- 三项评分指标是否通过；
- 生成轨迹图和逐步位置误差图。

评分阈值和本次实测结果为：

| 指标 | 阈值 | 本次结果 | 状态 |
| --- | ---: | ---: | --- |
| 仿真最终距离 | `<= 0.10 m` | `0.0403 m` | PASS |
| 实机最终距离 | `<= 0.10 m` | `0.0083 m` | PASS |
| Sim-to-real RMSE | `<= 0.20 m` | `0.0401 m` | PASS |

## 12. 蓝牙和安全处理

Windows BLE 单次连接超时时，程序最多重试三次，每次间隔三秒。连接成功前不会发送运动命令。

仿真和实机环境都使用 `try/finally`：

```python
try:
    yield env
finally:
    env.emergency_stop()
    env.close()
```

即使运行过程中出现异常，程序也会尝试立即停止机器人并释放连接。

## 13. 运行命令

只运行仿真并显示动画：

```powershell
python labs\lab1\lab1.py --sim
```

只运行仿真且不显示动画：

```powershell
python labs\lab1\lab1.py --sim --no-render
```

连接真实机器人并同时显示仿真：

```powershell
python labs\lab1\lab1.py --student-id 33377006
```

连接真实机器人但不显示动画：

```powershell
python labs\lab1\lab1.py --student-id 33377006 --no-render
```

验证正式 CSV：

```powershell
python labs\lab1\analyze_lab1.py labs\lab1\33377006_lab1.csv
```

## 14. 实现总结

本方案由以下部分组成：

```text
安全限速的距离 PD 控制器
+ 实机与仿真独立闭环
+ 基于实测参数的动力学模型
+ 相对起点坐标转换
+ 100 步轨迹记录和 CSV 输出
+ 自动评分指标验证
+ 蓝牙重试和紧急停止保护
```

最终结果同时满足真实定位精度、仿真定位精度和 sim-to-real 轨迹一致性要求。
