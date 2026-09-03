# Lab 1 Code Implementation Explained

## 1. Objective

The program controls the simulated robot and the real Sphero to reach `(0.5 m, 0.5 m)` relative to their own starting positions. It runs for 100 control steps with a period of `0.1 s`, giving an experiment duration of approximately 10 seconds.

After a real-robot run, it creates the automarker file:

```text
<student-id>_lab1.csv
```

The CSV contains 100 data rows and the four required columns: `sim_x`, `sim_y`, `real_x`, and `real_y`.

The execution flow is:

```text
Parse arguments
  -> create the simulator
  -> connect to the real Sphero for a formal run
  -> read the simulator and robot states
  -> run the same PD policy independently for each system
  -> execute and record 100 trajectory steps
  -> write the submission CSV
  -> calculate final errors and sim-to-real RMSE
```

## 2. Main Parameters

The main constants are defined in [`lab1.py`](./lab1.py):

```python
DT = 0.1
N_STEPS = 100
TARGET = np.array([0.5, 0.5], dtype=np.float32)
RAW_SPEED_LIMIT = 15
```

- `DT` is the control period in seconds;
- `N_STEPS` is the number of control updates;
- `TARGET` is the target relative to the starting position;
- `RAW_SPEED_LIMIT` limits the physical Sphero command to `15/255`.

The BP-2E84 low-speed test measured approximately `0.140 m` of travel in one second at raw speed `15/255`. This limit prevents an unexpectedly fast initial movement.

## 3. PD Position Controller

The controller first calculates the position error and target distance:

```python
error = TARGET - observation[:2]
distance = float(np.linalg.norm(error))
```

This corresponds to:

```text
error = target - position
distance = sqrt(error_x^2 + error_y^2)
```

The speed command is produced by distance-based PD control:

```text
speed = Kp * distance + Kd * d(distance)/dt
```

The current controller parameters are:

```python
kp = 0.80
kd = 0.08
derivative_filter = 0.70
max_speed = 0.15
min_speed = 0.025
stop_tolerance = 0.025
max_speed_increase_per_step = 0.01
```

The proportional term makes the robot move faster when far from the target and slow down as it approaches. While the robot approaches the target, the distance rate is normally negative, so the derivative term reduces the speed early and helps prevent overshoot and oscillation.

The distance rate is low-pass filtered:

```python
filtered_rate = alpha * previous_filtered_rate + (1 - alpha) * distance_rate
```

This reduces speed jitter caused by odometry noise and quantisation. There is no integral term, so this is specifically a PD controller rather than a complete PID controller.

## 4. Speed Limits and Stopping

The PD output is restricted to the permitted speed range:

```python
speed = np.clip(speed, min_speed, max_speed)
```

The speed may increase by at most `0.01 m/s` per control step:

```python
speed = min(speed, previous_speed + max_speed_increase_per_step)
```

This produces a gradual start instead of sending the maximum command immediately. The physical robot is additionally protected by the raw limit of `15/255`.

When the target distance is no more than `0.025 m`, the controller returns zero speed:

```python
if distance <= stop_tolerance:
    return np.array([0.0, observation[2]], dtype=np.float32)
```

The robot keeps its current heading and stops moving. Stopping near the target is therefore expected behaviour rather than a program failure.

## 5. Heading Calculation

The supplied environment uses this coordinate convention:

- heading `0 rad` points along `+y`;
- heading `pi/2 rad` points along `+x`.

The desired heading is therefore calculated as:

```python
heading = wrap_angle(np.arctan2(error_x, error_y))
```

The argument order differs from the common `atan2(y, x)` convention. `wrap_angle()` normalises the result to `[-pi, pi)` so that the heading error remains well defined.

## 6. Simulator Dynamics

[`dynamics.py`](./dynamics.py) uses the following state and action definitions:

```text
state  = [x, y, heading, speed]
action = [desired_speed, desired_heading]
```

The model includes:

- a maximum heading rate;
- a speed-command deadband;
- first-order speed response;
- acceleration and deceleration limits; and
- midpoint position integration.

The real robot cannot turn to the commanded heading instantly, so each heading update is limited:

```python
max_heading_step = max_turn_rate * dt
heading_step = np.clip(heading_error, -max_heading_step, max_heading_step)
```

The command deadband represents the observed behaviour where a small command may make the robot shake without producing clear forward motion:

```python
effective_command = max(abs(speed_command) - command_deadband, 0.0)
```

The first-order response models the time required for the motors to reach the desired speed:

```python
new_target = speed + (1 - exp(-dt / tau)) * (desired_speed - speed)
```

Position is then updated using the average speed and midpoint heading:

```python
x_new = x + speed_mid * sin(heading_mid) * dt
y_new = y + speed_mid * cos(heading_mid) * dt
```

Midpoint integration is more stable than using only the old or new state while the robot is turning and accelerating.

## 7. Coordinates Relative to the Start

The target is `(0.5 m, 0.5 m)` relative to the starting point, not an absolute map coordinate. After resetting the environments, the program stores both origins:

```python
real_origin = real_info["state_odom"][:2].copy()
sim_origin = sim_info["state_odom"][:2].copy()
```

Every observed position is converted using:

```python
relative_position = position - origin
```

The controller therefore requests the same relative displacement regardless of where the robot is initially placed.

## 8. Independent Simulator and Robot Feedback Loops

The formal run creates two controller instances:

```python
real_controller = PositionPDController()
sim_controller = PositionPDController()
```

Each action is calculated from the corresponding system state:

```text
real state -> real_controller -> real action
sim state  -> sim_controller  -> sim action
```

Both use the same algorithm and parameters, but their internal controller states are independent. This is essential because the real robot and simulator do not reach the target at exactly the same step. If the real robot stops first, the simulator can continue moving according to its own remaining error. This fixes the earlier problem where the simulator stopped prematurely when the real robot reached the target.

## 9. The 100-Step Control Loop

During each control step, the program:

1. calculates the real-robot action;
2. calculates the simulator action;
3. executes both actions;
4. records real odometry and the simulator's true state;
5. refreshes the animation when requested; and
6. waits for the next `0.1 s` control tick.

Timing uses a monotonic clock:

```python
next_tick += DT
time.sleep(max(0.0, next_tick - time.monotonic()))
```

Scheduling against an absolute next tick reduces accumulated timing drift compared with repeatedly calling `sleep(0.1)`.

## 10. CSV Output

The student ID is used only to form the automarker filename. For example:

```powershell
python labs\lab1\lab1.py --student-id 33377006
```

creates:

```text
labs/lab1/33377006_lab1.csv
```

The header is exactly:

```text
sim_x,sim_y,real_x,real_y
```

The program writes the simulator and real trajectories together using `zip(..., strict=True)`, which ensures that both trajectories have the same number of samples.

## 11. Metric Validation

The final target error is:

```text
final_distance = ||final_position - target||
```

The simulator-to-real trajectory RMSE is:

```text
RMSE = sqrt(mean((sim_x-real_x)^2 + (sim_y-real_y)^2))
```

[`analyze_lab1.py`](./analyze_lab1.py) also verifies:

- the four required column names;
- exactly 100 data rows;
- the absence of `NaN` and infinite values;
- all three assessment thresholds; and
- trajectory and pointwise-error plots.

The thresholds and verified result are:

| Metric | Threshold | Result | Status |
| --- | ---: | ---: | --- |
| Simulator final distance | `<= 0.10 m` | `0.0403 m` | PASS |
| Real-robot final distance | `<= 0.10 m` | `0.0083 m` | PASS |
| Simulator-to-real RMSE | `<= 0.20 m` | `0.0401 m` | PASS |

## 12. Bluetooth and Safety Handling

The program retries a Windows BLE timeout up to three times with a three-second delay. No movement command is sent before the connection succeeds.

Both environments are protected using `try/finally`:

```python
try:
    yield env
finally:
    env.emergency_stop()
    env.close()
```

Even if an exception occurs during the experiment, the program attempts to stop the robot immediately and release the connection.

## 13. Commands

Simulation with animation:

```powershell
python labs\lab1\lab1.py --sim
```

Simulation without animation:

```powershell
python labs\lab1\lab1.py --sim --no-render
```

Real robot with simulator animation:

```powershell
python labs\lab1\lab1.py --student-id 33377006
```

Real robot without animation:

```powershell
python labs\lab1\lab1.py --student-id 33377006 --no-render
```

Validate the formal CSV:

```powershell
python labs\lab1\analyze_lab1.py labs\lab1\33377006_lab1.csv
```

## 14. Summary

The completed implementation combines:

```text
distance-based PD control with safe speed limits
+ independent real and simulated feedback loops
+ a calibrated dynamics model
+ start-relative coordinates
+ 100-step trajectory recording and CSV output
+ automatic assessment metric validation
+ Bluetooth retry and emergency-stop protection
```

The verified run meets the real positioning, simulated positioning, and sim-to-real trajectory requirements.
