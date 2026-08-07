# Lab 1 Executable Solution

## Objective and Deliverable

This solution runs the same PD feedback strategy independently in the simulator and on the real Sphero for 100 control steps of 0.1 seconds each. Both systems move towards `(0.5 m, 0.5 m)` relative to their own starting positions.

After a real-robot run, the program creates the automarker file:

```text
<student-id>_lab1.csv
```

The file contains exactly 100 data rows and the four required columns: `sim_x`, `sim_y`, `real_x`, and `real_y`.

## Controller

The outer loop uses PD control on the distance to the target:

```text
distance = ||target - position||
speed = Kp * distance + Kd * d(distance)/dt
heading = atan2(error_x, error_y)
```

In the supplied environment, a heading of `0 rad` points along `+y`. This is why the heading calculation uses `atan2(error_x, error_y)` rather than the more common `atan2(y, x)` order.

The commanded speed is limited to `0.15 m/s`. The controller commands zero speed within `0.025 m` of the target. A low-pass filter is applied to the derivative term to reduce oscillation caused by odometry noise and quantisation. The speed command also increases gradually for safer starts.

## Simulator Dynamics

`dynamics.py` models the robot using:

- first-order speed response;
- acceleration and deceleration limits;
- a maximum heading rate;
- a speed-command deadband; and
- midpoint position integration.

The action is interpreted as `[desired speed, desired heading]`, matching the interface defined by the lab environment. The parameters are kept together in `MODEL_CONFIG` so that they can be tuned without changing the model equations.

## Simulation-Only Test

From the repository root, activate the virtual environment and run:

```powershell
.\.venv\Scripts\Activate.ps1
python labs\lab1\lab1.py --sim
```

To run without the animation window:

```powershell
python labs\lab1\lab1.py --sim --no-render
```

Simulation-only mode does not connect to or move a physical robot. It reports the simulator's final distance to the target and does not create an automarker submission CSV.

## Real-Robot Run

1. Prepare a clear, flat area of at least `1 m x 1 m`.
2. Wake the Sphero and place it at the starting point.
3. Replace `12345678` below with the actual student ID.
4. Run:

```powershell
python labs\lab1\lab1.py --student-id 12345678
```

The BP-2E84 low-speed test measured approximately `0.140 m` of travel in one second at raw speed `15/255`. The program therefore fixes `15/255` as the hardware speed limit and ramps up the command gradually.

To disable the simulator animation while still running the physical robot:

```powershell
python labs\lab1\lab1.py --student-id 12345678 --no-render
```

The program scans for a Sphero and establishes the Bluetooth connection before sending movement commands. A Windows BLE timeout is retried up to three times with a three-second delay. Emergency-stop and environment-close operations run even if an exception occurs.

## Validate the Submission

Run the validator after the experiment:

```powershell
python labs\lab1\analyze_lab1.py labs\lab1\12345678_lab1.csv
```

It verifies the required column names, the 100 data rows, and finite numeric values. It also calculates:

- real-robot final target distance, required to be no more than `0.10 m`;
- simulator final target distance, required to be no more than `0.10 m`; and
- simulator-to-real trajectory RMSE, required to be no more than `0.20 m`.

The validator also creates a trajectory and pointwise-error plot.

## Verified BP-2E84 Result

The successful 100-step run on 7 August 2026 produced:

| Metric | Result | Threshold | Status |
| --- | ---: | ---: | --- |
| Simulator final distance | `0.0403 m` | `<= 0.10 m` | PASS |
| Real-robot final distance | `0.0083 m` | `<= 0.10 m` | PASS |
| Simulator-to-real RMSE | `0.0401 m` | `<= 0.20 m` | PASS |

The generated submission file is `labs/lab1/33377006_lab1.csv`.

## Further Tuning

Only edit `MODEL_CONFIG` near the top of `dynamics.py` when tuning the simulator:

| Observed behaviour | Parameter to adjust first |
| --- | --- |
| Simulator consistently travels too far or not far enough | `speed_gain` |
| Simulator starts too quickly or too slowly | `speed_time_constant_s`, `max_acceleration_m_s2` |
| Simulated stopping distance is incorrect | `max_deceleration_m_s2` |
| Turning curvature does not match the robot | `max_turn_rate_rad_s` |
| The real robot does not move at small commands | `command_deadband_m_s` |

If the simulator passes but the real robot has a large final error, tune `ControllerConfig` in `lab1.py`. Adjust `kp` first, then make small changes to `kd`. Do not use simulator parameters to hide a real-controller error.

## Safety and Evidence Boundary

Always place the robot on the floor with a clear path before a real run. Simulation-only results verify software behaviour but cannot replace physical validation. Real positioning accuracy and simulator-to-real agreement must be demonstrated using the assigned Sphero.
