# Lab 2 Code Implementation Explained

## 1. Objective

Lab 2 uses an Extended Kalman Filter (EKF) to estimate the Sphero position while it follows a repeatable 200-step trajectory. The simulator supplies the comparison ground truth, and the real robot supplies heading and speed measurements during a hardware run.

The completed program:

```text
resets the simulator and robot
  -> aligns their initial headings
  -> follows four 50-step sides of a square
  -> predicts the state with the calibrated motion model
  -> corrects heading and speed with measurements
  -> records the posterior position and covariance
  -> stops the robot and closes Bluetooth after step 200
  -> writes and validates the automarker CSV
  -> prints both assessment metrics
  -> saves the analysis graph
```

The main files are:

- [`EKF.py`](./EKF.py): motion model and EKF implementation;
- [`lab2.py`](./lab2.py): experiment, robot connection, trajectory and output;
- [`analyze_lab2.py`](./analyze_lab2.py): CSV validation, metrics and graph generation;
- [`test_lab2.py`](./test_lab2.py): offline regression tests.

## 2. State, Control and Measurement Definitions

The EKF state is:

```text
x = [position_x, position_y, heading, speed]
```

The control input is:

```text
u = [speed_command, heading_command]
```

The measurement used by the filter is:

```text
z = [measured_heading, measured_speed]
```

The raw environment observation may contain position and collision entries, but the EKF deliberately selects only heading and speed. Noisy odometry position is therefore not incorrectly treated as a second independent position sensor.

The coordinate convention is:

- heading `0 rad` points along `+y`;
- heading `pi/2 rad` points along `+x`;
- angles are wrapped into `[-pi, pi)`.

## 3. Main Experiment Parameters

The principal constants in [`lab2.py`](./lab2.py) are:

```python
DT = 0.1
N_STEPS = 200
COMMAND_SPEED_LIMIT = 0.15
SIM_SPEED_LIMIT = MODEL_CONFIG["max_speed_m_s"]   # 0.50
RAW_SPEED_LIMIT = 15
```

Each experiment therefore lasts approximately 20 seconds. The scripted trajectory commands `0.10 m/s`, while the real robot also has the lower-level safety limit of `15/255`.

These two speed limits are **different physical quantities**. An earlier revision expressed both with a single `VELOCITY_LIMIT = 0.15`, which introduced a systematic error:

- `COMMAND_SPEED_LIMIT` is the **command cap**. `Robot` also uses it to convert a command into the raw speed byte (`speed_raw = speed_cmd / vel_limit * raw_speed_limit`), so changing it changes how fast the real robot drives. It must stay at `0.15`.
- `SIM_SPEED_LIMIT` is the **observation range**, i.e. the span the sensor can report. At the scripted `0.10` command the model settles at `2.69 x (0.10 - 0.0322) = 0.182 m/s`, which is 21% above `0.15`, so every simulated speed reading was returned saturated by `np.clip`. The range must cover whatever the model can produce, so it takes the model's own hard ceiling of `0.50`.

After this correction the simulated mean Mahalanobis distance fell from `2.29` to `0.053`, and position RMSE from `2.8 cm` to `3.8 mm`.

## 4. Calibrated Motion Model

The process model retains the nonlinear Lab 1 equations:

- speed-command deadband;
- first-order motor response;
- acceleration and deceleration limits;
- maximum heading rate; and
- midpoint position integration.

Lab 2 uses exactly the same speed gain as Lab 1:

```python
"speed_gain": 2.69
```

An earlier revision lowered the Lab 2 gain to `2.47` because the simulator square came out systematically larger than the EKF trajectory. **That conclusion has since been shown to be wrong.** The trajectory was short because the hardware speed "measurement" was read from `api.get_speed()`, which returns the commanded target rather than a sensor value (its docstring says "current target speed"), compounded by the observation clipping described above. Lowering the gain merely compensated for those two defects, at the cost of a motion model that no longer described the real robot.

Differentiating the locator track from the 2026-08-14 hardware log recovers the actual gain directly:

| Estimator | Leg 1 | Leg 2 | Leg 3 | Leg 4 | Mean |
| --- | --- | --- | --- | --- | --- |
| Per-step speed median | 2.93 | 2.60 | 2.52 | 2.42 | 2.62 |
| Net leg displacement | 3.42 | 2.53 | 2.72 | 2.60 | 2.82 |

`2.69` lies inside both spreads; `2.47` lies outside them. Once the measurement source and observation range are fixed, the two gains give practically identical simulation metrics (mean Mahalanobis `0.0531` versus `0.0532`), showing that the earlier gap came entirely from clipping rather than from model accuracy. Matching Lab 1 has a second benefit: the submitted `33377006_m1_2.py` and this filter then describe the same robot.

The position update uses midpoint speed and heading:

```text
x_next = x + speed_mid * sin(heading_mid) * dt
y_next = y + speed_mid * cos(heading_mid) * dt
```

This is more accurate than using only the old or new speed while the robot accelerates and turns.

## 5. EKF Prediction

The nonlinear prediction is:

```text
x_k^- = f(x_(k-1), u_k)
```

The covariance prediction is:

```text
P_k^- = F_k P_(k-1) F_k^T + Q
```

`F_k` is the Jacobian of the motion model. Because the model contains clipping, deadband, rate limits and angle wrapping, [`EKF.py`](./EKF.py) calculates it with a central numerical difference rather than a fragile global symbolic expression.

For state element `i`:

```text
F[:, i] = (f(x + epsilon_i, u) - f(x - epsilon_i, u)) / (2 * epsilon)
```

The internal dynamics calculation uses `float64` so that the small Jacobian differences are not lost to `float32` rounding.

## 6. Measurement Model and EKF Update

The measurement model is linear:

```text
h(x) = [heading, speed]
```

Its matrix is:

```python
H = [[0, 0, 1, 0],
     [0, 0, 0, 1]]
```

The update equations are:

```text
innovation:  nu = z - h(x^-)
innovation covariance: S = H P^- H^T + R
Kalman gain: K = P^- H^T S^-1
posterior state: x = x^- + K nu
```

The heading innovation is wrapped into `[-pi, pi)`. This prevents a measurement just below `-pi` and a prediction just below `+pi` from being interpreted as an almost `2*pi` error.

The code solves the linear system for the Kalman gain rather than explicitly calculating `S^-1`, which is numerically safer.

### 6.1 Where the measurement comes from (hardware differs from simulation)

`extract_measurement()` in `lab2.py` decides where `z` comes from, which matters most on hardware.

**Simulation**: the observation is ground truth plus Gaussian noise, so it is already a legitimate measurement and `obs[2:4]` is used directly.

**Hardware**: speed is taken from `info["velocity"]` (the motor encoders) instead of the observation, because both of the obvious `sphero_unsw` accessors only echo commands back:

```python
def get_speed(self):    # docstring: "current target speed"
    return self.__speed         # whatever the last set_speed() wrote
def get_heading(self):  # docstring: "target directional angle"
    return self.__heading       # whatever the last set_heading() wrote
```

The 200-step hardware log from 2026-08-14 confirms this directly: its `speed` column equals `speed_cmd` in all 200 rows and only ever takes the single value `0.10`, while `heading` differs from `heading_cmd` by at most `1.745e-02 rad`, which is exactly the 1 degree lost to `int(degrees(...))`.

Feeding a command echo back as a measurement makes the filter correct itself with its own input. The model predicted a steady-state speed of `0.182 m/s` against a "measurement" fixed at `0.10 m/s`, so the speed innovation sat permanently at `-0.082 m/s` — more than three times the assumed measurement sigma of `0.025`, and a constant bias rather than the zero-mean noise an EKF assumes.

Only the magnitude `hypot(vx, vy)` of the encoder velocity is used: the API documents `x`/`y` as right and forward but never states whether that frame is the body or the world one, and the magnitude is identical either way. The sign is taken from the command, where there is no ambiguity. If a dropped BLE frame leaves the field missing, the function falls back to the observation rather than admitting a `NaN` into the filter.

**Heading comes from the IMU yaw**, supplied by `ImuHeadingSensor`. The docstring for `get_orientation()` states that yaw is measured by the gyroscope, and instructions.md explicitly lists the gyroscope as a hardware measurement, so this is the intended source.

Using it required resolving two unknowns, only one of which needed an assumption:

- **The zero point is cancelled exactly, with no assumption.** The filter never uses the absolute yaw; the first frame is stored as a reference and only differences from it are used, so wherever the IMU's own origin sits is irrelevant.
- **The sign has to be assumed.** `set_heading()` and the IMU yaw are both documented as increasing clockwise seen from above, so `IMU_YAW_SIGN = +1` is the expected value — but it has never been measured on this robot.

What makes that assumption safe is that a wrong guess is **detectable and recoverable**. A mirrored sign puts the measured heading roughly 180 degrees from the commanded one and keeps it there for a whole leg (50 steps), whereas a genuine turn transient peaks at 90 degrees (every scripted heading step is 90 degrees) and decays within six steps at the `2.61 rad/s` turn limit. The gap between those two cases is clean, so `ImuHeadingSensor` uses a 120 degree threshold with a patience of 10 consecutive steps: on sustained disagreement it prints a warning, disables the IMU source and falls back to the observation. A wrong guess therefore costs one warning rather than the whole run.

The diagnostic log records both the raw `imu_yaw_deg` and the `z_heading` actually fed to the filter, so comparing them against `heading_cmd` after a run confirms the sign.

## 7. Covariance Update and Stability

The posterior covariance uses the Joseph form:

```text
P = (I - KH) P^- (I - KH)^T + K R K^T
```

The Joseph form better preserves symmetry and positive semidefiniteness under floating-point arithmetic than the short form `(I-KH)P`.

After prediction and update, the covariance is symmetrised and any tiny negative eigenvalue caused by numerical roundoff is lifted to a small positive floor. The CSV analyzer independently rejects any position covariance that is not positive definite.

## 8. Noise Matrices

The simulation process noise is:

```python
Q = diag([3.125e-6, 3.125e-6, 1.0e-4, 2.5e-5])
```

The real experiment uses larger position process noise:

```python
Q_real = diag([3.125e-5, 3.125e-5, 1.0e-4, 2.5e-5])
```

This represents unmodelled hardware motion, surface variation and sim-to-real residuals. It prevents the real filter from being unrealistically confident.

The measurement noise is:

```python
R = diag([0.025^2, 0.025^2])
```

The initial covariance is:

```python
P0 = diag([1e-6, 1e-6, 6.25e-4, 6.25e-4])
```

Position starts near zero uncertainty because both systems are reset at a known origin. Heading and speed begin with greater uncertainty.

## 9. The 200-Step Square Trajectory

The default scripted action divides the experiment into four equal legs:

| Steps | Heading relative to start | Motion direction |
| ---: | ---: | --- |
| `1-50` | `0` | straight forward along the initial `+y` direction |
| `51-100` | `pi/2` | right side |
| `101-150` | `pi` | backward side |
| `151-200` | `-pi/2` | left side back toward the origin |

Each leg uses:

```python
action = np.array([0.10, heading], dtype=np.float32)
```

The simulator heading is aligned with the robot's reset heading before motion starts. There is no separate calibration movement and no command toward `(0.5, 0.5)`.

## 10. Predict/Update Timing

Each control cycle performs:

1. select the scripted or teleoperation action;
2. step the simulator;
3. step the real robot when connected;
4. call `ekf.predict(action)`;
5. call `ekf.update(measurement)`;
6. send the posterior estimate and covariance to the renderer;
7. save one CSV record; and
8. wait for the next absolute `0.1 s` tick.

The timing code is:

```python
next_tick += DT
time.sleep(max(0.0, next_tick - time.monotonic()))
```

Using an absolute monotonic deadline limits accumulated timing drift.

## 11. Visualisation Lines

The animation uses:

- green: simulator ground truth;
- blue: noisy simulator odometry;
- magenta: EKF posterior estimate.

The green and magenta paths are the important comparison for assessment. The uncertainty circle/ellipse is derived from the EKF position covariance. The yellow point is the environment goal marker and does not control the scripted square trajectory.

## 12. CSV Output

The automarker filename is:

```text
<student-id>_lab2.csv
```

It contains exactly 200 rows and seven columns:

```text
sim_x,sim_y,real_x,real_y,P_xx,P_xy,P_yy
```

Here, `real_x` and `real_y` mean the EKF mean position estimate, not raw position odometry. `P_xx`, `P_xy` and `P_yy` reconstruct the symmetric posterior position covariance:

```text
P_position = [[P_xx, P_xy],
              [P_xy, P_yy]]
```

The program validates the row count, shape and finite values before atomically replacing the final CSV.

## 13. Assessment Metrics

For each step, the position error is:

```text
e_k = simulator_position_k - EKF_position_k
```

The squared Mahalanobis distance is:

```text
d_k^2 = e_k^T P_position_k^-1 e_k
```

The two assessed conditions are:

| Metric | Required threshold |
| --- | ---: |
| Mean squared Mahalanobis distance | `<= 4.0` |
| Fraction inside the 2-DoF 95% chi-square gate | `>= 0.90` |

A high statistic means the filter is overconfident: the actual error is larger than its covariance predicts. A consistently very low statistic can mean the filter is unnecessarily pessimistic.

## 14. Automatic Analysis Graph

After every successful run, the program saves:

```text
<student-id>_lab2_analysis.png
```

The graph contains:

1. simulator ground truth and EKF position trajectories; and
2. the stepwise squared Mahalanobis statistic with the 95% chi-square threshold.

[`analyze_lab2.py`](./analyze_lab2.py) also reports position RMSE and the minimum covariance eigenvalue as diagnostics, although these two values are not the published pass/fail thresholds.

## 15. Robot Stop, Bluetooth and Logging

The real environment is protected by context managers and `finally` blocks. After step 200, the program:

```text
calls emergency_stop
  -> stops diagnostic logging
  -> closes the Robot object
  -> exits SpheroEduAPI
  -> releases the Bluetooth connection
  -> optionally keeps only the simulator window open
```

Therefore `--hold` does not keep the physical robot connected. Closing the window or pressing `Q` is only required to close the final simulator view.

There are two logs:

```text
logs/lab2_robot.csv         written by the framework Visualiser: trajectory and commands
logs/lab2_diagnostics.csv   written by DiagnosticLog: raw sensors and filter internals
```

The diagnostic log records 26 columns per step: a high-resolution timestamp, the speed and heading commands, the observation, encoder `vx/vy`, IMU `yaw`, gyroscope `z`, **the `z_heading` / `z_speed` actually fed to the filter**, odometry position, EKF state, both innovations, NIS, and the position covariance. It exists so that **one** hardware run can answer the questions that can only be settled with the robot present: the frame `get_velocity()` reports in, whether `IMU_YAW_SIGN` is `+1`, and the true encoder measurement noise needed to calibrate `R`. Logging the `z_*` values rather than reconstructing them afterwards shows directly whether each step used a sensor or a fallback. It is written even when the run is aborted with `Q` — a run that had to be stopped is exactly the one worth reading.

Timestamps use `time.perf_counter()` rather than `time.monotonic()`. On this machine both `monotonic` and `time` are backed by `GetTickCount64()` with a resolution of `15.625 ms`, which is 16% of the `100 ms` control period: too coarse to measure jitter, and a comparable error source when used to pace the loop. `perf_counter` is backed by `QueryPerformanceCounter()` at `0.1 us`. After the change the measured step interval is `100.00 ms` with a standard deviation of `0.23 ms`.

Generated logs, automarker CSV files and analysis PNG files are excluded from Git (`logs/.gitignore` contains `*`).

## 16. Commands

Simulation with animation:

```powershell
.\.venv\Scripts\python.exe .\labs\lab2\lab2.py --sim --student-id 33377006
```

Simulation without animation:

```powershell
.\.venv\Scripts\python.exe .\labs\lab2\lab2.py --sim --no-render --student-id 33377006
```

Real robot, stopping Bluetooth after 200 steps and holding the final graph window:

```powershell
.\.venv\Scripts\python.exe .\labs\lab2\lab2.py --student-id 33377006 --hold
```

Validate an existing CSV and generate a graph:

```powershell
.\.venv\Scripts\python.exe .\labs\lab2\analyze_lab2.py .\labs\lab2\33377006_lab2.csv --plot
```

Run the Lab 2 tests:

```powershell
.\.venv\Scripts\python.exe -m pytest .\labs\lab2\test_lab2.py -q
```

## 17. Verified Results

Simulation verification (`seed=5178`, after correcting the observation range and the measurement source):

| Metric | Result | Status |
| --- | ---: | --- |
| Mean squared Mahalanobis distance | `0.0246` | PASS |
| Chi-square pass rate | `1.000` | PASS |
| Position RMSE | `0.0026 m` | diagnostic |
| Mean NIS | `2.006` | theoretical value for 2 DoF is `2.0` |

Before and after, averaged over seeds `5178 / 1 / 42`:

| Revision | Mean Mahalanobis | Position RMSE |
| --- | ---: | ---: |
| Observation range `0.15`, gain `2.47` | `0.8725` | `0.0172 m` |
| Observation range `0.15`, gain `2.69` | `2.2868` | `0.0279 m` |
| Observation range `0.50`, gain `2.47` | `0.0532` | `0.0038 m` |
| Observation range `0.50`, gain `2.69` (current) | `0.0531` | `0.0038 m` |

The last two rows are effectively identical, showing that once clipping is removed the gain no longer drives simulation consistency. `2.47` previously looked better only because its steady-state speed sat closer to the saturated range limit.

Latest hardware CSV verification (`33377006_lab2.csv`):

| Metric | Result | Status |
| --- | ---: | --- |
| Data rows | `200` | PASS |
| Mean squared Mahalanobis distance | `1.9720` | PASS |
| Chi-square pass rate | `1.000` | PASS |
| Position RMSE | `0.0779 m` | diagnostic |
| Minimum covariance eigenvalue | `3.22515e-05` | positive |

**Note**: this hardware CSV was recorded on 2026-08-14, before the corrections above; its speed measurement is still a command echo and its gain is still `2.47`. It continues to satisfy both published thresholds, but it does not represent the current code on hardware. Until the robot is run again, the hardware behaviour of the current code is unmeasured.

The offline regression suite contains 25 tests and passes completely (`pytest test_lab2.py`). The latest hardware square was also visually confirmed to have closely aligned green and magenta trajectories.

## 18. Summary

The completed Lab 2 implementation combines:

```text
calibrated nonlinear Lab 1 motion model
+ numerical-Jacobian EKF prediction
+ heading and speed measurement correction
+ stable Joseph covariance update
+ calibrated simulation and hardware noise
+ safe repeatable 200-step square motion
+ exact automarker CSV and consistency metrics
+ automatic trajectory/chi-square graph
+ robot stop, BLE cleanup and diagnostic logging
```

This provides a reproducible localisation baseline that can later be extended to circles, triangles, figure-eight paths and Lab 3 navigation.
