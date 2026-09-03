# Lab 2 Code Explanation

## 1. Goal and deliverable

Fuse a motion model with the robot's sensors in an Extended Kalman Filter to estimate position over 200 control steps, with a covariance `P` that honestly reflects the real error.

The run produces the automarker file:

```text
labs/lab2/33377006_lab2.csv
```

200 rows, seven columns: `sim_x, sim_y, real_x, real_y, P_xx, P_xy, P_yy`. The `sim_*` pair is the simulator's ground truth, `real_*` is the EKF's position estimate, and `P_*` are the three independent entries of the position covariance.

Overall flow:

```text
parse arguments → build simulator → connect to the real Sphero (hardware mode)
  → read each initial state → align the simulator's heading
  → 200 iterations: build action → step both environments
                    → take measurement → EKF predict → EKF update → record
  → write CSV → compute marker metrics → save analysis plot → write diagnostic log
```

## 2. Files

| File | Role |
| --- | --- |
| `lab2.py` | Main program: trajectory, measurement extraction, experiment loop, CSV output |
| `EKF.py` | Motion model `dynamics` and the `EKF` class (predict/update) |
| `analyze_lab2.py` | Validate the CSV, compute marker metrics, draw the analysis plot |
| `check_run.py` | After a hardware run, check the diagnostic log and say what to change |
| `calibrate_leg.py` | Measure control period and speed on the robot, size a lane segment |
| `test_lab2.py` / `test_check_run.py` | Unit tests, 68 in total |
| `33377006_m2_2.py` | The submitted EKF, self-contained, numpy only |

## 3. State, control and measurement

**State** `x = [x, y, heading, speed]`: position in metres, heading in radians, speed in m/s.

**Control** `u = [speed_cmd, heading_cmd]`. The second entry is a **desired angle**, not a turn rate.

**Measurement** `z = [heading, speed]`. Position is **not measured** — and that shapes the filter's behaviour: position can only be propagated by the model, so its uncertainty grows and is never contracted by an observation. This is exactly why Lab 3 needs a position measurement.

Coordinate convention, following the supplied environments:

- heading `0` points along `+y`, `π/2` along `+x`;
- so position integrates as `x += v·sin(θ)·dt`, `y += v·cos(θ)·dt`;
- and recovering a heading from a position error is `atan2(err_x, err_y)`, with the arguments in the opposite order to the usual `atan2(y, x)`.

## 4. Motion model and calibration

The model lives in `dynamics()` in [`EKF.py`](./EKF.py). Its equations are **identical** to the submitted Lab 1 file `33377006_m1_2.py`; only the constants were re-estimated against the robot. The brief asks both to reuse the Lab 1 motion model and to calibrate against the real robot, and those collide once a Lab 1 constant is contradicted by measurement. The resolution is to keep the **equations** fixed and re-estimate the **constants**, with `test_dynamics_equations_match_lab1_under_lab1_parameters` pinning the equations: it feeds Lab 1's parameters into Lab 2's `dynamics` and demands bit-identical output.

```python
MODEL_CONFIG = {
    "dt": 0.105,
    "max_speed_m_s": 0.50,
    "speed_gain": 1.957,
    "speed_time_constant_s": 0.216,
    "max_acceleration_m_s2": 1.79,
    "max_deceleration_m_s2": 1.33,
    "max_turn_rate_rad_s": 7.5,
    "command_deadband_m_s": 0.0537,
}
```

Five stages:

**① Turn-rate limit.** Heading may change by at most `max_turn_rate × dt` per step:

```python
heading_step = clip(heading_error, ±max_turn_rate·dt)
```

**② Command deadband.** A speed command below `command_deadband` only makes the robot rock in place. The model subtracts rather than thresholding to zero:

```python
effective = max(|speed_cmd| - deadband, 0)
desired   = sign(speed_cmd) · speed_gain · effective
```

Subtraction is continuous at the boundary (thresholding produces a jump), matches the physics of overcoming static friction before the remaining power becomes motion, and is differentiable almost everywhere, which suits the EKF's numerical Jacobian.

**③ First-order speed response.** The motors take time to reach the target:

```python
first_order = speed + (1 - exp(-dt/τ))·(desired - speed)
```

**④ Acceleration and deceleration limits.** Different ceilings (1.79 / 1.33 m/s²) because the robot accelerates faster than it brakes. Within this experiment's command range this stage rarely engages; it guards larger commands.

**⑤ Midpoint integration.** Position uses the midpoint heading and the mean speed:

```python
heading_mid = heading + 0.5·heading_step
speed_mid   = 0.5·(speed + speed_new)
x_new = x + speed_mid·sin(heading_mid)·dt
```

While turning and accelerating at once, integrating from the start state underestimates the displacement and from the end state overestimates it; the midpoint is second-order accurate.

### Where the parameters come from

| Parameter | Evidence |
| --- | --- |
| `speed_gain` = 1.957<br>`command_deadband` = 0.0537 | These are the slope and x-intercept of one line `v = k(u − u₀)`, so **a single working point cannot separate them**. Solved from two measured points: raw 7 → 0.0318 m/s and raw 10 → 0.0905 m/s, both of which the fit passes through exactly. |
| `max_turn_rate` = 7.5 rad/s | From the gyroscope: 511 °/s at the peak of a turn, integrating to 87.6° over two steps — the robot clears 90° in two. Lab 1's 2.61 needs six, which left the measured heading permanently ahead of the prediction. |
| `dt` = 0.105 | The control period actually achieved, 104.9 ms. The prediction is only correct when `dt` is the interval that really elapsed. |

Commands are quantised on the way to the robot: `raw = int(speed_cmd / 0.15 × 15)`, only the 16 integers 0–15, so **command resolution is 0.01 m/s**. The deadband 0.0537 falls between raw 5 and 6, which shows it is a fitted intercept rather than a directly observable threshold; its real uncertainty is about ±0.01.

## 5. EKF prediction

The prediction step advances state and covariance by one period:

```text
x⁻ = f(x, u)
P⁻ = F·P·Fᵀ + Q
```

`f` is `dynamics` above. `F` is its Jacobian, obtained by **central difference**:

```python
F[:, j] = (f(x + εeⱼ, u) − f(x − εeⱼ, u)) / (2ε)
```

The model contains clipping, a deadband, rate limits and angle wrapping, which make a single global analytic Jacobian error-prone — and a numerical one stays correct after the constants are re-calibrated. The angular component is wrapped before differencing so a crossing of ±π cannot corrupt the derivative.

## 6. Measurement model and update

The measurement model is linear: the filter observes heading and speed only.

```python
h(x) = [wrap(heading), speed]
H = [[0, 0, 1, 0],
     [0, 0, 0, 1]]
```

Update:

```text
ν = z − h(x⁻)          (the heading component wrapped to [−π, π))
S = H·P⁻·Hᵀ + R
K = P⁻·Hᵀ·S⁻¹
x = x⁻ + K·ν
```

The gain is obtained with `np.linalg.solve` rather than an explicit inverse, which is numerically better behaved.

### 6.1 Where the measurements come from (the heart of this lab)

**In simulation** the observation is ground truth plus Gaussian noise — a legitimate measurement, used as-is.

**On hardware, neither the heading nor the speed in the observation vector is a measurement.** `api.get_speed()` and `api.get_heading()` return the internal variables the last `set_speed()` / `set_heading()` wrote — the **command echoed straight back**. Feeding that to the filter makes it chase its own input: the innovation stops being zero-mean noise and becomes the constant offset between the command and the model's response to that command.

Hardware therefore uses two real sensors instead:

**Speed — wheel encoders.** `robot_speed_measurement()` reads `info["velocity"]`:

```python
speed = hypot(vx, vy) / 100.0     # the API reports cm/s
```

Only the magnitude is used: the wrapper documents `x`/`y` as right/forward but does not say whether that frame is the body or the world one, and the magnitude is the same either way; the sign comes from the command, which is unambiguous. A missing frame returns `None` and falls back to the observation, so a dropped packet degrades the measurement instead of injecting `NaN` or stalling the filter.

**Heading — IMU yaw.** `ImuHeadingSensor` reads `info["orientation"]["yaw"]`, a gyroscope-derived attitude (the brief lists the gyroscope among the measurements available on the real robot). Two things about it were unknown:

- **The zero point is removed exactly, with no assumption.** The absolute yaw is never used; the first reading becomes a reference and only differences from it are taken. Where the IMU's own origin sits cannot affect the result.
- **The sign has to be assumed, but a wrong guess is detectable and recoverable.** `IMU_YAW_SIGN = -1.0`, fixed from hardware data: regressing the measured yaw against the commanded heading gives a slope of −0.99, so the documented clockwise-positive convention does not hold for this robot. A wrong sign puts the measured heading roughly **180° from the command and keeps it there**, whereas a genuine turn transient peaks at 90° and decays within six steps. The gap between those cases is clean, so the guard uses a **120° threshold with a patience of 10 consecutive steps**: on sustained disagreement it prints a warning, disables the IMU source and falls back to the observation. A wrong guess costs one warning, not the run.

## 7. Covariance update and numerical stability

The covariance uses the **Joseph form**:

```python
P = (I − K·H)·P·(I − K·H)ᵀ + K·R·Kᵀ
```

One term more than the common `P = (I − K·H)·P`, but it stays symmetric and positive-definite even when `K` is not exactly the optimal gain — and with `Q` and `R` hand-tuned, `K` is not.

After every predict and update a further stabilisation runs:

```python
P = symmetrise(P), then clamp eigenvalues to ≥ 1e-12
```

This stops accumulated floating-point error from costing positive-definiteness over a long run, which would fail the marker's minimum-eigenvalue check.

## 8. Noise matrices

Initial covariance: the robot starts from a known origin, so position is nearly certain while heading and speed carry a little more doubt.

```python
P₀ = diag(1e-6, 1e-6, 6.25e-4, 6.25e-4)
```

Simulation and hardware use **different** noise matrices, because their noise has different sources:

| | Simulation | Hardware |
| --- | --- | --- |
| `Q` | `diag(3.125e-6, 3.125e-6, 1e-4, 2.5e-5)` | `REAL_PROCESS_NOISE = diag(3.125e-5, 3.125e-5, 3.0e-3, 2.5e-5)` |
| `R` | `diag(6.25e-4, 6.25e-4)` | `REAL_MEASUREMENT_NOISE = diag(1.03e-2, 1.08e-3)` |

**The hardware `R` has a heading term 16 times the simulator's**, because it is taken from the measured innovations: the heading innovation has a standard deviation of 0.10 rad against the simulator's assumed 0.025. The difference is that the IMU reading carries the robot's real yaw wander on uneven floor, not only sensor noise.

**The hardware `Q` opens its heading term to 3e-3** to absorb the same disturbance. The value is not arbitrary: replaying the hardware log offline while sweeping `Q` shows that raising it further makes the estimate trust the measurement too much and follow the disturbance, which makes the marker metric worse rather than better.

## 9. Trajectory: one lap of the yellow lane

The arena is a rounded-rectangle track with a barrier in the middle, so the trajectory follows the painted lane:

```python
TRACK_STRAIGHT_X = 0.385     # long straight
TRACK_STRAIGHT_Y = 0.395     # short straight
TRACK_RADIUS     = 0.065     # corner radius
TRACK_LAP        = 1.968     # lap length (computed)
```

The lane is described as eight `(arc length, heading change)` segments, and `track_heading(distance)` returns the tangent direction at that arc length: straights hold their heading, corners turn linearly with distance — which is what a constant-radius turn at constant speed does.

**Why not a square.** A square demands 90° within a single step. The robot obeys by pivoting at 511 °/s, which costs nearly all its forward speed and takes about two steps to rebuild. Eight of the ten near-motionless steps in a measured run fell inside that window — a high spot met while the robot is at low speed stops it outright. A 0.065 m corner at 0.09 m/s needs 1.39 rad/s, i.e. **8.4° per step against a demonstrated 45°**, so the robot turns while still driving and no low-speed window opens at all.

### 9.1 Driven by measured distance, not by the clock

`scripted_action(travelled, initial_heading)` takes the **distance actually covered**, not `step × speed × dt`.

The two diverge whenever the robot fails to move. An earlier version ran on the clock, so while the robot sat pinned against a high spot the schedule kept advancing: the corner was commanded after 0.219 m of real motion instead of the 0.32 m straight, cutting 0.10 m inside the lane and into the central barrier.

With measured displacement, **a stall costs time but not lane position** — the robot resumes the corner exactly where it left off. The source is the locator on hardware (which reports nothing while stuck) and the simulator's own truth in simulation, so both environments follow a single schedule.

Each step's displacement is also capped:

```python
travelled += min(moved, max_speed × dt)     # ceiling 52 mm
```

The locator occasionally jumps: one run reported 54–70 mm on four steps against encoder readings of 18–27 mm, which no speed the robot can reach explains. One such frame inside a corner skips part of the turn, and that run's 180° and 270° marks duly arrived 0.35 m late. When the reading is honest the cap costs nothing.

### 9.2 Stop after one lap

```python
completed = travelled >= TRACK_LAP
speed = 0.0 if completed else SCRIPT_SPEED
```

How far 200 steps carry the robot depends on the stall rate — about 1.1 laps when it runs clean, 0.7 at a 37% stall rate — so **no choice of speed fixes the lap count**. Deciding at the finish line does: hold still once the lap closes. The brief asks for 200 steps of **data**, not 200 steps of **driving**; and standing still is not wasted on the filter — it is the state the filter should predict best, and measured Mahalanobis distance over the stationary section is lower than over the moving one.

The commanded speed is `SCRIPT_SPEED = 0.12` (raw 12), 4.6 counts above the re-estimated deadband of 0.0537, which leaves margin against being stopped by a high spot.

## 10. Control period and Bluetooth throttling

`ThrottledRobot` subclasses the framework's `Robot` and overrides one method:

```python
def set_heading_and_speed(self, heading_deg, speed):
    with self._lock:
        if int(heading_deg) != self._last_heading_deg:
            self.api.set_heading(int(heading_deg))
            self._last_heading_deg = int(heading_deg)
        self.api.set_speed(int(clip(speed, 0, 255)))
```

The reason is in the underlying API: `set_heading()` and `set_speed()` both end in the **same** `roll_start(heading, speed)` command, and the one from `set_heading()` carries the **previous** speed, so it is overwritten by the second a full Bluetooth round trip later. When the heading has not changed that write buys nothing — and on the great majority of steps it has not.

Removing it took the control period from **209 ms to 104.9 ms**, which directly determines the filter's correctness: `EKF` predicts displacement from `dt`, so a period off by a factor of two means a predicted displacement off by a factor of two.

Not one byte of `src/sphero_env/` changed; this only overrides a public method.

## 11. What happens in one step

```text
1. build the action        scripted_action(travelled, initial_heading)
2. step the simulator      sim_env.step(action)
3. step the robot          robot_env.step(action)        (hardware mode)
4. take the measurement    extract_measurement(...)      → [heading, speed]
5. EKF predict             ekf.predict(action)
6. EKF update              ekf.update(measurement)
7. push the estimate back  update_estimate(...)          (for plots and logs)
8. record a submission row sim truth + EKF estimate + P
9. record a diagnostic row 26 columns of raw sensors and filter internals
10. advance the lane       travelled += min(moved, cap)
11. wait for the next tick perf_counter pacing
```

Step 11 uses `perf_counter` rather than `monotonic`: the latter resolves to 15.625 ms on this machine, 16% of a control period, which would swamp the very jitter being measured.

Simulation and hardware **receive the same action** but evolve independently: the simulator supplies the ground-truth trajectory the marker scores against, the robot supplies the measurements.

## 12. CSV output and marking

Writing goes to a temporary file and is then atomically replaced, so a failure part way through cannot truncate a previously valid submission. Row count, column count and finiteness are checked before the write.

Marker metrics (thresholds are constants in `analyze_lab2.py`, imported rather than restated):

| Metric | Threshold |
| --- | --- |
| Mean squared Mahalanobis distance of the position error | ≤ 4.0 |
| Chi-square pass rate (2 DoF, 95% gate) | ≥ 0.90 |

The Mahalanobis distance measures how large the estimation error is *relative to the uncertainty the filter claims*:

```text
d² = (x_sim − x_est)ᵀ · P⁻¹ · (x_sim − x_est)
```

It penalises both failure modes at once — an inaccurate estimate (large numerator) and an overconfident covariance (small denominator).

`analyze_lab2.py` also saves an analysis figure: trajectory comparison, 95% confidence ellipses, pointwise error and the NIS series. Following the brief's tuning guide, the NIS plot carries **both** chi-square bounds: above the upper bound means the filter is overconfident (raise `Q` or `R`), below the lower bound means it is too conservative.

## 13. Diagnostic and calibration tools

**`check_run.py`** — run it after a hardware session:

```bash
python labs/lab2/check_run.py
```

The marker metrics say whether the filter was self-consistent; they **cannot say whether it was fed real sensors**. A run can pass both thresholds with the speed "measurement" still being the command echoed back. This script answers that directly by replaying `ImuHeadingSensor`'s arithmetic against the logged raw yaw and comparing it with what actually reached the filter, and it prints a specific fix for every failed check.

Checks: run length, control period, speed sourced from the encoders, heading sourced from the IMU, IMU sign against the configured constant, systematic innovation bias, and filter consistency.

**`calibrate_leg.py`** — hardware calibration:

```bash
python labs/lab2/calibrate_leg.py --target-leg 0.40
```

Segment length is speed × steps × period, and two of those three are properties of the robot and the Bluetooth link rather than choices. The script runs in two phases: it times the period with the robot commanded to **zero speed and standing still** (the period is a software quantity and needs no floor space), then measures speed over a short straight, and prints the step count for a target segment.

## 14. Safety and fault handling

- both environments are wrapped in `try/finally`, calling `emergency_stop()` before closing on any exception;
- known transient Windows Bluetooth failures are retried 3 times at 3-second intervals, and no motion command is sent before the link is up;
- the hardware context exits immediately after step 200, closing Bluetooth before the `--hold` visualisation begins;
- the diagnostic log is written **even on an abort** — a run that had to be stopped is exactly the one whose sensor trace is worth reading. A failure there is deliberately swallowed, because raising from a `finally` block would replace the exception already propagating (a dropped Bluetooth link, say) and disguise the real fault as a disk error.

## 15. Commands

```powershell
# simulation only
python labs\lab2\lab2.py --sim --no-render --student-id 33377006

# with the real robot
python labs\lab2\lab2.py --student-id 33377006

# check the run that just finished
python labs\lab2\check_run.py

# validate the submission file
python labs\lab2\analyze_lab2.py labs\lab2\33377006_lab2.csv

# full test suite
python -m pytest
```

⚠️ Simulation and hardware write the same CSV and the same diagnostic log, so **running the simulator overwrites a hardware result**. Copy `33377006_lab2.csv` and `logs/lab2_diagnostics.csv` aside after a hardware run.

## 16. Verified results

**Hardware (2026-08-21, raw 12, one lap over 200 steps):**

| Item | Result |
| --- | --- |
| Mean squared Mahalanobis distance | **0.8732** (≤ 4.0) PASS |
| Chi-square pass rate | **1.000** (≥ 0.90) PASS |
| Position RMSE | 0.0514 m |
| Control period | 104.9 ms |
| Sensor checks | encoder speed, IMU heading and IMU sign all pass |

**Simulation:** mean Mahalanobis distance 0.0546, chi-square pass rate 1.000.

**Offline replay (the same hardware log, used to verify each calibration step):**

| Parameters | Mean NIS | Inside gate | Trajectory RMSE |
| --- | --- | --- | --- |
| Before calibration | 34.9 | 2% | 0.239 m |
| Two-point `gain`/`deadband` | 33.4 | 3% | 0.162 m |
| Plus `R` from measured variance | 4.0 | 90% | 0.172 m |
| Plus corrected turn rate, `Q` reduced | **2.04** | **96%** | **0.156 m** |

The theoretical value is 2.0 for 2 degrees of freedom. Each correction treats a different fault: `gain`/`deadband` the prediction bias, `R` an underestimated measurement noise, `max_turn_rate` a systematic lag in heading.

**Tests:** 68 passing.

## 17. Submitted artefacts

| File | Description |
| --- | --- |
| `33377006_lab2.csv` | 200 marked rows, produced by a hardware run |
| `33377006_m2_2.py` | The EKF as one self-contained file |
| `commit_history.txt` | `git log` export |

The constants in `33377006_m2_2.py` are those of the run that produced the CSV — replaying that run's log through the file reproduces every `real_x`, `real_y` and `P_xx` with a difference of `0.000e+00`.

## 18. Summary

```text
the Lab 1 motion model (equations fixed, constants re-estimated on hardware)
+ an EKF with a numerical Jacobian (Joseph-form covariance, eigenvalue clamping)
+ real sensor measurements (encoder speed, IMU heading, both with dropped-frame fallback)
+ separate noise matrices for simulation and hardware
+ a rounded-lane trajectory driven by measured displacement, stopping after one lap
+ Bluetooth throttling taking the control period from 209 ms to 105 ms
+ a 26-column diagnostic log and an automatic checker
+ 68 unit tests
```

One principle runs through the whole implementation: **a measurement must come from a real sensor, and the model must describe the real robot.** A consistency metric can always be "fixed" by inflating the covariance, but that only makes the filter admit it is inaccurate. The real improvements came from replacing the command echo with the encoders, the assumed turn rate with the gyroscope, and a clock-driven trajectory with a distance-driven one.
