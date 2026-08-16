# ECE5178 Sphero Project Instructions

Project-specific rules only. The workspace `CLAUDE.md` one level up carries the
general working method, verification, research-integrity, Git and MCP graph
rules, and they apply here unchanged.

Monash ECE5178 Intelligent Robotics. Student ID **33377006** — it is the CSV
filename prefix for every automarked submission.

## Environment and commands

Python 3.12 in a `uv`-managed virtualenv. Always call the interpreter through
the venv rather than a bare `python`:

```bash
.venv/Scripts/python.exe -m pytest labs/lab1/test_lab1.py labs/lab2/test_lab2.py -q
```

```bash
.venv/Scripts/python.exe labs/lab2/lab2.py --sim --no-render --student-id 33377006
```

Drop `--sim` for a hardware run; everything else stays the same. `--teleop` and
`--hold` both require the pygame window, so they conflict with `--no-render`.

After a hardware run, check what the filter was actually fed:

```bash
.venv/Scripts/python.exe labs/lab2/check_run.py
```

It replays `ImuHeadingSensor`'s arithmetic against the logged raw yaw, so it
distinguishes a real sensor reading from the command echoed back — which the
submission metrics cannot do — and prints the fix for each failed check.

Both simulation and hardware write the same `studentid_lab2.csv`, so a
simulation run overwrites a hardware result. Copy a hardware CSV aside before
re-running in simulation.

## Layout

```
labs/lab1/    lab1.py  dynamics.py  analyze_lab1.py  33377006_m1_2.py   [complete]
labs/lab2/    lab2.py  EKF.py  analyze_lab2.py                          [complete]
labs/lab3/    Planner.py  lab3.py                          [untouched skeleton]
labs/lab4/    Policy.py  lab4.py                           [untouched skeleton]
src/sphero_env/   course framework: SpheroEnv (sim) and Robot (hardware)
logs/         diagnostic output; contents are gitignored
```

`src/`, `examples/` and every `instructions.md` are course-supplied. Treat them
as read-only reference: fix problems in the lab files instead, and if a
framework change is genuinely required, say so explicitly.

Lab 3 depends on Lab 2's localisation, so EKF tuning here carries forward.

## Submission contracts

Strict — the automarker rejects any deviation.

| | Lab 1 | Lab 2 |
| --- | --- | --- |
| File | `labs/lab1/33377006_lab1.csv` | `labs/lab2/33377006_lab2.csv` |
| Data rows | 100 | 200 |
| Columns | `sim_x,sim_y,real_x,real_y` | `sim_x,sim_y,real_x,real_y,P_xx,P_xy,P_yy` |
| Thresholds | final distance ≤ 0.10 m, RMSE ≤ 0.20 m | mean Mahalanobis ≤ 4.0, chi-square pass rate ≥ 0.90 |

`analyze_lab1.py` and `analyze_lab2.py` hold these limits as constants; import
them rather than restating the numbers.

Lab 1 has been submitted. Its CSV records one specific hardware run, so
changing `lab1.py` behaviour would fork the code from what produced the marked
result — comment and documentation edits are fine, logic changes are not
without saying what it costs. Lab 2 has not been submitted yet.

## Hardware API traps

Each of these was found by reading the wrapper source or a diagnostic log, and
each one silently produces plausible-looking wrong numbers.

- **`api.get_speed()` and `api.get_heading()` echo the last command**, not a
  sensor. They return the values `set_speed()`/`set_heading()` wrote, so the
  observation's speed and heading entries are the filter's own input fed back.
  Real measurements live in `info`: `velocity` (wheel encoders, **cm/s**),
  `orientation.yaw` (IMU attitude, −179…180°), `gyroscope`, `location`.
- **`vel_limit` means two different things.** In `Robot` it also scales the raw
  0–15 speed byte (`speed_raw = speed_cmd / vel_limit * raw_speed_limit`), so
  changing it changes how fast the robot physically drives. In `SpheroEnv` it is
  the observation and action range, and it must cover what the model can
  produce or readings come back clipped.
- **`time.monotonic()` resolves to 15.625 ms on Windows** — 16% of a control
  period. Use `perf_counter` for loop pacing and timing logs. `lab1.py` still
  uses `monotonic`, deliberately left alone (see above).
- **Heading 0 points along +y, π/2 along +x.** The simulator picks a random
  initial heading on reset; hardware always starts from heading 0, and the run
  aligns the simulator to it.
- `EKF` hard-codes `dt = 0.1`, so any drift in the real control period goes
  straight into the predicted displacement.

## Unverified, pending hardware

Do not present any of these as calibrated. `logs/lab2_diagnostics.csv` records
what is needed to settle them in one 200-step hardware run.

- `IMU_YAW_SIGN` is assumed `+1` from documentation. A wrong sign is detected
  at runtime and falls back to the observation with a printed warning.
- `REAL_PROCESS_NOISE` is 10× the simulator's position Q. The evidence for that
  inflation has since been traced to two fixed measurement bugs, so it needs
  re-estimating downward.
- `R`'s speed entry is still the simulator's `obs_noise_std_vel`; encoder noise
  has never been measured on this robot.
- Whether `get_velocity()` reports in the body or the world frame.
