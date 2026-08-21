"""Measure the control period and driving speed, then size a square leg.

Leg length is ``speed x LEG_STEPS x period``, and two of those three are
properties of the robot and the Bluetooth link rather than choices.  Guessing
them wastes the one thing a lab session is short of.  This measures both and
prints the LEG_STEPS that fits a target leg.

It runs in two phases so the second one needs as little floor as possible:

* the period is a software quantity, so it is timed with the robot commanded
  to zero speed and standing still;
* the speed is then measured over a short straight line.

Usage:  python labs/lab2/calibrate_leg.py --target-leg 0.45
"""

from __future__ import annotations

import argparse
import time

import numpy as np

try:
    from .lab2 import DT, LEG_STEPS, SCRIPT_SPEED, open_real_env
except ImportError:
    from lab2 import DT, LEG_STEPS, SCRIPT_SPEED, open_real_env

STALL_METRES = 0.002


def _timed_run(robot, steps: int, speed: float, heading: float):
    """Return (periods, positions) over ``steps`` commands at one heading."""

    action = np.array([speed, heading], dtype=np.float32)
    periods: list[float] = []
    positions: list[np.ndarray] = []
    next_tick = time.perf_counter()
    previous = next_tick
    for _ in range(steps):
        _, _, _, _, info = robot.step(action)
        now = time.perf_counter()
        periods.append(now - previous)
        previous = now
        positions.append(np.asarray(info["state_odom"][:2], dtype=float))
        next_tick += DT
        time.sleep(max(0.0, next_tick - time.perf_counter()))
    return np.array(periods), np.array(positions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Size a square leg")
    parser.add_argument("--target-leg", type=float, default=0.45,
                        help="wanted leg length in metres")
    parser.add_argument("--speed", type=float, default=SCRIPT_SPEED,
                        help="speed command to characterise")
    parser.add_argument("--steps", type=int, default=20,
                        help="steps per phase")
    args = parser.parse_args()

    raw = int(args.speed / 0.15 * 15)
    print(f"\nCommand {args.speed:.2f} -> raw {raw}/15")
    print(f"Straight line needs roughly "
          f"{args.speed * 2.69 * args.steps * 0.21:.2f} m in the worst case.\n")

    with open_real_env() as robot:
        _, info = robot.reset(seed=0)
        heading = float(info["state_odom"][2])

        print(f"Phase 1: timing {args.steps} steps at zero speed...")
        idle_periods, _ = _timed_run(robot, args.steps, 0.0, heading)

        print(f"Phase 2: driving {args.steps} steps straight...")
        periods, positions = _timed_run(robot, args.steps, args.speed, heading)
        robot.emergency_stop()

    period = float(np.median(np.concatenate([idle_periods, periods])))
    steps_moved = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    stalled = float(np.mean(steps_moved < STALL_METRES))
    distance = float(np.linalg.norm(positions[-1] - positions[0]))
    elapsed = float(periods.sum())
    speed = distance / elapsed if elapsed > 0 else 0.0

    print(f"\n  control period   {period * 1000:6.1f} ms   "
          f"(idle {np.median(idle_periods) * 1000:.0f}, "
          f"driving {np.median(periods) * 1000:.0f}; filter assumes "
          f"{DT * 1000:.0f})")
    print(f"  distance         {distance:6.3f} m over {elapsed:.1f} s")
    print(f"  average speed    {speed:6.3f} m/s")
    print(f"  stalled steps    {stalled:6.0%}  (moved < {STALL_METRES * 1000:.0f} mm)")

    print()
    if stalled > 0.10:
        print(f"  The robot stalled on {stalled:.0%} of steps: this command is")
        print(f"  too close to the deadband. Raise --speed by 0.01 and repeat")
        print(f"  before trusting the numbers below.")
    if abs(period - DT) > 0.02:
        print(f"  The period is {period / DT:.1f}x what the filter assumes, so")
        print(f"  every predicted displacement is off by that factor. Set DT and")
        print(f"  EKF's MODEL_CONFIG['dt'] to {period:.3f}, or reduce the")
        print(f"  Bluetooth traffic further.")

    if speed > 0:
        recommended = max(1, round(args.target_leg / (speed * period)))
        actual = recommended * speed * period
        print(f"\n  For a {args.target_leg:.2f} m leg:  LEG_STEPS = {recommended}"
              f"  (gives {actual:.3f} m)")
        print(f"  Currently LEG_STEPS = {LEG_STEPS}"
              f"  (gives {LEG_STEPS * speed * period:.3f} m)")
        laps = 200 / (4 * recommended)
        print(f"  200 steps would be {laps:.1f} laps of the square.")


if __name__ == "__main__":
    main()
