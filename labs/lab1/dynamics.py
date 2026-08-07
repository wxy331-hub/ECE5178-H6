"""Calibratable dynamics model for Lab 1.

The Sphero action is ``[speed_cmd, heading_cmd]``.  This model treats speed as
a desired velocity and gives both speed and heading a finite response time.
The constants below are deliberately kept in one place so that they can be
updated from the real-robot logs without changing the model equations.
"""

from __future__ import annotations

import numpy as np


# Identified from the 100-step BP-2E84 run on 2026-08-07.
MODEL_CONFIG = {
    "dt": 0.1,
    "max_speed_m_s": 0.50,
    "speed_gain": 2.69,
    "speed_time_constant_s": 0.216,
    "max_acceleration_m_s2": 1.79,
    "max_deceleration_m_s2": 1.33,
    "max_turn_rate_rad_s": 2.61,
    "command_deadband_m_s": 0.0322,
}


def wrap_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi)."""

    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _move_towards(current: float, target: float, max_delta: float) -> float:
    return float(current + np.clip(target - current, -max_delta, max_delta))


def dynamics(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Return the next ``[x, y, heading, speed]`` state.

    Coordinate convention used by the supplied environments:

    * heading 0 points along +y;
    * heading +pi/2 points along +x;
    * speed is measured in m/s.

    The midpoint heading/speed are used for position integration.  This is
    more stable than integrating with only the old or new state during a turn.
    """

    x, y, heading, speed = np.asarray(state, dtype=float)
    speed_cmd, heading_cmd = np.asarray(action, dtype=float)

    dt = MODEL_CONFIG["dt"]
    max_speed = MODEL_CONFIG["max_speed_m_s"]

    heading_error = wrap_angle(float(heading_cmd) - float(heading))
    max_heading_step = MODEL_CONFIG["max_turn_rate_rad_s"] * dt
    heading_step = float(np.clip(heading_error, -max_heading_step, max_heading_step))
    heading_new = wrap_angle(float(heading) + heading_step)

    clipped_command = float(np.clip(speed_cmd, -max_speed, max_speed))
    effective_command = max(
        abs(clipped_command) - MODEL_CONFIG["command_deadband_m_s"],
        0.0,
    )
    desired_speed = (
        np.sign(clipped_command) * MODEL_CONFIG["speed_gain"] * effective_command
    )

    tau = max(MODEL_CONFIG["speed_time_constant_s"], 1e-6)
    first_order_target = float(speed + (1.0 - np.exp(-dt / tau)) * (desired_speed - speed))
    accelerating = abs(first_order_target) > abs(speed)
    rate_limit = (
        MODEL_CONFIG["max_acceleration_m_s2"]
        if accelerating
        else MODEL_CONFIG["max_deceleration_m_s2"]
    )
    speed_new = _move_towards(float(speed), first_order_target, rate_limit * dt)
    speed_new = float(np.clip(speed_new, -max_speed, max_speed))

    heading_mid = wrap_angle(float(heading) + 0.5 * heading_step)
    speed_mid = 0.5 * (float(speed) + speed_new)
    x_new = float(x) + speed_mid * np.sin(heading_mid) * dt
    y_new = float(y) + speed_mid * np.cos(heading_mid) * dt

    return np.array([x_new, y_new, heading_new, speed_new], dtype=np.float32)
