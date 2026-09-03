import numpy as np


def dynamics(state, action):
    """Return the next [x, y, heading, speed] simulator state."""

    x, y, heading, speed = np.asarray(state, dtype=float)
    speed_cmd, heading_cmd = np.asarray(action, dtype=float)

    dt = 0.1
    max_speed = 0.50
    speed_gain = 2.69
    speed_time_constant = 0.216
    max_acceleration = 1.79
    max_deceleration = 1.33
    max_turn_rate = 2.61
    command_deadband = 0.0322

    heading_error = float(
        (heading_cmd - heading + np.pi) % (2.0 * np.pi) - np.pi
    )
    max_heading_step = max_turn_rate * dt
    heading_step = float(
        np.clip(heading_error, -max_heading_step, max_heading_step)
    )
    heading_new = float(
        (heading + heading_step + np.pi) % (2.0 * np.pi) - np.pi
    )

    clipped_command = float(np.clip(speed_cmd, -max_speed, max_speed))
    effective_command = max(abs(clipped_command) - command_deadband, 0.0)
    desired_speed = np.sign(clipped_command) * speed_gain * effective_command

    response = 1.0 - np.exp(-dt / speed_time_constant)
    first_order_target = float(speed + response * (desired_speed - speed))
    rate_limit = (
        max_acceleration
        if abs(first_order_target) > abs(speed)
        else max_deceleration
    )
    speed_new = float(
        speed
        + np.clip(first_order_target - speed, -rate_limit * dt, rate_limit * dt)
    )
    speed_new = float(np.clip(speed_new, -max_speed, max_speed))

    heading_mid = float(
        (heading + 0.5 * heading_step + np.pi) % (2.0 * np.pi) - np.pi
    )
    speed_mid = 0.5 * (float(speed) + speed_new)
    x_new = float(x) + speed_mid * np.sin(heading_mid) * dt
    y_new = float(y) + speed_mid * np.cos(heading_mid) * dt

    return np.array([x_new, y_new, heading_new, speed_new], dtype=np.float32)
