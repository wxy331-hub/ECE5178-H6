"""Lab 2 EKF, self-contained for marking.

Filter state is ``[x, y, heading, speed]`` and the control input is
``[speed_cmd, heading_cmd]``.  Heading zero points along +y and heading pi/2
along +x, following the supplied environments.

The measurement is ``[heading, speed]``.  On hardware neither comes from the
observation vector: ``api.get_speed()`` and ``api.get_heading()`` return the
values the last ``set_speed``/``set_heading`` wrote, so the observation echoes
the command back.  Speed is taken from ``info["velocity"]`` (wheel encoders,
cm/s) and heading from ``info["orientation"]["yaw"]`` (gyroscope-derived
attitude), which are the sources instructions.md lists for the real robot.

Parameters are those used to produce 33377006_lab2.csv.  They are the Lab 1
equations with the constants re-estimated against the robot, which the brief
asks for separately from reusing the model:

    speed_gain / deadband   solved jointly from two measured working points,
                            raw 7 at 0.0318 m/s and raw 10 at 0.0905
    max_turn_rate_rad_s     from the gyroscope: 511 deg/s peak, 87.6 degrees
                            integrated over the two steps a corner takes
    dt                      the control period actually achieved, 104.9 ms
    R                       from the innovations of that run; the heading
                            innovation has sd 0.10 rad, four times what the
                            simulator assumes, because the IMU reading carries
                            the robot's real yaw wander on an uneven floor
"""

from __future__ import annotations

import numpy as np


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

PROCESS_NOISE = np.diag([3.125e-5, 3.125e-5, 3.0e-3, 2.5e-5])
MEASUREMENT_NOISE = np.diag([1.03e-2, 1.08e-3])
INITIAL_COVARIANCE = np.diag([1e-6, 1e-6, 6.25e-4, 6.25e-4])


def wrap_angle(angle: float) -> float:
    """Normalize an angle to [-pi, pi)."""

    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def dynamics(state, action):
    """Return the next [x, y, heading, speed] state.

    Same equations as the Lab 1 submission; only the constants differ.
    """

    x, y, heading, speed = np.asarray(state, dtype=float)
    speed_cmd, heading_cmd = np.asarray(action, dtype=float)

    dt = MODEL_CONFIG["dt"]
    max_speed = MODEL_CONFIG["max_speed_m_s"]

    heading_error = wrap_angle(float(heading_cmd) - float(heading))
    max_heading_step = MODEL_CONFIG["max_turn_rate_rad_s"] * dt
    heading_step = float(np.clip(heading_error, -max_heading_step, max_heading_step))
    heading_new = wrap_angle(float(heading) + heading_step)

    clipped = float(np.clip(speed_cmd, -max_speed, max_speed))
    effective = max(abs(clipped) - MODEL_CONFIG["command_deadband_m_s"], 0.0)
    desired_speed = np.sign(clipped) * MODEL_CONFIG["speed_gain"] * effective

    tau = max(MODEL_CONFIG["speed_time_constant_s"], 1e-6)
    first_order = float(speed + (1.0 - np.exp(-dt / tau)) * (desired_speed - speed))
    rate_limit = (
        MODEL_CONFIG["max_acceleration_m_s2"]
        if abs(first_order) > abs(speed)
        else MODEL_CONFIG["max_deceleration_m_s2"]
    )
    speed_new = float(
        speed + np.clip(first_order - speed, -rate_limit * dt, rate_limit * dt)
    )
    speed_new = float(np.clip(speed_new, -max_speed, max_speed))

    # Midpoint integration: more stable than either endpoint while the robot
    # is turning and accelerating at once.
    heading_mid = wrap_angle(float(heading) + 0.5 * heading_step)
    speed_mid = 0.5 * (float(speed) + speed_new)
    x_new = float(x) + speed_mid * np.sin(heading_mid) * dt
    y_new = float(y) + speed_mid * np.cos(heading_mid) * dt

    return np.array([x_new, y_new, heading_new, speed_new], dtype=np.float64)


def measurement_model(state):
    """h(x): the filter observes heading and speed, not position."""

    state = np.asarray(state, dtype=float)
    return np.array([wrap_angle(state[2]), state[3]], dtype=np.float64)


def process_jacobian(state, action, epsilon=1e-5):
    """Linearise the motion model by central difference.

    The deadband, the rate limits and the angle wrapping make a single global
    analytic Jacobian error-prone, and a numerical one stays correct when the
    constants above are re-calibrated.
    """

    state = np.asarray(state, dtype=float)
    action = np.asarray(action, dtype=float)
    jacobian = np.empty((4, 4), dtype=np.float64)
    for column in range(4):
        step = np.zeros(4)
        step[column] = epsilon
        plus = dynamics(state + step, action)
        minus = dynamics(state - step, action)
        difference = plus - minus
        difference[2] = wrap_angle(plus[2] - minus[2])
        jacobian[:, column] = difference / (2.0 * epsilon)
    return jacobian


class EKF:
    """Extended Kalman filter for Sphero position, heading and speed."""

    H = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)

    def __init__(
        self,
        initial_state=None,
        initial_covariance=None,
        process_noise=None,
        measurement_noise=None,
    ):
        self.state_est = (
            np.zeros(4, dtype=np.float64)
            if initial_state is None
            else np.asarray(initial_state, dtype=np.float64).copy()
        )
        self.P = np.array(
            INITIAL_COVARIANCE if initial_covariance is None else initial_covariance,
            dtype=np.float64,
        )
        self.Q = np.array(
            PROCESS_NOISE if process_noise is None else process_noise, dtype=np.float64
        )
        self.R = np.array(
            MEASUREMENT_NOISE if measurement_noise is None else measurement_noise,
            dtype=np.float64,
        )
        self.last_innovation = np.zeros(2, dtype=np.float64)
        self.last_innovation_covariance = np.eye(2, dtype=np.float64)

    @staticmethod
    def _stabilise(covariance):
        """Return a symmetric positive-semidefinite covariance."""

        symmetric = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        eigenvalues = np.maximum(eigenvalues, 1e-12)
        return (eigenvectors * eigenvalues) @ eigenvectors.T

    def predict(self, action):
        """Propagate the state and covariance through one control step."""

        prior = self.state_est.copy()
        F = process_jacobian(prior, action)
        self.state_est = dynamics(prior, action)
        self.state_est[2] = wrap_angle(self.state_est[2])
        self.P = self._stabilise(F @ self.P @ F.T + self.Q)
        return self.state_est.copy(), self.P.copy()

    def update(self, measurement):
        """Correct with a measured [heading, speed].

        Accepts the environment's [x, y, heading, speed, collision] as well;
        position and collision are ignored by this measurement model.
        """

        z = np.asarray(measurement, dtype=np.float64).reshape(-1)
        z = z[[2, 3]] if z.size >= 4 else z
        z = z.copy()
        z[0] = wrap_angle(z[0])

        innovation = z - measurement_model(self.state_est)
        innovation[0] = wrap_angle(innovation[0])

        S = self.H @ self.P @ self.H.T + self.R
        K = np.linalg.solve(S, (self.P @ self.H.T).T).T

        self.state_est = self.state_est + K @ innovation
        self.state_est[2] = wrap_angle(self.state_est[2])

        # Joseph form: stays symmetric and positive-definite even when K is
        # not the exactly optimal gain, which matters once Q and R are
        # hand-tuned rather than derived.
        identity = np.eye(4, dtype=np.float64)
        residual = identity - K @ self.H
        self.P = self._stabilise(residual @ self.P @ residual.T + K @ self.R @ K.T)

        self.last_innovation = innovation.copy()
        self.last_innovation_covariance = S.copy()
        return self.state_est.copy(), self.P.copy()


def robot_speed_measurement(info):
    """Encoder speed in m/s from a Robot.step() info dict, or None.

    Returns None rather than a number when the velocity frame is missing, so a
    dropped Bluetooth packet degrades the measurement instead of injecting a
    zero the filter would believe.
    """

    velocity = info.get("velocity")
    if not isinstance(velocity, dict):
        return None
    if "x" not in velocity and "y" not in velocity:
        return None
    try:
        vx = float(velocity.get("x", 0.0))
        vy = float(velocity.get("y", 0.0))
    except (TypeError, ValueError):
        return None
    speed = float(np.hypot(vx, vy)) / 100.0  # the API reports cm/s
    if not np.isfinite(speed):
        return None
    return -speed if float(info.get("speed_cmd", 0.0)) < 0.0 else speed
