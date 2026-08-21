"""Extended Kalman filter used by Lab 2.

The filter state is ``[x, y, heading, speed]`` and the control input is
``[speed_cmd, heading_cmd]``.  Position is predicted by the calibrated Lab 1
motion model.  Heading and speed are used as measurements; the noisy odometry
``x`` and ``y`` values are deliberately not treated as independent sensors.
"""

from __future__ import annotations

import numpy as np


# The equations come from the calibrated Lab 1 model; the parameters are
# re-estimated from hardware, which instructions.md asks for separately.
#
# speed_gain and the deadband are solved together from two working points
# measured on 2026-08-21, because they are the slope and the x-intercept of
# one line and a single point cannot separate them:
#
#     raw 7  -> 0.0318 m/s      raw 10 -> 0.0905 m/s
#     gain = 1.957   deadband = 0.0537   (both points fit exactly)
#
# Lab 1's 2.69 came from differentiating a locator track that included
# stalled steps, so it read the average of a stop-start motion as a steady
# speed. Replaying the 200-step log offline, this pair cuts the trajectory
# RMSE from 0.239 m to 0.162 m.
#
# dt is 0.105, not Lab 1's 0.1: with ThrottledRobot the control period
# measured 104.9 ms, and the prediction is only right if dt is the interval
# that actually elapsed.
MODEL_CONFIG = {
    "dt": 0.105,
    "max_speed_m_s": 0.50,
    "speed_gain": 1.957,
    "speed_time_constant_s": 0.216,
    "max_acceleration_m_s2": 1.79,
    "max_deceleration_m_s2": 1.33,
    # 7.5 rad/s, not Lab 1's 2.61.  The gyroscope on 2026-08-21 read 511 deg/s
    # at the peak of each turn and the robot cleared 90 degrees in two steps;
    # integrating the trace gives 87.6 degrees over those two steps, so the
    # rate that reproduces the timing is 7.5.  At 2.61 the model needed six
    # steps for a turn the robot finished in two, which left the measured
    # heading permanently ahead of the prediction -- the +6.7 degree bias in
    # the heading innovation, previously mistaken for yaw wander.
    "max_turn_rate_rad_s": 7.5,
    # Solved jointly with speed_gain above.  Lab 1's 0.0322 also disagrees
    # with its own run, which stalled 0.0633 m short while the controller was
    # still commanding 0.80 x 0.0633 = 0.0506 m/s.
    "command_deadband_m_s": 0.0537,
}


def wrap_angle(angle: float) -> float:
    """Normalize an angle to ``[-pi, pi)``."""

    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


def _move_towards(current: float, target: float, max_delta: float) -> float:
    return float(current + np.clip(target - current, -max_delta, max_delta))


def _dynamics_float64(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Evaluate the Lab 1 model without reducing Jacobian precision."""

    x, y, heading, speed = np.asarray(state, dtype=np.float64)
    speed_cmd, heading_cmd = np.asarray(action, dtype=np.float64)

    dt = MODEL_CONFIG["dt"]
    max_speed = MODEL_CONFIG["max_speed_m_s"]

    heading_error = wrap_angle(float(heading_cmd) - float(heading))
    max_heading_step = MODEL_CONFIG["max_turn_rate_rad_s"] * dt
    heading_step = float(
        np.clip(heading_error, -max_heading_step, max_heading_step)
    )
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
    response = 1.0 - np.exp(-dt / tau)
    first_order_target = float(speed + response * (desired_speed - speed))
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

    return np.array([x_new, y_new, heading_new, speed_new], dtype=np.float64)


def dynamics(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Return the next ``[x, y, heading, speed]`` state.

    This is the calibrated Lab 1 process model.  Heading zero points along
    ``+y`` and heading ``pi/2`` points along ``+x``.
    """

    state_array = np.asarray(state, dtype=np.float64)
    action_array = np.asarray(action, dtype=np.float64)
    if state_array.shape != (4,):
        raise ValueError("state must have shape (4,)")
    if action_array.shape != (2,):
        raise ValueError("action must have shape (2,)")
    if not np.all(np.isfinite(state_array)) or not np.all(np.isfinite(action_array)):
        raise ValueError("state and action must contain only finite values")

    return _dynamics_float64(state_array, action_array).astype(np.float32)


class EKF:
    """EKF for Sphero position, heading and speed estimation."""

    def __init__(
        self,
        dt: float = MODEL_CONFIG["dt"],
        initial_state: np.ndarray | None = None,
        initial_covariance: np.ndarray | None = None,
        process_noise: np.ndarray | None = None,
        measurement_noise: np.ndarray | None = None,
    ) -> None:
        if not np.isclose(dt, MODEL_CONFIG["dt"]):
            raise ValueError(
                f"dt must be {MODEL_CONFIG['dt']} s to match the Lab 1 model"
            )

        self.dt = float(dt)
        self.state_est = (
            np.zeros(4, dtype=np.float64)
            if initial_state is None
            else self._validate_vector(initial_state, 4, "initial_state")
        )

        # The robot and simulator are reset at a known origin.  Heading and
        # speed begin with slightly more uncertainty than position.
        default_p = np.diag([1e-6, 1e-6, 6.25e-4, 6.25e-4])
        self.P = self._validate_covariance(
            default_p if initial_covariance is None else initial_covariance,
            4,
            "initial_covariance",
        )

        # Safe initial values, not final calibration results.  Q's position
        # entry is the smallest value that passed both consistency limits
        # across a 100-seed simulation sweep.  R's speed entry is still the
        # simulator's obs_noise_std_vel of 0.025 m/s, which no longer
        # describes the source: hardware speed now comes from the wheel
        # encoders, whose noise has never been measured on this robot.  It is
        # left unchanged rather than guessed -- logs/lab2_diagnostics.csv
        # records the innovations needed to estimate it from one run.
        default_q = np.diag([3.125e-6, 3.125e-6, 1.0e-4, 2.5e-5])
        default_r = np.diag([6.25e-4, 6.25e-4])
        self.Q = self._validate_covariance(
            default_q if process_noise is None else process_noise,
            4,
            "process_noise",
        )
        self.R = self._validate_covariance(
            default_r if measurement_noise is None else measurement_noise,
            2,
            "measurement_noise",
        )

        # z = [heading, speed]
        self.H = np.array(
            [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.last_innovation = np.zeros(2, dtype=np.float64)
        self.last_innovation_covariance = np.eye(2, dtype=np.float64)

    @staticmethod
    def _validate_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},)")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        return array.copy()

    @staticmethod
    def _validate_covariance(
        value: np.ndarray, size: int, name: str
    ) -> np.ndarray:
        covariance = np.asarray(value, dtype=np.float64)
        if covariance.shape != (size, size):
            raise ValueError(f"{name} must have shape ({size}, {size})")
        if not np.all(np.isfinite(covariance)):
            raise ValueError(f"{name} must contain only finite values")
        covariance = 0.5 * (covariance + covariance.T)
        if np.min(np.linalg.eigvalsh(covariance)) < -1e-12:
            raise ValueError(f"{name} must be positive semidefinite")
        return covariance.copy()

    @staticmethod
    def _stabilize_covariance(covariance: np.ndarray) -> np.ndarray:
        """Return a symmetric positive-semidefinite covariance matrix."""

        symmetric = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        eigenvalues = np.maximum(eigenvalues, 1e-12)
        return (eigenvectors * eigenvalues) @ eigenvectors.T

    @staticmethod
    def measurement_model(state: np.ndarray) -> np.ndarray:
        """Measure heading and speed from a four-element state."""

        state_array = EKF._validate_vector(state, 4, "state")
        return np.array(
            [wrap_angle(state_array[2]), state_array[3]], dtype=np.float64
        )

    @staticmethod
    def _extract_measurement(measurement: np.ndarray) -> np.ndarray:
        """Accept either ``[heading, speed]`` or an environment observation."""

        measurement_array = np.asarray(measurement, dtype=np.float64).reshape(-1)
        if measurement_array.size == 2:
            selected = measurement_array
        elif measurement_array.size >= 4:
            selected = measurement_array[[2, 3]]
        else:
            raise ValueError(
                "measurement must be [heading, speed] or contain at least "
                "[x, y, heading, speed]"
            )
        if not np.all(np.isfinite(selected)):
            raise ValueError("measurement must contain only finite values")
        selected = selected.copy()
        selected[0] = wrap_angle(selected[0])
        return selected

    @staticmethod
    def process_jacobian(
        state: np.ndarray, action: np.ndarray, epsilon: float = 1e-5
    ) -> np.ndarray:
        """Numerically linearise the nonlinear Lab 1 motion model.

        A central difference is used because speed deadband, rate limits and
        heading wrapping make a single global analytic Jacobian error-prone.
        """

        state_array = EKF._validate_vector(state, 4, "state")
        action_array = EKF._validate_vector(action, 2, "action")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")

        jacobian = np.empty((4, 4), dtype=np.float64)
        for column in range(4):
            step = np.zeros(4, dtype=np.float64)
            step[column] = epsilon
            plus = _dynamics_float64(state_array + step, action_array)
            minus = _dynamics_float64(state_array - step, action_array)
            difference = plus - minus
            difference[2] = wrap_angle(plus[2] - minus[2])
            jacobian[:, column] = difference / (2.0 * epsilon)
        return jacobian

    def predict(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the next state and covariance from a control action."""

        action_array = self._validate_vector(action, 2, "action")
        prior_state = self.state_est.copy()
        transition_jacobian = self.process_jacobian(prior_state, action_array)

        self.state_est = _dynamics_float64(prior_state, action_array)
        self.state_est[2] = wrap_angle(self.state_est[2])
        self.P = self._stabilize_covariance(
            transition_jacobian @ self.P @ transition_jacobian.T + self.Q
        )
        return self.state_est.copy(), self.P.copy()

    def update(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Correct the prediction using measured heading and speed.

        ``measurement`` may be ``[heading, speed]`` or the environment's
        ``[x, y, heading, speed, collision]`` observation.  Position and
        collision entries are intentionally ignored by this measurement model.
        """

        z = self._extract_measurement(measurement)
        innovation = z - self.measurement_model(self.state_est)
        innovation[0] = wrap_angle(innovation[0])

        innovation_covariance = self.H @ self.P @ self.H.T + self.R
        p_h_transpose = self.P @ self.H.T
        kalman_gain = np.linalg.solve(innovation_covariance, p_h_transpose.T).T

        self.state_est = self.state_est + kalman_gain @ innovation
        self.state_est[2] = wrap_angle(self.state_est[2])

        identity = np.eye(4, dtype=np.float64)
        residual_transform = identity - kalman_gain @ self.H
        self.P = self._stabilize_covariance(
            residual_transform @ self.P @ residual_transform.T
            + kalman_gain @ self.R @ kalman_gain.T
        )

        self.last_innovation = innovation.copy()
        self.last_innovation_covariance = innovation_covariance.copy()
        return self.state_est.copy(), self.P.copy()
