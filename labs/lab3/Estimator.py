"""Pose estimator for Lab 3.

Lab 2's filter measures ``[heading, speed]`` and leaves the locator out on
purpose: its position comes from the same encoders and IMU the prediction
already integrates, so treating it as an independent sensor understates the
covariance -- and Lab 2 is marked on Mahalanobis distance and chi-square rate,
which measure exactly that.

Lab 3 is marked on position accuracy instead, and dead reckoning does not
deliver it.  Replaying the three hardware logs against the locator:

    log                stalled steps   locator   filter   final gap
    11:38 raw10             3 / 200    1.952 m   2.387 m     0.104 m
    12:47 raw12            46 / 200    2.253 m   2.578 m     0.498 m
    09-04 stuck            19 / 151    1.264 m   1.989 m     1.191 m

"stalled" counts steps where a speed was commanded and the encoder read
essentially zero -- the robot held against a bump.  The filter keeps
integrating the command through those, so the more it stalls the further it
runs away from where the robot is.  Lab 3's limit is 0.10 m.

So this filter measures the full state.  The correlation Lab 2 avoided is real
and still there; the cost of it is an optimistic covariance, which Lab 3 does
not score, and the alternative is being a metre out.
"""

from __future__ import annotations

import numpy as np

from EKF import EKF, MODEL_CONFIG, _dynamics_float64, wrap_angle

# Simulation defaults, taken from the environment's own observation noise:
# obs_noise_std_pos = 0.05 m, obs_noise_std_vel = 0.025 (heading and speed).
SIM_MEASUREMENT_NOISE = np.diag([2.5e-3, 2.5e-3, 6.25e-4, 6.25e-4])

# Hardware.  Heading and speed keep the values Lab 2 measured.  The position
# entries are an estimate, not a measurement: the locator's own noise has never
# been characterised on this robot, and it cannot be without ground truth.
# 0.02 m is chosen to sit below Q's 0.0056 m per-step position growth by enough
# that the locator actually pulls the estimate, while staying wide enough that
# the occasional 54-70 mm locator jump seen in the 12:47 log moves the estimate
# by a fraction of the jump rather than all of it.
REAL_MEASUREMENT_NOISE = np.diag([4.0e-4, 4.0e-4, 1.03e-2, 1.08e-3])


class Estimator:
    """EKF over ``[x, y, heading, speed]`` with a full-state measurement."""

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
            else EKF._validate_vector(initial_state, 4, "initial_state")
        )

        # The robot is placed on a known start plate, so position starts
        # confident; heading and speed carry the same uncertainty as Lab 2.
        default_p = np.diag([1e-6, 1e-6, 6.25e-4, 6.25e-4])
        self.P = EKF._validate_covariance(
            default_p if initial_covariance is None else initial_covariance,
            4,
            "initial_covariance",
        )

        default_q = np.diag([3.125e-6, 3.125e-6, 1.0e-4, 2.5e-5])
        self.Q = EKF._validate_covariance(
            default_q if process_noise is None else process_noise, 4, "process_noise"
        )
        self.R = EKF._validate_covariance(
            SIM_MEASUREMENT_NOISE if measurement_noise is None else measurement_noise,
            4,
            "measurement_noise",
        )

        # z = [x, y, heading, speed]: the whole state is measured.
        self.H = np.eye(4, dtype=np.float64)
        self.last_innovation = np.zeros(4, dtype=np.float64)
        self.last_innovation_covariance = np.eye(4, dtype=np.float64)

    @staticmethod
    def measurement_model(state: np.ndarray) -> np.ndarray:
        """Measure the whole state; only the heading needs normalising."""
        measured = EKF._validate_vector(state, 4, "state")
        measured[2] = wrap_angle(measured[2])
        return measured

    def predict(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the next state and covariance from a control action."""
        action_array = EKF._validate_vector(action, 2, "action")
        prior_state = self.state_est.copy()
        transition_jacobian = EKF.process_jacobian(prior_state, action_array)

        self.state_est = _dynamics_float64(prior_state, action_array)
        self.state_est[2] = wrap_angle(self.state_est[2])
        self.P = EKF._stabilize_covariance(
            transition_jacobian @ self.P @ transition_jacobian.T + self.Q
        )
        return self.state_est.copy(), self.P.copy()

    def update(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Correct the prediction using ``[x, y, heading, speed]``."""
        z = EKF._validate_vector(measurement, 4, "measurement")
        innovation = z - self.measurement_model(self.state_est)
        # Index 2, not 0: the heading sits third in this measurement vector.
        innovation[2] = wrap_angle(innovation[2])

        innovation_covariance = self.H @ self.P @ self.H.T + self.R
        p_h_transpose = self.P @ self.H.T
        kalman_gain = np.linalg.solve(innovation_covariance, p_h_transpose.T).T

        self.state_est = self.state_est + kalman_gain @ innovation
        self.state_est[2] = wrap_angle(self.state_est[2])

        # Joseph form: stays symmetric positive-definite even when the gain is
        # not the optimal one, which matters here because R is a guess.
        identity = np.eye(4, dtype=np.float64)
        residual_transform = identity - kalman_gain @ self.H
        self.P = EKF._stabilize_covariance(
            residual_transform @ self.P @ residual_transform.T
            + kalman_gain @ self.R @ kalman_gain.T
        )

        self.last_innovation = innovation.copy()
        self.last_innovation_covariance = innovation_covariance.copy()
        return self.state_est.copy(), self.P.copy()
