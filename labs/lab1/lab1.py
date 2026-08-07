"""Lab 1 PD controller for simulation and the BP-2E84 Sphero."""

from __future__ import annotations

import argparse
import csv
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from sphero_env.envs import SpheroEnv
from sphero_env.robot.connect import scan_and_connect
from sphero_env.robot.robot import Robot
from sphero_unsw.sphero_edu import SpheroEduAPI

try:
    from .dynamics import dynamics, wrap_angle
except ImportError:
    from dynamics import dynamics, wrap_angle


DT = 0.1
N_STEPS = 100
TARGET = np.array([0.5, 0.5], dtype=np.float32)
RAW_SPEED_LIMIT = 15  # BP-2E84: measured 0.140 m in 1 s at raw speed 15.
LAB_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ControllerConfig:
    kp: float = 0.80
    kd: float = 0.08
    derivative_filter: float = 0.70
    max_speed: float = 0.15
    min_speed: float = 0.025
    stop_tolerance: float = 0.025
    max_speed_increase_per_step: float = 0.01


class PositionPDController:
    """Point at the target and use PD control for forward speed."""

    def __init__(self, config: ControllerConfig = ControllerConfig()) -> None:
        self.config = config
        self.previous_distance: float | None = None
        self.filtered_distance_rate = 0.0
        self.previous_speed = 0.0

    def compute(self, observation: np.ndarray) -> np.ndarray:
        error = TARGET - observation[:2]
        distance = float(np.linalg.norm(error))

        if distance <= self.config.stop_tolerance:
            self.previous_speed = 0.0
            return np.array([0.0, observation[2]], dtype=np.float32)

        distance_rate = 0.0
        if self.previous_distance is not None:
            distance_rate = (distance - self.previous_distance) / DT
        self.previous_distance = distance

        alpha = self.config.derivative_filter
        self.filtered_distance_rate = (
            alpha * self.filtered_distance_rate + (1.0 - alpha) * distance_rate
        )

        speed = self.config.kp * distance + self.config.kd * self.filtered_distance_rate
        speed = float(np.clip(speed, self.config.min_speed, self.config.max_speed))
        speed = min(speed, self.previous_speed + self.config.max_speed_increase_per_step)
        self.previous_speed = speed

        # Heading 0 is +y in the supplied environment.
        heading = wrap_angle(float(np.arctan2(error[0], error[1])))
        return np.array([speed, heading], dtype=np.float32)


def make_sim_env(render: bool) -> SpheroEnv:
    return SpheroEnv(
        dt=DT,
        max_steps=N_STEPS,
        vel_limit=0.15,
        world_width=5.0,
        world_height=5.0,
        goal_pos=tuple(TARGET),
        goal_tolerance=0.1,
        occupancy_grid=None,
        dynamics=dynamics,
        obs_noise_std_pos=0.0,
        obs_noise_std_vel=0.0,
        process_noise_std_speed=0.0,
        process_noise_std_heading=0.0,
        render_mode="human" if render else None,
        window_size=(800, 800),
    )


def make_real_env(api: SpheroEduAPI) -> Robot:
    return Robot(
        api=api,
        dt=DT,
        max_steps=N_STEPS,
        vel_limit=0.15,
        raw_speed_limit=RAW_SPEED_LIMIT,
        world_width=5.0,
        world_height=5.0,
        goal_pos=tuple(TARGET),
        goal_tolerance=0.1,
        obs_noise_std_pos=0.0,
        obs_noise_std_vel=0.0,
        render_mode=None,
    )


def connect_with_retry(stack: ExitStack, selected_toy: object) -> SpheroEduAPI:
    """Try the transient Windows BLE connection up to three times."""

    last_error: TimeoutError | None = None
    for attempt in range(1, 4):
        try:
            return stack.enter_context(SpheroEduAPI(selected_toy))
        except TimeoutError as error:
            last_error = error
            if attempt < 3:
                print(f"Bluetooth timeout ({attempt}/3); retrying in 3 seconds...")
                time.sleep(3.0)

    raise RuntimeError("Bluetooth connection failed after 3 attempts") from last_error


@contextmanager
def open_sim_env(render: bool) -> Iterator[SpheroEnv]:
    env = make_sim_env(render)
    try:
        yield env
    finally:
        env.emergency_stop()
        env.close()


@contextmanager
def open_real_env() -> Iterator[Robot]:
    with ExitStack() as stack:
        selected_toy, _ = scan_and_connect()
        api = connect_with_retry(stack, selected_toy)
        print(f"Connected to: {selected_toy.name}")

        env = make_real_env(api)
        try:
            yield env
        finally:
            env.emergency_stop()
            env.close()


def relative_observation(observation: np.ndarray, origin: np.ndarray) -> np.ndarray:
    relative = observation.copy()
    relative[:2] -= origin
    return relative


def run_simulation(env: SpheroEnv, render: bool) -> list[np.ndarray]:
    observation, info = env.reset(seed=1)
    origin = info["state_odom"][:2].copy()
    controller = PositionPDController()
    trajectory: list[np.ndarray] = []

    for _ in range(N_STEPS):
        action = controller.compute(relative_observation(observation, origin))
        observation, _, _, _, info = env.step(action)
        trajectory.append((info["state_true"][:2] - origin).copy())
        if render:
            env.render()

    return trajectory


def run_real_and_sim(
    sim_env: SpheroEnv,
    real_env: Robot,
    render: bool,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    real_observation, real_info = real_env.reset(seed=1)
    sim_observation, sim_info = sim_env.reset(seed=1)
    real_origin = real_info["state_odom"][:2].copy()
    sim_origin = sim_info["state_odom"][:2].copy()

    # Apply the same feedback policy independently to each environment.
    real_controller = PositionPDController()
    sim_controller = PositionPDController()
    sim_trajectory: list[np.ndarray] = []
    real_trajectory: list[np.ndarray] = []
    next_tick = time.monotonic()

    print(f"Starting 100 steps; hardware speed limit is {RAW_SPEED_LIMIT}/255.")
    for _ in range(N_STEPS):
        real_action = real_controller.compute(
            relative_observation(real_observation, real_origin)
        )
        sim_action = sim_controller.compute(
            relative_observation(sim_observation, sim_origin)
        )
        real_observation, _, _, _, real_info = real_env.step(real_action)
        sim_observation, _, _, _, sim_info = sim_env.step(sim_action)

        real_trajectory.append((real_info["state_odom"][:2] - real_origin).copy())
        sim_trajectory.append((sim_info["state_true"][:2] - sim_origin).copy())
        if render:
            sim_env.render()

        next_tick += DT
        time.sleep(max(0.0, next_tick - time.monotonic()))

    return sim_trajectory, real_trajectory


def write_submission(
    student_id: str,
    sim_trajectory: list[np.ndarray],
    real_trajectory: list[np.ndarray],
) -> Path:
    output = LAB_DIR / f"{student_id}_lab1.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sim_x", "sim_y", "real_x", "real_y"])
        for sim_point, real_point in zip(sim_trajectory, real_trajectory, strict=True):
            writer.writerow([*map(float, sim_point), *map(float, real_point)])
    return output


def print_metrics(
    sim_trajectory: list[np.ndarray],
    real_trajectory: list[np.ndarray] | None = None,
) -> None:
    sim = np.asarray(sim_trajectory)
    sim_error = float(np.linalg.norm(sim[-1] - TARGET))
    print(f"{'PASS' if sim_error <= 0.10 else 'FAIL'} simulation final distance: {sim_error:.4f} m")
    if real_trajectory is None:
        return

    real = np.asarray(real_trajectory)
    real_error = float(np.linalg.norm(real[-1] - TARGET))
    rmse = float(np.sqrt(np.mean(np.sum((sim - real) ** 2, axis=1))))
    print(f"{'PASS' if real_error <= 0.10 else 'FAIL'} real final distance:       {real_error:.4f} m")
    print(f"{'PASS' if rmse <= 0.20 else 'FAIL'} sim-vs-real RMSE:          {rmse:.4f} m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 1 PD controller")
    parser.add_argument("--sim", action="store_true", help="simulation only")
    parser.add_argument("--no-render", action="store_true", help="disable animation")
    parser.add_argument("--student-id", help="student ID for the submission filename")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render = not args.no_render

    if args.sim:
        with open_sim_env(render) as sim_env:
            print_metrics(run_simulation(sim_env, render))
        return

    if not args.student_id or not args.student_id.isdigit():
        raise SystemExit("Real run requires --student-id followed by digits")

    # Enter the simulator first so the real robot is stopped first on exit.
    with open_sim_env(render) as sim_env, open_real_env() as real_env:
        sim_trajectory, real_trajectory = run_real_and_sim(sim_env, real_env, render)

    output = write_submission(args.student_id, sim_trajectory, real_trajectory)
    print_metrics(sim_trajectory, real_trajectory)
    print(f"Automarker CSV: {output}")


if __name__ == "__main__":
    main()
