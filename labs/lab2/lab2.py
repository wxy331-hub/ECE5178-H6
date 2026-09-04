"""Run the 200-step Sphero Lab 2 EKF experiment.

The default experiment follows a repeatable square trajectory.  ``--sim``
uses simulator observations, while a hardware run uses the real robot's
heading and speed measurements and keeps the simulator as the comparison
trajectory.  The automarker CSV is written only after all 200 posterior EKF
updates have completed successfully.
"""

from __future__ import annotations

import argparse
import csv
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pygame
from bleak.exc import BleakCharacteristicNotFoundError

from sphero_env.envs import SpheroEnv
from sphero_env.robot.connect import scan_and_connect
from sphero_env.robot.robot import Robot
from sphero_unsw.sphero_edu import SpheroEduAPI

try:
    from .EKF import EKF, dynamics, wrap_angle
    from .analyze_lab2 import analyze_values, save_plot
except ImportError:
    from EKF import EKF, dynamics, wrap_angle
    from analyze_lab2 import analyze_values, save_plot


DT = 0.1
N_STEPS = 200
VELOCITY_LIMIT = 0.15
RAW_SPEED_LIMIT = 15
SCRIPTED_SQUARE_SPEED = 0.12
SQUARE_LEG_STEPS = 50
CORNER_TURN_STEPS = 10
LAB_DIR = Path(__file__).resolve().parent
CSV_COLUMNS = (
    "sim_x",
    "sim_y",
    "real_x",
    "real_y",
    "P_xx",
    "P_xy",
    "P_yy",
)

# Calibrated from the 2026-08-28 200-step hardware runs at approximately
# 0.10 m/s.  The mean squared position Mahalanobis distance was close to the
# ideal 2-D value of 2 while all samples passed the 5.991 gate.
REAL_PROCESS_NOISE = np.diag(
    [2.0625e-5, 2.0625e-5, 1.0e-4, 2.5e-5]
)

SIM_PROCESS_NOISE = np.diag(
    [1.0e-5, 1.0e-5, 1.0e-4, 2.5e-5]
)


@dataclass(frozen=True)
class EstimateRecord:
    """One posterior estimate aligned with one simulator ground-truth step."""

    sim_x: float
    sim_y: float
    real_x: float
    real_y: float
    p_xx: float
    p_xy: float
    p_yy: float

    def as_row(self) -> tuple[float, ...]:
        return (
            self.sim_x,
            self.sim_y,
            self.real_x,
            self.real_y,
            self.p_xx,
            self.p_xy,
            self.p_yy,
        )

class ExperimentAborted(RuntimeError):
    """Raised when the operator closes the window or presses Q."""


class TeleopController:
    """Translate pygame keyboard state into safe Sphero actions."""

    def __init__(self, initial_heading: float) -> None:
        self.heading = wrap_angle(initial_heading)
        self.speed = 0.08

    def action(self, robot_env: Robot | None) -> np.ndarray:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise ExperimentAborted("window closed")
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_q:
                raise ExperimentAborted("Q pressed")
            if event.key == pygame.K_SPACE:
                if robot_env is not None:
                    robot_env.emergency_stop()
                self.speed = 0.0
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.speed = min(VELOCITY_LIMIT, self.speed + 0.01)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.speed = max(0.0, self.speed - 0.01)

        pressed = pygame.key.get_pressed()
        if pressed[pygame.K_a]:
            self.heading = wrap_angle(self.heading - 0.10)
        if pressed[pygame.K_d]:
            self.heading = wrap_angle(self.heading + 0.10)

        commanded_speed = 0.0
        if pressed[pygame.K_w]:
            commanded_speed = self.speed
        elif pressed[pygame.K_s]:
            commanded_speed = -self.speed

        return np.array(
            [commanded_speed, self.heading], dtype=np.float32
        )


def make_sim_env(render: bool) -> SpheroEnv:
    return SpheroEnv(
        dt=DT,
        max_steps=N_STEPS,
        vel_limit=VELOCITY_LIMIT,
        world_width=5.0,
        world_height=5.0,
        goal_pos=(0.25, 0.25),
        goal_tolerance=0.1,
        occupancy_grid=None,
        dynamics=dynamics,
        obs_noise_std_pos=0.05,
        process_noise_std_speed=0.005,
        process_noise_std_heading=0.01,
        obs_noise_std_vel=0.025,
        render_mode="human" if render else None,
        window_size=(550, 550),
    )


def make_real_env(api: SpheroEduAPI) -> Robot:
    return Robot(
        api=api,
        dt=DT,
        max_steps=N_STEPS,
        vel_limit=VELOCITY_LIMIT,
        raw_speed_limit=RAW_SPEED_LIMIT,
        world_width=5.0,
        world_height=5.0,
        goal_pos=(0.5, 0.5),
        goal_tolerance=0.1,
        obs_noise_std_pos=0.05,
        obs_noise_std_vel=0.025,
        render_mode=None,
    )


def _retryable_ble_error(error: BaseException) -> bool:
    if isinstance(error, (TimeoutError, BleakCharacteristicNotFoundError)):
        return True
    return isinstance(error, OSError) and getattr(error, "winerror", None) in {
        -2147023673,
        1223,
    }


def connect_with_retry(
    stack: ExitStack, selected_toy: object
) -> SpheroEduAPI:
    """Retry known transient Windows BLE connection failures."""

    last_error: BaseException | None = None
    for attempt in range(1, 4):
        try:
            return stack.enter_context(SpheroEduAPI(selected_toy))
        except (TimeoutError, OSError, BleakCharacteristicNotFoundError) as error:
            if not _retryable_ble_error(error):
                raise
            last_error = error
            if attempt < 3:
                if isinstance(error, BleakCharacteristicNotFoundError):
                    print(
                        f"Sphero BLE service was not discovered ({attempt}/3); "
                        "disconnecting and retrying..."
                    )
                else:
                    print(f"Bluetooth connection failed ({attempt}/3); retrying...")
                time.sleep(3.0)
    if isinstance(last_error, BleakCharacteristicNotFoundError):
        raise RuntimeError(
            "The selected device did not expose the Sphero control service after "
            "3 attempts. Power-cycle the robot, remove it from Windows Bluetooth "
            "devices, and select the correct BP-* robot when scanning."
        ) from last_error
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
        diagnostic_path = LAB_DIR.parents[1] / "logs" / "lab2_robot.csv"
        env.set_log_path(str(diagnostic_path))
        env.start_logging()
        try:
            yield env
        finally:
            env.stop_logging()
            env.emergency_stop()
            env.close()


def scripted_action(step: int, initial_heading: float) -> np.ndarray:
    """Move 0.48 m per side, then stop while turning through 90 degrees."""

    if not 0 <= step < N_STEPS:
        raise ValueError(f"step must be in [0, {N_STEPS})")
    relative_headings = (0.0, np.pi / 2.0, np.pi, -np.pi / 2.0)
    leg = min(step // SQUARE_LEG_STEPS, len(relative_headings) - 1)
    step_in_leg = step % SQUARE_LEG_STEPS
    straight_steps = SQUARE_LEG_STEPS - CORNER_TURN_STEPS
    turning = step_in_leg >= straight_steps

    if turning:
        current_heading = wrap_angle(initial_heading + relative_headings[leg])
        next_leg = (leg + 1) % len(relative_headings)
        target_heading = wrap_angle(initial_heading + relative_headings[next_leg])
        turn_step = step_in_leg - straight_steps + 1
        turn_fraction = turn_step / CORNER_TURN_STEPS
        heading = wrap_angle(
            current_heading
            + turn_fraction * wrap_angle(target_heading - current_heading)
        )
        speed = 0.0
    else:
        heading = wrap_angle(initial_heading + relative_headings[leg])
        speed = SCRIPTED_SQUARE_SPEED

    return np.array([speed, heading], dtype=np.float32)


def _poll_for_abort() -> None:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            raise ExperimentAborted("window closed")
        if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
            raise ExperimentAborted("Q pressed")


def _align_simulator_heading(sim_env: SpheroEnv, heading: float) -> None:
    """Align the simulator frame with the real robot's reset heading."""

    if sim_env.state_true is None or sim_env.state_odom is None:
        raise RuntimeError("simulator must be reset before heading alignment")
    aligned_heading = wrap_angle(heading)
    sim_env.state_true[2] = aligned_heading
    sim_env.state_odom[2] = aligned_heading


def _record(sim_info: dict, ekf: EKF) -> EstimateRecord:
    truth = np.asarray(sim_info["state_true"], dtype=np.float64)
    position_covariance = ekf.P[:2, :2]
    return EstimateRecord(
        sim_x=float(truth[0]),
        sim_y=float(truth[1]),
        real_x=float(ekf.state_est[0]),
        real_y=float(ekf.state_est[1]),
        p_xx=float(position_covariance[0, 0]),
        p_xy=float(position_covariance[0, 1]),
        p_yy=float(position_covariance[1, 1]),
    )


def run_experiment(
    sim_env: SpheroEnv,
    robot_env: Robot | None,
    *,
    render: bool,
    teleop: bool,
    seed: int,
) -> list[EstimateRecord]:
    """Run exactly 200 predict/update cycles and return posterior records."""

    sim_observation, sim_reset_info = sim_env.reset(seed=seed)
    if robot_env is None:
        initial_heading = float(sim_observation[2])
        initial_speed = float(sim_observation[3])
        control_state = np.asarray(
            sim_reset_info["state_odom"], dtype=np.float64
        )
    else:
        _, robot_info = robot_env.reset(seed=seed)
        initial_heading = float(robot_info["state_odom"][2])
        initial_speed = float(robot_info["state_odom"][3])
        control_state = np.asarray(robot_info["state_odom"], dtype=np.float64)

    _align_simulator_heading(sim_env, initial_heading)

    initial_state = np.array(
        [0.0, 0.0, initial_heading, initial_speed], dtype=np.float64
    )
    process_noise = (
        REAL_PROCESS_NOISE if robot_env is not None else SIM_PROCESS_NOISE
    )
    ekf = EKF(
        dt=DT,
        initial_state=initial_state,
        process_noise=process_noise,
    )
    sim_env.update_estimate(ekf.state_est, ekf.P)
    if render:
        # Initialise pygame before polling keyboard/window events.
        sim_env.render()

    controller = TeleopController(initial_heading) if teleop else None
    records: list[EstimateRecord] = []
    next_tick = time.monotonic()

    print(
        f"Starting {N_STEPS} steps; hardware speed limit is "
        f"{RAW_SPEED_LIMIT}/255."
    )
    for step in range(N_STEPS):
        if controller is not None:
            action = controller.action(robot_env)
        else:
            if render:
                _poll_for_abort()
            action = scripted_action(step, initial_heading)

        sim_observation, _, _, _, sim_info = sim_env.step(action)
        if robot_env is None:
            measurement = sim_observation
            control_state = np.asarray(sim_info["state_odom"], dtype=np.float64)
        else:
            measurement, _, _, _, robot_info = robot_env.step(action)
            control_state = np.asarray(robot_info["state_odom"], dtype=np.float64)

        ekf.predict(action)
        ekf.update(measurement)
        sim_env.update_estimate(ekf.state_est, ekf.P)
        records.append(_record(sim_info, ekf))

        if render:
            sim_env.render()
        if (step + 1) % 50 == 0:
            print(f"Completed {step + 1}/{N_STEPS} steps")

        next_tick += DT
        time.sleep(max(0.0, next_tick - time.monotonic()))

    if len(records) != N_STEPS:
        raise RuntimeError(f"expected {N_STEPS} records, got {len(records)}")
    return records


def write_submission(
    student_id: str, records: list[EstimateRecord]
) -> Path:
    """Atomically write a validated 200-row automarker CSV."""

    if not student_id.isdigit():
        raise ValueError("student_id must contain digits only")
    if len(records) != N_STEPS:
        raise ValueError(f"submission requires exactly {N_STEPS} records")

    values = np.asarray([record.as_row() for record in records])
    if values.shape != (N_STEPS, len(CSV_COLUMNS)):
        raise ValueError("submission rows have an invalid shape")
    if not np.all(np.isfinite(values)):
        raise ValueError("submission contains NaN or infinite values")

    output = LAB_DIR / f"{student_id}_lab2.csv"
    temporary = output.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(values)
    temporary.replace(output)
    return output


def print_submission_metrics(records: list[EstimateRecord]):
    """Print assessment metrics and start-to-final uncertainty clearly."""

    values = np.asarray([record.as_row() for record in records])
    result = analyze_values(values)
    mean_status = "PASS" if result.mean_passed else "FAIL"
    coverage_status = "PASS" if result.coverage_passed else "FAIL"

    first = records[0]
    final = records[-1]
    start_sigma_x = float(np.sqrt(first.p_xx))
    start_sigma_y = float(np.sqrt(first.p_yy))
    final_sigma_x = float(np.sqrt(final.p_xx))
    final_sigma_y = float(np.sqrt(final.p_yy))
    start_sigma_position = float(np.sqrt(first.p_xx + first.p_yy))
    final_sigma_position = float(np.sqrt(final.p_xx + final.p_yy))
    estimated_displacement = float(
        np.hypot(final.real_x - first.real_x, final.real_y - first.real_y)
    )
    sigma_change = final_sigma_position - start_sigma_position

    print("\nLab 2 assessment metrics")
    print()
    print(
        "Mean Mahalanobis distance of position error (sim - real): "
        f"{result.mean_mahalanobis:.4f} "
        f"({mean_status}; required <= 4.0)"
    )
    print()
    print(
        "Chi-square pass rate (2 DoF, 95% gate): "
        f"{result.chi_square_pass_rate:.3f} "
        f"({coverage_status}; required >= 0.90)"
    )
    print("\nPosition uncertainty comparison (1 standard deviation)")
    print()
    print(
        f"Start point: sigma_x={start_sigma_x:.4f} m, "
        f"sigma_y={start_sigma_y:.4f} m, "
        f"combined={start_sigma_position:.4f} m"
    )
    print()
    print(
        f"Final point: sigma_x={final_sigma_x:.4f} m, "
        f"sigma_y={final_sigma_y:.4f} m, "
        f"combined={final_sigma_position:.4f} m"
    )
    print()
    print(f"Change in combined uncertainty: {sigma_change:+.4f} m")
    print(f"Estimated start-to-final displacement: {estimated_displacement:.4f} m")
    return result


def hold_visualization(sim_env: SpheroEnv) -> None:
    """Keep the final trajectory visible until Q or window close."""

    print("Simulation complete. Press Q or close the window to exit.")
    while True:
        try:
            _poll_for_abort()
        except ExperimentAborted:
            return
        sim_env.render()
        time.sleep(0.05)


def stop_motion(sim_env: SpheroEnv, robot_env: Robot | None) -> None:
    """Stop both systems immediately while keeping their contexts open."""

    if robot_env is not None:
        robot_env.emergency_stop()
    sim_env.emergency_stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 2 EKF experiment")
    parser.add_argument("--sim", action="store_true", help="simulation only")
    parser.add_argument("--no-render", action="store_true", help="disable animation")
    parser.add_argument("--teleop", action="store_true", help="use keyboard control")
    parser.add_argument(
        "--hold",
        action="store_true",
        help="keep the final visualization open until Q or window close",
    )
    parser.add_argument("--seed", type=int, default=5178, help="simulation seed")
    parser.add_argument(
        "--student-id",
        required=True,
        help="digits used for the studentid_lab2.csv filename",
    )
    args = parser.parse_args()
    if not args.student_id.isdigit():
        parser.error("--student-id must contain digits only")
    if args.teleop and args.no_render:
        parser.error("--teleop requires the pygame window")
    if args.hold and args.no_render:
        parser.error("--hold requires the pygame window")
    return args


def main() -> None:
    args = parse_args()
    render = not args.no_render

    try:
        with open_sim_env(render) as sim_env:
            if args.sim:
                try:
                    records = run_experiment(
                        sim_env,
                        None,
                        render=render,
                        teleop=args.teleop,
                        seed=args.seed,
                    )
                finally:
                    stop_motion(sim_env, None)
            else:
                # Exit this context immediately after step 200.  This closes
                # the Robot and SpheroEduAPI/BLE connection before --hold.
                with open_real_env() as robot_env:
                    try:
                        records = run_experiment(
                            sim_env,
                            robot_env,
                            render=render,
                            teleop=args.teleop,
                            seed=args.seed,
                        )
                    finally:
                        stop_motion(sim_env, robot_env)
                print("Robot stopped; Bluetooth connection closed.")

            # At this point a hardware connection has already been closed.
            output = write_submission(args.student_id, records)
            evidence = "simulation-only" if args.sim else "robot + simulator"
            print(f"Automarker CSV ({evidence}): {output}")
            result = print_submission_metrics(records)
            values = np.asarray([record.as_row() for record in records])
            plot_path = output.with_name(f"{output.stem}_analysis.png")
            save_plot(values, result, plot_path)
            print(f"\nAnalysis graph:\n{plot_path}\n")
            if args.hold:
                hold_visualization(sim_env)
    except ExperimentAborted as error:
        raise SystemExit(f"Experiment aborted: {error}; no CSV written") from error

if __name__ == "__main__":
    main()
