"""Lab 4: drive the maze with the learned policy.

    .venv/Scripts/python.exe labs/lab4/lab4.py --sim        # simulation only
    .venv/Scripts/python.exe labs/lab4/lab4.py              # the robot, beside the simulator
    .venv/Scripts/python.exe labs/lab4/lab4.py --teacher    # the teacher drives instead, to collect data

Train first with train.py, which writes weight.pth beside this file.  The
network decides every action; the safety layer in runtime.py can only slow or
stop it, and the run prints how often it did.
"""

# Import necessary libraries
import argparse
import csv
import shutil
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import numpy as np
import torch

import runtime as R
from expert import expert_policy
from lab2 import ExperimentAborted, connect_with_retry, stop_motion
from Policy import Policy
from sphero_env.robot.connect import scan_and_connect

LAB_DIR = Path(__file__).resolve().parent
STUDENT_ID = "33377006"
CSV_COLUMNS = ("sim_x", "sim_y", "real_x", "real_y")
WEIGHTS = LAB_DIR / "weight.pth"


def load_policy(path=WEIGHTS):
    """The trained network, loaded the way the submission says it will be."""
    policy = Policy()
    policy.load_state_dict(torch.load(path))
    policy.eval()
    return policy


def as_controller(policy):
    def act(state):
        with torch.no_grad():
            return policy(torch.tensor(state, dtype=torch.float32)).numpy().astype(np.float64)
    return act


@contextmanager
def managed_env(sim: bool, render: bool = True, raw_speed: int = R.lab3.RAW_SPEED_LIMIT):
    """Yield ``(sim_env, robot_env)``; ``robot_env`` is None in simulation."""
    sim_env = R.make_sim_env(render=render)
    sim_env.set_log_path("logs/lab4_sim.csv")
    sim_env.start_logging()
    try:
        if sim:
            print("Simulation only: no robot, no Bluetooth.")
            yield sim_env, None
        else:
            print("Hardware mode: scanning for the robot. "
                  "Pass --sim to run without one, or press Ctrl+C to stop.")
            with ExitStack() as stack:
                selected_toy, _ = scan_and_connect()
                print(f"Selected: {selected_toy.name}")
                api = connect_with_retry(stack, selected_toy)
                real_env = R.make_real_env(api)
                # Robot.step sends int(speed_cmd / vel_limit * raw_speed_limit),
                # so the policy's 0.15 command goes out as this raw speed.  Speed
                # is observed from the encoders in cm/s, so nothing else rescales.
                real_env.raw_speed_limit = raw_speed
                real_env.set_log_path("logs/lab4_real.csv")
                real_env.start_logging()
                try:
                    yield sim_env, real_env
                finally:
                    real_env.close()
                    real_env.stop_logging()
    finally:
        sim_env.stop_logging()
        sim_env.close()


def write_submission(records, student_id=STUDENT_ID):
    """The automarker CSV: simulator truth against the filter's estimate.

    instructions.md names the file studentid_lab3.csv, which reads as a slip
    for lab4; confirm with the course before submitting.
    """
    path = LAB_DIR / f"{student_id}_lab4.csv"
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for row in records:
            writer.writerow(f"{value:.6f}" for value in row)
    return path


def write_step_log(rows):
    """Every hardware step as state -> action -> next state, for training later.

    obs_* is the estimate the action was chosen from, next_* the estimate
    after it.  policy_* is whatever drove the run -- the network, or the
    teacher under --teacher -- and applied_* what the safety layer let through.
    """
    path = Path("logs") / f"lab4_steps_{time.strftime('%m%d_%H%M%S')}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def print_metrics(records):
    """Report the four numbers the automarker checks, against the A* optimum."""
    names = {
        "final_sim": "final distance to goal (sim)", "final_real": "final distance to goal (real)",
        "path_sim": "distance to optimal path (sim)", "path_real": "distance to optimal path (real)",
    }
    values = R.metrics(records)
    print(f"{'metric':<32}{'value':>10}{'limit':>10}  result")
    for key, name in names.items():
        ok = values[key] <= R.LIMITS[key]
        print(f"{name:<32}{values[key]:>10.4f}{R.LIMITS[key]:>10.2f}  {'PASS' if ok else 'FAIL'}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Lab 4 learned navigation. Without --sim the robot is scanned "
                    "for over Bluetooth and driven alongside the simulator."
    )
    parser.add_argument("--sim", action="store_true", help="simulation only")
    parser.add_argument("--teacher", action="store_true",
                        help="drive with the teacher instead of the network, to collect data")
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--no-render", action="store_true", help="disable animation")
    parser.add_argument("--steps", type=int, default=R.MAX_STEPS, help="safety cap on steps")
    parser.add_argument("--raw-speed", type=int, default=R.lab3.RAW_SPEED_LIMIT,
                        help="raw Sphero speed (0-255) the robot drives at; Lab 3 used 15. "
                             "Assessment marks a run under 15 s higher. Hardware only.")
    parser.add_argument("--slope-boost", type=int, default=None, metavar="RAW",
                        help="raw speed for a robot that is told to drive but has stalled with "
                             "free floor ahead (the slope after the second corner). Off by default.")
    args = parser.parse_args(argv)
    if not 1 <= args.raw_speed <= 255:
        parser.error("--raw-speed must be between 1 and 255")
    if args.slope_boost is not None and not args.raw_speed < args.slope_boost <= 255:
        parser.error("--slope-boost must be above --raw-speed and at most 255")
    return args


def main(argv=None):
    args = parse_args(argv)
    render = not args.no_render
    # Loaded before any Bluetooth scan: a missing or broken weights file should
    # stop the run before a robot has been placed and paired for nothing.
    controller = expert_policy if args.teacher else as_controller(load_policy(args.weights))
    print(f"Driving with the {'teacher' if args.teacher else 'network from ' + str(args.weights)}.")
    records, steps = [], []
    hardware = False

    try:
        with managed_env(args.sim, render=render, raw_speed=args.raw_speed) as (sim_env, robot_env):
            hardware = robot_env is not None
            if hardware:
                print(f"Robot raw speed {args.raw_speed}"
                      + (f", {args.slope_boost} when stalled on a slope." if args.slope_boost else "."))
            try:
                R.run_episode(controller, sim_env, robot_env, max_steps=args.steps,
                              render=render, records=records,
                              step_log=steps if hardware else None, verbose=True,
                              slope_boost=args.slope_boost)
            finally:
                stop_motion(sim_env, robot_env)
    except ExperimentAborted as error:
        print(f"Run aborted: {error}")
    finally:
        try:
            if records:
                path = write_submission(records)
                print(f"Wrote {len(records)} rows to {path}")
                if hardware:
                    # The next run overwrites the submission; keep every robot run's copy.
                    kept = Path("logs") / f"lab4_submission_{time.strftime('%m%d_%H%M%S')}.csv"
                    kept.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, kept)
                    print(f"Kept a copy as {kept}")
                print_metrics(records)
        finally:
            if hardware and steps:
                for row in steps:
                    row["raw_speed"] = args.raw_speed
                try:
                    print(f"Wrote {len(steps)} steps to {write_step_log(steps)}")
                except OSError as error:
                    print(f"Could not write the step log: {error}")


if __name__ == "__main__":
    main()
