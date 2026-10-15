"""Measure the robot at speed: straight runs, stops, and turns while rolling.

    .venv/Scripts/python.exe project/calibrate.py straight --raw 30 50 80 --distance 0.6
    .venv/Scripts/python.exe project/calibrate.py turn --raw 30 50 --turn 90 --distance 0.3
    .venv/Scripts/python.exe project/calibrate.py idle --seconds 3

One connection for the whole session.  Before each trial the robot is placed
facing open floor and Enter is pressed; its aim and locator are reset where
it stands, so in that trial's log 'forward' is +y and the start is (0, 0).

Every sensor frame the robot streams is logged with its arrival time, and
every command with the time it was sent, one CSV per trial under
logs/calibration/:

  straight  drive at --raw until the locator has gone --distance, then stop
            and keep logging while it brakes and settles;
  turn      drive straight --distance at --raw, then turn by --turn degrees at
            the same speed -- all at once, or swung at --turn-rate deg/s --
            keep rolling 1 s more, then stop;
  idle      stand still and log, to see the frame rate and sensor noise.

--interval sets the sensor streaming period (the SDK starts at 150 ms).
Ctrl+C stops the robot and keeps what was logged.
"""

import argparse
import csv
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT.parent / "labs" / "lab4"))
import runtime as R  # noqa: E402,F401  (puts labs/lab3 and labs/lab2 on the path)
from lab2 import connect_with_retry  # noqa: E402
from sphero_env.robot.connect import scan_and_connect  # noqa: E402
from sphero_unsw.utils import ToyUtil  # noqa: E402

OUT = PROJECT.parent / "logs" / "calibration"
TRIAL_TIMEOUT_S = 6.0
SETTLE_AFTER_STOP_S = 1.5


class Recorder:
    """Keeps every streamed frame, flattened, with its arrival time."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.frames = []
        self.latest = {}
        self.lock = threading.Lock()

    def __call__(self, data):
        now = time.perf_counter() - self.t0
        row = {"t": now}
        for sensor, components in data.items():
            if isinstance(components, dict):
                for name, value in components.items():
                    row[f"{sensor}.{name}"] = value
        with self.lock:
            self.frames.append(row)
            self.latest = row

    def position(self):
        """Locator (x, y) in cm from the latest frame, or None."""
        with self.lock:
            row = dict(self.latest)
        for xk, yk in (("Locator.X", "Locator.Y"), ("locator.x", "locator.y")):
            if xk in row and yk in row:
                return np.array([row[xk], row[yk]], dtype=float)
        keys = {k.lower(): k for k in row}
        if "locator.x" in keys and "locator.y" in keys:
            return np.array([row[keys["locator.x"]], row[keys["locator.y"]]], dtype=float)
        return None


def send(toy, commands, recorder, heading, speed):
    ToyUtil.roll_start(toy, int(heading) % 360, int(speed))
    commands.append({"t": time.perf_counter() - recorder.t0, "heading": int(heading) % 360, "speed": int(speed)})


def reset_here(api, toy):
    ToyUtil.set_locator_flags(toy, False)
    api.reset_aim()
    api.stop_roll(heading=0)
    ToyUtil.reset_locator(toy)
    time.sleep(0.6)


# The firmware brakes a robot that has had no drive command for about 2 s
# (2026-10-09: every straight run stopped ~2.1 s after its one command, the
# raw 50 one rolling back 9 cm).  So a held command is re-sent this often.
KEEPALIVE_S = 0.2


def drive_until(toy, commands, recorder, heading, speed, distance_m, start_xy):
    """Drive until the locator has moved distance_m (or the trial times out), re-sending the command."""
    deadline = time.perf_counter() + TRIAL_TIMEOUT_S
    next_send = 0.0
    while time.perf_counter() < deadline:
        if time.perf_counter() >= next_send:
            send(toy, commands, recorder, heading, speed)
            next_send = time.perf_counter() + KEEPALIVE_S
        p = recorder.position()
        if p is not None and np.linalg.norm(p - start_xy) / 100.0 >= distance_m:
            return True
        time.sleep(0.005)
    return False


def hold(toy, commands, recorder, heading_of_time, speed, seconds):
    """Keep driving for ``seconds``, the heading a function of the time since this began."""
    began = time.perf_counter()
    while time.perf_counter() - began < seconds:
        send(toy, commands, recorder, heading_of_time(time.perf_counter() - began), speed)
        time.sleep(0.1)


def write_trial(name, recorder, commands, meta):
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%m%d_%H%M%S")
    with recorder.lock:
        frames = list(recorder.frames)
    fields = ["t"] + sorted({k for row in frames for k in row} - {"t"})
    frame_path = OUT / f"{stamp}_{name}_frames.csv"
    with open(frame_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(frames)
    with open(OUT / f"{stamp}_{name}_commands.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["t", "heading", "speed"])
        writer.writeheader()
        writer.writerows(commands)
    with open(OUT / f"{stamp}_{name}_meta.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(f"{k}={v}" for k, v in meta.items()) + "\n")
    period = np.diff([row["t"] for row in frames]) if len(frames) > 1 else np.array([np.nan])
    print(f"  wrote {len(frames)} frames (median period {np.median(period) * 1000:.0f} ms) to {frame_path.name}")
    summarise(frames, commands)


def summarise(frames, commands):
    """What the trial showed, printed at once: top speed, the stop, sideways drift, total travel."""
    rows = [r for r in frames if "locator.x" in r and "velocity.x" in r]
    if len(rows) < 3 or not commands:
        return
    t = np.array([r["t"] for r in rows])
    p = np.array([[r["locator.x"], r["locator.y"]] for r in rows])          # cm
    v = np.hypot([r["velocity.x"] for r in rows], [r["velocity.y"] for r in rows])  # cm/s
    go = next((c["t"] for c in commands if c["speed"] > 0), None)
    stop = next((c["t"] for c in reversed(commands) if c["speed"] == 0), None)
    if go is not None:
        moving = t[(t > go) & (v > 2.0)]
        if len(moving):
            print(f"  started moving {(moving[0] - go) * 1000:.0f} ms after the first drive command")
    print(f"  top speed {v.max() / 100:.2f} m/s; total travel {np.linalg.norm(p[-1] - p[0]):.1f} cm "
          f"(measure it with a tape: start to where it came to rest)")
    if stop is None:
        return
    i = int(np.searchsorted(t, stop))
    if i >= len(t):
        return
    at_stop, speed_at_stop = p[i], v[i]
    after = t > stop
    rest = t[after & (v < 2.0)]
    slide = np.linalg.norm(p[-1] - at_stop)
    forward = p[i] - p[0]
    sideways = 0.0
    if np.linalg.norm(forward) > 1.0:
        u = forward / np.linalg.norm(forward)
        sideways = float(np.max(np.abs((p[after] - at_stop) @ np.array([-u[1], u[0]])))) if after.any() else 0.0
    settle = f"{(rest[0] - stop) * 1000:.0f} ms" if len(rest) else "not seen"
    print(f"  STOP sent at {speed_at_stop / 100:.2f} m/s: slid {slide:.1f} cm, still after {settle}, "
          f"swerved {sideways:.1f} cm sideways")


def trial(api, toy, args, kind, raw):
    input(f"\n[{kind} raw {raw}] place the robot facing open floor, keep it still, press Enter: ")
    reset_here(api, toy)
    recorder = Recorder()
    toy.sensor_control.add_sensor_data_listener(recorder)
    commands = []
    try:
        time.sleep(0.5)                       # frames at rest before the command
        start = recorder.position()
        start = np.zeros(2) if start is None else start
        if kind == "straight":
            reached = drive_until(toy, commands, recorder, 0, raw, args.distance, start)
            send(toy, commands, recorder, 0, 0)
        elif kind == "turn":
            reached = drive_until(toy, commands, recorder, 0, raw, args.distance, start)
            if args.turn_rate > 0:
                # A ramp: the heading swings at turn_rate deg/s until it reaches the turn.
                ramp = abs(args.turn) / args.turn_rate
                sign = np.sign(args.turn)
                hold(toy, commands, recorder, lambda t: sign * min(abs(args.turn), args.turn_rate * t),
                     raw, ramp + 1.0)
            else:
                hold(toy, commands, recorder, lambda t: args.turn, raw, 1.2)
            send(toy, commands, recorder, args.turn, 0)
        else:
            reached = True
            time.sleep(args.seconds)
        time.sleep(SETTLE_AFTER_STOP_S)
        if not reached:
            print("  the locator never reached the distance: timed out")
    finally:
        ToyUtil.roll_start(toy, 0, 0)
        toy.sensor_control.remove_sensor_data_listener(recorder)
        meta = {"kind": kind, "raw": raw, "distance_m": args.distance, "turn_deg": args.turn,
                "turn_rate_deg_s": args.turn_rate, "interval_ms": args.interval}
        write_trial(f"{kind}_raw{raw}", recorder, commands, meta)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure the Sphero at speed.")
    parser.add_argument("kind", choices=["straight", "turn", "idle"])
    parser.add_argument("--raw", type=int, nargs="+", default=[30], help="raw speeds to try, one trial each")
    parser.add_argument("--distance", type=float, default=0.5, help="m of straight before stopping or turning")
    parser.add_argument("--turn", type=float, default=90.0, help="degrees to turn while rolling (negative: left)")
    parser.add_argument("--turn-rate", type=float, default=0.0,
                        help="deg/s to swing the heading at; 0 sends the whole turn at once")
    parser.add_argument("--seconds", type=float, default=3.0, help="idle: how long to log")
    parser.add_argument("--interval", type=int, default=50, help="sensor streaming period, ms")
    parser.add_argument("--repeat", type=int, default=1, help="trials per raw speed")
    args = parser.parse_args(argv)

    with ExitStack() as stack:
        toy, _ = scan_and_connect()
        print(f"Selected: {toy.name}")
        api = connect_with_retry(stack, toy)
        toy.sensor_control.set_interval(args.interval)
        print(f"Sensor streaming every {args.interval} ms.")
        try:
            for raw in (args.raw if args.kind != "idle" else [0]):
                for _ in range(args.repeat):
                    trial(api, toy, args, args.kind, raw)
        except KeyboardInterrupt:
            print("Stopped by the operator.")
        finally:
            ToyUtil.roll_start(toy, 0, 0)


if __name__ == "__main__":
    main()
