"""Read logs/calibration/*: per trial, what the robot did.

    .venv/Scripts/python.exe project/analyse_calibration.py            # every trial
    .venv/Scripts/python.exe project/analyse_calibration.py turn       # names containing 'turn'
"""

import csv
import sys
from pathlib import Path

import numpy as np

LOGS = Path(__file__).resolve().parents[1] / "logs" / "calibration"


def load(stem):
    frames = list(csv.DictReader(open(LOGS / f"{stem}_frames.csv", encoding="utf-8")))
    commands = list(csv.DictReader(open(LOGS / f"{stem}_commands.csv", encoding="utf-8")))
    f = lambda k: np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan for r in frames])  # noqa: E731
    data = {"t": f("t"), "x": f("locator.x"), "y": f("locator.y"), "vx": f("velocity.x"),
            "vy": f("velocity.y"), "yaw": f("attitude.yaw")}
    cmds = [(float(c["t"]), int(c["heading"]), int(c["speed"])) for c in commands]
    return data, cmds


def course(vx, vy):
    """Direction of travel, degrees clockwise from the robot's start-forward (+y)."""
    return np.degrees(np.arctan2(vx, vy))


def straight(d, cmds):
    t, p = d["t"], np.column_stack([d["x"], d["y"]])
    v = np.hypot(d["vx"], d["vy"])
    go = next(c[0] for c in cmds if c[2] > 0)
    stop = next(c[0] for c in reversed(cmds) if c[2] == 0)
    moving = t[(t > go) & (v > 2)]
    i_stop = int(np.searchsorted(t, stop))
    after = t > stop
    rest = t[after & (v < 2)]
    # speed while cruising: frames from 0.3 s after start-of-motion to the stop
    cruise = v[(t > (moving[0] + 0.3 if len(moving) else go)) & (t < stop)]
    return (f"start lag {(moving[0] - go) * 1000:4.0f} ms | cruise {np.median(cruise) / 100 if len(cruise) else np.nan:.2f} m/s "
            f"(max {v.max() / 100:.2f}) | stop at {v[min(i_stop, len(v) - 1)] / 100:.2f} m/s: slid "
            f"{np.linalg.norm(p[-1] - p[min(i_stop, len(p) - 1)]):5.1f} cm, still after "
            f"{(rest[0] - stop) * 1000 if len(rest) else np.nan:4.0f} ms | travel {np.linalg.norm(p[-1] - p[0]):5.1f} cm")


def turn(d, cmds):
    t, v = d["t"], np.hypot(d["vx"], d["vy"])
    tc = [c for c in cmds if c[2] > 0]
    t_turn, target = tc[1][0], tc[1][1]
    lines = [f"turn to {target} deg at t={t_turn:.2f}s, speed raw {tc[1][2]}"]
    for k in np.flatnonzero((t > t_turn - 0.3) & (t < t_turn + 1.6)):
        lines.append(f"    t{t[k] - t_turn:+.2f}s  speed {v[k] / 100:.2f} m/s  course {course(d['vx'][k], d['vy'][k]):+6.0f}"
                     f"  yaw {d['yaw'][k]:+6.0f}  pos ({d['x'][k]:+6.1f},{d['y'][k]:+6.1f}) cm")
    return "\n".join(lines)


def main():
    key = sys.argv[1] if len(sys.argv) > 1 else ""
    stems = sorted({p.name.rsplit("_", 1)[0] for p in LOGS.glob("*_frames.csv")})
    for stem in stems:
        if key not in stem:
            continue
        d, cmds = load(stem)
        period = np.diff(d["t"])
        head = f"{stem}: {len(d['t'])} frames, period median {np.median(period) * 1000:.0f} ms (p90 {np.quantile(period, .9) * 1000:.0f})"
        if "straight" in stem:
            print(head, "|", straight(d, cmds))
        elif "turn" in stem:
            print(head)
            print(turn(d, cmds))
        else:
            print(head, f"| locator spread {np.nanstd(d['x']):.2f}/{np.nanstd(d['y']):.2f} cm, yaw spread {np.nanstd(d['yaw']):.2f} deg")


if __name__ == "__main__":
    main()
