"""Training data for behaviour cloning: states, each labelled by the teacher.

Every label comes from expert.label_state, whatever the state's source, so the
dataset holds one rule.  The sources only decide which states get asked
about: points around the route, the teacher's own simulated runs, and the
positions the real robot passed through on the recorded hardware runs.
"""

import csv
from pathlib import Path

import numpy as np

import runtime as R
from expert import expert_policy, label_state

LAB_DIR = Path(__file__).resolve().parent
# Three good runs, the one that hit a wall and the one that strayed into the
# wrong plate and came back: the last two are where recovery is learned.
TRAIN_RUNS = ["data1.csv", "data2.csv", "data3.csv", "data4.csv",
              "data5(bad).csv", "data6(mid).csv"]
VALIDATION_RUNS = ["data7.csv", "data8.csv"]
# Estimated speeds, m/s: stopped, speeding up, cruising, and a sprint off a seam.
SPEEDS = np.array([0.0, 0.08, 0.18, 0.26])


def _speeds(n, rng):
    return np.clip(rng.choice(SPEEDS, n) + rng.normal(0.0, 0.02, n), 0.0, None)


def _headings(n, direction, rng):
    """Half near the given direction, half anywhere."""
    near = direction + rng.normal(0.0, np.deg2rad(30.0), n)
    anywhere = rng.uniform(-np.pi, np.pi, n)
    return np.where(rng.random(n) < 0.5, near, anywhere)


def _route_samples(n, rng):
    """Points spread along the optimal route, with its direction at each."""
    a, b = R.OPTIMAL_PATH[:-1], R.OPTIMAL_PATH[1:]
    lengths = np.linalg.norm(b - a, axis=1)
    leg = rng.choice(len(lengths), n, p=lengths / lengths.sum())
    t = rng.random(n)[:, None]
    points = a[leg] + t * (b[leg] - a[leg])
    direction = np.arctan2(b[leg, 0] - a[leg, 0], b[leg, 1] - a[leg, 1])
    return points, direction


def synthetic_states(n, rng):
    """States around the route, its corners and the goal, and some anywhere."""
    n_route, n_corner = int(0.6 * n), int(0.15 * n)
    n_any = n - n_route - n_corner
    points, direction = _route_samples(n_route, rng)
    points += rng.normal(0.0, 0.04, points.shape)
    route = np.column_stack([points, _headings(n_route, direction, rng), _speeds(n_route, rng)])

    corners = np.vstack([R.OPTIMAL_PATH[1:-1], [R.GOAL]])
    picked = corners[rng.integers(len(corners), size=n_corner)] + rng.normal(0.0, 0.05, (n_corner, 2))
    corner = np.column_stack([picked, rng.uniform(-np.pi, np.pi, n_corner), _speeds(n_corner, rng)])

    anywhere = np.column_stack([rng.uniform(-0.6, 0.6, (n_any, 2)),
                                rng.uniform(-np.pi, np.pi, n_any), _speeds(n_any, rng)])
    return np.vstack([route, corner, anywhere])


def hardware_positions(names):
    """(x, y) the filter estimated on the recorded runs: their real_x, real_y."""
    tracks = []
    for name in names:
        with open(LAB_DIR / name, newline="") as handle:
            rows = list(csv.DictReader(handle))
        tracks.append(np.array([[float(r["real_x"]), float(r["real_y"])] for r in rows]))
    return tracks


def hardware_states(names, per_point, rng):
    """Synthetic headings and speeds at the positions the robot really visited.

    The recorded runs keep positions only.  A heading read off the track is
    unreliable -- it says nothing while the robot turns on the spot, and
    collisions and filter corrections bend it -- so it is only used as the
    centre for half of the sampled headings.
    """
    states = []
    for track in hardware_positions(names):
        ahead = np.vstack([track[2:], np.repeat(track[-1:], 2, axis=0)])
        behind = np.vstack([np.repeat(track[:1], 2, axis=0), track[:-2]])
        step = ahead - behind
        direction = np.arctan2(step[:, 0], step[:, 1])
        for _ in range(per_point):
            states.append(np.column_stack([
                track, _headings(len(track), direction, rng), _speeds(len(track), rng),
            ]))
    return np.vstack(states)


def logged_states(folder=LAB_DIR.parents[1] / "logs"):
    """States the robot steered from on hardware runs lab4.py recorded.

    Each lab4_steps_*.csv row keeps the estimate its action was chosen from
    (obs_*).  Those states are relabelled like every other source, so it does
    not matter whether the network or the teacher drove that run.
    """
    columns = ("obs_x", "obs_y", "obs_heading", "obs_speed")
    states = []
    for path in sorted(Path(folder).glob("lab4_steps_*.csv")):
        with open(path, newline="", encoding="utf-8") as handle:
            states.extend([float(row[c]) for c in columns] for row in csv.DictReader(handle))
    return np.asarray(states, dtype=np.float64).reshape(-1, 4)


def random_scenario(rng):
    """A simulated robot and locator a little unlike the model, as hardware is."""
    return {
        "speed_scale": float(rng.uniform(0.85, 1.2)),
        "pos_offset": rng.uniform(-0.015, 0.015, 2),
        "heading_offset": float(rng.uniform(-np.deg2rad(4.0), np.deg2rad(4.0))),
    }


def run_scenario(policy_fn, scenario, step_log=None):
    env = R.make_sim_env(speed_scale=scenario["speed_scale"])
    try:
        return R.run_episode(policy_fn, env, pos_offset=scenario["pos_offset"],
                             heading_offset=scenario["heading_offset"], step_log=step_log)
    finally:
        env.close()


def visited_states(policy_fn, scenarios):
    """Every state a policy steered from, over a set of simulated runs."""
    states = []
    for scenario in scenarios:
        log = []
        run_scenario(policy_fn, scenario, step_log=log)
        states.extend([r["obs_x"], r["obs_y"], r["obs_heading"], r["obs_speed"]] for r in log)
    return np.asarray(states, dtype=np.float64)


def expert_run_states(n_runs, rng):
    return visited_states(expert_policy, [random_scenario(rng) for _ in range(n_runs)])


def label(states):
    """Keep the states the teacher can answer for, with its [speed, turn rate]."""
    kept, actions = [], []
    for state in states:
        action, _ = label_state(state)
        if action is not None:
            kept.append(state)
            actions.append(action)
    return np.asarray(kept, dtype=np.float64), np.asarray(actions, dtype=np.float64)
