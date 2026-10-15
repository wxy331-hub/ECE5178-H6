"""One race: where it starts, which way the robot faces, where it ends, and the route.

The maze is the one from Labs 3 and 4 (custom_maze_full).  Plates are 0.25 m
squares, addressed (col, row) as in that file: col 0..4 west to east, row 0..4
north to south, so (0, 4) is (-0.5, -0.5) and (4, 0) is (0.5, 0.5).

Headings follow the simulator and the Sphero: 0 points north (+y) and angles
grow clockwise, so 90 is east.

The route is the shortest one on the plate graph, and among the shortest the
one with fewest turns, counting the turn needed to set off from the start
heading.  On the robot every turn is a stop and a turn on the spot, about
1.4-2.1 s each on 2026-10-02.
"""

import heapq
import re
from dataclasses import dataclass

import numpy as np

from sphero_env.envs.custom_maze_full import BLOCKED_EDGES, cell_to_world

PLATE = 0.25
SIZE = 5
# North, east, south, west as (d_col, d_row): row grows southwards.
_STEPS = ((0, -1), (1, 0), (0, 1), (-1, 0))
_HEADINGS = (0, 90, 180, 270)


def _blocked(a, b):
    return (a, b) in BLOCKED_EDGES or (b, a) in BLOCKED_EDGES


def neighbours(cell):
    """(direction index, cell) for every plate one step from ``cell`` with no wall between."""
    col, row = cell
    for d, (dc, dr) in enumerate(_STEPS):
        other = (col + dc, row + dr)
        if 0 <= other[0] < SIZE and 0 <= other[1] < SIZE and not _blocked(cell, other):
            yield d, other


def reachable_from(cell):
    seen, stack = {cell}, [cell]
    while stack:
        for _, other in neighbours(stack.pop()):
            if other not in seen:
                seen.add(other)
                stack.append(other)
    return seen


def world_to_cell(xy):
    """The plate a world position lies on, or None outside the maze."""
    col = int(np.floor((float(xy[0]) + SIZE * PLATE / 2) / PLATE))
    row = int(np.floor((SIZE * PLATE / 2 - float(xy[1])) / PLATE))
    return (col, row) if 0 <= col < SIZE and 0 <= row < SIZE else None


def parse_place(text):
    """A position from the course's plate numbers, 'A1'..'E5', or metres.

    - '0,4' or '(0, 4)': whole numbers are a plate as the course numbers
      them (custom_maze_full's (col, row): col 0..4 west to east, row 0..4
      north to south), so (0, 4) is (-0.5, -0.5) and (4, 0) is (0.5, 0.5).
    - 'A1'..'E5': letters are columns west to east, digits rows south to
      north, so A1 is (0, 4) and E5 is (4, 0).
    - '-0.5,-0.5': numbers with a decimal point are metres, kept as given;
      they must lie on a plate.
    """
    text = text.strip()
    match = re.fullmatch(r"\(?\s*([0-4])\s*,\s*([0-4])\s*\)?", text)
    if match:
        return np.asarray(cell_to_world((int(match.group(1)), int(match.group(2)))), dtype=np.float64)
    match = re.fullmatch(r"([A-Ea-e])([1-5])", text)
    if match:
        col = "abcde".index(match.group(1).lower())
        row = SIZE - int(match.group(2))
        return np.asarray(cell_to_world((col, row)), dtype=np.float64)
    parts = [p for p in re.split(r"[,\s()]+", text) if p]
    if len(parts) != 2:
        raise ValueError(f"not a plate (A1..E5) or an 'x,y' position: {text!r}")
    xy = np.array([float(p) for p in parts], dtype=np.float64)
    if world_to_cell(xy) is None:
        raise ValueError(f"{text!r} is outside the maze")
    return xy


def plate_name(cell):
    """The course's (col, row) and the A1..E5 name, e.g. '(0,4)/A1'."""
    return f"({cell[0]},{cell[1]})/{'ABCDE'[cell[0]]}{SIZE - cell[1]}"


@dataclass(frozen=True)
class Mission:
    start_xy: np.ndarray       # m, where the robot is placed
    start_heading_deg: int     # 0, 90, 180 or 270
    goal_xy: np.ndarray        # m, the goal plate's centre

    @property
    def start_cell(self):
        return world_to_cell(self.start_xy)

    @property
    def goal_cell(self):
        return world_to_cell(self.goal_xy)

    @property
    def start_heading(self):
        return float(np.deg2rad(self.start_heading_deg))

    def describe(self):
        return (f"start {plate_name(self.start_cell)} ({self.start_xy[0]:+.2f}, {self.start_xy[1]:+.2f}) "
                f"facing {self.start_heading_deg} deg, goal {plate_name(self.goal_cell)} "
                f"({self.goal_xy[0]:+.2f}, {self.goal_xy[1]:+.2f})")


def make_mission(start, heading_deg, goal):
    """Check a race before it runs: on the maze, a legal heading, a reachable goal."""
    heading_deg = int(heading_deg) % 360
    if heading_deg not in _HEADINGS:
        raise ValueError(f"start heading must be 0, 90, 180 or 270, not {heading_deg}")
    start_xy = parse_place(start) if isinstance(start, str) else np.asarray(start, dtype=np.float64)
    goal_xy = parse_place(goal) if isinstance(goal, str) else np.asarray(goal, dtype=np.float64)
    start_cell, goal_cell = world_to_cell(start_xy), world_to_cell(goal_xy)
    if start_cell is None or goal_cell is None:
        raise ValueError("start and goal must both lie on the maze")
    # Checked on the plate graph itself: the lab planner snaps an unreachable
    # start onto the goal's region to cope with a drifting estimate, which
    # would hide a robot placed on the walled-in plate.
    if goal_cell not in reachable_from(start_cell):
        raise ValueError(f"no route from {plate_name(start_cell)} to {plate_name(goal_cell)}")
    # The goal is its plate's centre: the run is judged on the whole robot
    # being inside that plate.
    goal_centre = np.asarray(cell_to_world(goal_cell), dtype=np.float64)
    return Mission(start_xy, heading_deg, goal_centre)


def plan_cells(start_cell, start_heading_deg, goal_cell):
    """Plates from start to goal: fewest plates, then fewest turns, then least turning.

    Dijkstra over (plate, heading): at most 25 x 4 states.  A turn costs one
    turn and its size in quarter turns (1, or 2 for turning round).
    """
    start_dir = _HEADINGS.index(int(round(start_heading_deg / 90.0)) % 4 * 90)
    best = {(start_cell, start_dir): (0, 0, 0)}
    previous = {}
    queue = [((0, 0, 0), start_cell, start_dir)]
    while queue:
        cost, cell, d = heapq.heappop(queue)
        if cost > best.get((cell, d), cost):
            continue
        if cell == goal_cell:
            path = [cell]
            key = (cell, d)
            while key in previous:
                key = previous[key]
                path.append(key[0])
            return path[::-1]
        for nd, other in neighbours(cell):
            quarter = min((nd - d) % 4, (d - nd) % 4)
            step = (cost[0] + 1, cost[1] + (quarter > 0), cost[2] + quarter)
            if step < best.get((other, nd), (np.inf,)):
                best[(other, nd)] = step
                previous[(other, nd)] = (cell, d)
                heapq.heappush(queue, (step, other, nd))
    raise ValueError(f"no route from {start_cell} to {goal_cell}")


def route_waypoints(mission, cells=None):
    """The route as waypoints: every plate centre after the start, ending on the goal.

    Every plate centre is kept, as Lab 3's planner keeps one per plate: a
    waypoint every 0.25 m stops heading error building up along a long leg.
    The first waypoint is the start plate's centre only if the robot was not
    placed on it.
    """
    if cells is None:
        cells = plan_cells(mission.start_cell, mission.start_heading_deg, mission.goal_cell)
    points = [np.asarray(cell_to_world(c), dtype=np.float64) for c in cells]
    if np.linalg.norm(points[0] - mission.start_xy) < 0.02:
        points = points[1:]
    return points or [np.asarray(mission.goal_xy, dtype=np.float64)]


def count_turns(cells, start_heading_deg):
    d = int(round(start_heading_deg / 90.0)) % 4
    turns = 0
    for a, b in zip(cells, cells[1:]):
        nd = _STEPS.index((b[0] - a[0], b[1] - a[1]))
        turns += nd != d
        d = nd
    return turns
