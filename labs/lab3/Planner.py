import heapq
from collections import deque

import numpy as np

from sphero_env.envs.custom_maze_full import build_occupancy_grid

map = build_occupancy_grid()

# 4-connected: the maze corridors are one cell wide, so a diagonal step would
# always cut a corner through a wall.
_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))


### Implement a planner and controller for the Sphero robot to navigate to a goal position in the environment.
class Planner:
    def __init__(self, map, dt=0.1, grid_resolution=0.125):
        """
        The map (input) is the occupancy grid produced by the maze generator:
            map : a 2D numpy array of shape (h, w) with values in {0, 1}
                  1 = wall / obstacle cell
                  0 = free cell
        World origin (0, 0) is at the centre of the grid, so cell (i, j) maps
        to world coordinates via the environment's grid resolution.
        """
        self.map = np.asarray(map, dtype=np.uint8)
        self.dt = dt
        self.grid_resolution = float(grid_resolution)
        # A road plate spans two occupancy cells, so this leaves a waypoint on
        # every plate centre rather than only where the path turns.
        self.max_leg = 2.0 * self.grid_resolution

    # ---------------- grid <-> world ----------------
    # Both follow SpheroEnv._is_collision's convention, so a cell the planner
    # calls free is a cell the simulator also calls free.

    def world_to_grid(self, position):
        """Convert a world position (metres) to (row, column)."""
        height, width = self.map.shape
        column = int(np.floor(position[0] / self.grid_resolution + width / 2.0))
        row = int(np.floor(-position[1] / self.grid_resolution + height / 2.0))
        return row, column

    def grid_to_world(self, cell):
        """Convert (row, column) to the world position of that cell's centre."""
        height, width = self.map.shape
        row, column = cell
        return np.array(
            [
                (column + 0.5 - width / 2.0) * self.grid_resolution,
                (height / 2.0 - row - 0.5) * self.grid_resolution,
            ],
            dtype=np.float32,
        )

    # ---------------- occupancy queries ----------------

    def _in_bounds(self, cell):
        height, width = self.map.shape
        return 0 <= cell[0] < height and 0 <= cell[1] < width

    def _is_free(self, cell):
        return self._in_bounds(cell) and self.map[cell[0], cell[1]] == 0

    def is_clear(self, position):
        """True when a world position sits on a free cell.

        The control loop replans when this goes false: the estimate has drifted
        into a wall, so the waypoints it is still chasing were computed for a
        pose the robot is not in.
        """
        return self._is_free(self.world_to_grid(np.asarray(position, dtype=np.float64)))

    def _connected(self, cell):
        """Every free cell reachable from ``cell``."""
        reached = {cell}
        queue = deque([cell])
        while queue:
            current = queue.popleft()
            for step in _NEIGHBOURS:
                candidate = (current[0] + step[0], current[1] + step[1])
                if candidate not in reached and self._is_free(candidate):
                    reached.add(candidate)
                    queue.append(candidate)
        return reached

    def _nearest_free(self, cell, within=None):
        """Snap a cell onto the maze.

        Replanning starts from the estimated pose, and that estimate drifts, so
        the start cell can land inside a wall while the robot is in a corridor.
        Refusing to plan from there would strand the robot.

        ``within`` limits the answer to those cells.  The custom maze walls
        plate (0, 1) in on every side, and the first free cell north of the
        wall above (0, 2) is that plate, so an estimate that overruns that
        corner would otherwise snap somewhere with no route to the goal.
        """
        accept = self._is_free if within is None else within.__contains__
        height, width = self.map.shape
        cell = (
            int(np.clip(cell[0], 0, height - 1)),
            int(np.clip(cell[1], 0, width - 1)),
        )
        if accept(cell):
            return cell

        seen = {cell}
        queue = deque([cell])
        while queue:
            current = queue.popleft()
            for step in _NEIGHBOURS:
                candidate = (current[0] + step[0], current[1] + step[1])
                if candidate in seen or not self._in_bounds(candidate):
                    continue
                seen.add(candidate)
                if accept(candidate):
                    return candidate
                queue.append(candidate)
        raise ValueError("the occupancy grid contains no free cell")

    # ---------------- search ----------------

    def _search(self, start, goal):
        """A* over free cells; Manhattan is admissible for unit 4-connected steps."""
        frontier = [(0, 0, start)]
        came_from = {}
        best_cost = {start: 0}

        while frontier:
            _, spent, cell = heapq.heappop(frontier)
            if cell == goal:
                break
            if spent > best_cost.get(cell, spent):
                continue  # a cheaper route to this cell was already expanded
            for step in _NEIGHBOURS:
                candidate = (cell[0] + step[0], cell[1] + step[1])
                if not self._is_free(candidate):
                    continue
                cost = spent + 1
                if cost >= best_cost.get(candidate, cost + 1):
                    continue
                best_cost[candidate] = cost
                came_from[candidate] = cell
                priority = cost + abs(candidate[0] - goal[0]) + abs(candidate[1] - goal[1])
                heapq.heappush(frontier, (priority, cost, candidate))

        if goal not in best_cost:
            return None

        path = [goal]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        return path[::-1]

    # ---------------- path -> waypoints ----------------

    @staticmethod
    def _corners(cells):
        """Drop the cells in the middle of a straight run."""
        if len(cells) < 3:
            return list(cells)
        kept = [cells[0]]
        for previous, cell, following in zip(cells, cells[1:], cells[2:]):
            entering = (cell[0] - previous[0], cell[1] - previous[1])
            leaving = (following[0] - cell[0], following[1] - cell[1])
            if entering != leaving:
                kept.append(cell)
        kept.append(cells[-1])
        return kept

    def _resample(self, points):
        """Split any leg longer than one plate.

        The controller drives at whichever waypoint is current, so a long leg
        lets heading error carry the robot off the corridor centre before the
        next correction.
        """
        dense = [points[0]]
        for target in points[1:]:
            origin = dense[-1]
            span = float(np.linalg.norm(target - origin))
            pieces = max(1, int(np.ceil(span / self.max_leg)))
            for index in range(1, pieces + 1):
                dense.append(origin + (target - origin) * (index / pieces))
        return dense

    def plan(self, state, goal):
        """
        Fill in this function to implement a simple planner that computes the action based on the current state,
        the map and the goal position. The planner should return a sequence of 2D waypoints to reach the goal.

        The inputs are:
            state : the current state, same format as the previous labs:
                    [x, y, heading, speed]
            goal  : the goal position, a position-only array: [x, y]
        Returns:
            waypoints : a list of 2D waypoints to reach the goal, where each
                        waypoint is a position-only array: [x, y]
        """
        goal = np.asarray(goal, dtype=np.float32)[:2]
        goal_cell = self._nearest_free(self.world_to_grid(goal.astype(np.float64)))
        start_cell = self._nearest_free(
            self.world_to_grid(np.asarray(state, dtype=np.float64)),
            within=self._connected(goal_cell),
        )

        cells = self._search(start_cell, goal_cell)
        if cells is None:
            # Not reachable: the start was snapped into the goal's own region.
            # The fallback that used to sit here, [goal], drove the robot
            # straight at the goal through whatever walls lay between.
            raise ValueError("no path to the goal from the snapped start")

        waypoints = self._resample([self.grid_to_world(cell) for cell in self._corners(cells)])
        # The goal need not sit on a cell centre; finish on the goal itself so
        # the run is scored against where the marker expects the robot to stop.
        waypoints[-1] = goal
        return waypoints
