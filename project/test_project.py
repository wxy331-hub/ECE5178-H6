"""Offline tests for the project race: missions, routes, frames, and a whole run."""

import itertools
import sys
from collections import deque
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast_race  # noqa: E402
import race  # noqa: E402
from mission import (  # noqa: E402
    count_turns, make_mission, neighbours, parse_place, plan_cells, plate_name, reachable_from,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "labs" / "lab3"))
import test_lab3  # noqa: E402


# ---------------- missions ----------------

def test_the_course_plate_numbers_are_col_then_row_from_the_north_west():
    # As custom_maze_full numbers them: the labs' START_CELL (0, 4), GOAL_CELL (4, 0).
    np.testing.assert_allclose(parse_place("0,4"), [-0.5, -0.5])
    np.testing.assert_allclose(parse_place("(4, 0)"), [0.5, 0.5])
    np.testing.assert_allclose(parse_place("(1,2)"), [-0.25, 0.0])
    # With a decimal point it is metres.
    np.testing.assert_allclose(parse_place("0.0,0.0"), [0.0, 0.0])


def test_plates_name_the_maze_from_the_south_west():
    np.testing.assert_allclose(parse_place("A1"), [-0.5, -0.5])
    np.testing.assert_allclose(parse_place("e5"), [0.5, 0.5])
    np.testing.assert_allclose(parse_place("-0.25, 0"), [-0.25, 0.0])
    with pytest.raises(ValueError):
        parse_place("0.9,0")


def test_a_race_that_cannot_be_run_is_refused_before_it_starts():
    with pytest.raises(ValueError):
        make_mission("A1", 45, "E5")       # not a right angle
    with pytest.raises(ValueError):
        make_mission("A1", 0, "A4")        # the walled-in plate
    mission = make_mission("A1", 90, "E5")
    assert plate_name(mission.start_cell) == "(0,4)/A1" and mission.start_heading_deg == 90


def _shortest_paths(start, goal):
    """Every shortest plate path, by breadth-first layers."""
    dist, queue = {start: 0}, deque([start])
    while queue:
        cell = queue.popleft()
        for _, other in neighbours(cell):
            if other not in dist:
                dist[other] = dist[cell] + 1
                queue.append(other)
    paths = [[start]]
    for _ in range(dist[goal]):
        paths = [p + [o] for p in paths for _, o in neighbours(p[-1]) if dist.get(o) == dist[p[-1]] + 1]
    return [p for p in paths if p[-1] == goal]


def test_the_route_is_shortest_and_has_the_fewest_turns_from_the_start_heading():
    cells = [(c, r) for c in range(5) for r in range(5)]
    for start, goal in itertools.product(cells, cells):
        if start == goal or goal not in reachable_from(start):
            continue
        options = _shortest_paths(start, goal)
        for heading in (0, 90, 180, 270):
            route = plan_cells(start, heading, goal)
            assert len(route) == len(options[0])
            assert count_turns(route, heading) == min(count_turns(p, heading) for p in options)


# ---------------- frames ----------------

@pytest.mark.parametrize("theta0, moved", [(0, [0.0, 0.25]), (90, [0.25, 0.0]),
                                           (180, [0.0, -0.25]), (270, [-0.25, 0.0])])
def test_the_locator_turns_into_the_maze_by_the_start_heading(theta0, moved):
    # 25 cm straight ahead in the robot's own frame, from wherever it started.
    start = np.array([-0.25, 0.0])
    p = race.locator_to_maze([0.0, 0.25], [0.0, 0.0], start, np.deg2rad(theta0))
    np.testing.assert_allclose(p - start, moved, atol=1e-12)


def test_a_maze_heading_goes_out_relative_to_the_start_heading():
    # Facing east at the start, north is 90 degrees to the left.
    assert race.maze_to_sdk_heading(0.0, np.pi / 2) == pytest.approx(-np.pi / 2)
    assert race.maze_to_sdk_heading(np.pi / 2, np.pi / 2) == pytest.approx(0.0)


# ---------------- the safety pieces carried over ----------------

def test_a_driving_turn_is_checked_for_two_steps():
    # Codex's case from the Lab 4 experiment: past the third corner
    # northwards at 0.15 m/s, a 44 degree turn while driving slowly is clear
    # for one 105 ms step but not for the ~200 ms a new heading takes.
    state, speed, rate = np.array([-0.244, 0.545, -0.29, 0.15]), 0.105, 7.320666313171387
    assert race.R.SafetyLayer()._stops_clear(state, speed, rate)
    assert not race.RaceSafety()._stops_clear(state, speed, rate)


def test_a_held_heading_is_sent_only_if_it_is_safe():
    guard = race.RaceSafety()
    open_state = np.array([-0.5, -0.3, 0.0, 0.15])
    held = race.hold_heading_safely(guard, open_state, np.array([race.CRUISE_SPEED, 0.3]), 0.0)
    assert held[1] == 0.0


# ---------------- whole runs ----------------

def test_simulated_races_reach_the_goal_plate():
    env = race.R.make_sim_env()
    try:
        for start, heading, goal in (("A1", 0, "E5"), ("A1", 180, "E5"), ("E5", 270, "A1"), ("C3", 90, "A5")):
            result = race.run_race(make_mission(start, heading, goal), env, verbose=False)
            assert result["outcome"] == "arrived" and result["true_on_goal"], (start, heading, goal)
    finally:
        env.close()


def test_a_robot_set_down_facing_east_is_driven_in_maze_coordinates(monkeypatch):
    # The fake robot moves and reports in its own frame -- forward is +y, the
    # way the robot faced when its aim was reset -- as the real locator does.
    # Facing east on A1, the route north to E5 only works if every command
    # and every reading is turned by the start heading.
    monkeypatch.setattr(race.time, "sleep", lambda _: None)
    monkeypatch.setattr(race, "wait_for_frames", lambda *a: True)
    robot = test_lab3._fake_robot()
    robot.state_odom = np.zeros(4)
    robot.api.get_location.return_value = {"x": 0.0, "y": 0.0}
    robot.api.get_orientation.return_value = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    mission = make_mission("A1", 90, "E5")
    result = race.run_race(mission, robot_env=robot, toy=Mock(), raw_speed=20, verbose=False)
    assert result["outcome"] == "arrived"
    assert race.whole_ball_on_goal(result["estimate"], mission.goal_xy)
    # The first commands turn the robot left, to the maze's north: -90 degrees
    # in its own frame, reached 45 degrees a step.
    headings = [race.wrap_angle(c[0][0][1]) for c in robot.step.call_args_list[:3]]
    assert headings[0] == pytest.approx(-np.pi / 4, abs=0.01)
    assert headings[1] == pytest.approx(-np.pi / 2, abs=0.01)


def test_heading_and_speed_go_out_in_one_packet(monkeypatch):
    # Codex reviews 2026-10-09: two packets left a half-old command between
    # them, whichever went first.
    import threading
    from sphero_unsw.utils import ToyUtil

    sent = []
    monkeypatch.setattr(ToyUtil, "roll_start", staticmethod(lambda toy, h, v: sent.append((h, v))))
    robot = race.RaceRobot.__new__(race.RaceRobot)
    robot.api, robot._lock, robot._toy = Mock(), threading.Lock(), Mock()
    robot._last_heading_deg = 90
    robot.set_heading_and_speed(152, 10)
    robot.set_heading_and_speed(-30, 20)
    assert sent == [(152, 10), (330, 20)]
    assert robot.api.method_calls == []          # nothing through set_heading/set_speed
    # The SDK's own copies follow, for its stop_roll()/set_speed(0) at the end.
    assert robot.api._SpheroEduAPI__heading == 330 and robot.api._SpheroEduAPI__speed == 20


def test_the_robot_does_not_start_on_a_stale_locator(monkeypatch):
    # The SDK's getters return its cache; a locator still reading where the
    # robot was before the reset would put the start ~10 cm off.
    monkeypatch.setattr(race.time, "sleep", lambda _: None)
    monkeypatch.setattr(race, "wait_for_frames", lambda *a: True)
    robot = Mock()
    robot.api.get_location.return_value = {"x": 12.0, "y": -7.0}
    robot.api.get_orientation.return_value = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    with pytest.raises(RuntimeError):
        race.ready_robot(robot, Mock(), make_mission("A1", 90, "E5"))


def test_the_yaw_reference_is_read_after_the_robot_settles(monkeypatch):
    # Codex re-review 2026-10-09: a yaw cached before the reset (90) that the
    # stream replaces during robot_env.reset() (0) must not be the reference.
    monkeypatch.setattr(race.time, "sleep", lambda _: None)
    monkeypatch.setattr(race, "wait_for_frames", lambda *a: True)
    robot = Mock()
    robot.api.get_location.return_value = {"x": 0.0, "y": 0.0}
    robot.api.get_orientation.return_value = {"yaw": 90.0, "pitch": 0.0, "roll": 0.0}

    def settle(*a, **k):
        robot.api.get_orientation.return_value = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    robot.reset.side_effect = settle
    origin, yaw = race.ready_robot(robot, Mock(), make_mission("A1", 90, "E5"))
    assert yaw == 0.0
    np.testing.assert_allclose(origin, [0.0, 0.0])


def test_the_robot_does_not_start_without_fresh_frames(monkeypatch):
    monkeypatch.setattr(race, "READY_TIMEOUT_S", 0.05)
    robot = Mock()
    robot.api.get_location.return_value = {"x": 0.0, "y": 0.0}
    robot.api.get_orientation.return_value = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    with pytest.raises(RuntimeError):
        race.ready_robot(robot, Mock(), make_mission("A1", 90, "E5"))   # no frame ever streams in


def test_the_slope_boost_waits_until_the_robot_faces_its_heading():
    # 2026-10-09: boosted while still pivoting at a corner, the robot never
    # turned and drove off the map.
    args = Mock(boost=80)
    assert fast_race.boosted(40, 0.0, 0.0, args) == 120          # stalled, facing the leg
    assert fast_race.boosted(40, 0.0, 100.0, args) == 40         # still pivoting
    assert fast_race.boosted(40, 0.0, float("nan"), args) == 40  # no yaw yet
    assert fast_race.boosted(40, 0.4, 0.0, args) == 40           # moving fine
    assert fast_race.boosted(0, 0.0, 0.0, args) == 0             # stopping
