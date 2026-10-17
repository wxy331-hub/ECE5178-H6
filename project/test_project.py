"""Offline tests for the project race: missions, routes, frames, and a whole run."""

import itertools
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fast_race  # noqa: E402
from sphero_env.envs.custom_maze_full import cell_to_world  # noqa: E402
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


# ---------------- fast race: corners, walls, stalls ----------------

def _centre(cell):
    return np.asarray(cell_to_world(cell), dtype=float)


def test_a_closed_plate_side_sends_the_route_round_another_way():
    assert plan_cells((1, 2), 0, (4, 0))[1] == (1, 1)
    detour = plan_cells((1, 2), 0, (4, 0), {((1, 2), (1, 1))})
    assert (1, 1) not in detour and detour[-1] == (4, 0)


def test_legs_with_a_wall_all_along_one_side():
    assert fast_race.wall_side(_centre((1, 0)), _centre((4, 0))) == -1   # top row east: north wall on the left
    assert fast_race.wall_side(_centre((4, 0)), _centre((0, 0))) == 1    # top row west: on the right
    assert fast_race.wall_side(_centre((0, 4)), _centre((0, 2))) == -1   # column 0 north: the west wall
    assert fast_race.wall_side(_centre((1, 2)), _centre((1, 0))) == 0    # (1,2) is open on both sides
    # (1,1) is walled on both sides: no side to lean on.
    assert fast_race.wall_side(_centre((1, 1)) - [0.0, 0.1], _centre((1, 1))) == 0


def test_the_top_row_leans_on_the_north_wall_and_fast_corners_are_braked():
    args = SimpleNamespace(turn_lead=12.0, stop_lead=10.0, raw=40, corner_speed=0.35, wall_bias=8.0)
    top_row = [_centre((1, 0)), _centre((4, 0))]
    here, east = np.array([0.0, 0.52]), np.array([0.3, 0.0])
    heading, raw, _, stopping, turning, braking = fast_race.command_for(here, east, top_row, 0, False, args, [-1])
    assert heading == pytest.approx(82.0) and raw == 40 and not (stopping or turning or braking)
    # Without the wall the locator correction steers the "north of the line" ball south.
    heading = fast_race.command_for(here, east, top_row, 0, False, args, [0])[0]
    assert heading > 90.0
    # Towards the corner at (1,2): braked when fast, driven when not.
    legs = [_centre((0, 2)), _centre((1, 2)), _centre((1, 0))]
    fast = fast_race.command_for(np.array([-0.45, 0.0]), np.array([0.55, 0.0]), legs, 0, False, args, [0, 0])
    assert fast[1] == 0 and fast[5] and fast[2] == 0
    slow = fast_race.command_for(np.array([-0.45, 0.0]), np.array([0.30, 0.0]), legs, 0, False, args, [0, 0])
    assert slow[1] == 40 and not slow[5]


def test_the_plate_changes_only_through_an_open_side():
    assert fast_race.track_plate((0, 2), np.array([-0.365, 0.0])) == ((0, 2), None)       # 1 cm past the boundary
    assert fast_race.track_plate((0, 2), np.array([-0.350, 0.0])) == ((1, 2), (0, 2))     # 2.5 cm past
    assert fast_race.track_plate((0, 2), np.array([-0.5, 0.20])) == ((0, 2), None)        # (0,1) is behind a wall
    # Cutting the corner from (0,3) to (1,2): it came through (0,2), not from (0,3), so a
    # back-off does not go straight through the wall between (1,2) and (1,3) (Codex 1015).
    assert fast_race.track_plate((0, 3), np.array([-0.36, -0.07])) == ((1, 2), (0, 2))


def test_a_locator_that_stays_on_another_plate_is_believed():
    # 112149: read into (1,3), then west over the (0,3)|(1,3) wall; the plate stayed (1,3) for good.
    plate, came_from, lost = (1, 3), (1, 4), 0
    west = np.array([-0.38, -0.353])                       # on (0,3) by the locator
    for _ in range(fast_race.LOST_STEPS - 1):
        plate, came_from, lost = fast_race.follow_plate(plate, came_from, lost, west)
        assert plate == (1, 3)                             # a short reading over a wall is not believed
    plate, came_from, lost = fast_race.follow_plate(plate, came_from, lost, west)
    assert plate == (0, 3) and came_from is None and lost == 0   # a wall between: where it came from is unknown
    # Then on through open sides as usual.
    plate, came_from, lost = fast_race.follow_plate(plate, came_from, lost, np.array([-0.45, -0.10]))
    assert (plate, came_from) == ((0, 2), (0, 3))
    # A reading back on the plate clears the count.
    plate, came_from, lost = (1, 3), None, 3
    assert fast_race.follow_plate(plate, came_from, lost, np.array([-0.25, -0.25])) == ((1, 3), None, 0)


def test_a_stall_backs_off_to_a_plate_behind_and_avoids_a_side_that_stalls_twice():
    goal = (4, 0)
    legs = fast_race.legs_of(plan_cells((0, 4), 0, goal))      # north, east, north up the slope, east
    stalls, avoid = {}, set()
    stuck = np.array([-0.33, 0.08])                            # going north on (1,2), 8 cm past its centre
    back, route, new = fast_race.recovery(stuck, legs, 2, (1, 2), (0, 2), goal, stalls, avoid)
    assert back == (1, 2) and route[:2] == [(1, 2), (1, 1)] and not avoid
    np.testing.assert_allclose(new[0], stuck)
    np.testing.assert_allclose(new[1], _centre((1, 2)))
    # Just onto (1,1): its centre is ahead, so back to the plate it came from.
    back = fast_race.recovery(np.array([-0.25, 0.14]), legs, 2, (1, 1), (1, 2), goal, {}, set())[0]
    assert back == (1, 2)
    # The same plate and direction again: that side is closed while there is another way.
    back, route, new = fast_race.recovery(stuck, legs, 2, (1, 2), (0, 2), goal, stalls, avoid)
    assert ((1, 2), (1, 1)) in avoid and (1, 1) not in route and route[-1] == goal
    # Into (0,0) there is only one way: it stays open however often the ball stalls there
    # (125938: 1.6 cm into (0,0), still counted on (1,0)).
    legs = [_centre((1, 0)), _centre((0, 0))]
    stalls, avoid = {}, set()
    for _ in range(3):
        back, route, _ = fast_race.recovery(np.array([-0.39, 0.48]), legs, 0, (1, 0), (2, 0), (0, 0), stalls, avoid)
    assert back == (1, 0) and not avoid and route == [(1, 0), (0, 0)]
    # On the goal plate, heading for its centre, there is no side being left: nothing to close.
    stalls, avoid = {}, set()
    for _ in range(3):
        fast_race.recovery(np.array([-0.42, 0.48]), legs, 0, (0, 0), (1, 0), (0, 0), stalls, avoid)
    assert not avoid


def test_a_back_off_is_driven_to_its_point_before_the_ball_turns_round():
    # Codex 1015: with the ordinary lead the ball turned north again after backing 2-4 cm,
    # and a back-off shorter than the lead was skipped altogether.
    args = SimpleNamespace(turn_lead=12.0, stop_lead=10.0, raw=40, corner_speed=0.35, wall_bias=8.0)
    legs = fast_race.legs_of(plan_cells((0, 4), 0, (4, 0)))
    for stuck in ([-0.25, 0.08], [-0.33, 0.08], [-0.34, 0.08]):   # straight back, and off the line (135, 132 deg)
        back, _, new = fast_race.recovery(np.array(stuck), legs, 2, (1, 2), (0, 2), (4, 0), {}, set())
        assert back == (1, 2)
        target = new[1]
        way = (target - new[0]) / np.linalg.norm(target - new[0])
        sides = [0] * (len(new) - 1)
        for left_cm, expected_leg in ((7.0, 0), (3.0, 0), (1.0, 1)):
            here = target - way * left_cm / 100.0
            leg = fast_race.command_for(here, 0.15 * way, new, 0, False, args, sides, backoff_leg=0)[2]
            assert leg == expected_leg, (stuck, left_cm, leg)
    # Stalled 3 cm into the leg west from (1,0): no plate centre behind on the route, so back
    # to the open plate straight behind, (2,0), and the 28 cm back east is really driven.
    legs = [_centre((1, 0)), _centre((0, 0))]
    back, _, new = fast_race.recovery(np.array([-0.28, 0.50]), legs, 0, (1, 0), None, (0, 0), {}, set())
    assert back == (2, 0)
    heading, raw, leg = fast_race.command_for(np.array([-0.28, 0.50]), np.zeros(2), new, 0, True, args, [0, 0])[:3]
    assert leg == 0 and heading == pytest.approx(90.0, abs=1.0) and raw == 40


def test_a_ball_leaning_on_a_wall_of_the_goal_plate_is_on_it():
    assert fast_race.held_on_plate([0.5, 0.5], (4, 0))
    assert fast_race.held_on_plate([0.5, 0.592], (4, 0))       # against the north wall: the limit is 0.0885
    assert fast_race.held_on_plate([0.60, 0.60], (4, 0))       # in the corner, the locator reading past it
    assert not fast_race.held_on_plate([0.5, 0.40], (4, 0))    # south of (4,0) is open: half on (4,1)
    assert not fast_race.held_on_plate([0.40, 0.5], (4, 0))    # west is open: half on (3,0)
    assert fast_race.held_on_plate([-0.5, 0.40], (0, 0))       # south of (0,0) is a wall
    assert not fast_race.held_on_plate([-0.455, -0.09], (0, 0))  # stalled on (0,2), plates away (124624)
    assert not fast_race.held_on_plate([0.5, 0.75], (4, 0))       # 12.5 cm past the north wall: not leaning on it


def test_braking_holds_through_the_turn_until_the_ball_is_slow():
    # Codex 1015: on stepping to the next leg the braking used to be judged on that leg only.
    args = SimpleNamespace(turn_lead=12.0, stop_lead=10.0, raw=40, corner_speed=0.35, wall_bias=8.0)
    legs = fast_race.legs_of(plan_cells((0, 4), 0, (4, 0)))
    heading, raw, leg, _, turning, braking = fast_race.command_for(
        np.array([-0.385, 0.0]), np.array([0.40, 0.0]), legs, 1, False, args, [0] * 4)
    assert leg == 2 and heading == pytest.approx(0.0) and turning and braking and raw == 0
    heading, raw, leg, _, turning, braking = fast_race.command_for(
        np.array([-0.27, 0.01]), np.array([0.20, 0.10]), legs, 2, True, args, [0] * 4)
    assert raw == 40 and not braking

