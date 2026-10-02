"""Offline tests for Lab 4: the submitted interface, the teacher, the safety layer."""

import csv

import numpy as np
import pytest
import torch

import lab4
import runtime as R
from expert import expert_policy, label_state, target_for
from Policy import CRUISE_SPEED, MAX_TURN_RATE, Policy


# ---------------- the submitted interface ----------------

def test_policy_maps_one_state_and_a_batch():
    policy = Policy()
    one = policy(torch.tensor([-0.5, -0.5, 0.0, 0.0], dtype=torch.float32)).detach().numpy()
    batch = policy(torch.zeros((7, 4), dtype=torch.float32)).detach().numpy()
    assert one.shape == (2,)
    assert batch.shape == (7, 2)
    assert set(np.unique(batch[:, 0])) <= {0.0, np.float32(CRUISE_SPEED)}
    assert np.all(np.abs(batch[:, 1]) <= MAX_TURN_RATE)


def test_weights_load_the_way_the_submission_says(tmp_path):
    saved = Policy()
    torch.save(saved.state_dict(), tmp_path / "weight.pth")
    loaded = Policy()
    loaded.load_state_dict(torch.load(tmp_path / "weight.pth"))
    obs = torch.tensor([0.1, -0.2, 1.0, 0.1], dtype=torch.float32)
    assert torch.equal(saved(obs), loaded(obs))


def test_heading_plus_and_minus_pi_are_one_input():
    policy = Policy()
    a = policy(torch.tensor([0.0, 0.0, np.pi, 0.1], dtype=torch.float32))
    b = policy(torch.tensor([0.0, 0.0, -np.pi, 0.1], dtype=torch.float32))
    assert torch.allclose(a, b, atol=1e-5)


def test_turn_rate_becomes_a_heading_across_plus_minus_pi():
    state = np.array([0.0, 0.0, np.deg2rad(179.0), 0.0])
    rate = np.deg2rad(2.0) / R.DT
    _, heading = R.to_env_action(state, [0.0, rate])
    assert heading == pytest.approx(np.deg2rad(-179.0))


# ---------------- the teacher ----------------

def test_teacher_refuses_states_it_cannot_answer_for():
    assert label_state([-0.5, 0.25, 0.0, 0.0])[1] == "unreachable"  # plate (0, 1)
    assert label_state([0.9, 0.0, 0.0, 0.0])[1] == "outside"
    assert label_state([np.nan, 0.0, 0.0, 0.0])[1] == "invalid"
    action, reason = label_state([*R.GOAL, 0.0, 0.0])
    assert reason == "at-goal" and np.all(action == 0.0)


def test_teacher_answers_in_the_band_next_to_the_floor():
    # 2026-09-18 on plate (3, 0): the estimate in the wall band, the robot fine.
    action, reason = label_state([0.25, 0.4365, np.pi / 2, 0.1])
    assert reason == "ok"
    assert action[0] > 0.0  # drives on east rather than turning back


def test_teacher_reaches_a_corner_before_turning():
    # Facing north, 5.5 cm short of the first corner: go on to the corner.
    np.testing.assert_allclose(target_for(np.array([-0.5, -0.055]), 0.0), [-0.5, 0.0])
    # Facing east along the next leg, the same offset is drift: carry on.
    np.testing.assert_allclose(target_for(np.array([-0.5, -0.055]), np.pi / 2), [-0.25, 0.0])


def test_teacher_never_cuts_through_a_wall_end():
    # Lab 3's corridor test alone would send this straight past the end of the
    # wall between plates (1, 1) and (2, 1).
    target = target_for(np.array([-0.17, 0.38]), np.pi / 2)
    assert R.keeps_clear(np.array([-0.17, 0.38]), target)


# ---------------- the safety layer ----------------

def test_safety_stops_a_drive_into_a_wall():
    # Heading north on plate (3, 1), right under the wall to (3, 0).
    guard = R.SafetyLayer()
    applied, reason = guard.filter(np.array([0.25, 0.30, 0.0, 0.15]), [CRUISE_SPEED, 0.0])
    assert applied[0] == 0.0 and reason == "brake"


def test_safety_lets_a_clear_corridor_through():
    guard = R.SafetyLayer()
    applied, reason = guard.filter(np.array([-0.5, -0.4, 0.0, 0.15]), [CRUISE_SPEED, 0.0])
    assert applied[0] == CRUISE_SPEED and reason is None


def test_safety_pauses_after_a_collision_and_will_not_drive_through_a_turn():
    guard = R.SafetyLayer()
    state = np.array([-0.5, -0.4, 0.0, 0.1])
    for _ in range(R.SafetyLayer.PAUSE_STEPS):
        applied, reason = guard.filter(state, [CRUISE_SPEED, 0.0], collided=_ == 0)
        assert applied[0] == 0.0 and reason == "collision-pause"
    assert guard.filter(state, [CRUISE_SPEED, 0.0])[0][0] == CRUISE_SPEED
    # Open floor ahead: a large turn is taken slowly rather than on the spot.
    applied, reason = guard.filter(state, [CRUISE_SPEED, MAX_TURN_RATE])
    assert applied[0] == R.TURN_SPEED and reason == "slow-in-turn"
    # 17.5 cm short of the wall above the first corner: turn on the spot.
    applied, reason = guard.filter(np.array([-0.5, -0.05, 0.0, 0.1]), [CRUISE_SPEED, MAX_TURN_RATE])
    assert applied[0] == 0.0 and reason == "turn-before-drive"


# ---------------- closed loop ----------------

def _passes(controller):
    env = R.make_sim_env()
    try:
        out = R.run_episode(controller, env)
    finally:
        env.close()
    return out["stop_reason"] == "arrived" and R.passes(R.metrics(out["records"]))


def test_teacher_reaches_the_goal_in_simulation():
    assert _passes(expert_policy)


@pytest.mark.skipif(not lab4.WEIGHTS.exists(), reason="train.py has not been run")
def test_trained_policy_reaches_the_goal_in_simulation():
    assert _passes(lab4.as_controller(lab4.load_policy()))


def test_main_writes_the_lab4_csv_after_an_abort(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(lab4, "LAB_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    seen = {"n": 0}

    def abort_after_five():
        seen["n"] += 1
        if seen["n"] > 5:
            raise lab4.ExperimentAborted("test")

    monkeypatch.setattr(R, "_poll_for_abort", abort_after_five)
    lab4.main(["--sim", "--teacher"])

    assert "Run aborted" in capsys.readouterr().out
    with open(tmp_path / f"{lab4.STUDENT_ID}_lab4.csv", newline="") as handle:
        rows = list(csv.reader(handle))
    assert tuple(rows[0]) == lab4.CSV_COLUMNS
    assert len(rows) == 6  # header plus five steps


# ---------------- fixes from the first review ----------------

def test_teacher_refuses_a_band_position_that_snaps_across_the_wall():
    # South of the thin wall at y = 0.375 with room to spare, but in the band
    # the grid gives it, so the plan starts north of the wall.
    assert label_state([0.25, 0.33, 0.0, 0.0]) == (None, "across-wall")


def test_safety_brakes_straight_when_a_turn_would_swing_a_rolling_ball_at_a_wall():
    # Stopping and turning while still rolling at 0.18 m/s: the roll follows
    # the new heading toward the wall below plate (3, 0).
    guard = R.SafetyLayer()
    applied, reason = guard.filter(np.array([0.25, 0.312, np.pi / 3, 0.18]), [0.0, -3.936])
    assert reason == "brake"
    assert applied[0] == 0.0 and applied[1] == 0.0


def test_a_ball_pinned_against_a_wall_ends_the_run():
    # A policy that drives north for ever meets the wall above the first
    # corner and keeps pushing; the run has to stop rather than burn 500 steps.
    env = R.make_sim_env()
    try:
        out = R.run_episode(lambda state: np.array([CRUISE_SPEED, 0.0]), env)
    finally:
        env.close()
    assert out["stop_reason"] == "repeated_collision"
    assert out["steps"] < R.MAX_STEPS


def test_a_robot_without_imu_yaw_stops(monkeypatch):
    import test_lab3

    monkeypatch.setattr(R.time, "sleep", lambda _: None)
    robot = test_lab3._fake_robot()
    step = robot.step.side_effect

    def without_yaw(action):
        obs, reward, terminated, truncated, info = step(action)
        info["orientation"] = {}
        return obs, reward, terminated, truncated, info

    robot.step.side_effect = without_yaw
    env = R.make_sim_env()
    try:
        out = R.run_episode(expert_policy, env, robot)
    finally:
        env.close()
    assert out["stop_reason"] == "imu_lost"
    assert out["steps"] <= R.YAW_PATIENCE + 1


def test_recorded_hardware_steps_become_training_states(tmp_path):
    import dataset as D

    rows = [{"step": 0, "obs_x": -0.5, "obs_y": -0.5, "obs_heading": 0.0, "obs_speed": 0.0,
             "next_x": -0.5, "next_y": -0.48, "next_heading": 0.0, "next_speed": 0.1}]
    with open(tmp_path / "lab4_steps_0918_120000.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    states = D.logged_states(tmp_path)
    # The state the action was chosen from, not the one after it.
    np.testing.assert_allclose(states, [[-0.5, -0.5, 0.0, 0.0]])
    assert D.logged_states(tmp_path / "empty").shape == (0, 4)
