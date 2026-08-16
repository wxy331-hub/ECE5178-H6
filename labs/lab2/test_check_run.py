"""Tests for the hardware-run diagnostic checker.

The checker only ever runs on data that cannot be reproduced -- one hardware
session -- so its verdicts are pinned here against synthetic logs whose truth
is known by construction.  The heading in these logs ramps toward the command
at the model's turn-rate limit rather than stepping, because a checker that
only works on idealised data would pass here and mislead on the day.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from . import check_run
except ImportError:
    import check_run


TURN_LIMIT_PER_STEP = 2.61 * 0.1


def _hardware_log(
    *,
    sign: float = 1.0,
    imu_used: bool = True,
    imu_present: bool = True,
    encoder: bool = True,
    echo_speed: bool = False,
) -> dict[str, np.ndarray]:
    steps = 200
    leg = np.minimum(np.arange(steps) // 50, 3)
    commanded = np.array([0.0, np.pi / 2.0, np.pi, -np.pi / 2.0])[leg]

    heading = np.zeros(steps)
    for i in range(1, steps):
        error = check_run._wrap(np.array([commanded[i] - heading[i - 1]]))[0]
        heading[i] = check_run._wrap(
            np.array([heading[i - 1] + np.clip(error, -TURN_LIMIT_PER_STEP, TURN_LIMIT_PER_STEP)])
        )[0]

    encoder_speed = np.full(steps, 0.18)
    return {
        "step": np.arange(steps, dtype=float),
        "t_rel": np.arange(steps) * 0.1,
        "speed_cmd": np.full(steps, 0.10),
        "heading_cmd": commanded,
        "obs_heading": commanded.copy(),
        "obs_speed": np.full(steps, 0.10),
        "enc_speed_m_s": encoder_speed if encoder else np.full(steps, np.nan),
        "imu_yaw_deg": (
            np.rad2deg(check_run._wrap(sign * heading))
            if imu_present
            else np.full(steps, np.nan)
        ),
        "z_heading": heading.copy() if imu_used else commanded.copy(),
        "z_speed": np.full(steps, 0.10) if echo_speed else encoder_speed.copy(),
        "nu_heading": np.zeros(steps),
        "nu_speed": np.zeros(steps),
        "nis": np.random.default_rng(0).chisquare(2, steps),
    }


def _run(check, log) -> check_run.Report:
    report = check_run.Report()
    check(log, report)
    return report


def test_a_healthy_run_reports_no_failures():
    log = _hardware_log()
    for check in (
        check_run.check_speed_source,
        check_run.check_heading_source,
        check_run.check_imu_sign,
    ):
        assert not _run(check, log).failures


def test_a_command_echo_is_not_mistaken_for_an_encoder_reading():
    report = _run(check_run.check_speed_source, _hardware_log(encoder=False, echo_speed=True))
    assert len(report.failures) == 1


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_the_yaw_sign_is_recovered_from_the_log(sign):
    report = _run(check_run.check_imu_sign, _hardware_log(sign=sign))
    assert bool(report.failures) is (sign < 0)


def test_a_tracking_robot_does_not_look_like_a_command_echo():
    """The regression this checker was rewritten for.

    obs_heading is the commanded heading on hardware, so a robot that follows
    its command makes an honest IMU reading agree with the observation.  An
    earlier version compared the two and failed a perfectly good run.
    """

    log = _hardware_log()
    assert np.allclose(log["obs_heading"], log["heading_cmd"])
    assert not _run(check_run.check_heading_source, log).failures


def test_a_mid_run_fallback_to_the_observation_is_caught():
    log = _hardware_log(sign=-1.0, imu_used=False)
    assert _run(check_run.check_heading_source, log).failures


def test_a_missing_imu_stream_is_reported_rather_than_guessed():
    log = _hardware_log(imu_present=False, imu_used=False)
    assert _run(check_run.check_heading_source, log).failures
    assert not _run(check_run.check_imu_sign, log).failures  # skipped, not failed


def test_a_drifting_control_period_is_flagged():
    log = _hardware_log()
    log["t_rel"] = np.arange(200) * 0.14
    assert _run(check_run.check_timing, log).failures


def test_a_constant_innovation_offset_is_called_a_bias():
    log = _hardware_log()
    log["nu_speed"] = np.full(200, -0.082)
    assert _run(check_run.check_innovation_bias, log).failures
