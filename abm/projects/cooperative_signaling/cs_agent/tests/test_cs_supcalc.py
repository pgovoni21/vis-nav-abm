import numpy as np
import pytest
from unittest import mock

from abm.projects.cooperative_signaling.cs_agent import cs_supcalc
from abm.projects.cooperative_signaling.cs_agent.cs_supcalc import signaling


def test_random_walk():
    """Test random_walk()"""
    # set random seed
    np.random.seed(42)

    # passing values to override abm.movementparams values read from .env file
    dvel, dtheta = cs_supcalc.random_walk(desired_vel=1, exp_theta_min=-0.3,
                                          exp_theta_max=0.3)
    assert dvel == 1.0
    assert dtheta == -0.0752759286915825

    # test case: not passing min theta, read from module fixed var
    with mock.patch('abm.contrib.movement_params.exp_theta_min', -0.5):
        dvel, dtheta = cs_supcalc.random_walk(desired_vel=1, exp_theta_min=None,
                                              exp_theta_max=0.3)
        assert dvel == 1.0
        assert dtheta == 0.26057144512793295

    # test case: not passing velocity, read from module fixed var
    with mock.patch('abm.contrib.movement_params.exp_vel_max', 3):
        dvel, dtheta = cs_supcalc.random_walk(desired_vel=None,
                                              exp_theta_min=-0.3,
                                              exp_theta_max=0.3)
        assert dvel == 3
        assert dtheta == 0.13919636508684308


def test_reflection_from_circular_wall():
    """Test reflection_from_circular_wall()"""

    new_orientation = cs_supcalc.reflection_from_circular_wall(
        0, 1, np.pi / 2)
    assert new_orientation == np.pi * 3 / 2

    new_orientation = cs_supcalc.reflection_from_circular_wall(
        1, 0, np.pi)
    assert new_orientation == np.pi * 2

    # test very flat reflection angle
    orient = np.pi + np.pi / 6
    vec_i = [np.cos(orient), np.sin(orient)]
    # orientation inside the circle
    i_orientation = np.pi + np.arctan2(vec_i[1], vec_i[0])
    new_orientation = cs_supcalc.reflection_from_circular_wall(
        0, 1, orient)
    assert new_orientation == i_orientation


@pytest.mark.parametrize(
    "meter, prev_meter, prev_theta, taxis_dir, new_theta, new_taxis_dir",
    [
        # no change in meter, prev. taxis_dir is None
        # turn according to prev_theta and set it 1 if prev_theta 0
        (1, 1, 0, None, 1, None),
        (1, 1, 1, None, 1, None),
        (1, 1, -1, None, -1, None),

        # previous meter was larger
        # and prev_theta > 0
        (0, 1, 1, None, 0, -1),
        # prev_theta < 0
        (0, 1, -1, None, 0, 1),

        # previous meter was smaller
        # prev_theta > 0
        (1, 0, 1, None, 1, None),
        # prev_theta < 0
        (1, 0, -1, None, -1, None),

        # previous meter was smaller
        # prev_theta > 0
        (0.1, 0, 1, None, 0.1, None),
        # prev_theta < 0
        (0.1, 0, -1, None, -0.1, None),
    ]
)
def test_phototaxis(meter, prev_meter, prev_theta, taxis_dir, new_theta,
                    new_taxis_dir):
    """Test phototaxis()"""
    phototaxis_theta_step = 1

    _new_theta, _new_taxis_dir = cs_supcalc.phototaxis(
        meter, prev_meter, prev_theta, taxis_dir, phototaxis_theta_step)

    assert new_theta == _new_theta
    assert new_taxis_dir == _new_taxis_dir


@pytest.mark.parametrize(
    "meter, is_signaling, signaling_cost, probability_of_starting_signaling, "
    "rand_value, new_true_is_signaling",
    [
        # meter 0, not signaling
        (0, False, 0, 0, 0, False),
        # meter > 0 and already signaling
        (0.2, True, 0, 0, 0, True),
        # meter > 0 and probability_of_starting_signaling > rand_value
        (1, False, 0, 1, 0, True),
    ]
)
def test_signaling(meter, is_signaling, signaling_cost,
                   probability_of_starting_signaling, rand_value,
                   new_true_is_signaling):
    """Test signaling()"""
    new_is_signaling = signaling(meter, is_signaling, signaling_cost,
                                 probability_of_starting_signaling, rand_value)
    assert new_is_signaling == new_true_is_signaling