"""
calc.py : Supplementary methods and calculations necessary for agents
"""
import numpy as np

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def angle_between(v1, v2, v1_norm, v2_norm):
    """ 
    Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = v1 / v1_norm
    v2_u = v2 / v2_norm
    angle = np.arccos(np.dot(v1_u, v2_u))
    # sends RuntimeWarning for when v1_u = -v2_u // outputs angle = nan --> fix

    # marks left side with negative angles, taking into account flipped y-axis
    if v1_u[0] * v2_u[1] - v1_u[1] * v2_u[0] < 0: 
        angle = -angle

    return angle


def distance(coord1, coord2):
    """Euclidean distance between 2 agent class agents in the environment as pixels""" ## todo - hamming dist for calc speed
    return np.linalg.norm(coord1 - coord2)