"""
calc.py : Supplementary methods and calculations necessary for agents
"""
import numpy as np

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def angle_bw_vis(v1, v2, v1_norm, v2_norm):
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

    dot = np.dot(v1_u, v2_u)

    # accumulated calculation errors can give dot product above/below range
    # leads to RuntimeWarning with np.arccos when given scalar within -1:1 range + outputs angle=nan
    # practical solution: np.clip
    angle = np.arccos( np.clip(dot, -1, 1) )

    # # prints info when this occurs
    # if dot < -1 or dot > 1:
    #     print(f'dot product clipped, dot: {dot}')
    #     print(f'v1: {v1} \t v1_norm: {v1_norm}')
    #     print(f'v2: {v2} \t v2_norm: {v2_norm}')

    # marks left side with negative angles, taking into account flipped y-axis
    if v1_u[0] * v2_u[1] - v1_u[1] * v2_u[0] < 0: 
        angle = -angle

    return angle

def angle_bw_coll(v1, v2, v1_norm, v2_norm):
    v1_u = v1 / v1_norm
    v2_u = v2 / v2_norm

    dot = np.dot(v1_u, v2_u)

    angle = np.arccos( np.clip(dot, -1, 1) )

    # remaps collisions from south to abosolute orientation
    if v1_u[0] * v2_u[1] - v1_u[1] * v2_u[0] < 0: 
        angle = 2*np.pi - angle

    return angle

def distance(coord1, coord2):
    """Euclidean distance between 2 agent class agents in the environment as pixels""" ## todo - hamming dist for calc speed
    return np.linalg.norm(coord1 - coord2)

import pygame
def within_group_collision(sprite1, sprite2):
    """Custom colllision check that omits collisions of sprite with itself. This way we can use group collision
    detect WITHIN a single group instead of between multiple groups"""
    if sprite1 != sprite2:
        return pygame.sprite.collide_circle(sprite1, sprite2)
    return False