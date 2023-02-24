from vedo import Arrow
import numpy as np
from ncf_envs.utils.utils import pose_to_matrix


def draw_origin(scale=0.02):
    """
    Helper to draw axes in vedo.
    """
    return draw_pose(np.array([0, 0, 0, 0, 0, 0]), scale=scale)


def draw_pose(pose, scale=0.02, axes="rxyz", c=None):
    """
    Helper to draw a pose in vedo.
    """
    matrix = pose_to_matrix(pose, axes=axes)

    axes = [
        Arrow(start_pt=pose[:3], end_pt=pose[:3] + (scale * matrix[:3, 0]), c="r" if c is None else c),
        Arrow(start_pt=pose[:3], end_pt=pose[:3] + (scale * matrix[:3, 1]), c="g" if c is None else c),
        Arrow(start_pt=pose[:3], end_pt=pose[:3] + (scale * matrix[:3, 2]), c="b" if c is None else c),
    ]
    return axes
