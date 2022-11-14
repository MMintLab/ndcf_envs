from vedo import Arrow
import numpy as np
import utils


def draw_origin(scale=0.02):
    """
    Helper to draw axes in vedo.
    """
    return draw_pose(np.array([0, 0, 0, 0, 0, 0]), scale=scale)


def draw_pose(pose, scale=0.02, axes="rxyz"):
    """
    Helper to draw a pose in vedo.
    """
    matrix = utils.pose_to_matrix(pose, axes=axes)

    axes = [
        Arrow(start_pt=pose[:3], end_pt=pose[:3] + (scale * matrix[:3, 0]), c="r"),
        Arrow(start_pt=pose[:3], end_pt=pose[:3] + (scale * matrix[:3, 1]), c="g"),
        Arrow(start_pt=pose[:3], end_pt=pose[:3] + (scale * matrix[:3, 2]), c="b"),
    ]
    return axes
