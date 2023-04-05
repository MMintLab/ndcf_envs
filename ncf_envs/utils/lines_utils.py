import numpy as np

from ncf_envs.utils.utils import pose_to_matrix


def draw_origin(gym, viewer, env, scale=0.02):
    """
    Draw the origin in Isaac Gym with the Lines API.
    """
    draw_pose(gym, viewer, env, np.array([0, 0, 0, 0, 0, 0]), scale=scale)


def draw_pose(gym, viewer, env, pose=None, matrix=None, scale=0.02, axes="rxyz"):
    """
    Draw a pose in Isaac Gym with the Lines API.
    """
    if matrix is None:
        matrix = pose_to_matrix(pose, axes=axes)

    lines = [
        matrix[:3, 3],
        matrix[:3, 3] + (scale * matrix[:3, 0]),
        matrix[:3, 3],
        matrix[:3, 3] + (scale * matrix[:3, 1]),
        matrix[:3, 3],
        matrix[:3, 3] + (scale * matrix[:3, 2]),
    ]
    c = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    gym.add_lines(viewer, env, 3, lines, c)
