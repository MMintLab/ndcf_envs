import os

import cv2
import open3d as o3d
import mmint_utils

import utils
import numpy as np


def load_pointcloud(filename):
    pc: o3d.geometry.PointCloud = o3d.io.read_point_cloud(filename)
    points = np.asarray(pc.points)
    colors = np.asarray(pc.colors)
    if colors.shape[0] == points.shape[0]:
        return np.concatenate([points, colors], axis=1)
    else:
        return points


def load_image(filename):
    return cv2.imread(filename)


def load_observation_from_file(out_dir: str, example_name: str):
    # Realsense in pointcloud.
    realsense_fn = os.path.join(out_dir, "%s_realsense.ply" % example_name)
    realsense = load_pointcloud(realsense_fn)

    # Photoneo in pointcloud.
    photoneo_scanned = os.path.exists(os.path.join(out_dir, "%s_photoneo.ply" % example_name))
    if photoneo_scanned:
        photoneo_fn = os.path.join(out_dir, "%s_photoneo.ply" % example_name)
        photoneo = load_pointcloud(photoneo_fn)

    # RGB.
    rgb_fn = os.path.join(out_dir, "%s_rgb.png" % example_name)
    rgb_image = utils.load_image(rgb_fn)

    # Read the rest of the information to pkl file.
    data_fn = os.path.join(out_dir, "%s.pkl.gzip" % example_name)
    obs_dict = mmint_utils.load_gzip_pickle(data_fn)
    timestamp_dict = obs_dict.pop("timestamp")

    # Add visual data back to the dictionary.
    obs_dict["visual"] = {
        "rgb": (rgb_image, timestamp_dict["rgb"]),
        "realsense": (realsense, timestamp_dict["realsense"]),
    }
    if photoneo_scanned:
        obs_dict["visual"].update({
            "photoneo": (photoneo, timestamp_dict["photoneo_pc"]),
        })

    return obs_dict
