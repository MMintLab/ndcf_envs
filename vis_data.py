import argparse

import mmint_utils
import numpy as np
import matplotlib.pyplot as plt

import trimesh


def plot_points(points: np.ndarray, colors=None, size=0.04, vis=True):
    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=size)
    ax.set_xlim3d(points.min(), points.max())
    ax.set_ylim3d(points.min(), points.max())
    ax.set_zlim3d(points.min(), points.max())
    ax.grid()

    if vis:
        plt.show()


def vis_data(data_fn):
    # base_mesh: trimesh.Trimesh = trimesh.load(base_mesh_fn)
    # new_mesh = base_mesh.copy()
    data_dict = mmint_utils.load_gzip_pickle(data_fn)

    nodal_coords = data_dict["nodal_coords"]

    # new_mesh.vertices = nodal_coords[0]
    # new_mesh.export("test.ply")

    contact_points = data_dict["contact_points"]

    points = np.concatenate([nodal_coords[0],
                             np.array([list(ctc_pt) for ctc_pt in contact_points])], axis=0)
    colors = np.concatenate([
        np.array([[0.0, 0.0, 1.0]] * len(nodal_coords[0])),
        np.array([[1.0, 0.0, 1.0]] * len(contact_points)),
    ], axis=0)
    plot_points(points, colors, vis=True)

    plot_points(points[len(nodal_coords[0]):], colors[len(nodal_coords[0]):])

    rgb_image = np.array(data_dict["rgb"])
    rgb_image = rgb_image.reshape([512, 512, 4])
    depth_image = np.clip(data_dict["depth"], a_min=-1.0, a_max=0.0)
    segment_image = data_dict["segmentation"]

    ax1 = plt.subplot(131)
    ax1.axis("off")
    ax1.imshow(rgb_image)

    ax2 = plt.subplot(132)
    ax2.axis("off")
    ax2.imshow(depth_image)

    ax3 = plt.subplot(133)
    ax3.axis("off")
    ax3.imshow(segment_image)
    plt.show()

    print(data_dict["force"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="View simulated data.")
    # parser.add_argument("base_mesh", type=str)
    parser.add_argument("data_fn", type=str)
    args = parser.parse_args()

    vis_data(args.data_fn)
