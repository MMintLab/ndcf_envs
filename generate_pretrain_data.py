import argparse
import pdb

import mmint_utils
import numpy as np
import open3d as o3d
import vedo_utils
import utils
from vedo import Plotter, Points, Arrows


# TODO: Unified visualization toolkit.
def vis_pretrain_data(data_dict: dict):
    plt = Plotter(shape=(2, 2))
    plt.at(0).show(Points(data_dict["query_points"]), vedo_utils.draw_origin(), "All Sample Points")
    plt.at(1).show(Points(data_dict["query_points"][data_dict["sdf"] <= 0.0], c="b"), vedo_utils.draw_origin(),
                   "Occupied Points")
    plt.at(2).show(Points(data_dict["query_points"][data_dict["sdf"] == 0.0], c="b"), vedo_utils.draw_origin(),
                   "Surface Points")
    plt.at(3).show(Points(data_dict["query_points"][data_dict["sdf"] == 0.0], c="b"), vedo_utils.draw_origin(),
                   Arrows(data_dict["query_points"][data_dict["sdf"] == 0.0],
                          data_dict["query_points"][data_dict["sdf"] == 0.0] + 0.01 * data_dict["normals"][
                              data_dict["sdf"] == 0.0]), "Surface Normals")
    plt.interactive().close()


def generate_pretrain_data(mesh_fn: str, out_fn: str, n_off: int = 50000, n_on: int = 5000, vis: bool = False):
    mesh = o3d.io.read_triangle_mesh(mesh_fn)

    # Sample off surface points.
    query_points, sdf = utils.get_sdf_values(mesh, n_random=n_off // 2, n_off_surface=n_off // 2)

    # Sample surface points.
    surface_points, surface_normals, _ = utils.sample_surface_points(mesh, n_on)

    # Add mount height to query points to account for the tool mounting.
    mount_height = 0.036
    query_points[:, 2] += mount_height
    surface_points[:, 2] += mount_height

    data_dict = {
        "n_points": n_off + n_on,
        "query_points": np.concatenate([query_points, surface_points]),
        "sdf": np.concatenate([sdf, np.zeros(n_on)]),
        "normals": np.concatenate([np.zeros([n_off, 3]), surface_normals])
    }
    mmint_utils.save_gzip_pickle(data_dict, out_fn)

    if vis:
        vis_pretrain_data(data_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate SDF pretraining data.")
    parser.add_argument("mesh_fn", type=str, help="Mesh to generate SDF values for (use original .obj).")
    parser.add_argument("out_fn", type=str, help="Out file to save samples.")
    parser.add_argument('-v', '--vis', dest='vis', action='store_true', help='Visualize.')
    parser.set_defaults(vis=False)
    args = parser.parse_args()

    generate_pretrain_data(args.mesh_fn, args.out_fn, vis=args.vis)
