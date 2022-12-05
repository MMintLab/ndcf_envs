import argparse

import mmint_utils
import open3d as o3d
import vedo_utils
import utils
from vedo import Plotter, Points


# TODO: Unified visualization toolkit.
def vis_sdf_data(query_points, sdf):
    plt = Plotter(shape=(1, 2))
    plt.at(0).show(Points(query_points), vedo_utils.draw_origin(), "All Sample Points")
    plt.at(1).show(Points(query_points[sdf <= 0.0], c="b"), vedo_utils.draw_origin(), "Occupied Points")
    plt.interactive().close()


def generate_pretrain_data(mesh_fn: str, out_fn: str, n: int = 50000, vis: bool = False):
    mesh = o3d.io.read_triangle_mesh(mesh_fn)
    query_points, sdf = utils.get_sdf_values(mesh, n_random=n // 2, n_off_surface=n // 2)

    # Add mount height to query points to account for the tool mounting.
    mount_height = 0.036
    query_points[:, 2] += mount_height

    data_dict = {
        "n_points": n,
        "query_points": query_points,
        "sdf": sdf,
    }
    mmint_utils.save_gzip_pickle(data_dict, out_fn)

    if vis:
        vis_sdf_data(query_points, sdf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate SDF pretraining data.")
    parser.add_argument("mesh_fn", type=str, help="Mesh to generate SDF values for (use original .obj).")
    parser.add_argument("out_fn", type=str, help="Out file to save samples.")
    parser.add_argument('-v', '--vis', dest='vis', action='store_true', help='Visualize.')
    parser.set_defaults(vis=False)
    args = parser.parse_args()

    generate_pretrain_data(args.mesh_fn, args.out_fn, vis=args.vis)
