from vedo import TetMesh, Plotter, show, Arrow
import meshio
import mmint_utils
import argparse
import numpy as np
import utils


def draw_axes(scale=0.02):
    axes = [
        Arrow(end_pt=[scale, 0, 0], c="r"),
        Arrow(end_pt=[0, scale, 0], c="g"),
        Arrow(end_pt=[0, 0, scale], c="b"),
    ]
    return axes


def vis_data(data_fn, base_mesh_fn):
    data_dict = mmint_utils.load_gzip_pickle(data_fn)
    wrist_pose = data_dict["wrist_pose"]

    base_mesh = meshio.read(base_mesh_fn)
    vedo_mesh = TetMesh([base_mesh.points, base_mesh.cells_dict["tetra"]]).tomesh()
    vedo_deformed_mesh = TetMesh(
        [utils.transform_pointcloud(data_dict["nodal_coords"][0], np.linalg.inv(utils.pose_to_matrix(wrist_pose))),
         base_mesh.cells_dict["tetra"]]).tomesh()

    plt = Plotter(shape=(1, 2))
    plt.at(0).show(vedo_mesh, draw_axes())
    plt.at(1).show(vedo_deformed_mesh, draw_axes())
    plt.interactive().close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vis data (vedo).")
    parser.add_argument("data_fn", type=str)
    parser.add_argument("base_mesh", type=str)
    args = parser.parse_args()

    vis_data(args.data_fn, args.base_mesh)
