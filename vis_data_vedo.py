from vedo import TetMesh, Plotter, show, Arrow, Mesh, Points, Arrows
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

    # Transform deformed object points to wrist frame.
    wrist_pose = data_dict["wrist_pose"]
    deformed_vertices = utils.transform_pointcloud(data_dict["nodal_coords"][0],
                                                   np.linalg.inv(utils.pose_to_matrix(wrist_pose)))

    # Load base mesh and build tetra mesh of the undeformed and deformed mesh.
    base_mesh = meshio.read(base_mesh_fn)
    vedo_mesh = TetMesh([base_mesh.points, base_mesh.cells_dict["tetra"]]).tomesh()
    vedo_deformed_mesh = TetMesh([deformed_vertices, base_mesh.cells_dict["tetra"]]).tomesh()

    # Build surface triangle mesh.
    triangle_mesh = utils.tetrahedral_to_surface_triangles(deformed_vertices, base_mesh)
    vedo_surface_mesh = Mesh([triangle_mesh.vertices, triangle_mesh.faces])
    vedo_surface_mesh_alpha = Mesh([triangle_mesh.vertices, triangle_mesh.faces], alpha=0.2)

    # Build contact pointcloud.
    contact_points_w = np.array([list(ctc_pt) for ctc_pt in data_dict["contact_points"]])
    contact_points = utils.transform_pointcloud(contact_points_w,
                                                np.linalg.inv(utils.pose_to_matrix(wrist_pose)))
    vedo_contact_points = Points(contact_points, c="r")

    # Build contact forces.
    contact_forces = np.array(data_dict["contact_forces"]) * 0.001
    force_pts_end_w = contact_points_w + contact_forces
    force_pts_end = utils.transform_pointcloud(force_pts_end_w, np.linalg.inv(utils.pose_to_matrix(wrist_pose)))
    vedo_contact_forces = Arrows(contact_points, force_pts_end, c="r")

    plt = Plotter(shape=(2, 2))
    plt.at(0).show(vedo_mesh, draw_axes(), "Undeformed")
    plt.at(1).show(vedo_deformed_mesh, vedo_contact_points, draw_axes(), "Deformed (Tet)")
    plt.at(2).show(vedo_surface_mesh, vedo_contact_points, draw_axes(), "Deformed (Tri)")
    plt.at(3).show(vedo_surface_mesh_alpha, vedo_contact_points, vedo_contact_forces, draw_axes(), "Contact Points")
    plt.interactive().close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vis data (vedo).")
    parser.add_argument("data_fn", type=str)
    parser.add_argument("base_mesh", type=str)
    args = parser.parse_args()

    vis_data(args.data_fn, args.base_mesh)
