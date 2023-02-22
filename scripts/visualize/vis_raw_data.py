from vedo import TetMesh, Plotter, Arrow, Mesh, Points, Arrows
import mmint_utils
import argparse
import numpy as np
import utils
import vedo_utils


def vis_data(data_fn, base_tetra_mesh_fn):
    data_dict = mmint_utils.load_gzip_pickle(data_fn)

    # Get wrist pose.
    wrist_pose = data_dict["wrist_pose"]
    w_T_wrist_pose = utils.pose_to_matrix(wrist_pose, axes="rxyz")
    wrist_pose_T_w = np.linalg.inv(w_T_wrist_pose)
    mount_pose = data_dict["mount_pose"]

    # Load deformed object points.
    def_vert_w = data_dict["nodal_coords"][0]
    def_vert = utils.transform_pointcloud(def_vert_w, wrist_pose_T_w)

    plt = Plotter()
    plt.at(0).show(Points(def_vert_w), vedo_utils.draw_pose(wrist_pose), vedo_utils.draw_pose(mount_pose),
                   vedo_utils.draw_origin())
    plt.interactive().close()

    # Load base tetra mesh of the undeformed mesh.
    tet_vert, tet_tetra = utils.load_tetmesh(base_tetra_mesh_fn)
    # tet_vedo = TetMesh([tet_vert, tet_tetra]).tomesh()
    tet_def_vedo = TetMesh([def_vert, tet_tetra]).tomesh()

    # Convert tetra mesh to triangle mesh. Note, we use the deformed vertices.
    tri_vert, tri_triangles = utils.tetrahedral_to_surface_triangles(def_vert, tet_tetra)
    tri_vedo = Mesh([tri_vert, tri_triangles])
    tri_vedo_alpha = Mesh([tri_vert, tri_triangles], alpha=0.2)

    # Load contact point cloud.
    contact_points_w = np.array([list(ctc_pt) for ctc_pt in data_dict["contact_points"]])
    contact_points = utils.transform_pointcloud(contact_points_w, wrist_pose_T_w)
    vedo_contact_points = Points(contact_points, c="r")

    # Load contact forces.
    contact_forces_w = np.array(data_dict["contact_forces"])
    contact_forces = utils.transform_vectors(contact_forces_w, wrist_pose_T_w)
    vedo_contact_forces = Arrows(contact_points, contact_points + (0.001 * contact_forces))

    # Load wrist wrench.
    wrist_wrench = np.array(data_dict["wrist_wrench"])
    vedo_wrist_force = Arrow((0, 0, 0), 0.001 * wrist_wrench[:3], c="b")
    vedo_wrist_torque = Arrow((0, 0, 0), wrist_wrench[3:], c="y")

    mount_axes = vedo_utils.draw_pose(np.array([0.0, 0.0, 0.036, 0.0, 0.0, 0.0]))

    plt = Plotter(shape=(2, 2))
    # plt.at(0).show(tet_vedo, vedo_utils.draw_axes(), "Undeformed")
    plt.at(1).show(tet_def_vedo, vedo_contact_points, vedo_utils.draw_origin(), mount_axes, "Deformed (Tet)")
    plt.at(2).show(tri_vedo, vedo_utils.draw_origin(), mount_axes,
                   "Deformed (Tri)")
    plt.at(3).show(tri_vedo_alpha, vedo_contact_points, vedo_contact_forces, vedo_wrist_force, vedo_wrist_torque,
                   mount_axes, vedo_utils.draw_origin(), "Contact Points")
    plt.interactive().close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vis data (vedo).")
    parser.add_argument("data_fn", type=str)
    parser.add_argument("base_tetra_mesh", type=str)
    args = parser.parse_args()

    vis_data(args.data_fn, args.base_tetra_mesh)
