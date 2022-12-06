import os
import mmint_utils
import argparse
import numpy as np
import open3d as o3d
from tqdm import trange
from vedo import Plotter, Points, Arrows, Mesh
import utils
import vedo_utils


def vis_example_data(example_dict):
    all_points = example_dict["query_points"]
    sdf = example_dict["sdf"]
    in_contact = example_dict["in_contact"]
    normals = example_dict["normals"]

    plt = Plotter(shape=(2, 2))
    plt.at(0).show(Points(all_points), vedo_utils.draw_origin(), "All Sample Points")
    plt.at(1).show(Points(all_points[sdf <= 0.0], c="b"), Points(all_points[in_contact], c="r"),
                   vedo_utils.draw_origin(), "Occupied/Contact points")
    plt.at(2).show(Points(all_points[sdf == 0.0], c="b"), Points(all_points[in_contact], c="r"),
                   vedo_utils.draw_origin(), "Surface/Contact points")
    plt.at(3).show(Points(all_points[sdf == 0.0], c="b"),
                   Arrows(all_points[sdf == 0.0], all_points[sdf == 0.0] + 0.01 * normals[sdf == 0.0]))
    plt.interactive().close()


def process_sim_data_example(example_fn, base_tetra_mesh_fn, out_fn, vis=False):
    data_dict = mmint_utils.load_gzip_pickle(example_fn)

    # Get wrist pose.
    wrist_pose = data_dict["wrist_pose"]
    w_T_wrist_pose = utils.pose_to_matrix(wrist_pose, axes="rxyz")
    wrist_pose_T_w = np.linalg.inv(w_T_wrist_pose)

    # Load deformed object points.
    def_vert_w = data_dict["nodal_coords"]
    def_vert_prime = utils.transform_pointcloud(def_vert_w, wrist_pose_T_w)
    def_vert = data_dict["nodal_coords_wrist"]

    if vis:
        plt = Plotter(shape=(1, 2))
        plt.at(0).show(Points(def_vert_prime), vedo_utils.draw_origin())
        plt.at(1).show(Points(def_vert), vedo_utils.draw_origin())
        plt.interactive().close()

    # Load base tetra mesh of the undeformed mesh.
    tet_vert, tet_tetra = utils.load_tetmesh(base_tetra_mesh_fn)

    # Convert tetra mesh to triangle mesh. Note, we use the deformed vertices.
    tri_vert, tri_triangles = utils.tetrahedral_to_surface_triangles(def_vert, tet_tetra)
    tri_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(tri_vert),
                                         o3d.utility.Vector3iVector(tri_triangles))

    # Load contact point cloud.
    contact_points_w = np.array([list(ctc_pt) for ctc_pt in data_dict["contact_points"]])
    contact_points = utils.transform_pointcloud(contact_points_w, wrist_pose_T_w)

    # Load contact forces.
    contact_forces_w = np.array(data_dict["contact_forces"])
    contact_forces = utils.transform_vectors(contact_forces_w, wrist_pose_T_w)

    # Load contact normals.
    contact_normals_w = np.array(data_dict["contact_normals"])
    contact_normals = utils.transform_vectors(contact_normals_w, wrist_pose_T_w)

    # Get SDF values near object.
    query_points, sdf = utils.get_sdf_values(tri_mesh, n_random=20000, n_off_surface=20000)

    # Get samples on the surface of the object.
    contact_vertices, contact_triangles = utils.find_in_contact_triangles(tri_mesh, contact_points)

    surface_points, surface_normals, surface_contact_labels = \
        utils.sample_surface_points_with_contact(tri_mesh, contact_triangles, n=20000)

    # Some visualization for contact verts/tris.
    if vis:
        contact_points_vedo = Points(contact_points, c="r")
        tri_colors = [[255, 0, 0, 255] if c else [255, 255, 0, 255] for c in contact_triangles]
        tri_mesh_vedo_contact = Mesh([tri_vert, tri_triangles])
        tri_mesh_vedo_contact.celldata["CellIndividualColors"] = np.array(tri_colors).astype(np.uint8)
        tri_mesh_vedo_contact.celldata.select("CellIndividualColors")

        new_points_vedo = Points(surface_points, c="b")
        new_contact_points_vedo = Points(surface_points[surface_contact_labels], c="r")

        plt = Plotter(shape=(1, 2))
        plt.at(0).show(contact_points_vedo, tri_mesh_vedo_contact,
                       Arrows(contact_points, contact_points + 0.01 * contact_forces),
                       "Contact Points")
        plt.at(1).show(new_points_vedo, new_contact_points_vedo,
                       Arrows(contact_points, contact_points + 0.01 * contact_forces), "Contact Vertices")
        # plt.at(2).show(contact_points_vedo, tri_mesh_vedo_contact,
        #                Arrows(contact_points, contact_points + 0.01 * contact_normals), "Contact Normals")
        plt.interactive().close()

    # Build dataset.
    dataset_query_points = np.concatenate([query_points, contact_points, surface_points])
    dataset_sdf = np.concatenate([sdf, np.zeros(len(contact_points)), np.zeros(len(surface_points))])
    dataset_in_contact = np.concatenate([np.zeros(len(sdf), dtype=bool), np.ones(len(contact_points), dtype=bool),
                                         surface_contact_labels])
    dataset_normals = np.concatenate([np.zeros([len(query_points), 3]), contact_normals, surface_normals])

    assert len(dataset_query_points) == len(dataset_sdf) and len(dataset_query_points) == len(
        dataset_in_contact) and len(dataset_query_points) == len(dataset_normals)

    dataset_dict = {
        "n_points": len(dataset_query_points),
        "query_points": dataset_query_points,
        "sdf": dataset_sdf,
        "in_contact": dataset_in_contact,
        "normals": dataset_normals,
    }
    if out_fn is not None:
        mmint_utils.save_gzip_pickle(dataset_dict, out_fn)

    if vis:
        vis_example_data(dataset_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process sim data.")
    parser.add_argument("data_dir", type=str, help="Data dir.")
    parser.add_argument("base_tetra_mesh_fn", type=str, help="Base Tet mesh file.")
    parser.add_argument('-v', '--vis', dest='vis', action='store_true', help='Visualize.')
    parser.set_defaults(vis=False)
    args = parser.parse_args()

    # Load data fns.
    data_dir = args.data_dir
    data_fns = [f for f in os.listdir(data_dir) if "config_" in f]
    data_fns.sort(key=lambda a: int(a.replace(".pkl.gzip", "").split("_")[-1]))

    for data_idx in trange(len(data_fns)):
        data_fn = os.path.join(data_dir, data_fns[data_idx])
        out_fn_ = os.path.join(data_dir, "out_%d.pkl.gzip" % data_idx)

        process_sim_data_example(data_fn, args.base_tetra_mesh_fn, out_fn_, vis=args.vis)
