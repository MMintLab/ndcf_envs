import os

import mmint_utils
import argparse
import numpy as np
import open3d as o3d
from tqdm import trange
from vedo import Plotter, Points, Arrows, Mesh
import utils
import vedo_utils
import matplotlib.pyplot as plt


def vis_example_data(example_dict):
    all_points = example_dict["train"]["query_points"]
    sdf = example_dict["train"]["sdf"]
    in_contact = example_dict["train"]["in_contact"]
    normals = example_dict["train"]["normals"]

    vedo_plt = Plotter(shape=(2, 2))
    vedo_plt.at(0).show(Points(all_points), vedo_utils.draw_origin(), "All Sample Points")
    vedo_plt.at(1).show(Points(all_points[sdf <= 0.0], c="b"), Points(all_points[in_contact], c="r"),
                        vedo_utils.draw_origin(), "Occupied/Contact points")
    vedo_plt.at(2).show(Points(all_points[sdf == 0.0], c="b"), Points(all_points[in_contact], c="r"),
                        vedo_utils.draw_origin(), "Surface/Contact points")
    vedo_plt.at(3).show(Points(all_points[sdf == 0.0], c="b"),
                        Arrows(all_points[sdf == 0.0], all_points[sdf == 0.0] + 0.01 * normals[sdf == 0.0]))
    vedo_plt.interactive().close()


def vis_images(rgb, depth, segmentation):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.axis("off")
    ax1.imshow(rgb)
    ax2.axis("off")
    ax2.imshow(depth)
    ax3.axis("off")
    ax3.imshow(segmentation)
    plt.show()


def vis_partial_pc(gt_mesh, partial_pc_data):
    vedo_plt = Plotter(shape=(1, len(partial_pc_data)))
    vedo_mesh = Mesh([gt_mesh.vertices, gt_mesh.triangles])
    for idx in range(len(partial_pc_data)):
        pc_data = partial_pc_data[idx]
        pc_vedo = Points(pc_data["pointcloud"])
        vedo_plt.at(idx).show(vedo_utils.draw_origin(), vedo_mesh, pc_vedo)
    vedo_plt.interactive().close()


def deproject_depth_image(depth, projection_matrix, view_matrix, tool_segmentation, env_origin):
    view_matrix_centered = view_matrix.T
    vinv = np.linalg.inv(view_matrix_centered)
    vinv[:3, 3] -= env_origin

    fu = 2.0 / projection_matrix[0, 0]
    fv = 2.0 / projection_matrix[1, 1]

    width = depth.shape[1]
    centerU = width / 2
    height = depth.shape[0]
    centerV = height / 2

    points = []
    for i in range(width):
        for j in range(height):
            if tool_segmentation[j, i]:
                u = -(i - centerU) / width
                v = (j - centerV) / height
                d = depth[j, i]
                x2 = [d * fu * u, d * fv * v, d, 1]
                p2 = vinv @ x2
                points.append(p2[:3])

    return np.array(points)


def process_sim_data_example(example_fn, base_tetra_mesh_fn, out_dir, example_name, vis=False):
    data_dict = mmint_utils.load_gzip_pickle(example_fn)
    env_origin = data_dict["env_origin"]

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
    contact_vertices, contact_triangles, contact_area = utils.find_in_contact_triangles(tri_mesh, contact_points)
    surface_points, surface_normals, surface_contact_labels = \
        utils.sample_surface_points_with_contact(tri_mesh, contact_triangles, n=20000)

    # Calculate pressure (approx) for interaction.
    wrist_f = data_dict["wrist_wrench"][:3]
    wrist_f_norm = np.linalg.norm(wrist_f)
    pressure = wrist_f_norm / contact_area

    # Some visualization for contact verts/tris.
    if vis:
        contact_points_vedo = Points(contact_points, c="r")
        tri_colors = [[255, 0, 0, 255] if c else [255, 255, 0, 255] for c in contact_triangles]
        tri_mesh_vedo_contact = Mesh([tri_vert, tri_triangles])
        tri_mesh_vedo_contact.celldata["CellIndividualColors"] = np.array(tri_colors).astype(np.uint8)
        tri_mesh_vedo_contact.celldata.select("CellIndividualColors")

        contact_normals_vedo = Arrows(contact_points, contact_points + 0.01 * contact_normals)

        new_points_vedo = Points(surface_points, c="b")
        new_contact_points_vedo = Points(surface_points[surface_contact_labels], c="r")

        plt = Plotter(shape=(1, 3))
        plt.at(0).show(contact_points_vedo, tri_mesh_vedo_contact,
                       Arrows(contact_points, contact_points + 0.01 * contact_forces),
                       "Contact Points")
        plt.at(1).show(new_points_vedo, new_contact_points_vedo,
                       Arrows(contact_points, contact_points + 0.01 * contact_forces), "Contact Vertices")
        plt.at(2).show(contact_points_vedo, tri_mesh_vedo_contact,
                       contact_normals_vedo, "Contact Normals")
        plt.interactive().close()

    # Load and process partial views from cameras.
    camera_output = data_dict["cameras"]
    partial_pc_data = []
    for camera_out in camera_output:
        rgb = camera_out["rgb"]
        depth = camera_out["depth"]
        segmentation = camera_out["segmentation"]
        tool_segmentation = np.logical_and(segmentation == 0, np.logical_not(np.isinf(depth)))
        # vis_images(rgb, depth, tool_segmentation)

        # Deproject pointcloud to wrist frame.
        projection_matrix = camera_out["camera_proj_matrix"]
        camera_view_matrix = camera_out["camera_view_matrix"]
        pointcloud = deproject_depth_image(depth, projection_matrix, camera_view_matrix, tool_segmentation, env_origin)

        # plt = Plotter(shape=(1, 1))
        # plt.at(0).show(Points(pointcloud), vedo_utils.draw_origin())
        # plt.interactive().close()

        pointcloud = utils.transform_pointcloud(pointcloud, wrist_pose_T_w)

        partial_pc_data.append({
            "pointcloud": pointcloud,
        })
    # print(partial_pc_data)
    vis_partial_pc(tri_mesh, partial_pc_data)

    # Build dataset.
    dataset_query_points = np.concatenate([query_points, contact_points, surface_points])
    dataset_sdf = np.concatenate([sdf, np.zeros(len(contact_points)), np.zeros(len(surface_points))])
    dataset_in_contact = np.concatenate([np.zeros(len(sdf), dtype=bool), np.ones(len(contact_points), dtype=bool),
                                         surface_contact_labels])
    dataset_normals = np.concatenate([np.zeros([len(query_points), 3]), contact_normals, surface_normals])

    assert len(dataset_query_points) == len(dataset_sdf) and len(dataset_query_points) == len(
        dataset_in_contact) and len(dataset_query_points) == len(dataset_normals)

    dataset_dict = {
        "train": {
            "n_points": len(dataset_query_points),
            "query_points": dataset_query_points,
            "sdf": dataset_sdf,
            "in_contact": dataset_in_contact,
            "normals": dataset_normals,
            "wrist_wrench": data_dict["wrist_wrench"],
            "pressure": pressure,
        },
        "test": {
            "surface_points": surface_points,
            "surface_in_contact": surface_contact_labels,
        }
    }
    if out_dir is not None:
        mmint_utils.save_gzip_pickle(dataset_dict, os.path.join(out_dir, example_name + ".pkl.gzip"))
        o3d.io.write_triangle_mesh(os.path.join(out_dir, example_name + "_mesh.obj"), tri_mesh)

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
        example_name_ = "out_%d" % data_idx

        process_sim_data_example(data_fn, args.base_tetra_mesh_fn, data_dir, example_name_, vis=args.vis)
