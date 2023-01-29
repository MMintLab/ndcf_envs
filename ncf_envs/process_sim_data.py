import os

import mmint_utils
import argparse
import numpy as np
import open3d as o3d
import trimesh
from tqdm import trange
from vedo import Plotter, Points, Arrows, Mesh, Point, Line, Arrow
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


def vis_occupancy_data(points_iou, occ_tgt):
    vedo_plt = Plotter(shape=(1, 2))
    vedo_plt.at(0).show(Points(points_iou), vedo_utils.draw_origin(), "All Sample Points")
    vedo_plt.at(1).show(Points(points_iou[occ_tgt]), vedo_utils.draw_origin(), "Occupied Points")
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


def vis_partial_pc(gt_mesh, partial_pc_data, combined_pc):
    vedo_plt = Plotter(shape=(1, 1 + len(partial_pc_data)))
    vedo_mesh = Mesh([gt_mesh.vertices, gt_mesh.triangles])
    poses_vis = []
    for idx in range(len(partial_pc_data)):
        pc_data = partial_pc_data[idx]
        pc_vedo = Points(pc_data["pointcloud"])
        pose_vis = vedo_utils.draw_pose(pc_data["camera_pose"])
        poses_vis.append(pose_vis)
        vedo_plt.at(idx + 1).show(vedo_utils.draw_origin(), vedo_mesh, pc_vedo, pose_vis)
    vedo_plt.at(0).show(vedo_utils.draw_origin(), vedo_mesh, Points(combined_pc), *poses_vis)
    vedo_plt.interactive().close()


def deproject_depth_image(depth, projection_matrix, view_matrix, tool_segmentation, env_origin):
    """
    Based somewhat on https://gist.github.com/gavrielstate/8c855eb3b4b1f23e2990bc02c534792e
    """
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


def process_sim_data_example(example_fn, base_tetra_mesh_fn, out_dir, example_name, terrain_file: str = None,
                             vis=False):
    data_dict = mmint_utils.load_gzip_pickle(example_fn)

    # Get wrist pose.
    wrist_pose = data_dict["wrist_pose"]
    w_T_wrist_pose = utils.pose_to_matrix(wrist_pose, axes="rxyz")
    wrist_pose_T_w = np.linalg.inv(w_T_wrist_pose)

    # Load deformed object points.
    def_vert_w = data_dict["nodal_coords"]
    def_vert_prime = utils.transform_pointcloud(def_vert_w, wrist_pose_T_w)
    def_vert = data_dict["nodal_coords_wrist"]

    if vis and False:
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

    # Load terrain file.
    terrain_mesh: trimesh.Trimesh = trimesh.load(terrain_file)
    terrain_mesh.apply_transform(wrist_pose_T_w)

    # Load contact point cloud.
    contact_points_w = np.array([list(ctc_pt) for ctc_pt in data_dict["contact_points"]])
    contact_points = utils.transform_pointcloud(contact_points_w, wrist_pose_T_w)

    # Load all contact information.
    all_contacts = data_dict["all_contact"]

    if vis and False:
        for contact in all_contacts:
            contact_point = contact["bodyOffset"]
            normal = contact["normal"]
            normal = utils.transform_vectors(np.array([list(normal)]), wrist_pose_T_w)[0]
            contact_point = utils.transform_pointcloud(np.array([list(contact_point)]), wrist_pose_T_w)[0]
            particle_indices = np.array(list(contact["particleIndices"]))
            particle_barys = np.array(list(contact["particleBarys"]))
            contact_point_surface = (def_vert[particle_indices[0]] * particle_barys[0]) + (
                    def_vert[particle_indices[1]] * particle_barys[1]) + (
                                            def_vert[particle_indices[2]] * particle_barys[2])

            point_diff = contact_point - contact_point_surface
            point_diff /= np.linalg.norm(point_diff)

            plt = Plotter(shape=(1, 2))
            plt.at(0).show(  # Points(def_vert, c="black"),
                Mesh([tri_vert, tri_triangles]),
                Point(contact_point, c="red"),
                Point(contact_point_surface, c="purple"),
                Line(def_vert[particle_indices], c="blue", closed=True),
                # Arrow(start_pt=contact_point, end_pt=contact_point + 0.01 * normal, c="red"),
                # Arrow(start_pt=contact_point, end_pt=contact_point + 0.011 * point_diff, c="orange")
            )
            plt.at(1).show(  # Points(terrain_mesh.vertices, c="black"),
                Point(contact_point, c="red"),
                Mesh([terrain_mesh.vertices, terrain_mesh.faces]),
                # Arrow(start_pt=contact_point, end_pt=contact_point + 0.01 * normal, c="red"),
                # Arrow(start_pt=contact_point, end_pt=contact_point + 0.011 * point_diff, c="orange")
            )
            plt.interactive().close()

    # Load contact forces.
    contact_forces_w = np.array(data_dict["contact_forces"])
    contact_forces = utils.transform_vectors(contact_forces_w, wrist_pose_T_w)

    # Load contact normals.
    contact_normals_w = np.array(data_dict["contact_normals"])
    contact_normals = utils.transform_vectors(contact_normals_w, wrist_pose_T_w)

    # Get SDF values near object.
    query_points, sdf = utils.get_sdf_values(tri_mesh, n_random=20000, n_off_surface=20000)

    # Get samples on the surface of the object.
    # contact_vertices, contact_triangles, contact_area = utils.find_in_contact_triangles(tri_mesh, contact_points)
    contact_vertices, contact_triangles, contact_area = utils.find_in_contact_triangles_indices(
        tri_mesh, all_contacts["particleIndices"], def_vert
    )
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
    if "cameras" in data_dict:
        camera_output = data_dict["cameras"]
        env_origin = data_dict["env_origin"]
        partial_pc_data = []
        combined_pointcloud = []
        for camera_out in camera_output:
            rgb = camera_out["rgb"]
            depth = camera_out["depth"]
            segmentation = camera_out["segmentation"]
            tool_segmentation = np.logical_and(segmentation == 0, np.logical_not(np.isinf(depth)))
            vis_images(rgb, depth, tool_segmentation)

            # Deproject pointcloud to wrist frame.
            projection_matrix = camera_out["camera_proj_matrix"]
            camera_view_matrix = camera_out["camera_view_matrix"]
            pointcloud = deproject_depth_image(depth, projection_matrix, camera_view_matrix, tool_segmentation,
                                               env_origin)

            # Get camera pose w.r.t. wrist.
            c_T_w = camera_view_matrix.T
            w_T_c = np.linalg.inv(c_T_w)
            wrist_pose_T_c = wrist_pose_T_w @ w_T_c
            cam_wrist_pose = utils.matrix_to_pose(wrist_pose_T_c)

            pointcloud = utils.transform_pointcloud(pointcloud, wrist_pose_T_w)

            combined_pointcloud.append(pointcloud)
            partial_pc_data.append({
                "pointcloud": pointcloud,
                "camera_pose": cam_wrist_pose,
            })
        combined_pointcloud = np.concatenate(combined_pointcloud, axis=0)
        if vis:
            vis_partial_pc(tri_mesh, partial_pc_data, combined_pointcloud)
    else:
        partial_pc_data = []
        combined_pointcloud = np.empty([0, 3])

    # Generate ground truth occupancy values.
    points_iou, sdf_iou = utils.get_sdf_values(tri_mesh, 100000, n_off_surface=0, bound_extend=0.01)
    occ_tgt = sdf_iou <= 0.0
    if vis:
        vis_occupancy_data(points_iou, occ_tgt)

    # Build dataset.
    dataset_query_points = np.concatenate([query_points, surface_points])
    dataset_sdf = np.concatenate([sdf, np.zeros(len(surface_points))])
    dataset_in_contact = np.concatenate([np.zeros(len(sdf), dtype=bool), surface_contact_labels])
    dataset_normals = np.concatenate([np.zeros([len(query_points), 3]), surface_normals])

    assert len(dataset_query_points) == len(dataset_sdf) and len(dataset_query_points) == len(
        dataset_in_contact) and len(dataset_query_points) == len(dataset_normals)

    dataset_dict = {
        "train": {
            "n_points": len(dataset_query_points),
            "query_points": dataset_query_points,
            "sdf": dataset_sdf,
            "in_contact": dataset_in_contact,
            "normals": dataset_normals,
            "pressure": pressure,
        },
        "test": {
            "surface_points": surface_points,
            "surface_in_contact": surface_contact_labels,
            "points_iou": points_iou,
            "occ_tgt": occ_tgt,
        },
        "input": {
            "pointclouds": partial_pc_data,
            "combined_pointcloud": combined_pointcloud,
            "wrist_wrench": data_dict["wrist_wrench"],
        },
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
    parser.add_argument("-t", "--terrain", type=str, default=None, help="Terrain file used in interaction (if any).")
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

        process_sim_data_example(data_fn, args.base_tetra_mesh_fn, data_dir, example_name_, terrain_file=args.terrain,
                                 vis=args.vis)
