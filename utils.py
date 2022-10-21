import open3d as o3d
import numpy as np
import transforms3d as tf3d
import trimesh
from vedo import TetMesh, Plotter, show, Arrow, Mesh, Points, Arrows


def pointcloud_to_o3d(pointcloud):
    pointcloud_xyz = pointcloud[:, :3]
    if pointcloud.shape[1] == 4:
        color = np.zeros([pointcloud.shape[0], 3], dtype=float)
        color[:, 0] = pointcloud[:, 3]
        color[:, 1] = pointcloud[:, 3]
        color[:, 2] = pointcloud[:, 3]
    elif pointcloud.shape[1] > 4:
        color = pointcloud[:, 3:]
    else:
        color = None

    points = o3d.geometry.PointCloud()
    points.points = o3d.utility.Vector3dVector(pointcloud_xyz)
    if color is not None:
        points.colors = o3d.utility.Vector3dVector(color)
    return points


def transform_pointcloud(pointcloud, T):
    pointcloud_pcd: o3d.geometry.PointCloud = pointcloud_to_o3d(pointcloud)
    pointcloud_pcd.transform(T)
    if pointcloud.shape[1] > 3:
        return np.concatenate([np.asarray(pointcloud_pcd.points), pointcloud[:, 3:]], axis=1)
    else:
        return np.asarray(pointcloud_pcd.points)


def pose_to_matrix(pose, axes="sxyz"):
    matrix = np.eye(4, dtype=pose.dtype)
    matrix[:3, 3] = pose[:3]

    if len(pose) == 6:
        matrix[:3, :3] = tf3d.euler.euler2mat(pose[3], pose[4], pose[5], axes=axes)
    else:
        matrix[:3, :3] = tf3d.quaternions.quat2mat(pose[3:])

    return matrix


def tetrahedral_to_surface_triangles(points, tetramesh):
    # surface_points_z = np.logical_or(tetramesh.points[:, 2] == 0, tetramesh.points[:, 2] == -0.1)
    # surface_points_x = np.logical_or(tetramesh.points[:, 0] == -0.05, tetramesh.points[:, 0] == 0.05)
    # surface_points_y = np.logical_or(tetramesh.points[:, 1] == -0.05, tetramesh.points[:, 1] == 0.05)
    # surface_points = np.logical_or(np.logical_or(surface_points_x, surface_points_y), surface_points_z)
    #
    # vedo_surface_points = Points(tetramesh.points[surface_points])
    #
    # surface_triangles = []
    # for tetra in tetramesh.cells_dict["tetra"]:
    #     for face in [
    #         [tetra[0], tetra[1], tetra[2]],
    #         [tetra[0], tetra[2], tetra[3]],
    #         [tetra[0], tetra[1], tetra[3]],
    #         [tetra[1], tetra[2], tetra[3]]
    #     ]:
    #         if (surface_points[face[0]] and surface_points[face[1]]) and surface_points[face[2]]:
    #             surface_triangles.append(face)
    #
    # # plt = Plotter(shape=(1, 1))
    # # plt.at(0).show(vedo_surface_points)
    # # plt.interactive().close()

    surface = set()
    for tetra in tetramesh.cells_dict["tetra"]:
        for face in [
            [tetra[0], tetra[1], tetra[2]],
            [tetra[0], tetra[2], tetra[3]],
            [tetra[0], tetra[1], tetra[3]],
            [tetra[1], tetra[2], tetra[3]]
        ]:
            # Sort face.
            sorted_face = tuple(np.sort(face))
            if sorted_face in surface:
                surface.remove(sorted_face)
            else:
                surface.add(sorted_face)

    triangle_mesh = trimesh.Trimesh(points, list(surface))
    return triangle_mesh


def tet_to_triangle(vertices, tet):
    surface = set()
    all_triangles = []
    for tetra in tet:
        for face in [
            [tetra[0], tetra[1], tetra[2]],
            [tetra[0], tetra[2], tetra[3]],
            [tetra[0], tetra[1], tetra[3]],
            [tetra[1], tetra[2], tetra[3]]
        ]:
            # Sort face.
            sorted_face = tuple(np.sort(face))
            all_triangles.append(face)
            if sorted_face in surface:
                surface.remove(sorted_face)
            else:
                surface.add(sorted_face)

    return vertices, list(surface)  # all_triangles


def load_tetmesh(meshfn):
    """
    Based on https://github.com/NVlabs/DefGraspSim/blob/main/mesh_to_tet.py
    """
    mesh_file = open(meshfn, "r")

    mesh_lines = list(mesh_file)
    mesh_lines = [line.strip('\n') for line in mesh_lines]

    vertices_start = 3
    num_vertices = int(mesh_lines[2].split()[1])

    vertices = mesh_lines[vertices_start:vertices_start + num_vertices]

    tetrahedra_start = vertices_start + num_vertices + 2
    num_tetrahedra = int(mesh_lines[tetrahedra_start - 1].split()[1])
    tetrahedra = mesh_lines[tetrahedra_start:tetrahedra_start + num_tetrahedra]

    vertices = np.array([
        [float(v) for v in vs.split(" ")[1:]] for vs in vertices
    ])[:, :3]
    tetrahedra = np.array([
        [int(t) for t in ts.split(" ")[1:]] for ts in tetrahedra
    ])[:, :4]

    return vertices, tetrahedra
