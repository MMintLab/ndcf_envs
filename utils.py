import open3d as o3d
import numpy as np
import transforms3d as tf3d
import trimesh
from vedo import TetMesh, Plotter, show, Arrow, Mesh, Points, Arrows, Line


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


def tetrahedral_to_surface_triangles(verts, tetras):
    surface = set()
    all_triangles = []
    for tetra in tetras:
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

    points = []
    lines = []
    for tri in list(surface):
        points.extend([verts[tri[0]], verts[tri[1]], verts[tri[2]]])
        lines.extend([
            Line(verts[tri[0]], verts[tri[1]]),
            Line(verts[tri[0]], verts[tri[2]]),
            Line(verts[tri[2]], verts[tri[1]]),
        ])
    vedo_points = Points(points)

    # plt = Plotter(shape=(1, 1))
    # plt.at(0).show(vedo_points, lines)
    # plt.interactive().close()

    # Pull out only the surface points.
    tri_mesh_points = []
    point_idx_map = {}
    tri_mesh_triangles = []
    for tri in list(surface):
        for pt_idx in tri:
            if pt_idx not in point_idx_map:
                point = verts[pt_idx]
                new_pt_idx = len(tri_mesh_points)
                tri_mesh_points.append(point)
                point_idx_map[pt_idx] = new_pt_idx
        tri_mesh_triangles.append(
            [point_idx_map[tri[0]], point_idx_map[tri[1]], point_idx_map[tri[2]]]
        )

    # vedo_mesh = Mesh([tri_mesh_points, tri_mesh_triangles])
    # plt = Plotter(shape=(1, 1))
    # plt.at(0).show(vedo_mesh)
    # plt.interactive().close()

    return tri_mesh_points, tri_mesh_triangles


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
