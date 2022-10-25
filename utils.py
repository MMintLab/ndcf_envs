import open3d as o3d
import numpy as np
import transforms3d as tf3d
import trimesh
from vedo import TetMesh, Plotter, show, Arrow, Mesh, Points, Arrows, Line


def pointcloud_to_o3d(pointcloud):
    """
    Send pointcloud to open3d.
    """
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
    """
    Transform the given pointcloud by the given matrix transformation T.
    """
    pointcloud_pcd: o3d.geometry.PointCloud = pointcloud_to_o3d(pointcloud)
    pointcloud_pcd.transform(T)
    if pointcloud.shape[1] > 3:
        return np.concatenate([np.asarray(pointcloud_pcd.points), pointcloud[:, 3:]], axis=1)
    else:
        return np.asarray(pointcloud_pcd.points)


def transform_vectors(vectors, T):
    """
    Transform vectors (i.e., just apply rotation) from given matrix transformation T.
    """
    R = T[:3, :3]
    tf_vectors = (R @ vectors.T).T
    return tf_vectors


def pose_to_matrix(pose, axes="sxyz"):
    """
    Pose to matrix.
    """
    matrix = np.eye(4, dtype=pose.dtype)
    matrix[:3, 3] = pose[:3]

    if len(pose) == 6:
        matrix[:3, :3] = tf3d.euler.euler2mat(pose[3], pose[4], pose[5], axes=axes)
    else:
        matrix[:3, :3] = tf3d.quaternions.quat2mat(pose[3:])

    return matrix


def tetrahedral_to_surface_triangles(verts, tetras):
    """
    Get surface triangles from tetrahedral mesh.

    Surface triangles appear only once amongst all the tetrahedra.
    """
    surface = set()
    for tetra in tetras:
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

    return verts, np.array(list(surface))


def load_tetmesh(meshfn):
    """
    Load a tet (tetrahedral) mesh.

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


def get_sdf_values(tri_mesh: o3d.geometry.TriangleMesh, n: int = 10000):
    """
    Calculate SDF points for the given triangle mesh.
    """
    # Build o3d scene with triangle mesh.
    tri_mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(tri_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(tri_mesh_legacy)

    # Get SDF query points around mesh surface.
    min_bounds = np.array(tri_mesh.get_min_bound())
    max_bounds = np.array(tri_mesh.get_max_bound())
    min_bounds -= 0.03
    max_bounds += 0.03
    query_points_np = min_bounds + (np.random.random((n, 3)) * (max_bounds - min_bounds))

    # Compute SDF to surface.
    query_points = o3d.core.Tensor(query_points_np, dtype=o3d.core.Dtype.Float32)
    signed_distance = scene.compute_signed_distance(query_points)
    signed_distance_np = np.asarray(signed_distance)

    return query_points_np, signed_distance_np


def draw_axes(scale=0.02):
    axes = [
        Arrow(end_pt=[scale, 0, 0], c="r"),
        Arrow(end_pt=[0, scale, 0], c="g"),
        Arrow(end_pt=[0, 0, scale], c="b"),
    ]
    return axes
