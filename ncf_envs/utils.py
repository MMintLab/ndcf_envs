import numpy as np
import transforms3d as tf3d
import open3d as o3d
from scipy.spatial import KDTree
import trimesh
import trimesh.sample


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


def save_pointcloud(pointcloud, fn: str):
    pointcloud_pcd: o3d.geometry.PointCloud = pointcloud_to_o3d(pointcloud)
    o3d.io.write_point_cloud(fn, pointcloud_pcd)


def transform_vectors(vectors, T):
    """
    Transform vectors (i.e., just apply rotation) from given matrix transformation T.
    """
    R = T[:3, :3]
    tf_vectors = (R @ vectors.T).T
    return tf_vectors


def pose_to_matrix(pose, axes="rxyz"):
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


def matrix_to_pose(matrix, axes="rxyz"):
    """
    Matrix to pose.
    """
    pose = np.zeros(6)
    pose[:3] = matrix[:3, 3]
    pose[3:] = tf3d.euler.mat2euler(matrix, axes=axes)
    return pose


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


def get_sdf_values(tri_mesh: o3d.geometry.TriangleMesh, n_random: int = 10000, n_off_surface: int = 10000,
                   noise: float = 0.004, bound_extend: float = 0.03):
    """
    Calculate SDF points for the given triangle mesh.

    Sample n_random in random space around tool. Sample n_off_surface as points near surface.
    """
    # Build o3d scene with triangle mesh.
    tri_mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(tri_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(tri_mesh_legacy)

    # Get SDF query points around mesh surface.
    if n_random > 0:
        min_bounds = np.array(tri_mesh.get_min_bound())
        max_bounds = np.array(tri_mesh.get_max_bound())
        min_bounds -= bound_extend
        max_bounds += bound_extend
        query_points_random = min_bounds + (np.random.random((n_random, 3)) * (max_bounds - min_bounds))
    else:
        query_points_random = np.empty([0, 3], dtype=float)

    # Get SDF query points by sampling surface points and adding small amount of gaussian noise.
    if n_off_surface > 0:
        query_points_surface = tri_mesh.sample_points_uniformly(number_of_points=n_off_surface)
        query_points_surface = np.asarray(query_points_surface.points)
        query_points_surface += np.random.normal(0.0, noise, size=query_points_surface.shape)
    else:
        query_points_surface = np.empty([0, 3], dtype=float)

    # Compute SDF to surface.
    query_points_np = np.concatenate([query_points_random, query_points_surface])
    query_points = o3d.core.Tensor(query_points_np, dtype=o3d.core.Dtype.Float32)
    signed_distance = scene.compute_signed_distance(query_points)
    signed_distance_np = signed_distance.numpy()

    return query_points_np, signed_distance_np


def find_in_contact_triangles(tri_mesh: o3d.geometry.TriangleMesh, contact_points: np.ndarray):
    """
    Given the triangle mesh of the tool and the contact points (which are all vertices of the tool mesh),
    find the corresponding triangles in the mesh.
    """
    vertices = tri_mesh.vertices
    triangles = tri_mesh.triangles
    contact_vertices = np.zeros(len(vertices), dtype=bool)

    # Determine the vertices in contact.
    kd_tree = KDTree(vertices)
    _, contact_points_vert_idcs = kd_tree.query(contact_points)
    contact_vertices[contact_points_vert_idcs] = True

    # Determine if each triangle is in contact. Being in contact means ALL vertices of triangle are in contact.
    contact_triangles = np.array(
        [contact_vertices[tri[0]] and contact_vertices[tri[1]] and contact_vertices[tri[2]] for tri in triangles])

    # Find total area of contact patch.
    mesh = trimesh.Trimesh(tri_mesh.vertices, tri_mesh.triangles)
    contact_area = contact_triangles.astype(float) @ mesh.area_faces

    return contact_vertices, contact_triangles, contact_area


def sample_non_contact_surface_points(tri_mesh: o3d.geometry.TriangleMesh, contact_triangles: np.ndarray,
                                      n: int = 1000):
    """
    Sample points on the surface of the given mesh that are NOT in contact.
    """
    mesh = trimesh.Trimesh(tri_mesh.vertices, tri_mesh.triangles)
    triangle_weights = [1.0 if not c else 0.0 for c in contact_triangles]
    surface_points, _ = mesh.sample(n, return_index=True, face_weight=triangle_weights)
    return surface_points


def sample_surface_points(tri_mesh: o3d.geometry.TriangleMesh, n: int = 1000):
    mesh = trimesh.Trimesh(tri_mesh.vertices, tri_mesh.triangles)
    mesh.fix_normals()

    # Sample on the surface.
    surface_points, triangle_idcs = trimesh.sample.sample_surface(mesh, count=n)

    # Find normals from triangles of sampled points.
    surface_normals = mesh.face_normals[triangle_idcs]

    # Find normals using barycentric interpolation for smooth result.
    # bary = trimesh.triangles.points_to_barycentric(triangles=mesh.triangles[triangle_idcs], points=surface_points)
    # surface_normals = trimesh.unitize(
    #     (mesh.vertex_normals[mesh.faces[triangle_idcs]] * trimesh.unitize(bary).reshape((-1, 3, 1))).sum(axis=1))

    return surface_points, surface_normals, triangle_idcs


def sample_surface_points_with_contact(tri_mesh: o3d.geometry.TriangleMesh, contact_triangles: np.ndarray,
                                       n: int = 1000):
    surface_points, surface_normals, triangle_idcs = sample_surface_points(tri_mesh, n)

    # Determine contact labels based on whether the sampled points are on triangles labeled as in contact.
    contact_labels = np.array([contact_triangles[t_idx] for t_idx in triangle_idcs])

    return surface_points, surface_normals, contact_labels
