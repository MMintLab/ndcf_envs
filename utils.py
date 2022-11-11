import open3d as o3d
import numpy as np
import transforms3d as tf3d
from vedo import Arrow
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


def get_sdf_values(tri_mesh: o3d.geometry.TriangleMesh, n_random: int = 10000, n_off_surface: int = 10000,
                   noise: float = 0.004):
    """
    Calculate SDF points for the given triangle mesh.

    Sample n_random in random space around tool. Sample n_off_surface as points near surface.
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
    query_points_random = min_bounds + (np.random.random((n_random, 3)) * (max_bounds - min_bounds))

    # Get SDF query points by sampling surface points and adding small amount of gaussian noise.
    query_points_surface = tri_mesh.sample_points_uniformly(number_of_points=n_off_surface)
    query_points_surface = np.asarray(query_points_surface.points)
    query_points_surface += np.random.normal(0.0, noise, size=query_points_surface.shape)

    # Compute SDF to surface.
    query_points_np = np.concatenate([query_points_random, query_points_surface])
    query_points = o3d.core.Tensor(query_points_np, dtype=o3d.core.Dtype.Float32)
    signed_distance = scene.compute_signed_distance(query_points)
    signed_distance_np = signed_distance.numpy()

    return query_points_np, signed_distance_np


def draw_axes(scale=0.02):
    """
    Helper to draw axes in vedo.
    """
    axes = [
        Arrow(end_pt=[scale, 0, 0], c="r"),
        Arrow(end_pt=[0, scale, 0], c="g"),
        Arrow(end_pt=[0, 0, scale], c="b"),
    ]
    return axes


def find_in_contact_triangles(tri_mesh: o3d.geometry.TriangleMesh, contact_points: np.ndarray,
                              contact_forces: np.ndarray):
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

    # Get force at each vertex of mesh.
    vertices_forces = np.zeros(shape=[len(vertices), 3])
    for contact_pt_idx, (contact_pt, contact_force) in enumerate(zip(contact_points, contact_forces)):
        vertices_forces[contact_points_vert_idcs[contact_pt_idx]] = contact_force

    # Determine if each triangle is in contact. Being in contact means ALL vertices of triangle are in contact.
    contact_triangles = np.array(
        [contact_vertices[tri[0]] and contact_vertices[tri[1]] and contact_vertices[tri[2]] for tri in triangles])

    # Determine average contact forces for each triangle.
    contact_triangle_forces = np.zeros(shape=[contact_triangles.shape[0], 3])
    for t_idx, (tri, in_contact) in enumerate(zip(triangles, contact_triangles)):
        if in_contact:
            contact_triangle_forces[t_idx] = (vertices_forces[tri[0]] + vertices_forces[tri[1]] + vertices_forces[
                tri[2]]) / 3.0

    return contact_vertices, contact_triangles, contact_triangle_forces


def sample_non_contact_surface_points(tri_mesh: o3d.geometry.TriangleMesh, contact_triangles: np.ndarray,
                                      n: int = 1000):
    """
    Sample points on the surface of the given mesh that are NOT in contact.
    """
    mesh = trimesh.Trimesh(tri_mesh.vertices, tri_mesh.triangles)
    triangle_weights = [1.0 if not c else 0.0 for c in contact_triangles]
    surface_points, _ = mesh.sample(n, return_index=True, face_weight=triangle_weights)
    return surface_points


def sample_surface_points(tri_mesh: o3d.geometry.TriangleMesh, contact_triangles: np.ndarray,
                          contact_triangle_forces: np.ndarray, n: int = 1000):
    mesh = trimesh.Trimesh(tri_mesh.vertices, tri_mesh.triangles)

    # Sample on the surface.
    surface_points, triangle_idcs = trimesh.sample.sample_surface(mesh, count=n)

    # Determine contact labels based on whether the sampled points are on triangles labeled as in contact.
    contact_labels = np.array([contact_triangles[t_idx] for t_idx in triangle_idcs])

    # Determine contact forces. For now, use average of forces at vertices of sampled triangle.
    contact_forces = np.array([contact_triangle_forces[t_idx] for t_idx in triangle_idcs])

    return surface_points, contact_labels, contact_forces
