from matplotlib import pyplot as plt

import utils
import trimesh
import meshio
import numpy as np
import open3d as o3d


def get_sdf_query_points(mesh: o3d.geometry.TriangleMesh, n: int = 100000):
    min_bounds = np.array(mesh.get_min_bound())
    max_bounds = np.array(mesh.get_max_bound())
    min_bounds -= 0.03
    max_bounds += 0.03

    points = min_bounds + (np.random.random((n, 3)) * (max_bounds - min_bounds))
    return points


# tetra_file = "assets/meshes/sponge/sponge.tet"
# vert, tet = utils.load_tetmesh(tetra_file)
# tri_vert, tri = utils.tet_to_triangle(vert, tet)
# mesh = trimesh.Trimesh(vertices=tri_vert, faces=tri)
# mesh.show()

# tetra_file = "assets/meshes/sponge/sponge.mesh"
# tetra_mesh = meshio.read(tetra_file)
# tri_mesh = utils.tetrahedral_to_surface_triangles(None, tetra_mesh)
# tri_mesh.show()

tetra_file = "assets/meshes/sponge/sponge.tet"
vert, tet = utils.load_tetmesh(tetra_file)
tetra_mesh = o3d.geometry.TetraMesh(vertices=o3d.utility.Vector3dVector(vert), tetras=o3d.utility.Vector4iVector(tet))

tri_points, tri_triangles = utils.tetrahedral_to_surface_triangles(vert, tet)
tri_mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(tri_points),
                                     triangles=o3d.utility.Vector3iVector(tri_triangles))

# values = np.ones(len(vert), dtype=float)
# tri_mesh = tetra_mesh.extract_triangle_mesh(o3d.utility.DoubleVector(values), level=0.0)

# o3d.visualization.draw_geometries([o3d.geometry.LineSet.create_from_tetra_mesh(tetra_mesh)])
# o3d.visualization.draw_geometries([tetra_mesh])

# o3d.visualization.draw_geometries([tri_mesh])
# o3d.visualization.draw_geometries([o3d.geometry.LineSet.create_from_triangle_mesh(tri_mesh)])
# o3d.visualization.draw_geometries([tri_mesh.sample_points_uniformly(100000000)])

tri_mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(tri_mesh)
scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(tri_mesh_legacy)

query_points_np = get_sdf_query_points(tri_mesh)
query_points = o3d.core.Tensor(query_points_np, dtype=o3d.core.Dtype.Float32)
signed_distance = scene.compute_signed_distance(query_points)
occ = scene.compute_occupancy(query_points)

o3d.visualization.draw_geometries([o3d.geometry.LineSet.create_from_triangle_mesh(tri_mesh),
                                   o3d.geometry.PointCloud(
                                       o3d.utility.Vector3dVector(query_points_np))
                                   ])
o3d.visualization.draw_geometries([o3d.geometry.LineSet.create_from_triangle_mesh(tri_mesh),
                                   o3d.geometry.PointCloud(
                                       o3d.utility.Vector3dVector(query_points_np[np.asarray(occ) > 0.5]))
                                   ])
o3d.visualization.draw_geometries([o3d.geometry.LineSet.create_from_triangle_mesh(tri_mesh),
                                   o3d.geometry.PointCloud(
                                       o3d.utility.Vector3dVector(query_points_np[np.asarray(occ) < 0.5]))
                                   ])
