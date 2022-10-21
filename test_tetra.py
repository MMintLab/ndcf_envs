import utils
import trimesh

tetra_file = "assets/meshes/sponge/sponge.tet"

vert, tet = utils.load_tetmesh(tetra_file)

tri_vert, tri = utils.tet_to_triangle(vert, tet)

mesh = trimesh.Trimesh(vertices=tri_vert, faces=tri)
mesh.show()
