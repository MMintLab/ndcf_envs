from ncf_envs.terrain.random.terrain import DiscreteTerrainConfig as DTC
from ncf_envs.terrain.random.terrain import WaveTerrainConfig as WTC
from terrain_utils import *
import trimesh


def generate_terrain():
    N = 10
    for i in range(N):
        vertices, triangles, _ = generate_wave_terrain(WTC())
        mesh = trimesh.base.Trimesh(vertices, triangles)
        mesh.export('assets/meshes/terrain/terrain_wave_{}.obj'.format(i))

    for i in range(N):
        vertices, triangles, _ = generate_discrete_obstacles_terrain( DTC())
        mesh = trimesh.base.Trimesh(vertices, triangles)
        mesh.export('assets/meshes/terrain/terrain_discrete_{}.obj'.format(i))


if __name__ == '__main__':
    generate_terrain()
