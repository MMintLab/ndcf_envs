import os.path
from press_simulator import *
import argparse
import mmint_utils
from isaacgym.terrain_utils import *
from cfg.terrain import DiscreteTerrainConfig as DTC
from cfg.terrain import WaveTerrainConfig as WTC
from terrain_utils import *
import trimesh


def sample_sim_presses():
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
    sample_sim_presses()
