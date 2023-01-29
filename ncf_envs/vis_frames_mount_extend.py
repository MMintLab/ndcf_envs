import numpy as np
import trimesh
from vedo import Plotter, Mesh
import vedo_utils

if __name__ == '__main__':
    mount_extend_stl = "/home/markvdm/Documents/IsaacGym/ncf_envs/assets/meshes/wrist/mount_extend/mount_extend_v2.stl"
    mesh: trimesh.Trimesh = trimesh.load(mount_extend_stl)

    vedo_mesh = Mesh([mesh.vertices, mesh.faces])

    mount_extend_pose = np.array([0.01333, 0.0, 0.05219, 0.0, np.pi/4.0, 0.0])

    plt = Plotter()
    plt.at(0).show(vedo_mesh, vedo_utils.draw_origin(), vedo_utils.draw_pose(mount_extend_pose))
    plt.interactive().show()
