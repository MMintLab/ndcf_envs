import os

import numpy as np
import trimesh.creation


def generate_box(box_cfg: dict):
    box = trimesh.creation.box(box_cfg["extents"])
    xy_translation_lims = box_cfg["xy_translation_lims"]
    xy_translation = xy_translation_lims[0] + np.random.random([2]) * (xy_translation_lims[1] - xy_translation_lims[0])
    box.apply_translation([xy_translation[0], xy_translation[1], 0.05])
    return box


def generate_primitive_terrain(terrain_cfg: dict, vis=False):
    assets_dir = "assets"
    primitives_mesh_dir = "meshes/primitives/"
    primitives_urdf_dir = "urdf/primitives/"

    # Load base urdf.
    base_urdf_fn = os.path.join(assets_dir, "urdf/base.urdf")
    with open(base_urdf_fn, "r") as f:
        urdf_string = f.read()

    terrain_type = terrain_cfg["type"]

    if terrain_type == "box":
        mesh = generate_box(terrain_cfg)
    else:
        raise Exception("Unknown terrain type: %s" % terrain_type)

    if vis:
        mesh.show()
    mesh_path = os.path.join(primitives_mesh_dir, "terrain.obj")
    mesh.export(os.path.join(assets_dir, mesh_path))

    # Load into urdf.
    urdf_string = urdf_string % (mesh_path, mesh_path)
    urdf_path = os.path.join(primitives_urdf_dir, "terrain.urdf")
    with open(os.path.join(assets_dir, urdf_path), "w") as w:
        w.write(urdf_string)

    return os.path.splitext(urdf_path)[0], 0.1
