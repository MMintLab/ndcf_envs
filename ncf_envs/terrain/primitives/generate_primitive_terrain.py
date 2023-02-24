import os
import trimesh.creation


def generate_primitive_terrain(vis=False):
    assets_dir = "assets"
    primitives_mesh_dir = "meshes/primitives/"
    primitives_urdf_dir = "urdf/primitives/"

    # Load base urdf.
    base_urdf_fn = os.path.join(assets_dir, "urdf/base.urdf")
    with open(base_urdf_fn, "r") as f:
        urdf_string = f.read()

    # Generate object.
    extents = [0.2, 0.2, 0.03]  # TODO: Cfg
    box = trimesh.creation.box(extents)
    if vis:
        box.show()
    mesh_path = os.path.join(primitives_mesh_dir, "box.obj")
    box.export(os.path.join(assets_dir, mesh_path))

    # Load into urdf.
    urdf_string = urdf_string % (mesh_path, mesh_path)
    urdf_path = os.path.join(primitives_urdf_dir, "box.urdf")
    with open(os.path.join(assets_dir, urdf_path), "w") as w:
        w.write(urdf_string)

    return os.path.splitext(urdf_path)[0]
