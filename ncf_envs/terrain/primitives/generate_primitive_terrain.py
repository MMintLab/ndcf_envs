import os
import matplotlib.pyplot as plt
import numpy as np
import trimesh.creation
import shapely
from shapely.plotting import plot_polygon


def generate_box(box_cfg: dict, tool_width: float):
    height = box_cfg["height"]
    box = trimesh.creation.box([2 * tool_width, 2 * tool_width, height])

    # Half the time, lower straight into block. Other half, lower near edge.
    if np.random.random() < 0.5:
        xy_translation = (0.5 * tool_width) + (np.random.random([2]) * tool_width)
    else:
        xy_translation = np.zeros([2])
    box.apply_translation([xy_translation[0], xy_translation[1], height / 2.0])
    return box, height


def generate_cylinder(cylinder_cfg: dict, tool_width: float):
    radius = 0.02 + (np.random.random() * (0.2 - 0.02))
    height = tool_width
    cylinder = trimesh.creation.cylinder(radius, segment=[[-height, 0.0, 0.0], [height, 0.0, 0.0]])

    # Half the time, lower straight into cylinder. Other half, lower near edge.
    if np.random.random() < 0.5:
        x_translation = y_translation = 0.0
    else:
        x_translation = (0.5 * tool_width) + (np.random.random() * tool_width)
        y_translation = 0.0 + (np.random.random() * radius)
    cylinder.apply_translation([x_translation, y_translation, radius])

    return cylinder, 2 * radius


def generate_ridge(ridge_cfg: dict, tool_width: float):
    length = 2 * tool_width
    ridge_width = 0.01
    num_points = 100
    height = 0.03
    amplitude = 0.03

    x = np.arange(-tool_width, tool_width + 1e-6, length / num_points)
    if np.random.random() < 0.5:
        y = [0.0] * num_points
    else:
        coords = (x + tool_width) / length
        y = np.sin(coords * np.pi) * amplitude

    polygon_points = []
    for idx in range(num_points):
        polygon_points.append([x[idx], y[idx] + (ridge_width / 2.0)])
    for idx in range(num_points):
        polygon_points.append([x[num_points - 1 - idx], y[num_points - 1 - idx] - (ridge_width / 2.0)])

    ridge_polygon = shapely.Polygon(polygon_points)

    ridge_mesh = trimesh.creation.extrude_polygon(ridge_polygon, height)
    return ridge_mesh, height


def generate_primitive_terrain(terrain_cfg: dict, vis=False):
    assets_dir = "assets"
    primitives_mesh_dir = "meshes/primitives/"
    primitives_urdf_dir = "urdf/primitives/"
    tool_width = 0.046

    # Load base urdf.
    base_urdf_fn = os.path.join(assets_dir, "urdf/base.urdf")
    with open(base_urdf_fn, "r") as f:
        urdf_string = f.read()

    terrain_type = terrain_cfg["type"]

    if terrain_type == "box":
        mesh, offset = generate_box(terrain_cfg, tool_width)
    elif terrain_type == "cylinder":
        mesh, offset = generate_cylinder(terrain_cfg, tool_width)
    elif terrain_type == "ridge":
        mesh, offset = generate_ridge(terrain_cfg, tool_width)
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

    return os.path.splitext(urdf_path)[0], offset
