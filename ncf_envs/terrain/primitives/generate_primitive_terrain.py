import copy
import os
import numpy as np
import transforms3d.euler
import trimesh.creation
import shapely


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


def generate_curved_surface(cylinder_cfg: dict, tool_width: float):
    concave = np.random.random() < 0.5
    num_points = 100
    y = np.random.random() * 0.1
    l_ = tool_width + 0.01 + np.random.random() * 0.02 if concave else 0.02 + np.random.random() * 0.03
    r = np.sqrt(l_ ** 2 + y ** 2)
    theta_0 = np.arctan2(y, l_)

    polygon_points = np.array([
        [r * np.cos(theta), r * np.sin(theta) - y] for theta in
        np.linspace(theta_0, np.pi - theta_0, num=num_points)
    ])
    if concave:
        polygon_points[:, 1] *= -1.0
        polygon_points[:, 1] += abs(min(polygon_points[:, 1])) + 0.01
    else:
        polygon_points[:, 1] += 0.01
    polygon_points = list(polygon_points)
    polygon_points.extend([
        [-l_, 0], [l_, 0], [l_, 0.01]
    ])
    polygon = shapely.Polygon(polygon_points)

    curve_mesh = trimesh.creation.extrude_polygon(polygon, 2 * tool_width)

    transform_ = np.eye(4)
    transform_[:3, :3] = transforms3d.euler.euler2mat(np.pi / 2.0, 0.0, 0.0)
    transform_[1, 3] = tool_width

    # Half the time, lower straight into block. Other half, lower near edge.
    if not concave and np.random.random() < 0.5:
        xy_translation = np.array([0.0, (0.5 * tool_width) + (np.random.random() * tool_width)])
    else:
        xy_translation = np.zeros([2])
    transform_[:2, 3] += xy_translation

    curve_mesh.apply_transform(transform_)

    # Compute offset from the generated mesh.
    offset = compute_offset_from_mesh(curve_mesh, tool_width)

    return curve_mesh, offset


def generate_ridge(ridge_cfg: dict, tool_width: float):
    # TODO: Update to using semi-circle alg?
    length = 2 * tool_width
    ridge_width = 0.005 + (np.random.random() * 0.005)
    num_points = 100
    height = 0.03

    x = np.arange(-tool_width, tool_width + 1e-6, length / num_points)
    if np.random.random() < 0.5:
        amplitude = 0.0
        y = [0.0] * num_points
    else:
        amplitude = 0.0 + (np.random.random() * 0.06)
        coords = (x + tool_width) / length
        y = np.sin(coords * np.pi) * amplitude

    polygon_points = []
    for idx in range(num_points):
        polygon_points.append([x[idx], y[idx] + (ridge_width / 2.0)])
    for idx in range(num_points):
        polygon_points.append([x[num_points - 1 - idx], y[num_points - 1 - idx] - (ridge_width / 2.0)])

    ridge_polygon = shapely.Polygon(polygon_points)

    ridge_mesh = trimesh.creation.extrude_polygon(ridge_polygon, height)

    ridge_mesh.apply_translation([0.0, -amplitude + ((-tool_width / 2.0) + (np.random.random() * tool_width)), 0.0])
    return ridge_mesh, height


def compute_offset_from_mesh(mesh: trimesh.Trimesh, tool_width):
    vertices = copy.deepcopy(mesh.vertices)
    mask_vertices = vertices[vertices[:, 0] < tool_width / 2.0 + 0.01]
    mask_vertices = mask_vertices[mask_vertices[:, 0] > -tool_width / 2.0 - 0.01]
    mask_vertices = mask_vertices[mask_vertices[:, 0] < tool_width / 2.0 + 0.01]
    mask_vertices = mask_vertices[mask_vertices[:, 0] > -tool_width / 2.0 - 0.01]
    return max(mask_vertices[:, 2])


def generate_primitive_terrain(terrain_cfg: dict, idx: int, out_dir: str = None, vis=False):
    assets_dir = "assets"
    primitives_mesh_dir = "meshes/primitives/"
    primitives_urdf_dir = "urdf/primitives/"
    tool_width = 0.046  # TODO: Parameterize based on tool.

    # Load base urdf.
    base_urdf_fn = os.path.join(assets_dir, "urdf/base.urdf")
    with open(base_urdf_fn, "r") as f:
        urdf_string = f.read()

    terrain_type = terrain_cfg["type"]

    if terrain_type == "box":
        mesh, offset = generate_box(terrain_cfg, tool_width)
    elif terrain_type == "curve":
        mesh, offset = generate_curved_surface(terrain_cfg, tool_width)
    elif terrain_type == "ridge":
        mesh, offset = generate_ridge(terrain_cfg, tool_width)
    else:
        raise Exception("Unknown terrain type: %s" % terrain_type)

    if vis:
        mesh.show()
    mesh_path = os.path.join(primitives_mesh_dir, "terrain_%d.obj" % idx)
    mesh.export(os.path.join(assets_dir, mesh_path))

    # Also save to out dir (if provided).
    if out_dir is not None:
        mesh.export(os.path.join(out_dir, "terrain_%d.obj" % idx))

    # Load into urdf.
    urdf_string = urdf_string % (mesh_path, mesh_path)
    urdf_path = os.path.join(primitives_urdf_dir, "terrain_%d.urdf" % idx)
    with open(os.path.join(assets_dir, urdf_path), "w") as w:
        w.write(urdf_string)

    return os.path.splitext(urdf_path)[0], mesh, offset


def generate_primitive_terrains(terrain_cfg: dict, num_terrains: int, out_dir: str = None, offset_idx: int = 0,
                                vis=False):
    terrain_urdfs = []
    terrain_offsets = []
    terrain_meshes = []

    for terrain_idx in range(num_terrains):
        terrain_urdf, terrain_mesh, terrain_offset = generate_primitive_terrain(terrain_cfg, terrain_idx + offset_idx,
                                                                                out_dir, vis)
        terrain_urdfs.append(terrain_urdf)
        terrain_offsets.append(terrain_offset)
        terrain_meshes.append(terrain_mesh)

    return terrain_urdfs, terrain_meshes, np.array(terrain_offsets)
