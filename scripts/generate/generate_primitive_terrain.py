import argparse

import mmint_utils

from ncf_envs.terrain.primitives.generate_primitive_terrain import generate_box, generate_ridge, generate_curved_surface

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("terrain_cfg", type=str, help="Terrain configuration file.")
    parser.add_argument("-o", "--out", type=str, default=None, help="[Optional] file to save generated mesh.")
    args = parser.parse_args()

    terrain_cfg = mmint_utils.load_cfg(args.terrain_cfg)

    tool_width = 0.046  # TODO: Parameterize based on tool.
    terrain_type = terrain_cfg["type"]

    if terrain_type == "box":
        mesh, offset = generate_box(terrain_cfg, tool_width)
    elif terrain_type == "curve":
        mesh, offset = generate_curved_surface(terrain_cfg, tool_width)
    elif terrain_type == "ridge":
        mesh, offset = generate_ridge(terrain_cfg, tool_width)
    else:
        raise Exception("Unknown terrain type: %s" % terrain_type)

    if args.out is not None:
        mesh.export(args.out)
    else:
        mesh.show()
