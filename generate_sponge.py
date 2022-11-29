import argparse
import os
import mmint_utils
import trimesh.creation
from shapely.geometry import Polygon
import shapely


def generate_sponge(sponge_width: float, sponge_height: float):
    polygon = Polygon([(-sponge_width / 2, -sponge_width / 2), (-sponge_width / 2, sponge_width / 2),
                       (sponge_width / 2, sponge_width / 2), (sponge_width / 2, -sponge_width / 2),
                       (-sponge_width / 2, -sponge_width / 2)])

    mesh = trimesh.creation.extrude_polygon(polygon, height=sponge_height)
    return mesh


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate sponge mesh.")
    parser.add_argument("width", type=float, help="Sponge width.")
    parser.add_argument("height", type=float, help="Sponge height.")
    parser.add_argument("out_fn", type=str, help="File to write mesh to.")
    args = parser.parse_args()

    mesh = generate_sponge(args.width, args.height)

    mmint_utils.make_dir(os.path.dirname(args.out_fn))
    mesh.export(args.out_fn)
