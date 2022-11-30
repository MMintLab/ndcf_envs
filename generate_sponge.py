import argparse
import os
import mmint_utils
import trimesh.creation
from shapely.geometry import Polygon
import shapely


def generate_hollow_sponge(sponge_width: float, sponge_height: float, thickness: float):
    # Base/Top Parts.
    sponge_base_polygon = Polygon([(-sponge_width / 2, -sponge_width / 2), (-sponge_width / 2, sponge_width / 2),
                                   (sponge_width / 2, sponge_width / 2), (sponge_width / 2, -sponge_width / 2),
                                   (-sponge_width / 2, -sponge_width / 2)])

    # Base
    sponge_base = trimesh.creation.extrude_polygon(sponge_base_polygon, thickness)

    # Top.
    sponge_top = trimesh.creation.extrude_polygon(sponge_base_polygon, thickness)
    sponge_top.apply_translation([0.0, 0.0, sponge_height - thickness])

    # Wall Parts.
    sponge_wall_left_polygon = Polygon([(-sponge_width / 2, -sponge_width / 2), (-sponge_width / 2, sponge_width / 2),
                                        (-sponge_width / 2 + thickness, sponge_width / 2),
                                        (-sponge_width / 2 + thickness, -sponge_width / 2),
                                        (-sponge_width / 2, -sponge_width / 2)])
    sponge_wall_left = trimesh.creation.extrude_polygon(sponge_wall_left_polygon, sponge_height)

    sponge_wall_right_polygon = Polygon([(sponge_width / 2, -sponge_width / 2), (sponge_width / 2, sponge_width / 2),
                                         (sponge_width / 2 - thickness, sponge_width / 2),
                                         (sponge_width / 2 - thickness, -sponge_width / 2),
                                         (sponge_width / 2, -sponge_width / 2)])
    sponge_wall_right = trimesh.creation.extrude_polygon(sponge_wall_right_polygon, sponge_height)

    sponge_wall_top_polygon = Polygon([(-sponge_width / 2, sponge_width / 2),
                                       (sponge_width / 2, sponge_width / 2),
                                       (sponge_width / 2, sponge_width / 2 - thickness),
                                       (-sponge_width / 2, sponge_width / 2 - thickness),
                                       (-sponge_width / 2, sponge_width / 2)])
    sponge_wall_top = trimesh.creation.extrude_polygon(sponge_wall_top_polygon, sponge_height)

    sponge_wall_bottom_polygon = Polygon([(-sponge_width / 2, -sponge_width / 2),
                                          (sponge_width / 2, -sponge_width / 2),
                                          (sponge_width / 2, -sponge_width / 2 + thickness),
                                          (-sponge_width / 2, -sponge_width / 2 + thickness),
                                          (-sponge_width / 2, -sponge_width / 2)])
    sponge_wall_bottom = trimesh.creation.extrude_polygon(sponge_wall_bottom_polygon, sponge_height)

    # Union them all together.
    hollow_sponge = sponge_base.union(sponge_top, engine="scad")
    hollow_sponge = hollow_sponge.union(sponge_wall_left, engine="scad")
    hollow_sponge = hollow_sponge.union(sponge_wall_right, engine="scad")
    hollow_sponge = hollow_sponge.union(sponge_wall_top, engine="scad")
    hollow_sponge = hollow_sponge.union(sponge_wall_bottom, engine="scad")

    # Merge duplicated vertices.
    hollow_sponge.merge_vertices()

    hollow_sponge.show()
    return hollow_sponge


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
    parser.add_argument("--hollow", "-o", dest="hollow", action="store_true", help="Generate hollow mesh.")
    parser.add_argument("--thickness", "-t", type=float, default=0.005, help="Thickness of hollow mesh wall.")
    args = parser.parse_args()

    if args.hollow:
        mesh = generate_hollow_sponge(args.width, args.height, args.thickness)
    else:
        mesh = generate_sponge(args.width, args.height)

    mmint_utils.make_dir(os.path.dirname(args.out_fn))
    mesh.export(args.out_fn)
