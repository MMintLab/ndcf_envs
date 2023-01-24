from isaacgym.terrain_utils import *

# create all available terrain types

def new_sub_terrain(tc):
    return SubTerrain(width=tc.num_rows, length=tc.num_cols, vertical_scale=tc.vertical_scale, horizontal_scale=tc.horizontal_scale)

def add_discrete_obstacles_terrain(gym, sim, tc):
    tc.heightfield[0:tc.num_rows, :] = discrete_obstacles_terrain(new_sub_terrain(tc),
                                                                max_height=tc.max_height,
                                                                min_size=tc.min_size,
                                                                max_size=tc.max_size,
                                                                num_rects=tc.num_rects,
                                                                platform_size = tc.platform_size).height_field_raw
    vertices, triangles = convert_heightfield_to_trimesh(tc.heightfield, horizontal_scale=tc.horizontal_scale,
                                                         vertical_scale=tc.vertical_scale, slope_threshold=1.5)
    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = vertices.shape[0]
    tm_params.nb_triangles = triangles.shape[0]
    tm_params.transform.p.x = tc.transform_xyz[0]
    tm_params.transform.p.y = tc.transform_xyz[1]
    gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)