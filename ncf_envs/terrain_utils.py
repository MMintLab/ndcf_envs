from isaacgym.terrain_utils import *

# create all available terrain types

def new_sub_terrain(tc):
    return SubTerrain(width=tc.num_rows, length=tc.num_cols, vertical_scale=tc.vertical_scale, horizontal_scale=tc.horizontal_scale)

def generate_discrete_obstacles_terrain(tc):
    tc.heightfield[0:tc.num_rows, :] = discrete_obstacles_terrain(new_sub_terrain(tc),
                                                                max_height=tc.max_height,
                                                                min_size=tc.min_size,
                                                                max_size=tc.max_size,
                                                                num_rects=tc.num_rects,
                                                                platform_size = tc.platform_size).height_field_raw
    w, h = tc.heightfield.shape
    vertices, triangles = convert_heightfield_to_trimesh(tc.heightfield[: w//3, : h//3], horizontal_scale=tc.horizontal_scale,
                                                         vertical_scale=tc.vertical_scale, slope_threshold=1.5)
    vertices[:,0] += tc.transform_xyz[0]
    vertices[:,1] += tc.transform_xyz[1]
    vertices[:,2] += tc.transform_xyz[2]

    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = vertices.shape[0]
    tm_params.nb_triangles = triangles.shape[0]
    # tm_params.transform.p.x = tc.transform_xyz[0]
    # tm_params.transform.p.y = tc.transform_xyz[1]

    return vertices, triangles, tm_params

def generate_wave_terrain(tc):
    tc.heightfield[0:tc.num_rows, :] = wave_terrain(new_sub_terrain(tc),
                                                    num_waves= tc.num_waves,
                                                    amplitude= tc.amplitude).height_field_raw
    w, h = tc.heightfield.shape
    vertices, triangles = convert_heightfield_to_trimesh(tc.heightfield[: w//3, : h//3], horizontal_scale=tc.horizontal_scale,
                                                         vertical_scale=tc.vertical_scale)
    vertices[:,0] += tc.transform_xyz[0]
    vertices[:,1] += tc.transform_xyz[1]
    vertices[:,2] += tc.transform_xyz[2]


    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = vertices.shape[0]
    tm_params.nb_triangles = triangles.shape[0]
    # tm_params.transform.p.x = tc.transform_xyz[0]
    # tm_params.transform.p.y = tc.transform_xyz[1]
    # tm_params.transform.p.z = tc.transform_xyz[2]
    return vertices, triangles, tm_params


def add_discrete_obstacles_terrain(gym, sim, tc):
    vertices, triangles, tm_params = generate_discrete_obstacles_terrain(tc)
    gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)

def add_wave_terrain(gym, sim, tc):
    vertices, triangles, tm_params = generate_wave_terrain(tc)
    gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)