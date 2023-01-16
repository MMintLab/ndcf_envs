import copy
import os
from collections import defaultdict

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
import numpy as np
import transforms3d as tf3d
import torch
from tqdm import trange

import ncf_envs.utils as utils
import ncf_envs.real_utils as real_utils


# Simulate pressing deformable tool into surface.
# Some code borrowed from: https://sites.google.com/nvidia.com/tactiledata2

def load_real_world_examples(run_dir):
    # Load real world run data.
    example_names = [f.replace(".pkl.gzip", "") for f in os.listdir(run_dir) if ".pkl.gzip" in f]
    example_names.sort(key=lambda k: int(k.split(".")[0].split("_")[-1]))

    real_configs = []
    press_zs = []
    real_wrenches = []
    for example_name in example_names:
        real_dict = real_utils.load_observation_from_file(run_dir, example_name)

        # Get configuration from the real world data.
        real_config = real_dict["proprioception"]["tool_orn_config"]
        real_configs.append(real_config)

        # Find final z height of press.
        ee_pose = real_dict["proprioception"]["ee_pose"][0]
        table_height = 0.21  # TODO: Parameterize.
        press_z = ee_pose[0][2] - table_height
        press_zs.append(press_z)

        # Get wrench observed for real data.
        real_wrench = np.array(real_dict["tactile"]["ati_wrench"][-1][0])
        real_wrenches.append(real_wrench)

    return real_configs, press_zs, real_wrenches


def create_simulator(num_envs: int, use_viewer: bool = False):
    # Setup simulator.
    gym = gymapi.acquire_gym()

    # Setup sim object.
    sim = create_sim(gym)

    # Load table/wrist asset.
    urdf_dir = "assets"
    asset_options = set_asset_options()
    asset_handles = load_assets(gym, sim, urdf_dir, ['urdf/wrist', 'urdf/table'], asset_options, fix=True,
                                gravity=False)
    wrist_asset_handle = asset_handles[0]
    table_asset_handle = asset_handles[1]

    # Create scene.
    scene_props = set_scene_props(num_envs)
    env_handles, table_actor_handles, wrist_actor_handles, camera_handles = \
        create_scene(gym, sim, scene_props, wrist_asset_handle, table_asset_handle)

    # Setup wrist control properties.
    set_wrist_ctrl_props(gym, env_handles, wrist_actor_handles, [1e9, 50], [1e9, 50])

    # Create viewer.
    if use_viewer:
        viewer, _ = create_viewer(gym, sim)
    else:
        viewer = None

    # Get initial config of particles.
    init_particle_state = get_init_particle_state(gym, sim)

    return gym, sim, env_handles, wrist_actor_handles, camera_handles, viewer, init_particle_state


def set_scene_props(num_envs, env_dim=0.1):
    """
    Setup scene and environment properties.
    """
    envs_per_row = int(np.ceil(np.sqrt(num_envs)))
    env_lower = gymapi.Vec3(-env_dim, -env_dim, -env_dim)
    env_upper = gymapi.Vec3(env_dim, env_dim, env_dim)
    scene_props = {'num_envs': num_envs,
                   'per_row': envs_per_row,
                   'lower': env_lower,
                   'upper': env_upper}

    return scene_props


def create_scene(gym, sim, props, wrist_asset_handle, table_asset_handle):
    """
    Create scene.
    """
    # Add plane.
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.segmentation_id = 1
    plane_params.static_friction = 1
    plane_params.dynamic_friction = 1
    plane_params.distance = 0.01
    gym.add_ground(sim, plane_params)

    env_handles = []
    table_actor_handles = []
    wrist_actor_handles = []
    camera_handles = []

    for i in range(props["num_envs"]):
        # Create environment.
        env_handle = gym.create_env(sim, props['lower'], props['upper'], props['per_row'])
        env_handles.append(env_handle)

        # Create tabletop.
        pose = gymapi.Transform()
        table_actor_handle = gym.create_actor(env_handle, table_asset_handle, pose, "table", group=i, filter=0,
                                              segmentationId=1)
        table_actor_handles.append(table_actor_handle)

        # Create wrist.
        pose = gymapi.Transform()
        wrist_actor_handle = gym.create_actor(env_handle, wrist_asset_handle, pose, "wrist", group=i, filter=0,
                                              segmentationId=1)
        wrist_actor_handles.append(wrist_actor_handle)

        # Create camera.
        camera_props = gymapi.CameraProperties()
        camera_props.width = 512
        camera_props.height = 512
        camera_handle = gym.create_camera_sensor(env_handle, camera_props)
        gym.set_camera_location(camera_handle, env_handle, gymapi.Vec3(0.5, 0.0, 0.5), gymapi.Vec3(0.0, 0.0, 0.0))
        camera_handles.append(camera_handle)

    return env_handles, table_actor_handles, wrist_actor_handles, camera_handles


def create_sim(gym):
    """
    Setup simulation parameters.
    """

    sim_type = gymapi.SIM_FLEX
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0e-3  # Control frequency
    sim_params.substeps = 1  # Physics simulation frequency (multiplier)
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    sim_params.up_axis = gymapi.UpAxis.UP_AXIS_Z
    sim_params.use_gpu_pipeline = False

    sim_params.stress_visualization = True  # von Mises stress
    sim_params.stress_visualization_min = 1.0e4
    sim_params.stress_visualization_max = 1.0e5

    sim_params.flex.solver_type = 5  # PCR (GPU, global)
    sim_params.flex.num_outer_iterations = 10
    sim_params.flex.num_inner_iterations = 200
    sim_params.flex.relaxation = 0.75
    sim_params.flex.warm_start = 0.8
    sim_params.flex.deterministic_mode = True

    sim_params.flex.geometric_stiffness = 1.0
    sim_params.flex.shape_collision_distance = 1e-4  # Distance to be maintained between soft bodies and other bodies or ground plane
    sim_params.flex.shape_collision_margin = 1e-4  # Distance from rigid bodies at which to begin generating contact constraints

    sim_params.flex.friction_mode = 2  # Friction about all 3 axes (including torsional)
    sim_params.flex.dynamic_friction = 1.0

    gpu_physics = 0
    gpu_render = 0
    sim = gym.create_sim(gpu_physics, gpu_render, sim_type, sim_params)

    return sim


def set_asset_options():
    """
    Set asset options common to all assets.
    """

    options = gymapi.AssetOptions()
    options.flip_visual_attachments = False
    options.armature = 0.0
    options.thickness = 0.0
    options.linear_damping = 0.0
    options.angular_damping = 0.0
    options.default_dof_drive_mode = gymapi.DOF_MODE_POS
    options.min_particle_mass = 1e-20

    return options


def load_assets(gym, sim, base_dir, filenames, options, fix=True, gravity=False):
    """
    Load assets from specified URDF files.
    """
    options.fix_base_link = True if fix else False
    options.disable_gravity = True if not gravity else False
    handles = []
    for obj in filenames:
        handle = gym.load_asset(sim, base_dir, obj + '.urdf', options)
        handles.append(handle)

    return handles


def create_viewer(gym, sim):
    """Create viewer and axes objects."""

    camera_props = gymapi.CameraProperties()
    camera_props.horizontal_fov = 5.0
    camera_props.width = 1920
    camera_props.height = 1080
    viewer = gym.create_viewer(sim, camera_props)
    camera_pos = gymapi.Vec3(1.5, -2.0, 2.0)
    camera_target = gymapi.Vec3(0.0, 0.0, 0.0)
    gym.viewer_camera_look_at(viewer, None, camera_pos, camera_target)

    axes_geom = gymutil.AxesGeometry(0.1)

    return viewer, axes_geom


def set_wrist_ctrl_props(gym, envs, wrists, pos_pd_gains=[1.0e9, 0.0], orn_pd_gains=[1.0e9, 0.0]):
    """
    Set wrist control properties.
    """
    for env, wrist in zip(envs, wrists):
        wrist_dof_props = gym.get_actor_dof_properties(env, wrist)
        for dof_idx in range(3):
            wrist_dof_props['driveMode'][dof_idx] = gymapi.DOF_MODE_POS
            wrist_dof_props['stiffness'][dof_idx] = pos_pd_gains[0]
            wrist_dof_props['damping'][dof_idx] = pos_pd_gains[1]
        for dof_idx in range(3, 6):
            wrist_dof_props['driveMode'][dof_idx] = gymapi.DOF_MODE_POS
            wrist_dof_props['stiffness'][dof_idx] = orn_pd_gains[0]
            wrist_dof_props['damping'][dof_idx] = orn_pd_gains[1]
        gym.set_actor_dof_properties(env, wrist, wrist_dof_props)


def get_wrist_dof_info(gym, env, wrist):
    """
    Get wrist dof info.
    """
    dof_states = gym.get_actor_dof_states(env, wrist, gymapi.STATE_ALL)

    return dof_states["pos"].copy(), dof_states["vel"].copy()


def get_contact_info(gym, sim, rigid_body_per_env):
    """
    Extract the net force vector on the wrist.
    """
    contacts = gym.get_soft_contacts(sim)
    num_envs = gym.get_env_count(sim)
    contact_forces = defaultdict(list)
    contact_points = defaultdict(list)
    contact_normals = defaultdict(list)

    for contact in contacts:
        rigid_body_index = contact[4]
        contact_normal = np.array([*contact[6]])
        contact_force_mag = contact[7]
        env_index = rigid_body_index // rigid_body_per_env
        force_vec = contact_force_mag * contact_normal
        contact_forces[env_index].append(list(force_vec))
        contact_points[env_index].append(list(contact[5]))
        contact_normals[env_index].append(list(contact_normal))

    contact_forces_ = []
    contact_points_ = []
    contact_normals_ = []
    for env_idx in range(num_envs):
        contact_points_.append(contact_points[env_idx])
        contact_forces_.append(contact_forces[env_idx])
        contact_normals_.append(contact_normals[env_idx])

    return contact_points_, contact_forces_, contact_normals_


def get_init_particle_state(gym, sim):
    # Get particle state tensor and convert to PyTorch tensor - used to track nodes of tool mesh.
    particle_state_tensor = gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))
    gym.refresh_particle_state_tensor(sim)
    tool_state_init = copy.deepcopy(particle_state_tensor)
    return tool_state_init


def get_nodal_coords(gym, sim, particle_states):
    """
    Extract the nodal coordinates for the tool from each environment.

    Note: this assumes an equal number of nodes per environment.
    """
    # read tetrahedral and triangle data from simulation
    gym.refresh_particle_state_tensor(sim)
    num_envs = gym.get_env_count(sim)
    num_particles = len(particle_states)
    num_particles_per_env = int(num_particles / num_envs)
    nodal_coords = np.zeros((num_envs, num_particles_per_env, 3))
    for global_particle_index, particle_state in enumerate(particle_states):
        pos = particle_state[:3]
        env_index = global_particle_index // num_particles_per_env
        local_particle_index = global_particle_index % num_particles_per_env
        nodal_coords[env_index][local_particle_index] = pos.numpy()

    return nodal_coords


def reset_joint_state(gym, env, wrist, joint_state):
    num_dof = gym.get_actor_dof_count(env, wrist)
    dof_states = gym.get_actor_dof_states(env, wrist, gymapi.STATE_ALL)
    for dof_idx in range(num_dof):
        dof_states["pos"][dof_idx] = joint_state[dof_idx]
        dof_states["vel"][dof_idx] = 0.0
    gym.set_actor_dof_states(env, wrist, dof_states, gymapi.STATE_ALL)


def transform_points(points, transform, axes='rxyz'):
    """
    Transform given points.

    If 6 terms, assumed euler angles. Uses axes arg to interpret.
    If 7 terms, assumed wxyz quaternion.
    """
    if type(points) == torch.Tensor:
        points = points.cpu().numpy()

    # Build transform matrix.
    tf_matrix = np.eye(4)
    tf_matrix[:3, 3] = transform[:3]
    if len(transform) == 6:
        tf_matrix[:3, :3] = tf3d.euler.euler2mat(transform[3], transform[4], transform[5], axes=axes)
    else:
        tf_matrix[:3, :3] = tf3d.quaternions.quat2mat(transform[3:])

    # Perform transformation.
    points_tf = np.ones([points.shape[0], 4], dtype=points.dtype)
    points_tf[:, :3] = points
    return (tf_matrix @ points_tf.T).T[:, :3]


def reset_wrist(gym, sim, env, wrist, joint_state):
    """
    Reset wrist to starting pose (including joint position set points).

    We also reset the particle state of the deformable to avoid bad initialization.

    Joint state here is pose of wrist as [pos, rxyz euler angles].
    """
    reset_joint_state(gym, env, wrist, joint_state)
    gym.set_actor_dof_position_targets(env, wrist, joint_state)


def reset_wrist_offset(gym, sim, envs, wrists, tool_state_init, orientations, offset):
    z_offsets = []

    for env_idx, (env, wrist, orientation) in enumerate(zip(envs, wrists, orientations)):
        # Determine position for tool.
        base_R_w = tf3d.quaternions.quat2mat([0.0, 1.0, 0.0, 0.0])
        des_R_base = tf3d.euler.euler2mat(orientation[0], orientation[1], orientation[2], axes="rxyz")
        des_R_w = des_R_base @ base_R_w
        ax, ay, az = tf3d.euler.mat2euler(des_R_w, axes="rxyz")
        start_orientation = [0, 0, 0, ax, ay, az]
        z_offset = offset - min(transform_points(tool_state_init[env_idx, :, :3], start_orientation)[:, 2])
        z_offsets.append(z_offset)

        # Send to pose.
        pose = [0, 0, z_offset, ax, ay, az]
        reset_wrist(gym, sim, env, wrist, pose)

        # Transform particle positions to new pose.
        tool_state_init[env_idx, :, :3] = torch.from_numpy(transform_points(tool_state_init[env_idx, :, :3], pose)).to(
            "cuda:0")

    # Set particle states for tools to avoid bad initialization.
    gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(tool_state_init.reshape(-1, tool_state_init.shape[-1])))

    return z_offsets


def get_wrist_wrench(contact_points, contact_forces, wrist_pose):
    w_T_wrist_pose = utils.pose_to_matrix(wrist_pose, axes="rxyz")
    wrist_pose_T_w = np.linalg.inv(w_T_wrist_pose)

    # Load contact point cloud.
    contact_points_w = np.array([list(ctc_pt) for ctc_pt in contact_points])
    contact_points = utils.transform_pointcloud(contact_points_w, wrist_pose_T_w)

    # Load contact forces.
    contact_forces_w = np.array(contact_forces)
    contact_forces = -utils.transform_vectors(contact_forces_w, wrist_pose_T_w)

    wrist_wrench = np.zeros(6, dtype=float)
    wrist_wrench[:3] = contact_forces.sum(axis=0)
    wrist_wrench[3:] = np.cross(contact_points, contact_forces).sum(axis=0)

    return wrist_wrench


def get_results(gym, sim, envs, wrists, cameras, viewer, particle_state_tensor, render_cameras=False):
    # Render cameras.
    if render_cameras:
        gym.step_graphics(sim)
        gym.render_all_camera_sensors(sim)

    # Get mesh deformations for all envs.
    nodal_coords = get_nodal_coords(gym, sim, particle_state_tensor)

    # Get contact information for all envs.
    contact_points, contact_forces, contact_normals = get_contact_info(gym, sim, gym.get_env_rigid_body_count(envs[0]))

    results = []
    for env_idx, (env, wrist, camera) in enumerate(zip(envs, wrists, cameras)):
        gym.draw_env_rigid_contacts(viewer, env, gymapi.Vec3(1.0, 0.5, 0.0), 0.05, True)
        gym.draw_env_soft_contacts(viewer, env, gymapi.Vec3(0.6, 0.0, 0.6), 0.05, False, True)

        # Get wrist pose.
        wrist_pose, _ = get_wrist_dof_info(gym, env, wrist)
        w_T_wrist = utils.pose_to_matrix(wrist_pose, axes="rxyz")

        # Wrist to tool mount.
        wrist_T_mount = utils.pose_to_matrix(np.array([0.0, 0.0, 0.036, 0.0, 0.0, 0.0]), axes="rxyz")
        mount_pose = utils.matrix_to_pose(w_T_wrist @ wrist_T_mount)

        # For convenience, transform nodal coordinates to wrist frame.
        nodal_coords_wrist = utils.transform_pointcloud(nodal_coords[env_idx], np.linalg.inv(w_T_wrist))

        # Get wrist wrench from contact points/forces.
        wrist_wrench = get_wrist_wrench(contact_points[env_idx], contact_forces[env_idx], wrist_pose)

        results_dict = {
            "nodal_coords": nodal_coords[env_idx],
            "nodal_coords_wrist": nodal_coords_wrist,
            "contact_points": contact_points[env_idx],
            "contact_forces": contact_forces[env_idx],
            "contact_normals": contact_normals[env_idx],
            "wrist_pose": wrist_pose,
            "mount_pose": mount_pose,
            "wrist_wrench": wrist_wrench,
        }

        if render_cameras:
            # Get camera sensing.
            rgb_image = gym.get_camera_image(sim, env, camera, gymapi.IMAGE_COLOR).reshape(512, 512, 4)
            depth_image = gym.get_camera_image(sim, env, camera, gymapi.IMAGE_DEPTH)
            seg_image = gym.get_camera_image(sim, env, camera, gymapi.IMAGE_SEGMENTATION)

            results_dict.update({
                "rgb": rgb_image,
                "depth": depth_image,
                "segmentation": seg_image,
            })

        results.append(results_dict)
    return results


def close_sim(gym, sim, viewer, use_viewer):
    # Clean up
    if use_viewer:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    print("Done!")


def run_sim_loop(gym, sim, envs, wrists, cameras, viewer, use_viewer, configs, z_heights, init_particle_state):
    # Wrap particle
    particle_state_tensor = gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))
    gym.refresh_particle_state_tensor(sim)

    table_offset = 0.0001
    z_offset = table_offset
    lowering_speed = -0.2  # m/s
    indent_distance = 0.01
    dt = gym.get_sim_params(sim).dt
    num_envs = len(envs)
    num_configs = len(configs)
    num_rounds = int(np.ceil(float(num_configs) / float(num_envs)))
    z_height_provided = z_heights is not None

    if not z_height_provided:
        z_heights = [None] * num_configs

    results = []
    for range_idx in trange(num_rounds):
        round_configs = configs[range_idx * num_envs: min((range_idx + 1) * num_envs, num_configs)]
        round_goal_z_heights = z_heights[range_idx * num_envs: min((range_idx + 1) * num_envs, num_configs)]

        # Reset to new config.
        tool_state_init_ = copy.deepcopy(init_particle_state)
        tool_state_init_ = tool_state_init_.reshape(num_envs, -1, tool_state_init_.shape[-1])
        start_zs = reset_wrist_offset(gym, sim, envs, wrists, tool_state_init_, round_configs, z_offset)

        if not z_height_provided:
            for idx, start_z in enumerate(start_zs):
                round_goal_z_heights[idx] = start_z - table_offset - indent_distance

        # Lower until each environment reaches the desired height.
        t = 0
        while True:
            t += 1
            # Step simulator.
            gym.simulate(sim)
            gym.fetch_results(sim, True)

            # Visualize motion and deformation
            if use_viewer:
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
                gym.clear_lines(viewer)

            # Set goal motions for each wrist.
            complete = True
            for env, wrist, start_z, goal_z in zip(envs, wrists, start_zs, round_goal_z_heights):
                # Get current wrist pose.
                curr_pos, curr_vel = get_wrist_dof_info(gym, env, wrist)
                curr_z = curr_pos[2]

                # If we've reached our lowering goal, exit.
                if abs(curr_z - goal_z) > 0.001:
                    complete = False

                # Set new desired pose.
                des_z = start_z + (lowering_speed * dt * t)
                curr_pos[2] = max(des_z, goal_z)
                gym.set_actor_dof_position_targets(env, wrist, curr_pos)

            if complete:
                break

        # Let simulation settle a bit.
        for _ in range(10):
            # Step simulator.
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            if use_viewer:
                gym.step_graphics(sim)
                gym.draw_viewer(viewer, sim, True)
                gym.clear_lines(viewer)

        # Get results.
        results_ = get_results(gym, sim, envs, wrists, cameras, viewer, particle_state_tensor)[:len(round_configs)]

        results.extend(results_)
    return results
