import argparse
import copy
import os

import mmint_utils
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymutil
import numpy as np
from tqdm import trange
import transforms3d as tf3d

from mmint_utils.trajectory_planning import get_linear_trajectory
import torch

# Simulate pressing deformable tool into surface.
# Some code borrowed from: https://sites.google.com/nvidia.com/tactiledata2
from mmint_utils.images2gif import images2gif


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", "-o", type=str, default=None, help="Directory to store output.")
    parser.add_argument("--viewer", "-v", dest='viewer', action='store_true', help="Use viewer.")
    args = parser.parse_args()

    use_viewer = args.viewer
    out = args.out

    if out is not None:
        mmint_utils.make_dir(out)

    gym = gymapi.acquire_gym()

    # Setup sim object.
    sim = create_sim(gym)

    # Load assets.
    asset_options = set_asset_options()
    wrist_urdf_dir = "assets"
    wrist_asset_handle = load_assets(gym, sim, wrist_urdf_dir, ['urdf/wrist'], asset_options, fix=True, gravity=False)[
        0]

    # Create scene.
    env_handle, wrist_actor_handle, camera_handle = create_scene(gym, sim, wrist_asset_handle)

    # Setup wrist control properties.
    set_wrist_ctrl_props(gym, env_handle, wrist_actor_handle, [3e9, 50])

    # Create viewer.
    if use_viewer:
        viewer, axes = create_viewer(gym, sim)
    else:
        viewer = axes = None

    run_sim_loop(gym, sim, env_handle, wrist_actor_handle, camera_handle, viewer, axes, use_viewer, out)


def create_scene(gym, sim, wrist_asset_handle):
    """
    Create scene.
    """
    # Add plane.
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    plane_params.segmentation_id = 1
    plane_params.static_friction = 1
    plane_params.dynamic_friction = 1
    gym.add_ground(sim, plane_params)

    # Create environment.
    env_handle = gym.create_env(sim, gymapi.Vec3(-1.0, 0.0, -1.0), gymapi.Vec3(1.0, 1.0, 1.0), 1)

    # Create wrist.
    pose = gymapi.Transform()
    wrist_actor_handle = gym.create_actor(env_handle, wrist_asset_handle, pose, "wrist", segmentationId=1)

    # Create camera.
    camera_props = gymapi.CameraProperties()
    camera_props.width = 512
    camera_props.height = 512
    camera_handle = gym.create_camera_sensor(env_handle, camera_props)
    gym.set_camera_location(camera_handle, env_handle, gymapi.Vec3(0.5, 0.0, 0.5), gymapi.Vec3(0.0, 0.0, 0.0))

    return env_handle, wrist_actor_handle, camera_handle


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


def set_wrist_ctrl_props(gym, env, wrist, pd_gains=[1.0e9, 0.0]):
    """
    Set wrist control properties.
    """
    wrist_dof_props = gym.get_actor_dof_properties(env, wrist)
    for dof_idx in range(wrist_dof_props.shape[0]):
        wrist_dof_props['driveMode'][dof_idx] = gymapi.DOF_MODE_POS
        wrist_dof_props['stiffness'][dof_idx] = pd_gains[0]
        wrist_dof_props['damping'][dof_idx] = pd_gains[1]
    gym.set_actor_dof_properties(env, wrist, wrist_dof_props)


def get_wrist_dof_info(gym, env, wrist):
    """
    Get wrist dof info.
    """
    dof_states = gym.get_actor_dof_states(env, wrist, gymapi.STATE_ALL)

    return dof_states["pos"], dof_states["vel"]


def extract_net_forces(gym, sim):
    """
    Extract the net force vector on the wrist.

    TODO: Add torque as well.
    """

    contacts = gym.get_soft_contacts(sim)
    num_envs = gym.get_env_count(sim)
    net_force_vecs = np.zeros((num_envs, 3))
    contact_forces = []
    for contact in contacts:
        rigid_body_index = contact[4]
        contact_normal = np.array([*contact[6]])
        contact_force_mag = contact[7]
        env_index = rigid_body_index // 3
        force_vec = contact_force_mag * contact_normal
        contact_forces.append(force_vec)
        net_force_vecs[env_index] += force_vec
    net_force_vecs = -net_force_vecs

    return net_force_vecs, contact_forces


def extract_contact_points(gym, sim):
    # TODO: Extend to multi-environment setup.
    contacts = gym.get_soft_contacts(sim)
    num_envs = gym.get_env_count(sim)
    contact_points = []
    for contact in contacts:
        contact_points.append(contact[5])
    return contact_points


def extract_nodal_coords(gym, sim, particle_states):
    """
    Extract the nodal coordinates for the tool from each environment.
    """

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


def reset_wrist(gym, sim, env, wrist, tool_state_init, joint_state):
    """
    Reset wrist to starting pose (including joint position set points).

    We also reset the particle state of the deformable to avoid bad initialization.

    Joint state here is pose of wrist as [pos, rxyz euler angles].
    """
    reset_joint_state(gym, env, wrist, joint_state)
    gym.set_actor_dof_position_targets(env, wrist, joint_state)

    # Transform particle positions to new pose.
    tool_state_init[:, :3] = torch.from_numpy(transform_points(tool_state_init[:, :3], joint_state)).to("cuda:0")

    # return points, tf_matrix
    gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(tool_state_init))


def reset_wrist_offset(gym, sim, env, wrist, tool_state_init, orientation, offset):
    # Determine position for tool.
    start_orientation = [0, 0, 0, orientation[0], orientation[1], orientation[2]]
    z_offset = offset - min(transform_points(tool_state_init[:, :3], start_orientation)[:, 2])

    # Send to pose.
    pose = [0, 0, z_offset, orientation[0], orientation[1], orientation[2]]
    reset_wrist(gym, sim, env, wrist, tool_state_init, pose)

    return z_offset


def get_results(gym, sim, env, wrist, camera, viewer, particle_state_tensor):
    # Render cameras.
    gym.step_graphics(sim)
    gym.render_all_camera_sensors(sim)

    gym.draw_env_rigid_contacts(viewer, env, gymapi.Vec3(1.0, 0.5, 0.0), 0.05, True)
    gym.draw_env_soft_contacts(viewer, env, gymapi.Vec3(0.6, 0.0, 0.6), 0.05, False, True)

    # Get force.
    force, contact_forces = extract_net_forces(gym, sim)

    # Get mesh deformations.
    nodal_coords = extract_nodal_coords(gym, sim, particle_state_tensor)

    # Get contact points.
    contact_points = extract_contact_points(gym, sim)

    # Get camera sensing.
    rgb_image = gym.get_camera_image(sim, env, camera, gymapi.IMAGE_COLOR).reshape(512, 512, 4)
    depth_image = gym.get_camera_image(sim, env, camera, gymapi.IMAGE_DEPTH)
    seg_image = gym.get_camera_image(sim, env, camera, gymapi.IMAGE_SEGMENTATION)

    # Get wrist pose.
    wrist_pose, _ = get_wrist_dof_info(gym, env, wrist)

    results_dict = {
        "force": force,
        "nodal_coords": nodal_coords,
        "contact_points": contact_points,
        "contact_forces": contact_forces,
        "rgb": rgb_image,
        "depth": depth_image,
        "segmentation": seg_image,
        "wrist_pose": wrist_pose,
    }
    return results_dict


def run_sim_loop(gym, sim, env, wrist, camera, viewer, axes, use_viewer, out_dir):
    # Get particle state tensor and convert to PyTorch tensor - used to track nodes of tool mesh.
    particle_state_tensor = gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))
    gym.refresh_particle_state_tensor(sim)
    tool_state_init = copy.deepcopy(particle_state_tensor)

    z_offset = 0.01
    indent_distance = 0.01
    lowering_speed = -0.2  # m/s
    dt = gym.get_sim_params(sim).dt
    n = 10
    configs = np.array([-0.3, -0.3, -0.3]) + (np.random.random([n, 3]) * np.array([0.6, 0.6, 0.6]))

    for config_idx, config in enumerate(configs):
        # Reset to new config.
        tool_state_init_ = copy.deepcopy(tool_state_init)
        start_z = reset_wrist_offset(gym, sim, env, wrist, tool_state_init_, config, z_offset)
        goal_z = start_z - z_offset - indent_distance

        # Lower until in contact.
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

            curr_pos, curr_vel = get_wrist_dof_info(gym, env, wrist)

            if t % 1000 == 0:
                print("Curr z: %d" % curr_pos[2])

            if curr_pos[2] < goal_z:
                break

            curr_pos[2] += lowering_speed * dt
            gym.set_actor_dof_position_targets(env, wrist, curr_pos)

        # Let simulation settle a bit.
        for _ in range(100):
            # Step simulator.
            gym.simulate(sim)
            gym.fetch_results(sim, True)

        # Get results.
        results_dict = get_results(gym, sim, env, wrist, camera, viewer, particle_state_tensor)

        # Save results.
        if out_dir is not None:
            mmint_utils.save_gzip_pickle(results_dict, os.path.join(out_dir, "config_%d.pkl.gzip" % config_idx))

    # Clean up
    if use_viewer:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
    print("Done!")


if __name__ == '__main__':
    main()
