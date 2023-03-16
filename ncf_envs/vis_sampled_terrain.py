import argparse
from ncf_envs.terrain.primitives.generate_primitive_terrain import generate_primitive_terrains
from press_simulator import *
import mmint_utils


def vis_sampled_terrain():
    parser = argparse.ArgumentParser()
    parser.add_argument("terrain_cfg", type=str, help="Terrain configuration file.")
    parser.add_argument("--num_envs", "-e", type=int, default=1,
                        help="Number of environments to simultaneously simulate.")
    parser.add_argument("--cfg_s", type=str, default="cfg/scene.yaml",
                        help="path to scene config yaml file")
    args = parser.parse_args()
    num_envs = args.num_envs

    terrain_cfg = mmint_utils.load_cfg(args.terrain_cfg)
    cfg_s = mmint_utils.load_cfg(args.cfg_s)

    # Acquire singleton object.
    gym = gymapi.acquire_gym()

    # Generate terrain.
    terrain_files, _, terrain_offsets = generate_primitive_terrains(terrain_cfg, num_envs)

    # Setup environment.
    sim, env_handles, wrist_actor_handles, terrain_actor_handles, viewer, init_particle_state = \
        create_simulator(gym, num_envs, True, cfg_s, urdfs=['urdf/wrist'] + terrain_files)

    # Move to random init poses.
    num = 10 * num_envs
    configs = np.array([-0.3, -0.3, -0.3]) + (np.random.random([num, 3]) * np.array([0.6, 0.6, 0.6]))
    # configs = np.zeros([num, 3])

    for idx in range(10):
        tool_state_init_ = copy.deepcopy(init_particle_state)
        tool_state_init_ = tool_state_init_.reshape(num_envs, -1, tool_state_init_.shape[-1])
        reset_wrist_offset(gym, sim, env_handles, wrist_actor_handles, tool_state_init_,
                           configs[num_envs * idx:num_envs * (idx + 1)], terrain_offsets + 0.02)

    # Move terrain into position.
    for env_idx in range(num_envs):
        env_handle = env_handles[env_idx]
        terrain_handle = terrain_actor_handles[env_idx][0]

        state = gym.get_actor_rigid_body_states(env_handle, terrain_handle, gymapi.STATE_POS)
        state["pose"]["p"]["z"][:] = 0.0
        gym.set_actor_rigid_body_states(env_handle, terrain_handle, state, gymapi.STATE_POS)

    # Wrap particle
    particle_state_tensor = gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))
    gym.refresh_particle_state_tensor(sim)

    # for _ in range(10000):
    while not gym.query_viewer_has_closed(viewer):
        # gym.simulate(sim)
        # gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == '__main__':
    vis_sampled_terrain()
