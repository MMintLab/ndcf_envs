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

    # Generate terrain.
    terrain_files, terrain_offsets = generate_primitive_terrains(terrain_cfg, num_envs)

    # Setup environment.
    gym, sim, env_handles, wrist_actor_handles, camera_handles, viewer, init_particle_state = \
        create_simulator(num_envs, True, cfg_s, urdfs=['urdf/wrist'] + terrain_files)

    # Move to random init poses.
    num = 10 * num_envs
    configs = np.array([-0.3, -0.3, -0.3]) + (np.random.random([num, 3]) * np.array([0.6, 0.6, 0.6]))

    for idx in range(10):
        tool_state_init_ = copy.deepcopy(init_particle_state)
        tool_state_init_ = tool_state_init_.reshape(num_envs, -1, tool_state_init_.shape[-1])
        reset_wrist_offset(gym, sim, env_handles, wrist_actor_handles, tool_state_init_,
                           configs[num_envs * idx:num_envs * (idx + 1)], terrain_offsets + 0.02)

        for _ in range(10):
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == '__main__':
    vis_sampled_terrain()
