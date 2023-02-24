import argparse
from isaacgym import gymtorch
from ncf_envs.terrain.primitives.generate_primitive_terrain import generate_primitive_terrain
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
    terrain_file, terrain_offset = generate_primitive_terrain(terrain_cfg)
    print(terrain_file)

    # Setup environment.
    gym, sim, env_handles, wrist_actor_handles, camera_handles, viewer, init_particle_state = \
        create_simulator(num_envs, True, cfg_s, urdfs=['urdf/wrist', terrain_file])

    # Move to random init pose.
    configs = np.zeros([1, 3])
    tool_state_init_ = copy.deepcopy(init_particle_state)
    tool_state_init_ = tool_state_init_.reshape(num_envs, -1, tool_state_init_.shape[-1])
    reset_wrist_offset(gym, sim, env_handles, wrist_actor_handles, tool_state_init_, configs, terrain_offset + 0.03)

    while True:
        # gym.simulate(sim)
        # gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)


if __name__ == '__main__':
    vis_sampled_terrain()
