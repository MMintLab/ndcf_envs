import argparse

from tqdm import trange
from isaacgym import gymapi
from ncf_envs.terrain.primitives.generate_primitive_terrain import generate_primitive_terrains
from press_simulator import *
import mmint_utils


def sample_sim_presses():
    parser = argparse.ArgumentParser()
    parser.add_argument("terrain_cfg", type=str, help="Terrain configuration file.")
    parser.add_argument('--cuda_id', type=int, default=0, help="Cuda device id to use.")
    parser.add_argument("--out", "-o", type=str, help="Directory to store results")
    parser.add_argument("--num", "-n", type=int, default=100, help="Number of presses to simulate.")
    parser.add_argument("--viewer", "-v", dest='viewer', action='store_true', help="Use viewer.")
    parser.add_argument("--num_envs", "-e", type=int, default=4,
                        help="Number of environments to simultaneously simulate.")
    parser.add_argument("--cfg_s", type=str, default="cfg/scene.yaml",
                        help="path to scene config yaml file")
    args = parser.parse_args()
    use_viewer = args.viewer
    num_envs = args.num_envs
    num = args.num
    cuda_id = args.cuda_id
    assert num % num_envs == 0, "Num sims to run should be divisible by num environments."

    # Acquire singleton object.
    gym = gymapi.acquire_gym()

    terrain_cfg = mmint_utils.load_cfg(args.terrain_cfg)
    cfg_s = mmint_utils.load_cfg(args.cfg_s)

    # Sample random configs to run in sim.
    configs = np.array([-0.3, -0.3, -0.3]) + (np.random.random([num, 3]) * np.array([0.6, 0.6, 0.6]))

    # Setup out directory.
    out = args.out
    if out is not None:
        mmint_utils.make_dir(out)

    # Generate terrain.
    terrain_files, terrain_meshes, terrain_offsets = generate_primitive_terrains(terrain_cfg, num, out_dir=out)

    # Setup environment.
    sim, env_handles, wrist_actor_handles, terrain_actor_handles, viewer, init_particle_state = \
        create_simulator(gym, num_envs, use_viewer, cfg_s, urdfs=['urdf/wrist'] + terrain_files, cuda_id=cuda_id)

    # Run simulation with sampled configurations.
    run_sim_loop(gym, sim, env_handles, wrist_actor_handles, terrain_actor_handles, [], viewer, use_viewer,
                 configs, None, init_particle_state, terrain_offsets + 0.005, out)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == '__main__':
    sample_sim_presses()
