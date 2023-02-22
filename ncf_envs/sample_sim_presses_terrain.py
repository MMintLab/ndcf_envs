import os.path
from press_simulator import *
import argparse
import mmint_utils
from isaacgym.terrain_utils import *
from terrain_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("terrain_file", type=str, help="Terrain file to load.")
parser.add_argument("--out", "-o", type=str, help="Directory to store results")
parser.add_argument("--num", "-n", type=int, default=100, help="Number of presses to simulate.")
parser.add_argument("--viewer", "-v", dest='viewer', action='store_true', help="Use viewer.")
parser.add_argument("--num_envs", "-e", type=int, default=4,
                    help="Number of environments to simultaneously simulate.")
parser.add_argument("--cfg_s", type=str, default=4,
                    help="path to scene config yaml file")
args = parser.parse_args()
use_viewer = args.viewer
num_envs = args.num_envs
num = args.num
terrain_file = args.terrain_file


def sample_sim_presses(i, type='discrete'):
    cfg_s = mmint_utils.load_cfg(args.cfg_s)

    # Sample random configs to run in sim.
    configs = np.array([-0.3, -0.3, -0.3]) + (np.random.random([num, 3]) * np.array([0.6, 0.6, 0.6]))

    # Setup out directory.
    out = args.out
    if out is not None:
        mmint_utils.make_dir(out)
    print("make directory")

    # Setup environment.
    gym, sim, env_handles, wrist_actor_handles, camera_handles, viewer, init_particle_state = \
        create_simulator(num_envs, use_viewer, cfg_s, urdfs=['urdf/wrist', terrain_file])

    print("setup done")

    # out_folder = f'{out}/{type}/terrain_{i}'
    # try:
    #     os.mkdir(out_folder)
    # except:
    #     print("folder already exists")

    # Run simulation with sampled configurations.
    # input("ready")
    run_sim_loop_v2(gym, sim, env_handles, wrist_actor_handles, camera_handles, viewer, use_viewer,
                    configs, None, init_particle_state, out)


if __name__ == '__main__':
    # N = 2
    # i = 0
    # type = 'wave'
    # while i < N:
    # sample_sim_presses(1, type)
    # i += 1

    N = 2
    i = 0
    type = 'discrete'
    # while i < N:
    sample_sim_presses(1, type)
    # i += 1
