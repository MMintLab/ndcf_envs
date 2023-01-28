import os.path
from press_simulator import *
import argparse
import mmint_utils
from isaacgym.terrain_utils import *
from cfg.terrain import DiscreteTerrainConfig as DTC
from cfg.terrain import WaveTerrainConfig as WTC
from terrain_utils import *

parser = argparse.ArgumentParser()
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


def sample_sim_presses(i, type = 'discrete'):
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
        create_simulator(num_envs, use_viewer, cfg_s, urdfs = ['urdf/wrist', f'urdf/terrain_{type}_{i}'])

    print("setup done")

    out_folder = f'output/{type}/terrain_{i}'
    try:
        os.mkdir(out_folder)
    except:
        print("folder already exists")

    # Run simulation with sampled configurations.
    run_sim_loop_v2(gym, sim, env_handles, wrist_actor_handles, camera_handles, viewer, use_viewer,
                           configs, None, init_particle_state, out_folder)


if __name__ == '__main__':
    N = 10
    i = 0
    type = 'wave'
    while i < N:
        sample_sim_presses(i, type)
        i += 1


    N = 10
    i = 0
    type = 'discrete'
    while i < N:
        sample_sim_presses(i, type)
        i += 1
