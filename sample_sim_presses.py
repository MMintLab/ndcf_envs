import argparse
import os.path
import pdb

import mmint_utils
import numpy as np

from press_simulator import *


def sample_sim_presses():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", "-o", type=str, help="Directory to store results")
    parser.add_argument("--num", "-n", type=int, default=100, help="Number of presses to simulate.")
    parser.add_argument("--viewer", "-v", dest='viewer', action='store_true', help="Use viewer.")
    parser.add_argument("--num_envs", "-e", type=int, default=4,
                        help="Number of environments to simultaneously simulate.")
    args = parser.parse_args()
    use_viewer = args.viewer
    num_envs = args.num_envs
    num = args.num

    # Sample random configs to run in sim.
    configs = np.array([-0.3, -0.3, -0.3]) + (np.random.random([num, 3]) * np.array([0.6, 0.6, 0.6]))

    # Setup out directory.
    out = args.out
    if out is not None:
        mmint_utils.make_dir(out)

    # Setup environment.
    gym, sim, env_handles, wrist_actor_handles, camera_handles, viewer, init_particle_state = \
        create_simulator(num_envs, use_viewer)

    # Run simulation with sampled configurations.
    results = run_sim_loop(gym, sim, env_handles, wrist_actor_handles, camera_handles, viewer, use_viewer,
                           configs, None, init_particle_state)

    if out is not None:
        for config_idx, result in enumerate(results):
            mmint_utils.save_gzip_pickle(result, os.path.join(out, "config_%d.pkl.gzip" % config_idx))

    # Cleanup.
    close_sim(gym, sim, viewer, use_viewer)


if __name__ == '__main__':
    sample_sim_presses()
