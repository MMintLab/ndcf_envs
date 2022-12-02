import argparse
import mmint_utils
import time

from press_simulator import *


def recreate_real_presses():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str, help="Real world data directory.")
    parser.add_argument("--out", "-o", type=str, default=None, help="Directory to store output.")
    parser.add_argument("--viewer", "-v", dest='viewer', action='store_true', help="Use viewer.")
    parser.add_argument("--num_envs", "-e", type=int, default=4,
                        help="Number of environments to simultaneously simulate.")
    args = parser.parse_args()
    use_viewer = args.viewer
    num_envs = args.num_envs

    # Get real press info.
    real_configs, press_zs, _ = load_real_world_examples(args.run_dir)

    # Setup out directory.
    out = args.out
    if out is not None:
        mmint_utils.make_dir(out)

    # Setup environment.
    gym, sim, env_handles, wrist_actor_handles, camera_handles, viewer, init_particle_state = \
        create_simulator(num_envs, use_viewer)

    # Run simulation with the configuration used in the real world.
    start_time = time.time()
    run_sim_loop(gym, sim, env_handles, wrist_actor_handles, camera_handles, viewer, use_viewer,
                 real_configs, press_zs, init_particle_state)
    end_time = time.time()
    run_time = end_time - start_time
    print("Run time: %f" % run_time)

    # Cleanup.
    close_sim(gym, sim, viewer, use_viewer)


if __name__ == '__main__':
    recreate_real_presses()
