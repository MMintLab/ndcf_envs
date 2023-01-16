import argparse
import mmint_utils

from ncf_envs.press_simulator import *


def optim_real_presses_de():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str, help="Real world data directory.")
    parser.add_argument("--viewer", "-v", dest='viewer', action='store_true', help="Use viewer.")
    parser.add_argument("--num_envs", "-e", type=int, default=4,
                        help="Number of environments to simultaneously simulate.")
    args = parser.parse_args()
    use_viewer = args.viewer

    # Load real world run data.
    real_configs, press_zs, _ = load_real_world_examples(args.run_dir)

    # Setup environment.
    gym, sim, env_handle, wrist_actor_handle, camera_handle, viewer, init_particle_state = create_simulator(use_viewer)

    # TODO: Update to address multiple examples case.
    def optim_func(soft_params):
        # Change actor soft materials.
        soft_mat = gym.get_actor_soft_materials(env_handle, wrist_actor_handle)[0]
        soft_mat.youngs = soft_params[0]
        soft_mat.poissons = soft_params[1]
        gym.set_actor_soft_materials(env_handle, wrist_actor_handle, [soft_mat])

        # TODO: Run simulation with the configuration used in the real world.

        # wrench_loss = np.linalg.norm(real_wrench - sim_wrench)
        wrench_loss = 0.0
        return wrench_loss

    optim_func([4.39117435e+04, 3.81013133e-01])

    # Cleanup.
    close_sim(gym, sim, viewer, use_viewer)


def optim_real_presses_grid_search():
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

    def optim_func(soft_params):
        # Change actor soft materials.
        for env_handle, wrist_actor_handle in zip(env_handles, wrist_actor_handles):
            soft_mat = gym.get_actor_soft_materials(env_handle, wrist_actor_handle)[0]
            soft_mat.youngs = soft_params[0]
            soft_mat.poissons = 0.1
            gym.set_actor_soft_materials(env_handle, wrist_actor_handle, [soft_mat])

        # Run simulation with the configuration used in the real world.
        results = run_sim_loop(gym, sim, env_handles, wrist_actor_handles, camera_handles, viewer, use_viewer,
                               real_configs, press_zs, init_particle_state)

        return results

    # Setup soft params.
    youngs = np.arange(1e3, 2e4 + 1.0, 1e3)
    youngs_results = {young: optim_func([young]) for young in youngs}

    if out is not None:
        mmint_utils.save_gzip_pickle(youngs_results, os.path.join(out, "all_results.pkl.gzip"))


if __name__ == '__main__':
    optim_real_presses_grid_search()
