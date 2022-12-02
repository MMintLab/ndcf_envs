import argparse

import mmint_utils
import numpy as np
import matplotlib.pyplot as plt

from ncf_envs.press_simulator import load_real_world_examples


def real_vs_sim():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str, help="Real world data directory.")
    parser.add_argument("sim_file", type=str, help="File where simulated results are stored.")
    args = parser.parse_args()

    # Get real press info.
    _, _, real_wrenches = load_real_world_examples(args.run_dir)
    real_wrenches = np.array(real_wrenches)

    # Load simulated data.
    sim_results = mmint_utils.load_gzip_pickle(args.sim_file)

    youngs = list(sim_results.keys())
    errors_all = []
    errors_mean = []

    for young in youngs:
        sim_wrenches = []
        for res in sim_results[young]:
            sim_wrenches.append(res["wrist_wrench"])
        sim_wrenches = np.array(sim_wrenches)

        wrench_errors = np.linalg.norm(real_wrenches - sim_wrenches, axis=1)
        errors_all.append(wrench_errors)
        errors_mean.append(np.mean(wrench_errors))

    idx = np.argmin(errors_mean)
    print("Best: %f, loss: %f" % (youngs[idx], errors_mean[idx]))

    plt.plot(youngs, errors_mean)
    for y, er_all in zip(youngs, errors_all):
        plt.scatter([y] * len(er_all), er_all)
    plt.xlabel("Young's Modulus")
    plt.ylabel("L2 Wrench Error")
    plt.show()


if __name__ == '__main__':
    real_vs_sim()
