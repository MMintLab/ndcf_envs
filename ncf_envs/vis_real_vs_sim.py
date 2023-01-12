import argparse

import real_utils
import numpy as np
from vedo import Plotter, Arrow, Points

import mmint_utils

import utils
import vedo_utils


def vis_real_vs_sim(real_dir: str, real_example: str, sim_fn: str):
    sim_dict = mmint_utils.load_gzip_pickle(sim_fn)
    real_dict = real_utils.load_observation_from_file(real_dir, real_example)

    # Load simulated data.
    simulated_points = sim_dict["nodal_coords_wrist"]
    simulated_points_vedo = Points(simulated_points, c="black")

    simulated_wrench = sim_dict["wrist_wrench"]
    vedo_sim_force = Arrow((0, 0, 0), 0.00001 * simulated_wrench[:3], c="b")
    vedo_sim_torque = Arrow((0, 0, 0), 0.01 * simulated_wrench[3:], c="y")

    # Load real data.
    real_points_w = real_dict["visual"]["photoneo"][0]
    wrist_pose = real_dict["proprioception"]["ee_pose"][0]
    wrist_pose = np.array(
        [wrist_pose[0][0], wrist_pose[0][1], wrist_pose[0][2], wrist_pose[1][3], wrist_pose[1][0], wrist_pose[1][1],
         wrist_pose[1][2]])
    w_T_wrist = utils.pose_to_matrix(wrist_pose)
    wrist_T_w = np.linalg.inv(w_T_wrist)
    real_points = utils.transform_pointcloud(real_points_w, wrist_T_w)
    real_points_vedo = Points(real_points[:, :3], c="black")

    real_wrench = np.array(real_dict["tactile"]["ati_wrench"][-1][0])
    vedo_real_force = Arrow((0, 0, 0), 0.1 * real_wrench[:3], c="b")
    vedo_real_torque = Arrow((0, 0, 0), real_wrench[3:], c="y")

    # Plot data side by side.
    plt = Plotter(shape=(1, 2))
    plt.at(0).show(simulated_points_vedo, vedo_utils.draw_origin(), "Sim",
                   vedo_sim_force, vedo_sim_torque)
    plt.at(1).show(real_points_vedo, vedo_utils.draw_origin(), "Real",
                   vedo_real_force, vedo_real_torque)
    plt.interactive().close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vis real vs sim data.")
    parser.add_argument("real_dir", type=str, help="Real data dir.")
    parser.add_argument("real_example", type=str, help="Real example name to load.")
    parser.add_argument("sim_fn", type=str, help="Simulated data file.")
    args = parser.parse_args()

    vis_real_vs_sim(args.real_dir, args.real_example, args.sim_fn)
