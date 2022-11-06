import argparse
import real_utils
import utils
from vedo import Plotter, Points, Arrows, Mesh


def vis_real_data(data_dir, example_name):
    obs_dict = real_utils.load_observation_from_file(data_dir, example_name)

    # Pull out data.
    pointcloud = obs_dict["visual"]["photoneo"]
    ee_pose = obs_dict["proprioception"]["ee_pose"]
    ee_pose = [ee_pose[0][0], ee_pose[0][1], ee_pose[0][2], ee_pose[1][0], ee_pose[1][1], ee_pose[1][2], ee_pose[1][3]]

    # Transform pointcloud to EE pose.
    pointcloud_ee = utils.transform_pointcloud(pointcloud, utils.pose_to_matrix(ee_pose))

    plt = Plotter(shape=(1, 1))
    plt.at(0).show(Points(pointcloud_ee), utils.draw_axes(), "Real Data")
    plt.interactive().close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vis real data.")
    parser.add_argument("data_dir", type=str, help="Dataset dir.")
    parser.add_argument("example_name", type=str, help="Example name.")
    args = parser.parse_args()

    vis_real_data(args.data_dir, args.example_name)
