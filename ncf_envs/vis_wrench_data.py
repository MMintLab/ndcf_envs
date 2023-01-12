import argparse
import matplotlib.pyplot as plt
import os
import mmint_utils
import numpy as np


def vis_wrench_data(dataset_dir: str):
    out_files = [f for f in os.listdir(dataset_dir) if "out" in f]
    data_wrenches = []
    for out_file in out_files:
        out_dict = mmint_utils.load_gzip_pickle(os.path.join(dataset_dir, out_file))
        data_wrenches.append(out_dict["wrist_wrench"])
    data_wrenches = np.array(data_wrenches)

    fig, axs = plt.subplots(2, 3)
    fig.suptitle("Simulated Wrench Data")
    for w_idx, label in zip(range(6), ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]):
        axs[w_idx // 3, w_idx % 3].set_title(label)
        axs[w_idx // 3, w_idx % 3].plot(data_wrenches[:, w_idx], c="b")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize dataset wrench data.")
    parser.add_argument("dataset_dir", type=str, help="Dataset directory.")
    args = parser.parse_args()

    vis_wrench_data(args.dataset_dir)
