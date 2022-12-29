import argparse
import pdb

from vedo import Plotter, Points
import mmint_utils
import vedo_utils


def vis_object_module_pretraining(data_fn: str):
    pred_dict = mmint_utils.load_gzip_pickle(data_fn)

    query_points = pred_dict["query_points"]
    pred_sdf = pred_dict["pred_sdf"]
    sdf = pred_dict["sdf"]

    plt = Plotter(shape=(1, 3))
    plt.at(0).show(Points(query_points), vedo_utils.draw_origin(), "All Sample Points")
    plt.at(1).show(Points(query_points[sdf <= 0.0], c="b"), vedo_utils.draw_origin(), "Occupied Points (GT)")
    plt.at(2).show(Points(query_points[pred_sdf <= 0.0], c="b"), vedo_utils.draw_origin(), "Occupied Points (Pred)")
    plt.interactive().close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vis object module pretraining.")
    parser.add_argument("data_fn", type=str, help="File with saved predictions.")
    args = parser.parse_args()

    vis_object_module_pretraining(args.data_fn)
