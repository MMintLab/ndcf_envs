import argparse
import pdb

import mmint_utils
from vedo import Plotter, Points, Arrows, Mesh
import utils


def vis_prediction_vs_dataset(pred_fn):
    pred_dict = mmint_utils.load_gzip_pickle(pred_fn)

    # Dataset data.
    all_points = pred_dict["gt"]["query_point"]
    sdf = pred_dict["gt"]["sdf"]
    in_contact = pred_dict["gt"]["in_contact"] > 0.5
    forces = pred_dict["gt"]["force"]

    # Prediction data.
    pred_sdf = pred_dict["pred"]["sdf"]
    pred_contact = pred_dict["pred"]["contact_prob"] > 0.5
    pred_forces = pred_dict["pred"]["contact_force"]

    plt = Plotter(shape=(1, 3))
    plt.at(0).show(Points(all_points[sdf <= 0.0], c="b"), Points(all_points[in_contact], c="r"),
                   Arrows(all_points[in_contact], all_points[in_contact] + 0.01 * forces[in_contact]),
                   utils.draw_axes(), "Ground Truth")
    plt.at(1).show(Points(all_points[pred_sdf <= 0.0], c="b"),
                   utils.draw_axes(), "Predicted Surface")
    plt.at(2).show(Points(all_points[pred_sdf <= 0.0], c="b"), Points(all_points[pred_contact], c="r"),
                   Arrows(all_points[pred_contact],
                          all_points[pred_contact] + 0.01 * pred_forces[pred_contact]),
                   utils.draw_axes(), "Predicted")
    plt.interactive().close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vis pred vs data.")
    parser.add_argument("pred", type=str)
    args = parser.parse_args()

    vis_prediction_vs_dataset(args.pred)
