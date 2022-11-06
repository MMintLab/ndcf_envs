import argparse
import mmint_utils
from vedo import Plotter, Points, Arrows, Mesh
import utils


def vis_prediction_vs_dataset(data_fn, pred_fn):
    example_dict = mmint_utils.load_gzip_pickle(data_fn)
    pred_data = mmint_utils.load_gzip_pickle(pred_fn)

    # Dataset data.
    all_points = example_dict["query_points"]
    sdf = example_dict["sdf"]
    in_contact = example_dict["in_contact"]
    forces = example_dict["forces"]

    # Prediction data.
    pred_all_points = pred_data[:, :3]
    pred_sdf = pred_data[:, 3]
    pred_contact = pred_data[:, 4] > 0.5
    pred_forces = pred_data[:, 5:]

    plt = Plotter(shape=(1, 2))
    plt.at(0).show(Points(all_points[sdf <= 0.0], c="b"), Points(all_points[in_contact], c="r"),
                   Arrows(all_points[in_contact], all_points[in_contact] + 0.01 * forces[in_contact]),
                   utils.draw_axes(), "Ground Truth")
    plt.at(1).show(Points(pred_all_points[pred_sdf <= 0.0], c="b"), Points(pred_all_points[pred_contact], c="r"),
                   Arrows(pred_all_points[pred_contact],
                          pred_all_points[pred_contact] + 0.01 * pred_forces[pred_contact]),
                   utils.draw_axes(), "Predicted")
    plt.interactive().close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vis pred vs data.")
    parser.add_argument("data", type=str)
    parser.add_argument("pred", type=str)
    args = parser.parse_args()

    vis_prediction_vs_dataset(args.data, args.pred)
