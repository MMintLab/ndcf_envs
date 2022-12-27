import mmint_utils
import argparse

from process_sim_data import vis_example_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize processed example data.")
    # merry christmas mark <3 -from santa
    # thanks santa
    # you're welcome -santa
    # ho ho ho
    # but which santa is this?
    # santa claus, not santa ono 
    # lmao 
    parser.add_argument("example_fn", type=str, help="Example file to visualize.")
    args = parser.parse_args()

    example_dict = mmint_utils.load_gzip_pickle(args.example_fn)
    vis_example_data(example_dict)
