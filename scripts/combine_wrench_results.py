import argparse
import os

import mmint_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Combine results from real2sim opt.")
    parser.add_argument("out_dir", type=str, help="Directory to write combined results to.")
    parser.add_argument("in_dirs", nargs='+', default=[], help="Dirs to combine.")
    args = parser.parse_args()

    out_dir = args.out_dir
    mmint_utils.make_dir(out_dir)

    comb_dict = {}
    for in_dir in args.in_dirs:
        comb_dict = mmint_utils.combine_dict(comb_dict,
                                             mmint_utils.load_gzip_pickle(os.path.join(in_dir, "all_results.pkl.gzip")))

    mmint_utils.save_gzip_pickle(comb_dict, os.path.join(out_dir, "all_results.pkl.gzip"))
