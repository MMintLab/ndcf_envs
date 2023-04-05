import mmint_utils
import os
import argparse

import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Combine datasets.")
    parser.add_argument("gen_dirs", nargs="+", help="Generated directories.")
    parser.add_argument("out_dir", type=str, help="Out dir to combine outputs to.")
    parser.add_argument("--num", "-n", type=int, default=np.inf, help="Limit number of configs used from each dir.")
    args = parser.parse_args()

    gen_dirs = args.gen_dirs
    out_dir = args.out_dir

    mmint_utils.make_dir(out_dir)

    out_dir_idx = 0

    for gen_dir in gen_dirs:
        num_cfg_files = len([f for f in os.listdir(gen_dir) if "config" in f])
        num_terrain_files = len([f for f in os.listdir(gen_dir) if "terrain" in f])
        assert num_cfg_files == num_terrain_files, "Number of config files and terrain files must be equal."

        for cfg_idx in range(min(num_cfg_files, args.num)):
            cfg_file = "config_%d.pkl.gzip" % cfg_idx
            terrain_file = "terrain_%d.obj" % cfg_idx
            os.symlink(os.path.abspath(os.path.join(gen_dir, cfg_file)),
                       os.path.join(out_dir, "config_%d.pkl.gzip" % out_dir_idx))
            os.symlink(os.path.abspath(os.path.join(gen_dir, terrain_file)),
                       os.path.join(out_dir, "terrain_%d.obj" % out_dir_idx))
            out_dir_idx += 1
