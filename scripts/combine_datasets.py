import mmint_utils
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Combine datasets.")
    parser.add_argument("gen_dirs", nargs="+", help="Generated directories.")
    parser.add_argument("out_dir", type=str, help="Out dir to combine outputs to.")
    args = parser.parse_args()

    gen_dirs = args.gen_dirs
    out_dir = args.out_dir

    mmint_utils.make_dir(out_dir)

    out_dir_idx = 0

    for gen_dir in gen_dirs:
        cfg_files = [f for f in os.listdir(gen_dir) if "config" in f]
        terrain_files = [f for f in os.listdir(gen_dir) if "terrain" in f]

        for cfg_file, terrain_file in zip(cfg_files, terrain_files):
            os.symlink(os.path.abspath(os.path.join(gen_dir, cfg_file)),
                       os.path.join(out_dir, "config_%d.pkl.gzip" % out_dir_idx))
            os.symlink(os.path.abspath(os.path.join(gen_dir, terrain_file)),
                       os.path.join(out_dir, "terrain_%d.obj" % out_dir_idx))
            out_dir_idx += 1
