import argparse
import os
import subprocess


def num_generated(out_dir: str):
    """
    Check out_dir for number of successfully completed examples.
    """
    if os.path.exists(out_dir) is False:
        return 0
    successful_completion_fns = [f for f in os.listdir(out_dir) if "config_" in f]
    return len(successful_completion_fns)


def sample_sim_presses_proc():
    """
    Sample simulated presses in separate process.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("terrain_cfg", type=str, help="Terrain configuration file.")
    parser.add_argument('--cuda_id', type=int, default=0, help="Cuda device id to use.")
    parser.add_argument("--out", "-o", type=str, help="Directory to store results")
    parser.add_argument("--num", "-n", type=int, default=100, help="Number of presses to simulate.")
    parser.add_argument("--viewer", "-v", dest='viewer', action='store_true', help="Use viewer.")
    parser.add_argument("--num_envs", "-e", type=int, default=4,
                        help="Number of environments to simultaneously simulate.")
    parser.add_argument("--cfg_s", type=str, default="cfg/scene.yaml",
                        help="path to scene config yaml file")
    args = parser.parse_args()
    out_dir = args.out

    # Call sample_sim_presses in separate process.
    while True:
        num_generated_so_far = num_generated(out_dir)
        if num_generated_so_far >= args.num:
            break
        num_to_generate = args.num - num_generated_so_far
        cmd = f"python ncf_envs/sample_sim_presses_gen_terrain.py {args.terrain_cfg} --out {out_dir} --num {num_to_generate} --num_envs {args.num_envs} --cfg_s {args.cfg_s} --cuda_id {args.cuda_id} --offset {num_generated_so_far}"
        if args.viewer:
            cmd += " --viewer"
        print(cmd)
        subprocess.call(cmd, shell=True)
    print("Done generating examples.")


if __name__ == '__main__':
    sample_sim_presses_proc()
