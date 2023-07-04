# NDCF Environments

Isaac Gym (Preview) environments for the Neural Deforming Contact Fields project.

## Setup

We use two environments: one for running the simulation and another for post-processing the resulting
meshes, pointclouds, wrenches etc. into data ready to train our networks.

### Simulation Environment

Follow Nvidia's instructions [here](https://developer.nvidia.com/isaac-gym) to install Isaac Gym to a conda environment.
Other required packages can be found in `sim_environment.yaml`.

### Processing Environment

Setup processing conda environment with `proc_environment.yaml`.

## Running

### Running the simulation

To collect a new dataset of simulated presses, use your simulation environment `python` to run the following:

```
python ncf_envs/sample_sim_presses_gen_terrain_proc.py cfg/primitives/ridge.yaml -o <out> -n <num> -e 1
```

This will run the simulation for `<num>` episodes and save the results to `<out>`. Switch out `cfg/primitives/ridge.yaml`
for other configuration files found in `cfg/primitives/` to run with different environments. This script does wraps Isaac
Gym in a process to catch simulation failures that are tricky to detect and automatically starts from where it left off.

### Post-Processing

Once the dataset has been collected, use your processing env `python` to run the following to generate
SDF samples, pointclouds, meshes, etc. to be used during training and evaluation.

```
python ncf_envs/process_sim_data.py <output directory used by simulator> <path to tool .tet file used in simulation>
```

Default path for sponge is: `assets/meshes/sponge/sponge_2/sponge_2.tet`
