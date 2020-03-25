# gps-rllab3


### Dependencies

python 3.5

This codebase requires a valid installation of `rllab`. Please refer to the [rllab repository](https://github.com/rll/rllab) for installation instructions.

The environments are built in Mujoco 1.31: follow the instructions [here](https://github.com/openai/mujoco-py/tree/0.5) to install Mujoco 1.31 if not already done. You are required to have a Mujoco license to run any of the environments.


### Usage

```bash
source activate rllab3
cd gps/python
python gps/gps_main.py rllab3_ant_example
python gps/gps_main.py rllab3_hopper_example
python gps/gps_main.py rllab3_hopper_traj_opt
python gps/gps_main.py rllab3_cheetah_example
```
