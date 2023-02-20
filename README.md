# ccai
Constrained Control-as-inference 

## Requirements:
The following requirements can be installed via pip
- NumPy
- PyTorch 
Other dependencies:
- Isaac Gym (https://developer.nvidia.com/isaac-gym)
- isaacgym-arm-envs https://github.com/UM-ARM-Lab/isaacgym-arm-envs

## Install
Navigate to directory and install with `pip install -e .` 

## Scripts:
Example scripts are in the `examples` folder. 
`double_integrator_on_sphere.py` will run the planner for a 3D double integrator constraint to travel on the unit sphere.

`victor_table_surface.py` will run the planner on a task where the robot must move the end-effector to a goal location while maintaining contact with the table. On launch the user should adjust the camera and move via ctrl-c. Once planning is completed the script will prompt the user for a key-press to execute. 

`run_victor_wrench_sim.py` will run the planner on a task where the robot must turn a wrench

`run_victor_wrench_real.py` Same as above, but for running on the real robot in the lab