# ccai
Constrained Control-as-inference 

## Requirements:
The following requirements can be installed via pip
- NumPy
- PyTorch 
Other dependencies:
- Isaac Gym (https://developer.nvidia.com/isaac-gym)
- isaacgym-arm-envs https://github.com/UM-ARM-Lab/isaacgym-arm-envs
- pytorch_kinematics https://github.com/UM-ARM-Lab/pytorch_kinematics/tree/kinematic_hessian
## Install
Navigate to directory and install with `pip install -e .` 

## Scripts:
Example scripts are in the `examples` folder. 

`double_integrator_on_sphere.py` will run the planner for a 3D double integrator constraint to travel on the unit sphere.

`victor_table_surface.py` will run the planner on a task where the robot must move the end-effector to a goal location while maintaining contact with the table. 

`run_victor_wrench_sim.py` will run the planner on a task where the robot must turn a wrench. 

`run_victor_wrench_real.py` Same as above, but for running on the real robot in the lab

The planning configuration files for these examples are found in `config/planning_configs` in `.yaml` format.

### Training generative models to improve planning
`quadrotor_learn_to_sample.py` will train a generative model for the quadrotor example
`victor_table_learn_to_sample.py` will train a generative model for the victor table example

The training configuration files are found in `config/training_configs` in `.yaml` format. Using these configs you can train a diffusion model or a normalizing flow model (either by max-likelihood or flow matching)

Saved models and plots are stored in `data/training`, and the training data for these models is stored in `data/training_data`.
