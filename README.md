# ccai
Constrained Control-as-inference 

##Requirements:
The following requirements can be installed via pip
- NumPy
- PyTorch 
- nullspace_optimizer (use command `pip install nullspace_optimizer[osqp,colored,matplotlib]`)
Other dependencies:
- Isaac Gym (https://developer.nvidia.com/isaac-gym)
- isaacgym-arm-envs https://github.com/UM-ARM-Lab/isaacgym-arm-envs

##Scripts:

`victor_example_task_space.py` will run the planner on a task where the robot must move the end-effector to a goal location while maintaining contact with the table. On launch the user should adjust the camera and move via ctrl-c. Once planning is completed the script will prompt the user for a key-press to execute. 

`victor_example_joint_space.py` is the same as above but planning occurs in joint configuration space rather than task space. 

`victor_block_pushing.py` this is a WIP. There is now a block on the table and the robot will iteratively attempt to push the block to a goal location and train a dynamics model of pushing dynamics, while maintaining contact with the table. 
