# Requirements:
Most dependencies are in the `requirements.txt`. However, some need to be installed from github/downloaded online. These should be installed before the dependencies from the requirements.txt
Github:
- Isaac Gym (https://developer.nvidia.com/isaac-gym)
- Pytorch Volumetric (collision_cost_closest_point branch) https://github.com/UM-ARM-Lab/pytorch-volumetric
- Pytorch Kinematics (chain_jacobian_at_links_IK branch) https://github.com/UM-ARM-Lab/pytorch-kinematics
- isaacgym-arm-envs (touchlegro branch) https://github.com/honda-research-institute/isaacgym-arm-envs
- torch_cg https://github.com/sbarrat/torch_cg

After installing the above packages, install other requirements with `xargs -a requirements.txt -n 1 -I {} sh -c 'pip install "{}" || echo "Failed to install {}"'`. Compared to the default `pip install -r requirements.txt`, this command will not halt installation if a single requirement fails, which might occur due to some packages being installed through ROS instead.

# Installation
The ccai package itself also needs to be installed. Install with `pip install -e .` 

# Overview
This document provides instructions on how to generate data for training the task and recovery diffusion models as well as how to deploy them. 

# Task Data Generation
Task data is generated under nominal conditions, without perturbation.
To generate task data for the Allegro hand with Touchlab sensors, run the following:
```
python -u allegro_screwdriver.py touchlegro_screwdriver_task_data_gen > ./logs/touchlegro_screwdriver_task_data_gen.log
```
Data is stored in `data/experiments/<experiment_name>`, where `<experiment_name>` is specified in `touchlegro_screwdriver_task_data_gen.yaml`
# Task Model Training
To train the task diffusion model on the task data, run the following:
```
python train_allegro_screwdriver.py --config touchlegro_screwdriver_task_diffusion.yaml
```
Make sure the `data_directory` field in the yaml file points to task data directory.
# Recovery Data Generation
Recovery data is generated with perturbations applied to the environment. For the screwdriver, this is done using external wrenches that are randomly applied during task execution.
To generate recovery data, run the following:
```
python -u allegro_screwdriver.py touchlegro_screwdriver_recovery_data_gen > ./logs/touchlegro_screwdriver_recovery_data_gen.log
```
# Recovery Model Training
To train the recovery diffusion model on the recovery data, run the following:
```
python train_allegro_screwdriver.py --config touchlegro_screwdriver_recovery_diffusion.yaml
```
# Recovery Model Simulation Evaluation
To evaluate the trained recovery model in simulation, run the following:
```
python -u allegro_screwdriver.py touchlegro_screwdriver_recovery_model_sim_eval > ./logs/touchlegro_screwdriver_recovery_model_sim_eval.log
```
The script will run 50 trials in which perturbations will be applied during task execution. Trials end when the screwdriver is dropped or after 100 steps.
# Recovery Model Hardware Evaluation
There are multiple things that need to be set up before running the `allegro_screwdriver` script.
## Mocap
To receive mocap data, run the following:
```
roslaunch vrpn_client_ros sample.launch server:=<IP>
```
The \<IP> is the IP address of the motion capture server. Last used value: 172.16.2.209
## Hand controller
To launch the hand controller, run the following:
```
roslaunch allegro_hand allegro_hand_modified.launch
```
Press \<h> to put the hand in the "home pose" before launching the `allegro_screwdriver` script.
Alternatively, pressing \<z> will put the hand in gravity compensation mode, where the fingers can be moved around. Joint commands sent in gravity compensation mode will not be respected. This mode is useful for manually adjusting configurations, after which the joint values can be read out using `rostopic echo /allegroHand/joint_states` or `rostopic echo /allegroHand_right/joint_states`.
## Task + Recovery Script
To evaluate the trained recovery model on hardware, run the following:
```
python -u allegro_screwdriver.py touchlegro_screwdriver_recovery_model_hardware_eval > ./logs/touchlegro_screwdriver_recovery_model_hardware_eval.log
```
Once this script is launched, a text prompt (printed in the log file) will ask the user to press \<ENTER> to perform the pregrasp. This is the initial grasp before task execution. The screwdriver should be held upright while this is happening as all fingers will lift off the screwdriver before grasping. The user will then need to press \<ENTER> again after the pregrasp to begin task execution.


# Potential Improvements
At the time of this writing, the turning and thumb/middle regrasp behavior work reliably. The execution of the index regrasp works as well, but there is an issue where the diffusion model does not plan to execute that regrasp as often as it should. This could be due to multiple issues:
- Initial hand configuration: When running the task, we start the hand in an initial configuration. The choice of this configuration is important in that when performing the task, the initial grasps will be based on the initial configuration, modified locally to ensure contact. These contact points on the object during task execution will propagate through the method as the recovery will try to make contact on the object in the same points as contact was made during nominal task execution. This configuration is defined in the `default_dof_pos` variable in `allegro_screwdriver.py`. If changed, the value also needs to be changed for the hardware in the `UMichCollabration_DexManip/Allegro-Hand-Controller-DIME-prehensile_manip/src/allegro_hand_parameters/initial_position.yaml`. The finger order in both files is index, middle, ring, thumb. The recommendation is to find a reasonable looking initial configuration on hardware using gravity compensation mode and set the simulated state accordingly. 
- Data imbalance: When generating training data with default parameters, approximately 5% of trajectories are index regrasps and 95% are thumb/middle regrasps. While ideally the diffusion model should be able to diffuse index regrasps when necessary due to the initial state conditioning, this imbalance could still pose an issue. Some dataset balancing techniques at training time were explored and had limited success. It would probably be better to change how perturbations are applied to organically increase the percentage of index regrasps in the training dataset. Currently, these perturbations take the form of random wrenches, or pokes, applied to the screwdriver during task execution. 2 parameters control how these are applied: `rand_pct`, which specifies the probability that a poke will be applied each timestep, and `random_force_magnitude`, which specifies the maximum force magnitude of the poke. As these are changed, the types of states the system visits will change as well.
- Diffusion model training: The diffusion models are trained by default for 15000 epochs. it is possible that with longer training, the model performance could improve in sampling the index regrasp trajectories.