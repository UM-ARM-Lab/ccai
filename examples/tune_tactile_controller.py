#!/usr/bin/env python3
"""
Ray Tune hyperparameter tuning for tactile controller parameters.
Tunes K_e, w_q, w_p, w_f, w_u, and w_ori parameters for optimal turning performance.
Uses the actual do_trial function from allegro_screwdriver.py for real controller testing.
"""

import os
import sys
import yaml
import pathlib
import tempfile
import numpy as np
from copy import deepcopy
import time
from typing import Dict, Any

# Ray Tune imports
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

# Add the ccai path to sys.path
CCAI_PATH = pathlib.Path(__file__).parent.parent.absolute()
sys.path.append(str(CCAI_PATH))

# Import Isaac Gym modules first (before PyTorch)
try:
    from isaac_victor_envs.tasks.allegro import AllegroScrewdriverTurningEnv, AllegroValveTurningEnv
    from isaac_victor_envs.utils import get_assets_dir
    import pytorch_kinematics as pk
    ISAAC_GYM_AVAILABLE = True
    print("✓ Successfully imported Isaac Gym environment")
except ImportError as e:
    print(f"Warning: Isaac Gym not available: {e}")
    ISAAC_GYM_AVAILABLE = False

# Import required modules with error handling
# from examples.allegro_screwdriver import do_trial, CCAI_PATH
from examples.allegro_valve_turning import do_trial, CCAI_PATH
from ccai.models.management.model_manager import ModelManager
ALLEGRO_AVAILABLE = True
import torch

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def setup_environment_and_models(config):
    """
    Set up the environment and load models exactly like allegro_screwdriver.py does.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (env, trajectory_sampler, trajectory_sampler_orig, classifier, chain)
    """
    if not ISAAC_GYM_AVAILABLE:
        raise ImportError("Isaac Gym environment required for real controller testing")
    
    # Force simulation mode for tuning
    config['mode'] = 'simulation'
    config['visualize'] = False
    
    num_envs = 1
    
    # Create environment
    # env = AllegroScrewdriverTurningEnv(
    #     num_envs, 
    #     control_mode='joint_impedance',
    #     use_cartesian_controller=False,
    #     viewer=False,  # No visualization during tuning
    #     steps_per_action=60,
    #     friction_coefficient=2.5,
    #     device=config['sim_device'],
    #     video_save_path=None,
    #     joint_stiffness=config['kp'],
    #     fingers=config['fingers'],
    #     gradual_control=False,
    #     gravity=True, 
    #     randomize_obj_start=config.get('randomize_obj_start', False),
    #     randomize_rob_start=config.get('randomize_rob_start', False),
    #     external_wrench_perturb=config.get('external_wrench_perturb', False),
    #     force_sensors=config.get('tactile_controller', False)
    # )

    env = AllegroValveTurningEnv(num_envs, control_mode='joint_impedance',
                                        use_cartesian_controller=False,
                                        viewer=config['visualize'],
                                        steps_per_action=60,
                                        friction_coefficient=config['friction_coefficient'] * 1.0,
                                        device=config['sim_device'],
                                        video_save_path=None,
                                        joint_stiffness=config['kp'],
                                        fingers=config['fingers'],
                                        gravity=True, 
                                        randomize_obj_start=config['randomize_obj_start'],
                                        randomize_rob_start=config['randomize_rob_start']
                                        )
    
    # Build kinematic chain
    asset = f'{get_assets_dir()}/xela_models/allegro_hand_right.urdf'
    chain = pk.build_chain_from_urdf(open(asset).read())
    
    # Load models using ModelManager
    params = config.copy()
    params.pop('controllers', None)
    if 'controllers' in config and 'csvgd' in config['controllers']:
        params.update(config['controllers']['csvgd'])
    
    model_manager = ModelManager(config, params, CCAI_PATH)
    trajectory_sampler, trajectory_sampler_orig, classifier = model_manager.load_trajectory_samplers()

    
    return env, trajectory_sampler, trajectory_sampler_orig, classifier, chain


def run_tactile_trial_with_environment(config_params: Dict[str, Any], env, trajectory_sampler, 
                                     trajectory_sampler_orig, classifier, chain, config, params) -> Dict[str, float]:
    """
    Run a single trial using the actual do_trial function with tactile controller parameters.
    Uses provided environment and models.
    
    Args:
        config_params: Dictionary containing hyperparameters to tune
        env: Isaac Gym environment
        trajectory_sampler: Trajectory sampler
        trajectory_sampler_orig: Original trajectory sampler
        classifier: Classifier model
        chain: Kinematic chain
        config: Base configuration
        params: Base parameters
        
    Returns:
        Dictionary with performance metrics
    """
    
    # Update config with tuned parameters
    trial_config = config.copy()
    trial_config.update(config_params)
    
    # Update params with tuned parameters
    trial_params = params.copy()
    trial_params.update(config_params)
    
    # Create temporary directory for trial data
    with tempfile.TemporaryDirectory() as temp_dir:
        fpath = pathlib.Path(temp_dir) / 'trial_data'
        fpath.mkdir(exist_ok=True)
        
        # Reset environment
        env.reset()
        
        # Record initial state
        initial_state = env.get_state()
        # initial_ori = initial_state['screwdriver_ori'][0] if 'screwdriver_ori' in initial_state else torch.zeros(3)
        # initial_yaw = initial_ori[2].item() if len(initial_ori) > 2 else 0.0


        initial_ori = initial_state['valve'][0] if 'valve' in initial_state else torch.zeros(3)
        initial_yaw = initial_ori[0].item()
        
        # Run the actual do_trial function
        yaw_change, dropped = do_trial(
            env=env,
            params=trial_params,
            fpath=fpath,
            sim_viz_env=None,
            ros_copy_node=None,
            inits_noise=None,
            noise_noise=None,
            sim=None,
            seed=0,
            proj_path=None,
            perturb_this_trial=False,
            trajectory_sampler=trajectory_sampler,
            trajectory_sampler_orig=trajectory_sampler_orig,
            config=trial_config,
            classifier=classifier
        )
                
        completed = not dropped
        
        # Performance score (to be maximized)
        performance_score = -yaw_change * float(completed)
        
        return {
            'completion_rate': float(completed),
            'performance_score': performance_score,
            'yaw_change': yaw_change,
            'initial_yaw': initial_yaw,
            'dropped': dropped
        }


def run_tactile_experiment_with_environment(config_params: Dict[str, Any], env, trajectory_sampler, 
                                          trajectory_sampler_orig, classifier, chain, config, params) -> Dict[str, float]:
    """
    Run multiple trials and aggregate results using provided environment.
    
    Args:
        config_params: Dictionary containing hyperparameters to tune
        env: Isaac Gym environment
        trajectory_sampler: Trajectory sampler
        trajectory_sampler_orig: Original trajectory sampler
        classifier: Classifier model
        chain: Kinematic chain
        config: Base configuration
        params: Base parameters
        
    Returns:
        Dictionary with aggregated performance metrics
    """
    num_trials = 15  # Run 3 trials per hyperparameter configuration
    all_results = []
    
    for trial_idx in range(num_trials):
        trial_result = run_tactile_trial_with_environment(
            config_params, env, trajectory_sampler, trajectory_sampler_orig, 
            classifier, chain, config, params
        )
        all_results.append(trial_result)
        
        # Early stopping if trial failed catastrophically
        if trial_result['performance_score'] < -5.0:
            break
    
    if not all_results:
        return {
            'success_rate': 0.0,
            'completion_rate': 0.0,
            'performance_score': -10.0
        }
    
    # Aggregate results
    mean_completion = np.mean([r['completion_rate'] for r in all_results])
    mean_performance = np.mean([r['performance_score'] for r in all_results])
    
    return {
        'completion_rate': mean_completion,
        'performance_score': mean_performance,
        'num_trials_completed': len(all_results)
    }

def objective(config_params):
    """Objective function for Ray Tune optimization."""
    # Create environment and models within this worker process
    # Load base config
    # base_config_path = CCAI_PATH / 'examples/config/screwdriver/allegro_screwdriver_diff_tactile_control.yaml'
    # if not base_config_path.exists():
    #     base_config_path = CCAI_PATH / 'examples/config/screwdriver/allegro_screwdriver_diff_only.yaml'

    base_config_path = CCAI_PATH / 'examples/config/valve/allegro_valve_csvto_diff_tactile_control.yaml'
    if not base_config_path.exists():
        base_config_path = CCAI_PATH / 'examples/config/valve/allegro_valve_csvto_diff_only.yaml'
    
    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    
    # Set tuning-specific overrides
    config['num_episodes'] = 1  # Only run 1 episode per trial
    config['visualize'] = False
    config['mode'] = 'simulation'
    config['tactile_controller'] = True  # Enable tactile controller
    config['experiment_name'] = 'tune_tactile_controller_valve_csvto_diff'
    
    # Ensure required fields are set
    if 'recovery_controller' not in config:
        config['recovery_controller'] = 'csvgd'
    if 'sim_device' not in config:
        config['sim_device'] = 'cpu'
    if 'goal' not in config:
        config['goal'] = -1.5
    
    # Update config with tuned parameters
    config.update(config_params)
    
    # Create environment and models within this worker
    env, trajectory_sampler, trajectory_sampler_orig, classifier, chain = setup_environment_and_models(config)
    
    # Prepare parameters for do_trial
    params = config.copy()
    params.pop('controllers', None)
    if 'controllers' in config and 'csvgd' in config['controllers']:
        params.update(config['controllers']['csvgd'])
        
    params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Add required fields
    params['controller'] = 'csvgd'
    params['valve_goal'] = torch.tensor([0, 0, float(config['goal'])]).to(device=params['device'])
    params['chain'] = chain.to(device=params['device'])
    params['obj_dof'] = 3
    
    # Run the actual experiment with real environment
    results = run_tactile_experiment_with_environment(
        config_params, env, trajectory_sampler, trajectory_sampler_orig,
        classifier, chain, config, params
    )
    
    # Report results to Ray Tune using the correct API
    from ray.air import session
    session.report({
        'performance_score': results['performance_score'],
        'completion_rate': results['completion_rate'],
        'num_trials_completed': results['num_trials_completed']
    })

def main():
    """Main hyperparameter tuning function."""
    
    print("Starting tactile controller hyperparameter tuning...")
    print(f"Allegro modules available: {ALLEGRO_AVAILABLE}")
    print(f"Isaac Gym available: {ISAAC_GYM_AVAILABLE}")
    
    if not ALLEGRO_AVAILABLE:
        print("Warning: Running in mock mode - allegro modules not available")
    if not ISAAC_GYM_AVAILABLE:
        print("Warning: Isaac Gym not available, some functionality may be limited")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Define search space for tactile controller hyperparameters
    search_space = {
        'K_e': tune.loguniform(100.0, 500.0),      # Environment stiffness
        'w_q': tune.loguniform(1, 50.0),        # Position weight  
        'w_p': tune.loguniform(1, 10.0),         # Velocity weight
        'w_f': tune.loguniform(1, 50.0),         # Force weight
        'w_u': tune.loguniform(1, 10.0),         # Control effort weight
        'w_ori': tune.loguniform(0.1, 1.0),       # Orientation weight
    }
    
    # Configure ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="performance_score",
        mode="max",  # Maximize performance score
        max_t=1,     # Each config gets 1 evaluation (we handle multiple trials internally)
        grace_period=1,
        reduction_factor=2
    )
    
    # Configure reporter
    reporter = CLIReporter(
        metric_columns=["performance_score", "completion_rate", "num_trials_completed"],
        max_report_frequency=30
    )
    
    # Run hyperparameter tuning
    num_samples = 680 if ALLEGRO_AVAILABLE and ISAAC_GYM_AVAILABLE else 3  # Fewer samples if in mock mode
    print(f"\nRunning {num_samples} hyperparameter configuration(s)")
    print("Note: Environment and models instantiated once before tuning")
    
    # Create absolute path for storage
    storage_path = pathlib.Path("./ray_results").resolve()
    
    search_alg = HyperOptSearch(
        space=search_space,
        metric="performance_score",
        mode="max",
        random_state_seed=42,
        # points_to_evaluate=[
        #     {
        #         'K_e': 113.058390,
        #         'w_q': 49.967775,
        #         'w_p': 2.994464,
        #         'w_f': 5.410153,
        #         'w_u': 4.369370,
        #         'w_ori': 0.263349
        #     }
        # ]
    )
    
    # search_alg.restore_from_dir(
    #     pathlib.Path("./ray_results/tactile_controller_tuning_diff_partial_patch/")
    # )
    
    # # Create a wrapper function that passes the pre-instantiated objects
    # def objective_wrapper(config_params):
    #     return objective(config_params, env, trajectory_sampler, trajectory_sampler_orig, 
    #                    classifier, chain, base_config, base_params)
    
    analysis = tune.run(
        objective,
        # config=search_space,
        num_samples=num_samples,
        # scheduler=scheduler,
        progress_reporter=reporter,
        name="tactile_controller_tuning_valve_csvto_diff_partial_patch",
        storage_path=str(storage_path),  # Use absolute path as string
        resources_per_trial={"cpu": 8, "gpu": .5},  # CPU only for stability
        max_failures=5,  # Allow more failures since we're doing complex trials
        raise_on_failed_trial=False,  # Don't crash on individual trial failures
        search_alg=search_alg,
        resume=False,
        # max_concurrent_trials=4  # Ensure only 1 configuration is tested at a time
    )
    
    # Get best hyperparameters
    best_config = analysis.get_best_config(metric="performance_score", mode="max")
    best_result = analysis.get_best_trial(metric="performance_score", mode="max").last_result
    
    print("\n" + "="*60)
    print("TACTILE CONTROLLER HYPERPARAMETER TUNING RESULTS")
    print("="*60)
    print(f"Best performance score: {best_result['performance_score']:.4f}")
    print(f"Best completion rate: {best_result['completion_rate']:.4f}")
    print(f"Trials completed: {best_result.get('num_trials_completed', 'N/A')}")
    print("\nBest hyperparameters:")
    for param, value in best_config.items():
        print(f"  {param}: {value:.6f}")
    
    # Save best config to file
    output_dir = pathlib.Path("./tuning_results")
    output_dir.mkdir(exist_ok=True)
    
    # base_config_path = CCAI_PATH / 'examples/config/screwdriver/allegro_screwdriver_diff_tactile_control.yaml'
    # if not base_config_path.exists():
    #     base_config_path = CCAI_PATH / 'examples/config/screwdriver/allegro_screwdriver_diff_only.yaml'

    base_config_path = CCAI_PATH / 'examples/config/valve/allegro_valve_csvto_diff_tactile_control.yaml'
    if not base_config_path.exists():
        base_config_path = CCAI_PATH / 'examples/config/valve/allegro_valve_csvto_diff_only.yaml'

    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Save best config as YAML
    best_config_path = output_dir / f"best_tactile_controller_config_{config['experiment_name']}.yaml"
    with open(best_config_path, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False)
    
    # Save complete results summary
    results_summary = {
        'best_config': best_config,
        'best_metrics': {
            'performance_score': best_result['performance_score'],
            'completion_rate': best_result['completion_rate'],
            'num_trials_completed': best_result.get('num_trials_completed', 'N/A')
        },
        'search_space': {param: str(space) for param, space in search_space.items()},
        'total_trials': len(analysis.trials),
        'allegro_available': ALLEGRO_AVAILABLE,
        'isaac_gym_available': ISAAC_GYM_AVAILABLE,
        'test_mode': False,
        'note': 'Environment and models instantiated once before tuning'
    }
    
    results_path = output_dir / f"tuning_results_summary_{config['experiment_name']}.yaml"
    with open(results_path, 'w') as f:
        yaml.dump(results_summary, f, default_flow_style=False)
    
    print(f"\nResults saved to:")
    print(f"  Best config: {best_config_path}")
    print(f"  Summary: {results_path}")
    print(f"  Full results: ./ray_results/tactile_controller_tuning/")
    
    print("\n✅ Tuning completed successfully!")
    
    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    main() 
    # import json
    # def get_max(path):
    #     exp = json.loads(open(f'{path}', 'r').read())
    #     all_results = []
    #     for i in range(len(exp['trial_data'])):
    #         exp['trial_data'][i][1] = json.loads(exp['trial_data'][i][1])
    #         if 'performance_score' in exp['trial_data'][i][1]['last_result'].keys():
    #             all_results.append((exp['trial_data'][i][1]['last_result']['performance_score'], exp['trial_data'][i][1]['last_result']['config']))
    #     if len(all_results) > 0:
    #         try:
    #             print(max(all_results))
    #         except:
    #             print('No performance score found')
    #     else:
    #         print('No performance score found')
        
    # dir_ = './examples/ray_results/tactile_controller_tuning_csvto_diff'
    
    # # Iterate through dir, run get_max for every experiment_state*.json file
    # for file in os.listdir(dir_):
    #     if file.endswith('.json') and 'experiment_state' in file:
    #         print(file)
    #         get_max(os.path.join(dir_, file))
    #         print('-'*100)

