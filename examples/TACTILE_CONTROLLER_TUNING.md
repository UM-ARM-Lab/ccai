# Tactile Controller Hyperparameter Tuning

This directory contains tools for tuning tactile controller hyperparameters using Ray Tune with the **real controller implementation** from `allegro_screwdriver.py`.

## Files

- `tune_tactile_controller.py` - Main Ray Tune hyperparameter optimization script
- `test_tactile_tuning_setup.py` - Test script to validate setup before running tuning
- `run_tactile_tuning.py` - Simple wrapper script to run tuning experiments
- `TACTILE_CONTROLLER_TUNING.md` - This documentation file

## Quick Start

### 1. Test the Setup

First, run the test script to validate your environment:

```bash
python examples/test_tactile_tuning_setup.py
```

This will check:
- Ray Tune imports
- Configuration file loading  
- Environment creation (real or mock)
- Tuning function imports

### 2. Run Hyperparameter Tuning

Execute the main tuning script (uses real `do_trial` function from `allegro_screwdriver.py`):

```bash
python examples/tune_tactile_controller.py
```

Or use the wrapper script:

```bash
python examples/run_tactile_tuning.py
```

### 3. Monitor Progress

The script will:
- Test 12 different hyperparameter combinations (3 in mock mode)
- Run 3 evaluation trials per configuration using the **real tactile controller**
- Execute the full `do_trial` function with actual trajectory planning and control
- Display progress in the terminal
- Save results to `./ray_results/tactile_controller_tuning/`

### 4. View Results

Results are saved to:
- `./tuning_results/best_tactile_controller_config.yaml` - Best hyperparameters
- `./tuning_results/tuning_results_summary.yaml` - Complete summary
- `./ray_results/tactile_controller_tuning/` - Raw Ray Tune results

## Hyperparameters Being Tuned

The script optimizes these tactile controller parameters:

- `K_e` (50-1000): Environment stiffness
- `w_q` (1-100): Position weight
- `w_p` (0.1-10): Velocity weight  
- `w_f` (0.1-10): Force weight
- `w_u` (0.1-10): Control effort weight
- `w_ori` (0.01-1): Orientation weight

## Metrics

The optimization maximizes a composite performance score based on:

- **Success Rate** (2x weight): Whether the screwdriver turning goal was achieved
- **Completion Rate** (1x weight): Whether the screwdriver was not dropped
- **Distance to Goal** (0.5x penalty): How close the final orientation was to target

## Real Controller Integration

The tuning script now uses the **actual `do_trial` function** from `allegro_screwdriver.py`, which means:

- **Full Trajectory Planning**: Uses CSVGD controller with diffusion model sampling
- **Real Tactile Feedback**: Integrates tactile controller parameters into the control loop
- **Complete Pipeline**: Includes pregrasp, turning, and recovery phases
- **Actual Performance**: Measures real screwdriver turning performance with physics simulation
- **Proper Dependencies**: Loads trajectory samplers, classifiers, and kinematic chains

This ensures the hyperparameter tuning optimizes the **actual controller performance**, not a simplified approximation.

## Configuration

The script uses configuration from:
1. `examples/config/screwdriver/allegro_screwdriver_diff_tactile_control.yaml` (primary)
2. `examples/config/screwdriver/allegro_screwdriver_diff_only.yaml` (fallback)
3. Built-in defaults if neither file is available

## Dependencies

### Required
- Python 3.7+
- PyTorch
- Ray Tune: `pip install 'ray[tune]'`
- PyYAML: `pip install pyyaml`
- NumPy

### Optional (will use mock/fallback if missing)
- Isaac Gym environments (`isaac_victor_envs`)
- Open3D (`open3d`)
- CCAI tactile controller modules

## Mock Mode

If Isaac Gym environments are not available, the script automatically runs in mock mode using a simulated environment. This is useful for:
- Testing the tuning pipeline
- Validating hyperparameter search spaces
- Development without full Isaac Gym setup

## Customization

### Modify Search Space

Edit the `search_space` dictionary in `tune_tactile_controller.py`:

```python
search_space = {
    'K_e': tune.loguniform(50.0, 1000.0),      # Adjust ranges
    'w_q': tune.loguniform(1.0, 100.0),        
    # Add new parameters:
    'new_param': tune.uniform(0.1, 1.0),
}
```

### Adjust Tuning Settings

Modify these variables in the `main()` function:

```python
num_samples=15,          # Number of configurations to try
max_failures=3,          # Max failed trials allowed
resources_per_trial={"cpu": 1, "gpu": 0},  # Resource allocation
```

### Change Evaluation Metrics

Edit the `run_single_trial()` function to modify:
- Success criteria
- Trial termination conditions  
- Performance scoring

## Troubleshooting

### Common Issues

1. **Ray Tune not found**
   ```bash
   pip install 'ray[tune]'
   ```

2. **Missing configuration files**
   - Script will use fallback configs or defaults
   - Check that you're running from the correct directory

3. **Isaac Gym not available**  
   - Script will automatically use mock mode
   - Install isaac-gym environments for real simulation

4. **CUDA errors**
   - Set `sim_device: 'cpu'` in config files
   - Ensure CUDA is properly installed if using GPU

### Debug Mode

Run with additional debugging:

```python
# In tune_tactile_controller.py, add:
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Tips

1. **Faster Tuning**: Reduce `num_samples` and `num_eval_trials`
2. **Parallel Execution**: Increase Ray resource allocation
3. **Early Stopping**: Use ASHA scheduler (already configured)
4. **Mock Mode**: Use for rapid prototyping

## Results Interpretation

The tuning will output:

```
Best performance score: 2.1543
Best distance to goal: 0.0234
Best success rate: 0.8333
Best completion rate: 1.0000

Best hyperparameters:
  K_e: 234.56
  w_q: 45.78
  w_p: 2.34
  ...
```

Higher performance scores indicate better overall performance, combining success rate, completion rate, and goal distance. 