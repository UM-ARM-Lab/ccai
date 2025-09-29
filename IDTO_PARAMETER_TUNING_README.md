# IDTO Parameter Tuning for Allegro Screwdriver Task

This directory contains scripts for systematically tuning IDTO (Inverse Dynamics Trajectory Optimization) parameters to optimize the screwdriver turning performance of the Allegro hand.

## Overview

The tuning framework searches over 9 key IDTO parameters using a logspace grid search:

- **Qq_hand**: Goal cost for hand joint positions
- **Qq**: Goal cost for screwdriver angles  
- **Qv_hand**: Goal cost for hand joint velocities
- **Qv**: Goal cost for screwdriver angular velocities
- **hand_R**: Control cost for hand joints
- **screw_r**: Control cost for screwdriver joints
- **Qf_q_hand**: Final goal cost for hand positions
- **Qf_q**: Final goal cost for screwdriver angles
- **Qf_v**: Final goal cost for screwdriver angular velocities

The performance metric is **clockwise rotation** measured as `original_yaw - final_yaw`, where higher values indicate better performance.

## Files

### Core Scripts

- **`tune_idto_parameters.py`** - Main tuning script that runs grid search over parameter space
- **`test_tuning_small.py`** - Small-scale test script for validation before full tuning
- **`analyze_tuning_results.py`** - Analysis script to process and visualize tuning results

### Helper Files

- **`ccai/idto/python_examples/screwdriver_task.py`** - Original parameterized problem definition
- **`ccai/idto/allegro_screwdriver_idto.py`** - MPC controller implementation

## Usage

### 1. Test the Framework (Recommended First)

Before running the full parameter sweep, test with a small subset:

```bash
cd /home/abhinav/Documents/ccai
python test_tuning_small.py
```

This runs:
- 1 parameter configuration with 2 trials (Test 1)  
- 16 parameter configurations with 2 trials each (Test 2)
- 50-100 MPC cycles per trial

### 2. Full Parameter Tuning

Run the complete parameter search:

```bash
python tune_idto_parameters.py
```

**Default Configuration:**
- 9 parameters with 3 values each = **3^9 = 19,683 configurations**
- 5 trials per configuration = **98,415 total trials**
- Up to 200 MPC cycles per trial
- **Parallel execution**: 4 configurations simultaneously, 8 threads each
- Results saved to `tuning_results/` directory

**Performance:** With parallel execution, expected ~4x speedup over sequential processing.

**Warning:** This is computationally intensive but parallelization significantly reduces runtime!

### 3. Analyze Results

After tuning completes, analyze the results:

```bash
python analyze_tuning_results.py tuning_results/tuning_results_TIMESTAMP.pkl
```

This generates:
- Statistical summary of all trials
- Top-performing parameter configurations  
- Correlation analysis showing parameter importance
- Visualizations (heatmaps, distributions, scatter plots)
- Processed data as CSV file

## Parameter Ranges

The default logspace ranges (base 10) are:

| Parameter | Min Exp | Max Exp | Points | Range |
|-----------|---------|---------|--------|--------|
| Qq_hand   | -3      | -1      | 3      | [1e-3, ~3.16e-3, 1e-1] |
| Qq        | 0       | 2       | 3      | [1, 10, 100] |
| Qv_hand   | -4      | -2      | 3      | [1e-4, ~3.16e-4, 1e-2] |
| Qv        | -1      | 1       | 3      | [0.1, 1, 10] |
| hand_R    | -3      | -1      | 3      | [1e-3, ~3.16e-3, 1e-1] |
| screw_r   | 0       | 2       | 3      | [1, 10, 100] |
| Qf_q_hand | -3      | -1      | 3      | [1e-3, ~3.16e-3, 1e-1] |
| Qf_q      | 2       | 4       | 3      | [100, 1000, 10000] |
| Qf_v      | 0       | 2       | 3      | [1, 10, 100] |

## Parallel Execution

The framework supports parallel execution to significantly speed up parameter tuning:

### Configuration Options

```python
results = run_parameter_tuning(
    # ... other parameters ...
    parallel=True,                    # Enable/disable parallel execution
    num_parallel_configs=4,           # Number of configurations to run simultaneously
    threads_per_config=8,             # IDTO solver threads per configuration
    devices=['cuda:0', 'cuda:1'],     # CUDA devices to use (optional)
)
```

### Resource Allocation

- **Total Threads**: `num_parallel_configs × threads_per_config` (default: 4×8 = 32 threads)
- **Memory**: Each parallel process needs ~2-4GB GPU memory
- **GPU Devices**: Configurations are distributed across available devices

### Multi-GPU Setup

For systems with multiple GPUs:

```python
devices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3']  # Use 4 GPUs
num_parallel_configs=4                              # 1 config per GPU
```

### Performance Scaling

- **2 parallel configs**: ~2x speedup
- **4 parallel configs**: ~4x speedup  
- **8+ parallel configs**: Limited by CPU/memory bottlenecks

**Note**: Ensure your system has sufficient CPU cores and GPU memory before increasing parallelism.

## Customization

### Modify Parameter Ranges

Edit the `param_ranges` dictionary in `tune_idto_parameters.py`:

```python
param_ranges = {
    'Qq_hand': (-3, -1, 5),    # 5 points from 1e-3 to 1e-1
    'Qq': (0, 2, 4),           # 4 points from 1 to 100
    # ... etc
}
```

### Adjust Trial Settings

Modify these variables in the `main()` function:

```python
results = run_parameter_tuning(
    param_ranges=param_ranges,
    num_trials_per_config=3,      # Change number of trials
    max_cycles_per_trial=150,     # Change max MPC cycles  
    output_dir="my_tuning_results", # Change output directory
)
```

### Simulation Configuration

Modify `config_base` in `main()` to change simulation settings:

```python
config_base = {
    'visualize': True,           # Enable visualization
    'sim_device': 'cpu',         # Use CPU instead of GPU
    'kp': 50.0,                 # Change joint stiffness
    'randomize_obj_start': True, # Enable object randomization
    # ... etc
}
```

## Output Files

The tuning script saves several files in the output directory:

- **`tuning_results_TIMESTAMP.pkl`** - Complete results (Python pickle format)
- **`tuning_summary_TIMESTAMP.json`** - Summary statistics (JSON format)
- **`tuning_TIMESTAMP.log`** - Detailed execution log
- **`intermediate_results_TIMESTAMP.pkl`** - Periodic backups during execution

The analysis script generates:

- **`processed_results.csv`** - All successful trials as CSV
- **`performance_distribution.png`** - Histogram of performance metrics
- **`parameter_correlations.png`** - Parameter correlation heatmap
- **`parameter_vs_performance.png`** - Parameter vs. performance scatter plots

## Performance Considerations

### Computational Requirements

- **Full tuning**: ~20k trials × 200 cycles × 6 timesteps = ~24M simulation steps
- **GPU recommended**: CUDA-enabled GPU significantly speeds up Isaac Gym simulation
- **Memory**: ~8-16GB RAM recommended for large result datasets
- **Storage**: Results can be several GB for full parameter sweeps

### Optimization Tips

1. **Start small**: Always test with `test_tuning_small.py` first
2. **Use parallel execution**: Enable `parallel=True` for ~4x speedup with default settings
3. **Multi-GPU**: Use multiple CUDA devices if available for maximum performance
4. **Thread allocation**: Balance `num_parallel_configs × threads_per_config ≤ CPU cores`
5. **Early stopping**: Trials stop early if the screwdriver tips over (roll/pitch > 0.35 rad)
6. **Intermediate saves**: Results are saved every 5 configurations to prevent data loss

### Memory Management

The script reuses computation where possible [[memory:2647243]] by:
- Reusing the same Drake plant model across trials
- Caching parsed URDF models
- Cleaning up Isaac Gym resources after each trial

## Troubleshooting

### Common Issues

1. **"pyidto not available"** - Ensure IDTO is properly built and installed
2. **CUDA errors** - Try setting `'sim_device': 'cpu'` in config_base
3. **Memory errors** - Reduce number of parallel threads or batch size
4. **File not found errors** - Check that URDF files exist in Isaac assets directory

### Debugging

- Enable visualization: `'visualize': True` (slows down execution)
- Increase logging: Set `params.verbose = True` in solver parameters
- Check intermediate results: Monitor `intermediate_results_*.pkl` files

## Expected Results

Good parameter configurations typically achieve:
- **Clockwise rotation**: 0.1-0.5 radians (5-30 degrees)
- **Successful cycles**: 50-200 MPC cycles before failure
- **Success rate**: 60-90% of trials complete successfully

The analysis will identify:
- Which parameters most strongly correlate with performance
- Optimal parameter ranges for future fine-tuning  
- Trade-offs between different cost terms

## Next Steps

After completing the parameter tuning:

1. **Select best configurations** from the analysis results
2. **Fine-tune** the top configurations with narrower parameter ranges
3. **Validate** the best parameters on additional test scenarios
4. **Integrate** the optimal parameters into the main IDTO controller

## References

- IDTO documentation: [link to IDTO docs]
- Isaac Gym: [link to Isaac Gym docs]
- Drake: [link to Drake docs]
