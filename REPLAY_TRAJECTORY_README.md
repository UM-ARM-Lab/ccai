# Trajectory Replay Scripts

This directory contains scripts for replaying saved trajectories in the AllegroScrewdriverTurningEnv.

## Scripts

### `replay_trajectory.py`
Main script to replay trajectories from pickle files in the simulation environment.

**Usage:**
```bash
python replay_trajectory.py <pickle_file> [options]
```

**Arguments:**
- `pickle_file`: Path to pickle file containing trajectory tensor of shape T x dx

**Options:**
- `--no-visualize`: Disable visualization (run headless)
- `--step-delay FLOAT`: Delay between steps in seconds (default: 0.1)
- `--device DEVICE`: Device to use (default: cuda:0)
- `--save-video`: Save video frames during replay
- `--video-path PATH`: Path to save video frames (default: ./replay_video)

**Examples:**
```bash
# Basic replay with visualization
python replay_trajectory.py my_trajectory.pkl

# Faster replay with shorter delays
python replay_trajectory.py my_trajectory.pkl --step-delay 0.02

# Save video while replaying
python replay_trajectory.py my_trajectory.pkl --save-video --video-path ./output_video

# Run without visualization (headless)
python replay_trajectory.py my_trajectory.pkl --no-visualize
```

### `create_test_trajectory.py`
Helper script to create test trajectories for demonstration and validation.

**Usage:**
```bash
python create_test_trajectory.py
```

This creates a test trajectory file `test_trajectory.pkl` with smooth finger movements and screwdriver rotation.

## Supported Trajectory Formats

The replay script supports pickle files containing:
- `torch.Tensor` of shape T x dx
- `numpy.ndarray` of shape T x dx  
- `list` of tensors or arrays (will be stacked)

Where:
- T = number of timesteps
- dx = state dimension (typically 19 for Allegro hand + screwdriver: 16 joint angles + 3 object orientation)

## Environment Configuration

The replay script creates an AllegroScrewdriverTurningEnv with these default settings:
- 1 environment
- Joint impedance control
- Visualization enabled (unless --no-visualize)
- 60 simulation steps per action
- Friction coefficient: 2.5
- Joint stiffness: 300
- Fingers: ['index', 'middle', 'thumb']
- Gravity enabled

## Quick Start

1. Create a test trajectory:
   ```bash
   python create_test_trajectory.py
   ```

2. Replay the test trajectory:
   ```bash
   python replay_trajectory.py test_trajectory.pkl
   ```

3. For faster playback:
   ```bash
   python replay_trajectory.py test_trajectory.pkl --step-delay 0.02
   ```

## Notes

- The script automatically handles different tensor formats and devices
- Press Ctrl+C to interrupt replay at any time
- The environment will automatically close after replay completes
- Video saving creates individual frame images in the specified directory
- Ensure you have the required dependencies from the main project installed
