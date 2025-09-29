# Inverse Kinematics for Hardware Environment

This document describes the inverse kinematics functionality added to the hardware environment, which allows you to calculate where the screwdriver should be in mocap world frame given an arm configuration.

## Overview

The original `hardware_env.py` takes mocap data and calculates the required arm configuration to position the hand correctly. The new `hardware_env_inverse.py` does the reverse - it takes an arm configuration and calculates where the screwdriver should be positioned in mocap world frame.

## Files

- `hardware/hardware_env_inverse.py` - Main inverse kinematics implementation
- `test_inverse_kinematics.py` - Test script to verify functionality
- `hardware/hardware_env_inverse_example.py` - Comprehensive example showing integration

## Key Classes

### InverseObjectPoseCalculator

The main class that performs the inverse calculation:

```python
from hardware.hardware_env_inverse import InverseObjectPoseCalculator

# Initialize
inverse_calc = InverseObjectPoseCalculator(obj='blue_screwdriver', device='cuda:0')

# Calculate screwdriver pose from arm configuration
arm_config = torch.tensor([50.92, -73.15, 106.4, 64.1, 40.81, -119.07, -20.78]) / 180 * np.pi
arm_base_trans = np.array([0.1, 0.2, 0.3])  # Arm base position in mocap world
arm_base_euler = np.array([0.1, 0.2, 0.3])  # Arm base orientation in mocap world

screwdriver_trans, screwdriver_euler = inverse_calc.get_screwdriver_pose_from_arm_config(
    arm_config, arm_base_trans, arm_base_euler
)
```

## Transformation Chain

The inverse calculation follows this transformation chain:

1. **Arm Configuration → Hand Pose (Arm Victor Frame)**
   - Uses forward kinematics with pytorch_kinematics
   - `hand_pose_arm_victor = chain.forward_kinematics(arm_config)`

2. **Hand Pose (Arm Victor) → Hand Pose (Arm Mocap)**
   - Applies the transformation matrix between arm victor and arm mocap frames
   - `hand_to_arm_mocap = arm_victor_to_arm_mocap * hand_to_arm_victor`

3. **Hand Pose (Arm Mocap) → Hand Pose (Mocap World)**
   - Uses arm base pose from mocap data
   - `hand_to_mocap_world = arm_mocap_to_mocap_world * hand_to_arm_mocap`

4. **Hand Pose (Mocap World) → Screwdriver Pose (Mocap World)**
   - Applies the object-to-hand transformation
   - `screwdriver_to_mocap_world = hand_to_mocap_world * screwdriver_to_hand`

## Usage Examples

### Basic Usage

```python
import torch
import numpy as np
from hardware.hardware_env_inverse import InverseObjectPoseCalculator

# Initialize
device = 'cuda:0'
inverse_calc = InverseObjectPoseCalculator(obj='blue_screwdriver', device=device)

# Define arm configuration (7 joint angles in radians)
arm_config = torch.tensor([0.5, -0.3, 0.8, -0.2, 0.6, -0.4, 0.1], device=device)

# Define arm base pose in mocap world frame
arm_base_trans = np.array([0.1, 0.2, 0.3])
arm_base_euler = np.array([0.05, 0.1, 0.15])

# Calculate screwdriver pose
screwdriver_trans, screwdriver_euler = inverse_calc.get_screwdriver_pose_from_arm_config(
    arm_config, arm_base_trans, arm_base_euler
)

print(f"Screwdriver position: {screwdriver_trans}")
print(f"Screwdriver orientation: {screwdriver_euler}")
```

### Without Arm Base Pose

If you don't have arm base pose data, you can still get the hand pose in arm mocap frame:

```python
# Calculate hand pose in arm mocap frame
hand_trans, hand_euler = inverse_calc.get_screwdriver_pose_from_arm_config(arm_config)
print(f"Hand position (arm mocap frame): {hand_trans}")
print(f"Hand orientation (arm mocap frame): {hand_euler}")
```

### Integration with Existing Code

```python
from hardware.hardware_env_inverse_example import InverseHardwareEnv

# Initialize extended environment
env = InverseHardwareEnv(device='cuda:0')

# Get arm config from mocap (original functionality)
arm_config = env.get_arm_config_from_mocap()

# Get expected mocap data from arm config (new functionality)
expected_trans, expected_euler = env.get_expected_mocap_from_arm_config(
    arm_config, arm_base_trans, arm_base_euler
)

# Verify the arm configuration matches mocap data
is_valid = env.verify_arm_config(arm_config)
```

## Testing

Run the test script to verify functionality:

```bash
python test_inverse_kinematics.py
```

This will:
1. Test the inverse calculation with different arm configurations
2. Compare results with forward kinematics to verify correctness
3. Show position and orientation calculations

## Key Features

1. **Bidirectional**: Works in both forward (mocap → arm) and inverse (arm → mocap) directions
2. **Flexible**: Can work with or without arm base pose data
3. **Verified**: Includes comparison with forward kinematics for validation
4. **Compatible**: Uses the same transformation matrices as the original code
5. **Efficient**: Reuses computation where possible [[memory:2647243]]

## Dependencies

- pytorch_kinematics
- torch
- numpy
- scipy
- rospy (for mocap integration)

## Notes

- The transformation matrices are hardcoded and should match those in the original `hardware_env.py`
- The URDF path is hardcoded and should be updated if the robot model changes
- The code assumes the same coordinate frames and conventions as the original implementation
- All angles are in radians unless otherwise specified
