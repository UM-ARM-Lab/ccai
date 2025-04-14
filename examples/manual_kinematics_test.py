import os
import re
import time
import torch
import importlib.util
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

from pytorch_kinematics.urdf_parser_py.urdf import URDF
from pytorch_kinematics.urdf import build_serial_chain_from_urdf
from pytorch_kinematics.chain import Chain


class SymbolicKinematics:
    """Symbolic kinematics for multi-finger robots with drop-in replacement for pytorch_kinematics."""
    
    def __init__(self, urdf_path: str, fingers: List[Dict[str, str]], 
                 cache_dir: str = "./kinematics", use_cache: bool = True,
                 joint_counts: Optional[Dict[str, int]] = None) -> None:
        """
        Initialize kinematics for multiple fingers with optional caching.
        
        Args:
            urdf_path: Path to the robot URDF file
            fingers: List of dicts with keys 'name', 'root', 'end' for each finger
            cache_dir: Directory for cached PyTorch function files
            use_cache: Whether to use cached files if they exist
            joint_counts: Optional pre-defined joint counts per finger
        """
        import os
        import re
        import importlib.util
        import numpy as np
        import torch
        import sympy as sp
        
        self.cache_dir = cache_dir
        self.joint_counts = joint_counts or {}
        self.root_link = "allegro_hand_base_link"  # Common base link for all fingers
        
        # Check if we can load from cache
        if use_cache and self._check_cached_files(fingers):
            print(f"Using cached kinematics functions from {cache_dir}")
            return
            
        # If cache doesn't exist or we're not using it, process from scratch
        print(f"Building kinematics from {urdf_path}")
        
        with open(urdf_path, "r") as f:
            self.urdf_str = f.read()
        
        self.urdf_robot = URDF.from_xml_string(self.urdf_str)
        self.fingers = {}
        
        # Process valid fingers
        for finger in fingers:
            name = finger["name"]
            end = finger["end"]
            
            # Always use the base link as root for all fingers
            root = self.root_link
            
            try:
                # Check if links exist in the URDF
                links = [link.name for link in self.urdf_robot.links]
                if root not in links:
                    raise ValueError(f"Root link '{root}' not found in URDF")
                if end not in links:
                    raise ValueError(f"End link '{end}' for finger '{name}' not found in URDF")
                
                # Create serial chain from base to end-effector
                serial_chain = build_serial_chain_from_urdf(
                    self.urdf_str, end_link_name=end, root_link_name=root
                )
                
                # Find joint chain from root to end effector
                joint_chain = []
                link_name = end
                while link_name != root:
                    # Find the joint that has this link as child
                    for joint in self.urdf_robot.joints:
                        if joint.child == link_name:
                            joint_chain.insert(0, joint)
                            link_name = joint.parent
                            break
                    else:
                        # If no joint found, this is an error
                        raise ValueError(f"Could not find joint chain from {root} to {end}")
                
                # Store only essential information
                movable_joints = [j for j in joint_chain if j.type == "revolute"]
                q_vars = sp.symbols(f"q:{len(movable_joints)}")
                
                self.fingers[name] = {
                    "chain": serial_chain,
                    "joints": joint_chain,
                    "movable_joints": movable_joints,
                    "q_vars": q_vars,
                    "root": root,  # Always use hand base as root
                    "end": end,
                    "num_joints": len(movable_joints)
                }
                
                # Store joint count in our dictionary for future use
                self.joint_counts[name] = len(movable_joints)
                
                # Get joint limits
                try:
                    low_list, high_list = serial_chain.get_joint_limits()
                    self.fingers[name]["limits"] = (
                        torch.tensor(low_list, dtype=torch.float32),
                        torch.tensor(high_list, dtype=torch.float32)
                    )
                except Exception:
                    # Default to -π to π
                    num_joints = len(movable_joints)
                    self.fingers[name]["limits"] = (
                        -torch.ones(num_joints, dtype=torch.float32) * np.pi,
                        torch.ones(num_joints, dtype=torch.float32) * np.pi
                    )
                
                # Save joint names for reference
                self.fingers[name]["joint_names"] = [j.name for j in movable_joints]
                
            except Exception as e:
                print(f"Error processing finger {name}: {str(e)}")
                continue
        
        if not self.fingers:
            raise ValueError("No valid fingers could be processed from URDF")
        
        # Compute symbolic kinematics
        self._compute_symbolic_kinematics()
        
        # Export to PyTorch files for future use
        self.export_pytorch_functions(self.cache_dir)
    
    def _check_cached_files(self, fingers: List[Dict[str, str]]) -> bool:
        """
        Check if cached PyTorch implementation files exist and load them.
        
        Args:
            fingers: List of finger definitions
        
        Returns:
            bool: True if all files exist and were loaded successfully
        """
        if not os.path.exists(self.cache_dir):
            return False
        
        # Check if both file types exist for all fingers
        for finger in fingers:
            name = finger["name"]
            fk_path = os.path.join(self.cache_dir, f"{name}_fk.py")
            jac_path = os.path.join(self.cache_dir, f"{name}_jacobian.py")
            if not (os.path.exists(fk_path) and os.path.exists(jac_path)):
                return False
        
        # All files exist, try to load them
        self.fingers = {}
        for finger in fingers:
            name = finger["name"]
            try:
                # Import the generated modules
                fk_module = self._import_module_from_path(
                    f"{name}_fk", os.path.join(self.cache_dir, f"{name}_fk.py")
                )
                jac_module = self._import_module_from_path(
                    f"{name}_jacobian", os.path.join(self.cache_dir, f"{name}_jacobian.py")
                )
                
                # Get the forward kinematics and jacobian functions
                fk_func = getattr(fk_module, f"{name}_forward_kinematics")
                jac_func = getattr(jac_module, f"{name}_jacobian")
                
                # Try multiple methods to determine joint count
                num_joints = None
                
                # 1. Try to get from pre-defined joint counts
                if name in self.joint_counts:
                    num_joints = self.joint_counts[name]
                
                # 2. Try to extract from docstring pattern
                if num_joints is None and fk_func.__doc__:
                    match = re.search(r"expected (\d+) joints", fk_func.__doc__, re.IGNORECASE)
                    if match:
                        num_joints = int(match.group(1))
                
                # 3. Try to extract from filename
                if num_joints is None:
                    # Check if there's a metadata file with joint counts
                    meta_path = os.path.join(self.cache_dir, "metadata.py")
                    if os.path.exists(meta_path):
                        meta_module = self._import_module_from_path("kinematics_metadata", meta_path)
                        if hasattr(meta_module, "JOINT_COUNTS") and name in meta_module.JOINT_COUNTS:
                            num_joints = meta_module.JOINT_COUNTS[name]
                
                # 4. Default to common values if we can't determine
                if num_joints is None:
                    print(f"Warning: Could not determine joint count for {name}, defaulting to 4")
                    num_joints = 4  # Common for fingers
                
                # Store minimal information needed for runtime
                self.fingers[name] = {
                    "fk_fn": fk_func,
                    "jac_fn": jac_func,
                    "num_joints": num_joints,
                    "root": finger["root"],
                    "end": finger["end"],
                }
                
            except Exception as e:
                print(f"Error loading cached files for {name}: {str(e)}")
                return False
            
        return True
    
    def _import_module_from_path(self, module_name: str, file_path: str) -> Any:
        """
        Import a Python module from file path.
        
        Args:
            module_name: Name to give the imported module
            file_path: Path to the module file
            
        Returns:
            The imported module object
        """
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    def _compute_symbolic_kinematics(self) -> None:
        """
        Compute symbolic forward kinematics and Jacobian for all fingers.
        Uses SymPy for symbolic mathematics to derive analytical expressions.
        """
        import sympy as sp
        
        print("Computing symbolic kinematics...")
        for name, finger in self.fingers.items():
            print(f"Computing symbolic kinematics for {name}...")
            
            # Get joint information from the URDF
            joint_chain = finger["joints"]
            q_vars = finger["q_vars"]
            root_link = finger["root"]
            end_link = finger["end"]
            
            # Initialize identity transform at the root
            link_poses = {root_link: sp.eye(4)}
            current_transform = sp.eye(4)
            
            # Keep track of joint variables for revolute joints
            q_idx = 0
            
            # Store intermediates for Jacobian computation
            intermediates = []
            
            # Process each joint in the chain to build transformations
            for joint in joint_chain:
                parent = joint.parent
                child = joint.child
                
                # Skip if we don't have the parent transform yet
                if parent not in link_poses:
                    continue
                
                # Get transform up to parent link
                current_transform = link_poses[parent]
                
                # Create transform for this joint
                joint_transform = sp.eye(4)
                
                # Apply origin transformation first (fixed component)
                if hasattr(joint, 'origin') and joint.origin is not None:
                    xyz = joint.origin.xyz if hasattr(joint.origin, 'xyz') and joint.origin.xyz is not None else [0, 0, 0]
                    rpy = joint.origin.rpy if hasattr(joint.origin, 'rpy') and joint.origin.rpy is not None else [0, 0, 0]
                    
                    # Create homogeneous transform for origin
                    T_origin = sp.eye(4)
                    
                    # Set translation
                    T_origin[0, 3] = xyz[0]
                    T_origin[1, 3] = xyz[1]
                    T_origin[2, 3] = xyz[2]
                    
                    # Apply RPY rotation (ZYX convention)
                    R_x = sp.Matrix([
                        [1, 0, 0],
                        [0, sp.cos(rpy[0]), -sp.sin(rpy[0])],
                        [0, sp.sin(rpy[0]), sp.cos(rpy[0])]
                    ])
                    
                    R_y = sp.Matrix([
                        [sp.cos(rpy[1]), 0, sp.sin(rpy[1])],
                        [0, 1, 0],
                        [-sp.sin(rpy[1]), 0, sp.cos(rpy[1])]
                    ])
                    
                    R_z = sp.Matrix([
                        [sp.cos(rpy[2]), -sp.sin(rpy[2]), 0],
                        [sp.sin(rpy[2]), sp.cos(rpy[2]), 0],
                        [0, 0, 1]
                    ])
                    
                    # Combine rotations in ZYX order (as per URDF convention)
                    T_origin[:3, :3] = R_z * R_y * R_x
                    
                    # Apply origin transform
                    joint_transform = joint_transform * T_origin
                
                # Store the position before applying joint motion for Jacobian calculation
                if joint.type == "revolute":
                    # Get joint axis (default Z if not specified)
                    axis = joint.axis if hasattr(joint, 'axis') and joint.axis else [0, 0, 1]
                    
                    # Store intermediate transform and joint info for Jacobian
                    intermediates.append((q_idx, current_transform * joint_transform, axis))
                    
                    # Create joint motion transform using the symbolic variable
                    T_motion = sp.eye(4)
                    
                    # Get the joint variable
                    q = q_vars[q_idx]
                    
                    # Apply rotation based on joint axis, accounting for direction
                    if abs(axis[0]) > 0.99:  # X-axis rotation
                        # Handle negative X-axis by flipping the angle
                        sign = -1 if axis[0] < 0 else 1
                        theta = sign * q
                        R_joint = sp.Matrix([
                            [1, 0, 0],
                            [0, sp.cos(theta), -sp.sin(theta)],
                            [0, sp.sin(theta), sp.cos(theta)]
                        ])
                    elif abs(axis[1]) > 0.99:  # Y-axis rotation
                        # Handle negative Y-axis by flipping the angle
                        sign = -1 if axis[1] < 0 else 1
                        theta = sign * q
                        R_joint = sp.Matrix([
                            [sp.cos(theta), 0, sp.sin(theta)],
                            [0, 1, 0],
                            [-sp.sin(theta), 0, sp.cos(theta)]
                        ])
                    else:  # Z-axis rotation (default)
                        # Handle negative Z-axis by flipping the angle
                        sign = -1 if axis[2] < 0 else 1
                        theta = sign * q
                        R_joint = sp.Matrix([
                            [sp.cos(theta), -sp.sin(theta), 0],
                            [sp.sin(theta), sp.cos(theta), 0],
                            [0, 0, 1]
                        ])
                    
                    T_motion[:3, :3] = R_joint
                    
                    # Apply joint motion transform
                    joint_transform = joint_transform * T_motion
                    
                    # Increment joint variable index
                    q_idx += 1
                
                # Compute the full transform to this link
                link_transform = current_transform * joint_transform
                
                # Store the transform for this link
                link_poses[child] = link_transform
            
            # Get end-effector transform
            T_fk = link_poses[end_link]
            
            # Store symbolic FK
            finger["T_fk_sym"] = T_fk
            finger["link_poses"] = link_poses
            
            # Now compute Jacobians for all links
            jacobians = {}
            
            for link_name, link_pose in link_poses.items():
                if link_name == root_link:
                    continue  # Skip root link
                
                # Get end position for this link
                p_link = link_pose[:3, 3]
                
                # Initialize Jacobian (6×num_joints)
                num_joints = len(q_vars)
                J = sp.zeros(6, num_joints)
                
                # Process each joint's contribution to this link's motion
                for inter_idx, (j_idx, T_j, axis) in enumerate(intermediates):
                    # Skip joints that come after this link in the chain
                    found_in_chain = False
                    for j in joint_chain:
                        if j.child == link_name:
                            found_in_chain = True
                            break
                    
                    if found_in_chain and inter_idx >= joint_chain.index(j):
                        continue
                    
                    # Get rotation matrix and position for this joint
                    R_j = T_j[:3, :3]
                    p_j = T_j[:3, 3]
                    
                    # Convert joint axis to global frame, respecting axis direction
                    # For negative axis components, we need to properly apply the direction
                    norm_axis = [axis[i]/abs(axis[i]) if abs(axis[i]) > 0.01 else 0 for i in range(3)]
                    z_j = R_j * sp.Matrix(norm_axis)
                    
                    # Linear velocity component (cross product of rotation axis and lever arm)
                    J_v = z_j.cross(p_link - p_j)
                    
                    # Angular velocity component is the rotation axis
                    J_ω = z_j
                    
                    # Set the Jacobian columns
                    J[:3, j_idx] = J_v
                    J[3:, j_idx] = J_ω
                
                # Store the Jacobian
                jacobians[link_name] = sp.simplify(J)
            
            # Store all Jacobians
            finger["jacobians"] = jacobians
            
            print(f"Completed symbolic kinematics for {name}")

    def _create_transform_matrix(self, xyz: List[float], rpy: List[float]) -> Any:
        """
        Create a precise homogeneous transformation matrix from position and orientation.
        
        Args:
            xyz: Position vector [x, y, z]
            rpy: Rotation in roll-pitch-yaw [r, p, y]
            
        Returns:
            4×4 transformation matrix
        """
        import sympy as sp
        
        # Extract position components
        x, y, z = xyz
        
        # Extract rotation components
        roll, pitch, yaw = rpy
        
        # Create elementary rotation matrices
        # Roll: Rotation around X
        Rx = sp.Matrix([
            [1, 0, 0],
            [0, sp.cos(roll), -sp.sin(roll)],
            [0, sp.sin(roll), sp.cos(roll)]
        ])
        
        # Pitch: Rotation around Y
        Ry = sp.Matrix([
            [sp.cos(pitch), 0, sp.sin(pitch)],
            [0, 1, 0],
            [-sp.sin(pitch), 0, sp.cos(pitch)]
        ])
        
        # Yaw: Rotation around Z
        Rz = sp.Matrix([
            [sp.cos(yaw), -sp.sin(yaw), 0],
            [sp.sin(yaw), sp.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combine rotations according to the specified convention (ZYX for URDF)
        # Note: URDF uses the ZYX convention (yaw, pitch, roll)
        R = Rz @ Ry @ Rx
        
        # Create the full transformation matrix
        T = sp.eye(4)
        T[:3, :3] = R
        T[0, 3] = x
        T[1, 3] = y
        T[2, 3] = z
        
        # Simplify the matrix to reduce computational complexity
        T = sp.simplify(T)
        return T
    
    def forward_kinematics(self, th: Dict[str, torch.Tensor], 
                           joint_indices: Optional[Dict[str, List[int]]] = None) -> Dict[str, torch.Tensor]:
        """
        Compute forward kinematics for multiple fingers.
        
        Args:
            th: Dictionary of joint angles with finger names as keys
                If joint_indices is None, each tensor should have exactly the joints for that finger
                If joint_indices is provided, each tensor can have all joints, and indices will be used
            joint_indices: Optional mapping of finger name to list of joint indices to use
                
        Returns:
            Dictionary of end-effector poses with finger names as keys
        """
        results = {}
        for name, q in th.items():
            if name not in self.fingers:
                continue
                
            finger = self.fingers[name]
            num_joints = finger["num_joints"]
            
            # Extract the correct joints if indices are provided
            if joint_indices and name in joint_indices:
                indices = joint_indices[name]
                if len(indices) != num_joints:
                    raise ValueError(f"Finger '{name}' expects {num_joints} indices, but got {len(indices)}")
                
                if q.dim() == 1:
                    # Single vector case
                    q_finger = torch.tensor([q[i] for i in indices], dtype=q.dtype, device=q.device)
                else:
                    # Batched case
                    q_finger = torch.stack([q[:, i] for i in indices], dim=1)
            else:
                # Check if the input has exact number of joints or too many
                if q.shape[-1] > num_joints:
                    # If too many joints, use only the first num_joints
                    q_finger = q[..., :num_joints]
                    print(f"Warning: Using first {num_joints} joints for finger '{name}'")
                else:
                    # Use as-is if dimensions match
                    if q.shape[-1] != num_joints:
                        raise ValueError(f"Finger '{name}' expects {num_joints} joints, but got {q.shape[-1]}")
                    q_finger = q
            
            # Call the appropriate function based on whether we're using cached or dynamically generated functions
            if callable(finger.get("fk_fn")):
                # If device is provided, use it
                device_arg = q_finger.device if hasattr(q_finger, "device") else None
                results[name] = finger["fk_fn"](q_finger, device=device_arg)
            else:
                # Fall back to dynamic computation if available
                if "torch_fk_fn" in finger:
                    results[name] = finger["torch_fk_fn"](q_finger)
                else:
                    raise RuntimeError(f"No forward kinematics function available for finger '{name}'")
                
        return results
    
    def jacobian(self, th: Dict[str, torch.Tensor], links: Optional[Dict[str, List[str]]] = None,
                 joint_indices: Optional[Dict[str, List[int]]] = None) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute Jacobian for specified links of multiple fingers.
        
        Args:
            th: Dictionary of joint angles with finger names as keys
            links: Dictionary with finger names as keys and lists of link names as values
                   If None, compute Jacobians for end-effectors only
            joint_indices: Optional mapping of finger name to list of joint indices to use
                   
        Returns:
            Dictionary with finger names as keys and dictionaries of link Jacobians as values
        """
        results = {}
        for name, q in th.items():
            if name not in self.fingers:
                continue
                
            finger = self.fingers[name]
            num_joints = finger["num_joints"]
            
            # Extract the correct joints if indices are provided
            if joint_indices and name in joint_indices:
                indices = joint_indices[name]
                if len(indices) != num_joints:
                    raise ValueError(f"Finger '{name}' expects {num_joints} indices, but got {len(indices)}")
                
                if q.dim() == 1:
                    # Single vector case
                    q_finger = torch.tensor([q[i] for i in indices], dtype=q.dtype, device=q.device)
                else:
                    # Batched case
                    q_finger = torch.stack([q[:, i] for i in indices], dim=1)
            else:
                # Check if the input has exact number of joints or too many
                if q.shape[-1] > num_joints:
                    # If too many joints, use only the first num_joints
                    q_finger = q[..., :num_joints]
                    print(f"Warning: Using first {num_joints} joints for finger '{name}'")
                else:
                    # Use as-is if dimensions match
                    if q.shape[-1] != num_joints:
                        raise ValueError(f"Finger '{name}' expects {num_joints} joints, but got {q.shape[-1]}")
                    q_finger = q
            
            # Determine which links to include
            link_list = links.get(name, [finger["end"]]) if links else [finger["end"]]
            
            # Compute Jacobian for each link
            finger_results = {}
            for link in link_list:
                if callable(finger.get("jac_fn")):
                    # If device is provided, use it
                    device_arg = q_finger.device if hasattr(q_finger, "device") else None
                    finger_results[link] = finger["jac_fn"](q_finger, link, device=device_arg)
                else:
                    # Fall back to dynamic computation if available
                    if "torch_jac_fns" in finger and link in finger["torch_jac_fns"]:
                        finger_results[link] = finger["torch_jac_fns"][link](q_finger)
                    else:
                        print(f"Warning: No Jacobian function available for link '{link}' of finger '{name}'")

            if finger_results:
                results[name] = finger_results
                
        return results
    
    def get_joint_limits(self, finger_name: Optional[str] = None) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get joint limits for specified fingers.
        
        Args:
            finger_name: Optional name of finger to get limits for.
                         If None, returns limits for all fingers.
                         
        Returns:
            Dictionary with finger names as keys and (lower, upper) limit tuples as values
        """
        if finger_name:
            if finger_name in self.fingers and "limits" in self.fingers[finger_name]:
                return {finger_name: self.fingers[finger_name]["limits"]}
            return {}
        
        return {name: finger["limits"] for name, finger in self.fingers.items() 
                if "limits" in finger}

    def export_pytorch_functions(self, output_dir: str) -> None:
        """
        Export symbolic FK and Jacobian expressions to PyTorch-based Python files.
        
        Args:
            output_dir: Directory to save the generated Python files
        """
        import sympy as sp  # Import only when needed
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create an __init__.py to make the directory a package
        with open(os.path.join(output_dir, "__init__.py"), 'w') as f:
            f.write('# Auto-generated kinematics package\n')
        
        # Create a metadata file with joint counts
        self._export_metadata(output_dir)
        
        for finger_name, finger in self.fingers.items():
            # Skip if both files already exist
            fk_file_path = os.path.join(output_dir, f"{finger_name}_fk.py")
            jac_file_path = os.path.join(output_dir, f"{finger_name}_jacobian.py")
            if os.path.exists(fk_file_path) and os.path.exists(jac_file_path):
                print(f"Files for {finger_name} already exist, skipping generation")
                continue
                
            # Generate FK function
            self._export_pytorch_fk_function(finger_name, finger, fk_file_path)
            
            # Generate Jacobian functions
            self._export_pytorch_jacobian_functions(finger_name, finger, jac_file_path)
            
            print(f"  Saved to {fk_file_path} and {jac_file_path}")

    def _export_metadata(self, output_dir: str) -> None:
        """
        Export metadata about the kinematics to help with loading cached files.
        
        Args:
            output_dir: Directory to save the metadata file
        """
        meta_path = os.path.join(output_dir, "metadata.py")
        
        # Gather joint counts
        joint_counts = {}
        end_effectors = {}
        root_links = {}
        
        for name, finger in self.fingers.items():
            joint_counts[name] = finger["num_joints"]
            end_effectors[name] = finger["end"]
            root_links[name] = finger["root"]
        
        # Create metadata content with properly formatted dictionaries
        meta_content = [
            "# Auto-generated kinematics metadata",
            "# Contains information needed to load cached kinematics functions",
            "",
            "# Joint counts for each finger",
            f"JOINT_COUNTS = {repr(joint_counts)}",
            "",
            "# End-effector links for each finger",
            f"END_EFFECTORS = {repr(end_effectors)}",
            "",
            "# Root links for each finger",
            f"ROOT_LINKS = {repr(root_links)}",
        ]
        
        # Write to file
        with open(meta_path, 'w') as f:
            f.write('\n'.join(meta_content))

    def _export_pytorch_fk_function(self, finger_name: str, finger: Dict[str, Any], file_path: str) -> None:
        """
        Export direct forward kinematics function with PyTorch implementation.
        Avoids pre-computation for more reliable code generation.
        
        Args:
            finger_name: Name of the finger
            finger: Finger data dictionary containing symbolic expressions
            file_path: Path where to save the function file
        """
        import sympy as sp
        import os
        
        # Get symbolic FK and important information
        if "T_fk_sym" not in finger:
            raise ValueError(f"No symbolic FK found for {finger_name}. Run _compute_symbolic_kinematics first.")
                
        T_fk_sym = finger["T_fk_sym"]
        q_vars = finger["q_vars"]
        num_joints = finger["num_joints"]
        
        # Get joint names for documentation
        joint_names = finger.get("joint_names", [f"joint_{i}" for i in range(num_joints)])
        joint_names_str = ", ".join(joint_names)
        
        # Create implementation without intermediate variables
        fk_code = [
            "import torch",
            "from typing import Union, Tuple, List, Optional",
            "from functools import lru_cache",
            "",
            f"def {finger_name}_forward_kinematics_impl(q: torch.Tensor) -> torch.Tensor:",
            '    """',
            f"    Forward kinematics implementation for {finger_name} finger.",
            f"    Takes joint angles and returns 4x4 transformation matrices.",
            "",
            f"    Args:",
            f"        q: Joint angles tensor of shape [batch_size, {num_joints}]",
            "",
            f"    Returns:", 
            f"        Transformation matrices of shape [batch_size, 4, 4]",
            '    """',
            "    batch_size = q.shape[0]",
            "    device = q.device",
            "    dtype = q.dtype",
            "",
            "    # Create transformation matrix",
            "    T = torch.eye(4, dtype=dtype, device=device).repeat(batch_size, 1, 1)",
            ""
        ]
        
        # Process transformation matrix elements
        fk_code.append("    # Set transformation matrix elements directly")
        for i in range(4):
            for j in range(4):
                expr = T_fk_sym[i, j]
                # Skip identity elements
                if (i == j and expr == 1) or (i != j and expr == 0):
                    continue
                
                # Convert to string representation
                expr_str = str(expr)
                
                # Ensure all trigonometric functions use torch namespace
                expr_str = expr_str.replace("sin(", "torch.sin(")
                expr_str = expr_str.replace("cos(", "torch.cos(")
                
                # Replace symbolic variables with tensor indexing
                for k, var in enumerate(q_vars):
                    var_str = str(var)
                    expr_str = expr_str.replace(var_str, f"q[:, {k}]")
                
                # Add matrix element assignment
                fk_code.append(f"    T[:, {i}, {j}] = {expr_str}")
        
        # Return the result
        fk_code.extend([
            "",
            "    return T",
            ""
        ])
        
        # Add the cached single-input helper
        fk_code.extend([
            "@lru_cache(maxsize=128)",
            f"def {finger_name}_forward_kinematics_single(q_tuple: Tuple[float, ...]) -> torch.Tensor:",
            '    """Cached implementation for single joint configuration."""',
            "    q = torch.tensor([q_tuple], dtype=torch.float32)",
            f"    return {finger_name}_forward_kinematics_impl(q)[0]",
            ""
        ])
        
        # Add the main API function
        fk_code.extend([
            f"def {finger_name}_forward_kinematics(q: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:",
            '    """',
            f"    Forward kinematics for the {finger_name} finger with PyTorch.",
            "",
            f"    Args:",
            f"        q: Joint angles in radians, shape [batch_size, {num_joints}] or [{num_joints}]",
            f"        device: Optional device for computation (CPU/GPU)",
            "",
            f"    Returns:",
            f"        Transformation matrices, shape [batch_size, 4, 4] or [4, 4]",
            "",
            f"    Notes:",
            f"        Expects {num_joints} joints: {joint_names_str}",
            '    """',
            f"    # Input validation",
            f"    if q.shape[-1] != {num_joints}:",
            f'        raise ValueError(f"Expected {num_joints} joint values, got {{q.shape[-1]}}")',
            "",
            "    # Handle device placement",
            "    if device is None:",
            "        device = q.device if torch.is_tensor(q) else torch.device('cpu')",
            "    elif torch.is_tensor(q) and q.device != device:",
            "        q = q.to(device)",
            "",
            "    # Handle various input formats",
            "    unbatched = False",
            "    if not torch.is_tensor(q):",
            "        q = torch.tensor(q, dtype=torch.float32, device=device)",
            "",
            "    if q.dim() == 1:",
            "        unbatched = True",
            "        q_tuple = tuple(q.cpu().numpy().tolist())",
            "        return {}_forward_kinematics_single(q_tuple).to(device)".format(finger_name),
            "",
            "    if q.dim() != 2:",
            "        q = q.reshape(-1, {})".format(num_joints),
            "",
            "    # Call optimized implementation",
            "    T = {}_forward_kinematics_impl(q)".format(finger_name),
            "",
            "    # Handle unbatched case",
            "    return T.squeeze(0) if unbatched else T",
            ""
        ])
        
        # Write to file and load
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write('\n'.join(fk_code))
        
        # Create function object for immediate use
        try:
            import importlib.util
            import sys
            
            module_name = f"{finger_name}_fk"
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            finger["fk_fn"] = getattr(module, f"{finger_name}_forward_kinematics")
            finger["torch_fk_fn"] = finger["fk_fn"]
            print(f"Successfully loaded {finger_name} FK function from file")
        except Exception as e:
            print(f"Warning: Could not compile FK function for {finger_name}: {e}")
            
            try:
                print(f"Trying fallback method for {finger_name}")
                namespace = {}
                exec('\n'.join(fk_code), namespace)
                finger["fk_fn"] = namespace[f"{finger_name}_forward_kinematics"]
                finger["torch_fk_fn"] = finger["fk_fn"]
                print(f"Successfully loaded {finger_name} FK function via fallback")
            except Exception as e2:
                print(f"Fatal error loading {finger_name}: {e2}")

    def _export_pytorch_jacobian_functions(self, finger_name: str, finger: Dict[str, Any], file_path: str) -> None:
        """
        Export direct Jacobian functions with PyTorch implementation.
        Avoids pre-computation for more reliable code generation.
        
        Args:
            finger_name: Name of the finger
            finger: Finger data dictionary containing symbolic Jacobian expressions
            file_path: Path where to save the function file
        """
        import sympy as sp
        import os
        
        # Get symbolic Jacobians and important information
        if "jacobians" not in finger:
            raise ValueError(f"No symbolic Jacobians found for {finger_name}. Run _compute_symbolic_kinematics first.")
        
        jacobians = finger["jacobians"]  # Dictionary of link name to symbolic Jacobian
        q_vars = finger["q_vars"]        # Joint symbolic variables
        num_joints = finger["num_joints"]
        
        # Find all unique link names with Jacobians
        link_names = list(jacobians.keys())
        link_names.append(finger['end'])  # Add end effector
        
        # Remove duplicates while preserving order
        unique_links = []
        for link in link_names:
            if link not in unique_links:
                unique_links.append(link)
        
        # Generate code header for the Jacobian function
        jac_code = [
            "import torch",
            "from typing import Union, Tuple, List, Dict, Optional",
            "from functools import lru_cache",
            "",
            f"def {finger_name}_jacobian_impl(q: torch.Tensor, link_name: str) -> torch.Tensor:",
            '    """',
            f"    Jacobian calculation for {finger_name} finger.",
            f"    Takes joint angles and link name, returns 6xN Jacobian matrix.",
            "",
            f"    Args:",
            f"        q: Joint angles tensor of shape [batch_size, {num_joints}]",
            f"        link_name: Name of the link to compute Jacobian for",
            "",
            f"    Returns:",
            f"        Jacobian matrix of shape [batch_size, 6, {num_joints}]",
            '    """',
            "    batch_size = q.shape[0]",
            "    device = q.device",
            "    dtype = q.dtype",
            "",
            f"    # Initialize Jacobian matrix",
            f"    J = torch.zeros(batch_size, 6, {num_joints}, dtype=dtype, device=device)",
            ""
        ]
        
        # Generate link-specific Jacobian calculations
        first_link = True
        for link_name in unique_links:
            if link_name not in jacobians:
                continue
                
            jacobian_sym = jacobians[link_name]
            
            # Check if this link has any non-zero elements
            has_nonzero_elements = False
            for i in range(6):
                for j in range(num_joints):
                    if jacobian_sym[i, j] != 0:
                        has_nonzero_elements = True
                        break
                if has_nonzero_elements:
                    break
            
            if not has_nonzero_elements:
                continue
                
            # Add if/elif statement for this link
            if first_link:
                jac_code.append(f"    if link_name == '{link_name}':")
                first_link = False
            else:
                jac_code.append(f"    elif link_name == '{link_name}':")
            
            # Process each non-zero element in the Jacobian directly without precomputation
            for i in range(6):
                for j in range(num_joints):
                    expr = jacobian_sym[i, j]
                    if expr == 0:
                        continue
                    
                    # Convert to string representation
                    expr_str = str(expr)
                    
                    # Ensure all trigonometric functions use torch namespace
                    expr_str = expr_str.replace("sin(", "torch.sin(")
                    expr_str = expr_str.replace("cos(", "torch.cos(")
                    
                    # Replace symbolic variables with tensor indexing
                    for k, var in enumerate(q_vars):
                        var_str = str(var)
                        expr_str = expr_str.replace(var_str, f"q[:, {k}]")
                    
                    # Add Jacobian element assignment
                    jac_code.append(f"        J[:, {i}, {j}] = {expr_str}")
        
        # Handle the end effector and unknown links
        if first_link:
            # No blocks were generated, default case with end effector
            jac_code.append(f"    if link_name == '{finger['end']}':")
            jac_code.append("        pass  # End effector will return zeros")
        else:
            # Add default case for end effector
            jac_code.append(f"    elif link_name == '{finger['end']}':")
            jac_code.append("        pass  # End effector with empty block")
        
        # Add else case for unknown links 
        jac_code.extend([
            "    else:",
            f"        raise ValueError(f\"Unknown link name {{link_name}} for {finger_name} finger\")",
            "",
            "    # Return the Jacobian matrix",
            "    return J",
            ""
        ])
        
        # Add the cached single-input helper
        jac_code.extend([
            "@lru_cache(maxsize=128)",
            f"def {finger_name}_jacobian_single(q_tuple: Tuple[float, ...], link_name: str) -> torch.Tensor:",
            '    """Cached Jacobian implementation for single joint configuration."""',
            "    q = torch.tensor([q_tuple], dtype=torch.float32)",
            f"    return {finger_name}_jacobian_impl(q, link_name)[0]",
            ""
        ])
        
        # Add the main function with optimized input handling
        jac_code.extend([
            f"def {finger_name}_jacobian(q: torch.Tensor, link_name: Optional[str] = None, device: Optional[torch.device] = None) -> torch.Tensor:",
            '    """',
            f"    Jacobian calculation for the {finger_name} finger with PyTorch.",
            "",
            f"    Args:",
            f"        q: Joint angles in radians, shape [batch_size, {num_joints}] or [{num_joints}]",
            f"        link_name: Optional name of the link to compute Jacobian for",
            f"        device: Optional device for computation (CPU/GPU)",
            "",
            f"    Returns:",
            f"        Jacobian matrix of shape [batch_size, 6, {num_joints}] or [6, {num_joints}]",
            "",
            f"    Notes:",
            f"        Defaults to end effector link if link_name is not specified.",
            '    """',
            f"    # Input validation",
            f"    if q.shape[-1] != {num_joints}:",
            f'        raise ValueError(f"Expected {num_joints} joint values, got {{q.shape[-1]}}")',
            "",
            "    # Handle device placement",
            "    if device is None:",
            "        device = q.device if torch.is_tensor(q) else torch.device('cpu')",
            "    elif torch.is_tensor(q) and q.device != device:",
            "        q = q.to(device)",
            "",
            "    # Default to end effector if not specified",
            f"    if link_name is None or link_name == '':",
            f"        link_name = '{finger['end']}'",
            "",
            "    # Handle various input formats",
            "    unbatched = False",
            "    if not torch.is_tensor(q):",
            "        q = torch.tensor(q, dtype=torch.float32, device=device)",
            "",
            "    if q.dim() == 1:",
            "        unbatched = True",
            "        q_tuple = tuple(q.cpu().numpy().tolist())",
            "        return {}_jacobian_single(q_tuple, link_name).to(device)".format(finger_name),
            "",
            "    if q.dim() != 2:",
            "        q = q.reshape(-1, {})".format(num_joints),
            "",
            "    # Call implementation function",
            "    J = {}_jacobian_impl(q, link_name)".format(finger_name),
            "",
            "    # Handle unbatched case",
            "    return J.squeeze(0) if unbatched else J",
            ""
        ])
        
        # Write to file and load
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write('\n'.join(jac_code))
        
        # Create function object for immediate use
        try:
            import importlib.util
            import sys
            
            module_name = f"{finger_name}_jacobian"
            if module_name in sys.modules:
                del sys.modules[module_name]
                
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
                
            if hasattr(module, f"{finger_name}_jacobian"):
                finger["jac_fn"] = getattr(module, f"{finger_name}_jacobian")
                print(f"Successfully loaded {finger_name} Jacobian function from file")
            else:
                raise AttributeError(f"Module does not contain function '{finger_name}_jacobian'")
        except Exception as e:
            print(f"Warning: Could not compile Jacobian function for {finger_name}: {e}")
                
            # Try direct execution as fallback
            try:
                print(f"Trying fallback method for {finger_name} jacobian")
                namespace = {}
                exec('\n'.join(jac_code), namespace)
                finger["jac_fn"] = namespace[f"{finger_name}_jacobian"]
                print(f"Successfully loaded {finger_name} Jacobian function via fallback")
            except Exception as e2:
                print(f"Fatal error loading {finger_name} jacobian: {e2}")

def main() -> None:
    """
    Main function to test symbolic kinematics with caching and compare to PyTorch Kinematics.
    Also compares with the optimized HandKinematicsModel using JIT compilation.
    
    Returns:
        None
    """
    import pytorch_kinematics as pk
    import os
    import re
    from pathlib import Path
    import torch
    from typing import Dict, List, Optional, Tuple, Any
    
    # Import the optimized HandKinematicsModel
    from kinematics.optimized_model import HandKinematicsModel
    
    # URDF path verification
    urdf_path = '/home/abhinav/Documents/github/isaacgym-arm-envs/isaac_victor_envs/assets/xela_models/allegro_hand_right.urdf'
    
    # Check if file exists
    urdf_file = Path(urdf_path)
    if not urdf_file.exists():
        print(f"ERROR: URDF file not found at {urdf_path}")
        # Try to find the file in a different location
        alternative_paths = [
            "./allegro_hand_right.urdf",
            "/home/abhinav/Documents/ccai/examples/allegro_hand_right.urdf"
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                urdf_path = alt_path
                print(f"Found URDF at alternative location: {urdf_path}")
                break
        else:
            print("ERROR: Could not find URDF file. Please check the path.")
            return
    
    # Parse URDF file
    try:
        with open(urdf_path, 'r') as f:
            urdf_content = f.read()
        
        # Parse URDF with URDF parser
        urdf_robot = URDF.from_xml_string(urdf_content)
        print(f"URDF file parsed successfully")
    except Exception as e:
        print(f"ERROR: Failed to parse URDF file: {e}")
        return
    
    # Common base link for all fingers
    base_link = "allegro_hand_base_link"
    
    # Define fingers in the correct order: index, middle, ring, thumb
    fingers = [
        {"name": "index", "root": base_link, "end": "hitosashi_ee"},
        {"name": "middle", "root": base_link, "end": "naka_ee"},
        {"name": "ring", "root": base_link, "end": "kusuri_ee"},
        {"name": "thumb", "root": base_link, "end": "oya_ee"}
    ]
    
    # Extract prefixes from end links for joint identification
    finger_prefixes = {}
    for finger in fingers:
        # Extract prefix before '_ee'
        prefix = finger["end"].split("_ee")[0]
        finger_prefixes[finger["name"]] = prefix
    
    print("Identifying finger joints based on naming patterns...")
    
    # Get all joints from URDF
    all_joints = urdf_robot.joints
    all_joint_names = [joint.name for joint in all_joints if joint.type == "revolute"]
    print(f"Found {len(all_joint_names)} revolute joints in URDF: {all_joint_names}")
    
    # Identify finger-specific joints using naming patterns
    finger_joint_names: Dict[str, List[str]] = {}
    finger_joint_limits: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    
    for finger_name, prefix in finger_prefixes.items():
        # Find all joints that contain the finger prefix
        matching_joints = [
            joint for joint in all_joints 
            if joint.type == "revolute" and prefix in joint.name
        ]
        
        # Sort joints by their numerical index if possible
        # This ensures correct order: joint_0, joint_1, etc.
        def extract_joint_index(joint_name: str) -> int:
            """Extract numeric index from joint name or return 0 if not found."""
            match = re.search(r'(\d+)$', joint_name.split('_joint_')[-1])
            return int(match.group(1)) if match else 0
        
        matching_joints.sort(key=lambda j: extract_joint_index(j.name))
        
        # Store joint names
        joint_names = [joint.name for joint in matching_joints]
        finger_joint_names[finger_name] = joint_names
        
        # Extract joint limits
        lower_limits = []
        upper_limits = []
        
        for joint in matching_joints:
            if hasattr(joint, 'limit') and joint.limit is not None:
                lower_limits.append(joint.limit.lower)
                upper_limits.append(joint.limit.upper)
            else:
                # Default limits if not specified
                lower_limits.append(-3.14)
                upper_limits.append(3.14)
        
        # Store joint limits as tensors
        finger_joint_limits[finger_name] = (
            torch.tensor(lower_limits, dtype=torch.float32),
            torch.tensor(upper_limits, dtype=torch.float32)
        )
        
        print(f"Finger {finger_name}: {len(joint_names)} joints: {joint_names}")
        print(f"  Joint limits: Low: {lower_limits}, High: {upper_limits}")
    
    # Create mapping to indices in the complete joint list
    joint_indices: Dict[str, List[int]] = {}
    for name, joint_list in finger_joint_names.items():
        indices = []
        for joint in joint_list:
            if joint in all_joint_names:
                indices.append(all_joint_names.index(joint))
        joint_indices[name] = indices
        print(f"Finger {name} joint indices in full chain: {indices}")
    
    # Define joint counts for each finger
    joint_counts = {name: len(joints) for name, joints in finger_joint_names.items()}
    
    # Initialize PyTorch Kinematics for comparison (if possible)
    pk_comparison = True
    try:
        # Build the complete kinematic chain
        full_chain = pk.build_chain_from_urdf(urdf_content)
        full_chain = full_chain.to(device=torch.device('cuda:0'))
        pk_all_joint_names = full_chain.get_joint_parameter_names()
        
        # Create serial chains for each finger
        finger_serial_chains = {}
        
        for finger in fingers:
            name = finger["name"]
            end = finger["end"]
            try:
                # Create serial chain from base to end effector
                serial_chain = pk.build_serial_chain_from_urdf(
                    urdf_content,
                    end_link_name=end,
                    root_link_name=base_link
                )
                finger_serial_chains[name] = serial_chain
            except Exception as e:
                print(f"Warning: Could not build serial chain for {name}: {e}")
        
        print("PyTorch Kinematics chains initialized successfully")
    except Exception as e:
        print(f"ERROR: Failed to initialize PyTorch Kinematics: {e}")
        print("Skipping PyTorch Kinematics comparison")
        pk_comparison = False
    
    # Set up cache directory for symbolic kinematics
    cache_dir = "./kinematics"
    
    # Initialize symbolic kinematics with correct joint counts
    print("Initializing symbolic kinematics...")
    hand = SymbolicKinematics(
        urdf_path=urdf_path,
        fingers=fingers,
        cache_dir=cache_dir,
        use_cache=True,
        joint_counts=joint_counts
    )
    
    # Initialize optimized HandKinematicsModel with and without JIT
    print("Initializing optimized HandKinematicsModel...")
    hand_model_jit = HandKinematicsModel(use_jit=True)
    hand_model_no_jit = HandKinematicsModel(use_jit=False)
    print("HandKinematicsModel initialization complete")
    
    # Create batch of joint angles within limits
    batch_size = 5
    
    # Sample joint angles within limits for each finger
    joint_angles: Dict[str, torch.Tensor] = {}
    
    # If PyTorch Kinematics is available, create full joint tensor
    if pk_comparison:
        all_joints = torch.zeros(batch_size, len(all_joint_names), dtype=torch.float32, device='cuda:0')
    else:
        all_joints = None
    
    print("\nSampling joint angles within limits...")
    for name in joint_counts.keys():
        lower, upper = finger_joint_limits[name]
        num_joints = len(lower)
        
        # Sample uniformly between lower and upper limits
        finger_joints = torch.zeros(batch_size, num_joints, dtype=torch.float32, device='cuda:0' if torch.cuda.is_available() else 'cpu')
        for j in range(num_joints):
            finger_joints[:, j] = torch.rand(batch_size) * (upper[j] - lower[j]) + lower[j]
        
        # Store the sampled joints
        joint_angles[name] = finger_joints
        
        # Place these joints in the all_joints tensor at the correct indices
        if all_joints is not None:
            for j, idx in enumerate(joint_indices[name]):
                all_joints[:, idx] = finger_joints[:, j]
                
        print(f"  {name}: sampled {num_joints} joints within limits [{lower[0]:.2f}, {upper[0]:.2f}]")
    
    print("\n========== FORWARD KINEMATICS TIMING ==========")
    
    # ===== Method 1: Symbolic Kinematics with full joint vector and indices =====
    print("\n--- Method 1: Symbolic FK with Joint Indices ---")
    start = time.perf_counter()
    
    # Create input dictionary for each finger with full joint vector
    finger_inputs = {}
    for name in joint_angles.keys():
        if all_joints is not None:
            finger_inputs[name] = all_joints  # Full joint vector for each finger
        else:
            finger_inputs[name] = joint_angles[name]  # Use finger-specific joints as fallback
    
    sym_fk_results1 = hand.forward_kinematics(
        th=finger_inputs, 
        joint_indices=joint_indices
    )
    sym_fk_time1 = time.perf_counter() - start
    print(f"FK time: {sym_fk_time1 * 1000:.2f} ms")
    
    # ===== Method 2: Symbolic Kinematics with per-finger tensors =====
    print("\n--- Method 2: Symbolic FK with Per-finger Tensors ---")
    start = time.perf_counter()
    sym_fk_results2 = hand.forward_kinematics(joint_angles)
    sym_fk_time2 = time.perf_counter() - start
    print(f"FK time: {sym_fk_time2 * 1000:.2f} ms")
    
    # ===== Method 5: HandKinematicsModel with JIT compilation =====
    print("\n--- Method 5: HandKinematicsModel with JIT ---")
    start = time.perf_counter()
    jit_fk_results = {}
    for name, q in joint_angles.items():
        jit_fk_results[name] = hand_model_jit.forward_kinematics(name, q)
    jit_fk_time = time.perf_counter() - start
    print(f"FK time: {jit_fk_time * 1000:.2f} ms")
    
    # ===== Method 6: HandKinematicsModel without JIT compilation =====
    print("\n--- Method 6: HandKinematicsModel without JIT ---")
    start = time.perf_counter()
    no_jit_fk_results = {}
    for name, q in joint_angles.items():
        no_jit_fk_results[name] = hand_model_no_jit.forward_kinematics(name, q)
    no_jit_fk_time = time.perf_counter() - start
    print(f"FK time: {no_jit_fk_time * 1000:.2f} ms")
    
    # ===== Method 7: HandKinematicsModel forward pass =====
    print("\n--- Method 7: HandKinematicsModel forward() Method ---")
    start = time.perf_counter()
    model_forward_results = hand_model_jit.forward(joint_angles, compute_jacobian=False)
    model_forward_time = time.perf_counter() - start
    print(f"FK time (using forward() method): {model_forward_time * 1000:.2f} ms")
    
    # Only run PyTorch Kinematics comparison if initialization succeeded
    if pk_comparison:
        # ===== Method 3: PyTorch Kinematics with complete chain from URDF =====
        print("\n--- Method 3: PyTorch Kinematics with Complete Chain ---")
        
        # Calculate FK using the full chain for each finger
        start = time.perf_counter()
        pk_fk_results = {}
        
        # Compute FK with the full chain
        all_poses = full_chain.forward_kinematics(all_joints)
        pk_fk_time = time.perf_counter() - start
        
        # Extract end effector poses for each finger
        for finger in fingers:
            name = finger["name"]
            end = finger["end"]
            pk_fk_results[name] = all_poses[end]
            
        print(f"FK time: {pk_fk_time * 1000:.2f} ms")
        
        # ===== Method 4: PyTorch Kinematics with Serial Chains (per finger) =====
        print("\n--- Method 4: PyTorch Kinematics with Serial Chains ---")
        start = time.perf_counter()
        
        serial_fk_results = {}
        for name, finger_joints in joint_angles.items():
            if name in finger_serial_chains:
                chain = finger_serial_chains[name]
                serial_fk_results[name] = chain.forward_kinematics(finger_joints)
                
        serial_fk_time = time.perf_counter() - start
        print(f"FK time (with serial chains): {serial_fk_time * 1000:.2f} ms")
    
    print("\n========== JACOBIAN TIMING ==========")
    
    # ===== Method 1: Symbolic Jacobian with full joint vector and indices =====
    print("\n--- Method 1: Symbolic Jacobian with Joint Indices ---")
    start = time.perf_counter()
    
    sym_jac_results1 = hand.jacobian(
        th=finger_inputs,
        joint_indices=joint_indices
    )
    sym_jac_time1 = time.perf_counter() - start
    print(f"Jacobian time: {sym_jac_time1 * 1000:.2f} ms")
    
    # ===== Method 2: Symbolic Jacobian with per-finger tensors =====
    print("\n--- Method 2: Symbolic Jacobian with Per-finger Tensors ---")
    start = time.perf_counter()
    sym_jac_results2 = hand.jacobian(joint_angles)
    sym_jac_time2 = time.perf_counter() - start
    print(f"Jacobian time: {sym_jac_time2 * 1000:.2f} ms")
    
    # ===== Method 5: HandKinematicsModel with JIT compilation =====
    print("\n--- Method 5: HandKinematicsModel with JIT ---")
    start = time.perf_counter()
    jit_jac_results = {}
    for name, q in joint_angles.items():
        jit_jac_results[name] = {
            finger["end"]: hand_model_jit.jacobian(name, q)
            for finger in fingers if finger["name"] == name
        }
    jit_jac_time = time.perf_counter() - start
    print(f"Jacobian time: {jit_jac_time * 1000:.2f} ms")
    
    # ===== Method 6: HandKinematicsModel without JIT compilation =====
    print("\n--- Method 6: HandKinematicsModel without JIT ---")
    start = time.perf_counter()
    no_jit_jac_results = {}
    for name, q in joint_angles.items():
        no_jit_jac_results[name] = {
            finger["end"]: hand_model_no_jit.jacobian(name, q)
            for finger in fingers if finger["name"] == name
        }
    no_jit_jac_time = time.perf_counter() - start
    print(f"Jacobian time: {no_jit_jac_time * 1000:.2f} ms")
    
    # ===== Method 7: HandKinematicsModel forward pass with Jacobian =====
    print("\n--- Method 7: HandKinematicsModel forward() Method with Jacobian ---")
    start = time.perf_counter()
    model_forward_jac_results = hand_model_jit.forward(joint_angles, compute_jacobian=True)
    model_forward_jac_time = time.perf_counter() - start
    print(f"Jacobian time (using forward() method): {model_forward_jac_time * 1000:.2f} ms")
    
    # Only run PyTorch Kinematics comparison if initialization succeeded
    if pk_comparison:
        # ===== Method 3: PyTorch Kinematics Jacobian with Serial Chains =====
        print("\n--- Method 3: PyTorch Kinematics Jacobian with Serial Chains ---")
        
        # Import the Jacobian calculation function
        from pytorch_kinematics.jacobian import calc_jacobian
        
        # Calculate Jacobian using the serial chains for each finger (batched)
        start = time.perf_counter()
        pk_jac_results = {}
        
        # Process each finger using its dedicated serial chain
        for name, finger_joints in joint_angles.items():
            if name not in finger_serial_chains:
                print(f"Warning: No serial chain available for {name}")
                continue
                
            # Get the serial chain for this finger
            chain = finger_serial_chains[name]
            
            # Calculate Jacobian for the entire batch at once
            jacobians = chain.jacobian(finger_joints)
            # Store the result
            pk_jac_results[name] = jacobians

        pk_jac_time = time.perf_counter() - start
        print(f"Jacobian time: {pk_jac_time * 1000:.2f} ms")
    
    # Performance comparison summary
    print("\n========== PERFORMANCE COMPARISON ==========")
    print(f"Forward Kinematics:")
    print(f"  Symbolic (Joint Indices): {sym_fk_time1 * 1000:.2f} ms")
    print(f"  Symbolic (Per-finger): {sym_fk_time2 * 1000:.2f} ms")
    print(f"  HandKinematicsModel with JIT: {jit_fk_time * 1000:.2f} ms")
    print(f"  HandKinematicsModel without JIT: {no_jit_fk_time * 1000:.2f} ms")
    print(f"  HandKinematicsModel forward(): {model_forward_time * 1000:.2f} ms")
    
    if pk_comparison:
        print(f"  PyTorch Kinematics (Full Chain): {pk_fk_time * 1000:.2f} ms")
        print(f"  PyTorch Kinematics (Serial Chains): {serial_fk_time * 1000:.2f} ms")
    
    # Calculate speedups relative to PyTorch Kinematics if available
    if pk_comparison:
        print("\nSpeedups (relative to PyTorch Kinematics Full Chain):")
        print(f"  Symbolic (Joint Indices): {pk_fk_time / sym_fk_time1:.2f}x")
        print(f"  Symbolic (Per-finger): {pk_fk_time / sym_fk_time2:.2f}x")
        print(f"  HandKinematicsModel with JIT: {pk_fk_time / jit_fk_time:.2f}x")
        print(f"  HandKinematicsModel without JIT: {pk_fk_time / no_jit_fk_time:.2f}x")
        print(f"  HandKinematicsModel forward(): {pk_fk_time / model_forward_time:.2f}x")
        print(f"  PyTorch Kinematics (Serial Chains): {pk_fk_time / serial_fk_time:.2f}x")
    
    print(f"\nJacobian:")
    print(f"  Symbolic (Joint Indices): {sym_jac_time1 * 1000:.2f} ms")
    print(f"  Symbolic (Per-finger): {sym_jac_time2 * 1000:.2f} ms")
    print(f"  HandKinematicsModel with JIT: {jit_jac_time * 1000:.2f} ms")
    print(f"  HandKinematicsModel without JIT: {no_jit_jac_time * 1000:.2f} ms")
    print(f"  HandKinematicsModel forward() with Jacobian: {model_forward_jac_time * 1000:.2f} ms")
    
    if pk_comparison:
        print(f"  PyTorch Kinematics: {pk_jac_time * 1000:.2f} ms")
    
    # Calculate speedups for Jacobian
    if pk_comparison:
        print("\nSpeedups (relative to PyTorch Kinematics):")
        print(f"  Symbolic (Joint Indices): {pk_jac_time / sym_jac_time1:.2f}x")
        print(f"  Symbolic (Per-finger): {pk_jac_time / sym_jac_time2:.2f}x")
        print(f"  HandKinematicsModel with JIT: {pk_jac_time / jit_jac_time:.2f}x")
        print(f"  HandKinematicsModel without JIT: {pk_jac_time / no_jit_jac_time:.2f}x")
        print(f"  HandKinematicsModel forward() with Jacobian: {pk_jac_time / model_forward_jac_time:.2f}x")


if __name__ == "__main__":
    main()