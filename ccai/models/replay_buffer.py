from typing import Tuple, List, Dict, Any, Optional, Union
import numpy as np
import torch
import random
from collections import deque
from dataclasses import dataclass


class SumTree:
    """
    A sum tree data structure for efficient priority-based sampling.
    
    The leaf nodes contain priority values while internal nodes store the sum
    of their children's priorities. This allows O(log n) updates and sampling
    based on priority weights.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize the SumTree with a fixed capacity.
        
        Args:
            capacity: Maximum number of leaf nodes (experiences)
        """
        # Tree capacity must be a power of 2 for a complete binary tree
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2
            
        # Build a complete binary tree with 2*capacity - 1 nodes
        # (capacity leaf nodes + capacity-1 internal nodes)
        self.tree = np.zeros(2 * self.capacity - 1)
        
        # Data storage for experiences (same size as leaf nodes)
        self.data = [None] * self.capacity
        
        # Index to track the next position to write in the circular buffer
        self.write_index = 0
        
        # Counter for the number of entries in the buffer
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        """
        Propagate priority changes upward through the tree.
        
        Args:
            idx: Node index to start propagation from
            change: Change in value to be propagated
        """
        # Get parent index
        parent = (idx - 1) // 2
        
        # Update parent's value
        self.tree[parent] += change
        
        # Continue propagating up if not at root
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """
        Retrieve the leaf node index containing value s.
        
        Args:
            idx: Current node index
            s: Target value to find
            
        Returns:
            Index of the leaf node
        """
        # Calculate indices of left and right children
        left = 2 * idx + 1
        right = left + 1
        
        # If at leaf node, return the current index
        if left >= len(self.tree):
            return idx
        
        # Search left subtree if s <= left child's value
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        # Otherwise, search right subtree with adjusted s value
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """
        Get the total priority in the tree.
        
        Returns:
            Sum of all priorities
        """
        return self.tree[0]

    def add(self, priority: float, data: Any) -> None:
        """
        Add a new experience with given priority.
        
        Args:
            priority: Priority value for the experience
            data: Experience data to store
        """
        # Calculate tree index for the current write position
        tree_idx = self.write_index + self.capacity - 1
        
        # Store data
        self.data[self.write_index] = data
        
        # Update tree with new priority
        self.update(tree_idx, priority)
        
        # Move write index to next position (circular)
        self.write_index = (self.write_index + 1) % self.capacity
        
        # Increment entry counter if not at full capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, tree_idx: int, priority: float) -> None:
        """
        Update priority of a specific leaf node.
        
        Args:
            tree_idx: Index of the leaf node
            priority: New priority value
        """
        # Calculate change in priority
        change = priority - self.tree[tree_idx]
        
        # Update leaf node
        self.tree[tree_idx] = priority
        
        # Propagate changes upward
        self._propagate(tree_idx, change)

    def get(self, s: float) -> Tuple[int, int, Any]:
        """
        Get an experience based on a sample value s.
        
        Args:
            s: Sample value between 0 and total priority
            
        Returns:
            Tuple of (tree_idx, priority, data)
        """
        # Find the leaf node containing value s
        tree_idx = self._retrieve(0, s)
        
        # Calculate the corresponding data index
        data_idx = tree_idx - self.capacity + 1
        
        return tree_idx, self.tree[tree_idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer that samples important transitions more frequently.
    
    Uses a sum tree structure for efficient sampling based on priorities.
    Implements importance sampling weights to correct for the bias introduced by prioritized sampling.
    """
    
    def __init__(self, 
                 capacity: int, 
                 alpha: float = 0.6, 
                 beta: float = 0.4, 
                 beta_increment: float = 0.001,
                 epsilon: float = 0.01):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            alpha: Controls the amount of prioritization (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Increment for beta parameter annealing
            epsilon: Small constant added to priorities to ensure non-zero sampling probability
        """
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        self.max_priority = 1.0  # Initial max priority
        
    def add(self, experience: Any, error: Optional[float] = None) -> None:
        """
        Add an experience to the buffer with calculated priority.
        
        Args:
            experience: Experience data to store
            error: TD error or other priority measure (if None, use max priority)
        """
        # If error is not provided, use max priority to ensure exploration
        if error is None:
            priority = self.max_priority
        else:
            # Convert error to priority using alpha parameter
            priority = (abs(error) + self.epsilon) ** self.alpha
            
        # Update max priority if needed
        self.max_priority = max(self.max_priority, priority)
        
        # Add to sum tree
        self.tree.add(priority, experience)
    
    def sample(self, batch_size: int) -> Tuple[List[Any], List[int], np.ndarray]:
        """
        Sample a batch of experiences based on their priorities.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (experiences, indices, importance_sampling_weights)
        """
        # Storage for sampled experiences and their indices
        experiences = []
        indices = []
        weights = np.zeros(batch_size, dtype=np.float32)
        
        # Calculate segment size
        total = self.tree.total()
        segment = total / batch_size
        
        # Increment beta for annealing
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Calculate min priority for normalization
        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / total if self.tree.n_entries == self.tree.capacity else 0
        
        # Sample experiences
        for i in range(batch_size):
            # Uniformly sample from segment
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            
            # Get experience by sampling value
            idx, priority, experience = self.tree.get(s)
            
            # Calculate sampling probability
            sampling_prob = priority / total
            
            # Calculate importance sampling weight
            weights[i] = (sampling_prob / max(min_prob, 1e-10)) ** (-self.beta)
            
            indices.append(idx)
            experiences.append(experience)
        
        # Normalize weights to scale updates correctly
        weights = weights / np.max(weights)
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: List[int], errors: List[float]) -> None:
        """
        Update priorities of experiences based on new TD errors.
        
        Args:
            indices: Tree indices of the experiences to update
            errors: New TD errors for priority calculation
        """
        for idx, error in zip(indices, errors):
            # Convert error to priority using alpha parameter
            priority = (abs(error) + self.epsilon) ** self.alpha
            
            # Update max priority if needed
            self.max_priority = max(self.max_priority, priority)
            
            # Update in the sum tree
            self.tree.update(idx, priority)
    
    def __len__(self) -> int:
        """
        Get the current number of experiences in the buffer.
        
        Returns:
            Number of experiences stored
        """
        return self.tree.n_entries
