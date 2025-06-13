"""
Model management functionality for Allegro screwdriver recovery.
Contains trajectory sampler loading and configuration logic.
"""

import torch
from torch import nn
import pathlib

from ccai.models.trajectory_samplers_sac import TrajectorySampler


class ModelManager:
    """Handles model loading and configuration for recovery experiments."""
    
    def __init__(self, config, params, ccai_path):
        self.config = config
        self.params = params
        self.ccai_path = ccai_path
        
    def load_trajectory_samplers(self):
        """Load trajectory samplers based on configuration."""
        trajectory_sampler = None
        trajectory_sampler_orig = None
        classifier = None
        
        model_path = self.config.get('model_path', None)
        task_model_path = self.config.get('task_model_path', None)
        
        if model_path is not None:
            problem_for_sampler = None
            if 'type' not in self.config:
                self.config['type'] = 'diffusion'

            loading_recovery_model = self.params.get('task_model_path', None) is not None
            
            if self.params['recovery_controller'] != 'mppi':
                T_for_diff = self.config['T'] if loading_recovery_model else self.config['T_orig']
                trajectory_sampler = self._load_sampler(
                    model_path, dim_mults=(1,2,4), T=T_for_diff, recovery=loading_recovery_model)

            if task_model_path is not None:
                trajectory_sampler_orig = self._load_sampler(
                    task_model_path, dim_mults=(1,2,4), T=self.config['T_orig'], recovery=False)
                
                if not self.config.get('generate_context', False):
                    classifier = self._create_classifier()
                    
            else:
                trajectory_sampler_orig = trajectory_sampler
                
        return trajectory_sampler, trajectory_sampler_orig, classifier
    
    def _load_sampler(self, path, dim_mults=(1,2), T=None, recovery=False):
        """Load a single trajectory sampler."""
        if T is None:
            T = self.config['T']
            
        dx = 15 + (1 if self.config['sine_cosine'] else 0)
        
        trajectory_sampler = TrajectorySampler(
            T=T + 1, 
            dx=dx, 
            du=21, 
            type=self.config['type'],
            timesteps=256, 
            hidden_dim=128,
            context_dim=3, 
            problem=None,
            guided=self.config.get('use_guidance', False),
            state_control_only=self.config.get('state_control_only', False),
            initial_threshold=self.config.get('likelihood_threshold', -15),
            new_projection=True,
            generate_context=recovery,
            trajectory_condition=True,
        )
        
        d = torch.load(f'{self.ccai_path}/{path}', map_location=torch.device(self.params['device']))
        
        trajectory_sampler.model.diffusion_model.classifier = None
        d = {k:v for k, v in d.items() if 'classifier' not in k}
        trajectory_sampler.load_state_dict(d, strict=recovery)
        trajectory_sampler.to(device=self.params['device'])
        trajectory_sampler.send_norm_constants_to_submodels()
        trajectory_sampler.model.diffusion_model.subsampled_t = '5_10_15' in self.config['experiment_name']
        trajectory_sampler.model.diffusion_model.classifier = None
        trajectory_sampler.model.diffusion_model.cutoff_timesteps = 128
        
        return trajectory_sampler
    
    def _create_classifier(self):
        """Create and load the contact mode classifier."""
        classifier = nn.Sequential(
            nn.Linear(16, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3),  # 3 binary outputs for contact mode
        ).to(self.params['device'])
        
        model_path_classifier = self.config.get('model_path_classifier', None)
        if model_path_classifier:
            classifier_d = torch.load(f'{self.ccai_path}/{model_path_classifier}', 
                                    map_location=torch.device(self.params['device']))
            classifier.load_state_dict(classifier_d)
            classifier.eval()
            
        return classifier 