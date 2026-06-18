"""
Model management functionality for Allegro screwdriver recovery.
Contains trajectory sampler loading and configuration logic.
"""

import torch
from torch import nn
import pathlib
import os
import re

from ccai.models.trajectory_samplers_sac import TrajectorySampler


class ModelManager:
    """Handles model loading and configuration for recovery experiments."""
    
    def __init__(self, config, params, ccai_path):
        self.config = config
        self.params = params
        self.ccai_path = ccai_path
        
    def load_trajectory_samplers(self, obj_dof=3):
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
                    model_path, dim_mults=(1,2,4), T=T_for_diff, recovery=loading_recovery_model, obj_dof=obj_dof)
                # trajectory_sampler.warmup_model(warmup_batch_size=16, warmup_horizon=None)

            if task_model_path is not None:
                trajectory_sampler_orig = self._load_sampler(
                    task_model_path, dim_mults=(1,2,4), T=self.config['T_orig'], recovery=False)
                # trajectory_sampler_orig.warmup_model(warmup_batch_size=16, warmup_horizon=None)
                
                if not self.config.get('generate_context', False):
                    classifier = self._create_classifier()
                    
            else:
                trajectory_sampler_orig = trajectory_sampler
                
        return trajectory_sampler, trajectory_sampler_orig, classifier
    
    def _load_sampler(self, path, dim_mults=(1,2), T=None, recovery=False, obj_dof=3):
        """Load a single trajectory sampler."""
        if T is None:
            T = self.config['T']
            
        dx = 12 + obj_dof + (1 if self.config['sine_cosine'] else 0)
        checkpoint_path = f'{self.ccai_path}/{path}'
        d = self._unwrap_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
        d = {k:v for k, v in d.items() if 'classifier' not in k}

        role_prefix = 'model' if recovery else 'task_model'
        hidden_dim = int(self.config.get(f'{role_prefix}_hidden_dim', self.config.get('hidden_dim', 128)))
        inferred = self._infer_sampler_architecture(d)
        if inferred.get('hidden_dim') is not None:
            hidden_dim = inferred['hidden_dim']
        dim_mults = self._normalize_dim_mults(self.config.get(f'{role_prefix}_dim_mults', dim_mults))
        if inferred.get('dim_mults') is not None:
            dim_mults = inferred['dim_mults']
        
        trajectory_sampler = TrajectorySampler(
            T=T + 1, 
            dx=dx, 
            du=21, 
            type=self.config['type'],
            timesteps=256,#128 if recovery else 256, 
            hidden_dim=hidden_dim,
            context_dim=3, 
            problem=None,
            guided=self.config.get('use_guidance', False),
            state_control_only=self.config.get('state_control_only', False),
            initial_threshold=self.config.get('likelihood_threshold', -15),
            new_projection=True,
            generate_context=recovery,
            trajectory_condition=True,
            dim_mults=dim_mults,
        )
        
        trajectory_sampler.model.diffusion_model.classifier = None
        trajectory_sampler.load_state_dict(d, strict=recovery)
        trajectory_sampler.to(device=self.params['device'])
        trajectory_sampler.send_norm_constants_to_submodels()
        trajectory_sampler.model.diffusion_model.subsampled_t = '5_10_15' in self.config['experiment_name']
        trajectory_sampler.model.diffusion_model.classifier = None
        trajectory_sampler.model.diffusion_model.cutoff_timesteps = 128
        
        # Set up compilation cache directory for the diffusion model
        cache_dir = os.path.join(self.ccai_path, 'compiled_models_cache')
        if hasattr(trajectory_sampler.model.diffusion_model, 'set_compilation_cache_dir'):
            trajectory_sampler.model.diffusion_model.set_compilation_cache_dir(cache_dir)
            print(f"Set compilation cache directory to: {cache_dir}")
        
        return trajectory_sampler

    @staticmethod
    def _unwrap_state_dict(checkpoint):
        """Accept raw state_dict checkpoints and common wrapped checkpoint forms."""
        if isinstance(checkpoint, dict):
            for key in ('state_dict', 'model_state_dict'):
                value = checkpoint.get(key)
                if isinstance(value, dict):
                    return value
        return checkpoint

    @staticmethod
    def _normalize_dim_mults(dim_mults):
        if dim_mults is None:
            return None
        if isinstance(dim_mults, str):
            dim_mults = dim_mults.strip().strip('[]()')
            if not dim_mults:
                return None
            return tuple(int(part.strip()) for part in dim_mults.split(',') if part.strip())
        return tuple(int(part) for part in dim_mults)

    @staticmethod
    def _infer_sampler_architecture(state_dict):
        pattern = re.compile(
            r'^model\.diffusion_model\.model\.(?:temporal_unet\.)?'
            r'downs\.(\d+)\.0\.blocks\.0\.block\.0\.weight$'
        )
        channels_by_level = {}
        for key, tensor in state_dict.items():
            match = pattern.match(key)
            if match is None or not hasattr(tensor, 'shape') or len(tensor.shape) < 1:
                continue
            channels_by_level[int(match.group(1))] = int(tensor.shape[0])
        if not channels_by_level:
            return {}

        channels = [channels_by_level[idx] for idx in sorted(channels_by_level)]
        hidden_dim = channels[0]
        if hidden_dim <= 0:
            return {}
        dim_mults = tuple(channel // hidden_dim for channel in channels)
        return {'hidden_dim': hidden_dim, 'dim_mults': dim_mults}
    
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
