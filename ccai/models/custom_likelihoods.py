import torch
import gpytorch
from gpytorch.likelihoods import Likelihood
from typing import Any, Optional, Tuple
import numpy as np

class CDFBernoulliLikelihood(Likelihood):
    """
    Custom Bernoulli likelihood that uses a sigmoid activation with learnable
    bias and temperature parameters.
    
    This replaces the standard GPyTorch BernoulliLikelihood (which uses normal CDF)
    with a sigmoid implementation that has learnable parameters to better adapt to
    the diffusion model likelihood distribution.
    
    Attributes:
        bias: Learnable bias parameter (initialized with threshold)
    """
    
    def __init__(
        self, 
        initial_threshold: float = -15.0, 
        # initial_temp: float = .1, 
        num_samples: int = 15
    ):
        """
        Initialize custom Bernoulli likelihood.
        
        Args:
            initial_threshold: Initial value for the bias parameter
            initial_temp: Initial value for the temperature parameter
            num_samples: Number of MC samples for expected log prob calculations
        """
        super().__init__()
        self.threshold = torch.nn.Parameter(torch.tensor([initial_threshold]))
        self.log_global_scale_increase = torch.nn.Parameter(torch.tensor([np.log(10.0)]))
        # self.temperature = torch.nn.Parameter(torch.tensor([initial_temp]))
        self.num_samples = num_samples
        
    def forward(self, function_samples: torch.Tensor) -> torch.distributions.Bernoulli:
        """
        Applies the likelihood to compute probabilities from latent function values.
        
        Args:
            function_samples: Samples from the latent function (GP output)
            
        Returns:
            A Bernoulli distribution with probabilities determined by sigmoid(temperature * (input + bias))
        """
        # Apply temperature and bias, then sigmoid
        output_probs = function_samples.cdf(self.threshold)
        
        # Return Bernoulli distribution
        return torch.distributions.Bernoulli(probs=output_probs)
    
    def expected_log_prob(self, target: torch.Tensor, function_dist: gpytorch.distributions.MultivariateNormal) -> torch.Tensor:
        """
        Compute the expected log likelihood of the observations given the function distribution.
        
        Args:
            target: Binary observations (0 or 1)
            function_dist: GPyTorch MultivariateNormal output from the GP
            
        Returns:
            Expected log likelihood
        """
        # Monte Carlo integration for the expected log prob
        function_samples = function_dist.rsample(torch.Size([self.num_samples]))
        
        # Get probabilities for each sample
        probs = function_dist.cdf(target)
        
        
        # Calculate log probabilities for the binary observations
        log_probs = torch.zeros_like(probs)
        log_probs = target * torch.log(probs + 1e-8) + (1 - target) * torch.log(1 - probs + 1e-8)
        
        # Average over MC samples
        return log_probs.mean(0)
    
    def log_marginal(self, observations: torch.Tensor, function_dist: gpytorch.distributions.MultivariateNormal) -> torch.Tensor:
        """
        Compute the log marginal likelihood.
        
        Args:
            observations: Binary observations (0 or 1)
            function_dist: GPyTorch MultivariateNormal output from the GP
            
        Returns:
            Log marginal likelihood
        """
        # Use Monte Carlo integration to approximate the integral
        function_samples = function_dist.rsample(torch.Size([self.num_samples]))
        probs = function_dist.cdf(observations)

        
        # Calculate log probabilities
        obs = observations.unsqueeze(0).expand(self.num_samples, *observations.shape)
        log_probs = obs * torch.log(probs + 1e-8) + (1 - obs) * torch.log(1 - probs + 1e-8)
        
        # Average over samples
        return log_probs.mean(0)
    
    def marginal(self, function_dist: gpytorch.distributions.MultivariateNormal) -> torch.distributions.Bernoulli:
        """
        Returns the marginal distribution integrating out the function values.
        
        Args:
            function_dist: GPyTorch MultivariateNormal output from the GP
            
        Returns:
            Bernoulli distribution representing the marginal
        """
        # Use Monte Carlo integration to compute the marginal probabilities
        # function_samples = function_dist.rsample(torch.Size([self.num_samples]))

        # Extract single variate normal from multivariatenormal using mean, variance
        loc = function_dist.mean
        scale = function_dist.variance.sqrt() + torch.exp(self.log_global_scale_increase)
        output_probs = torch.distributions.Normal(loc, scale).cdf(self.threshold)

        
        # Average probabilities over samples
        marginal_probs = output_probs.mean(0)
        
        # Return Bernoulli with these average probabilities
        return torch.distributions.Bernoulli(probs=marginal_probs)
