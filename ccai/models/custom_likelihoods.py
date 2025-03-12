import torch
import gpytorch
from gpytorch.likelihoods import Likelihood
from typing import Any, Optional, Tuple

class SigmoidBernoulliLikelihood(Likelihood):
    """
    Custom Bernoulli likelihood that uses a sigmoid activation with learnable
    bias and temperature parameters.
    
    This replaces the standard GPyTorch BernoulliLikelihood (which uses normal CDF)
    with a sigmoid implementation that has learnable parameters to better adapt to
    the diffusion model likelihood distribution.
    
    Attributes:
        bias: Learnable bias parameter (initialized with negative threshold)
        temperature: Learnable temperature parameter for sigmoid scaling
    """
    
    def __init__(
        self, 
        initial_bias: float = -15.0, 
        initial_temp: float = .1, 
        num_samples: int = 15
    ):
        """
        Initialize custom Bernoulli likelihood.
        
        Args:
            initial_bias: Initial value for the bias parameter
            initial_temp: Initial value for the temperature parameter
            num_samples: Number of MC samples for expected log prob calculations
        """
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor([initial_bias]))
        self.temperature = torch.nn.Parameter(torch.tensor([initial_temp]))
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
        probs = torch.sigmoid(self.temperature * (function_samples + self.bias))
        
        # Return Bernoulli distribution
        return torch.distributions.Bernoulli(probs=probs)
    
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
        probs = torch.sigmoid(self.temperature * (function_samples + self.bias))
        
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
        probs = torch.sigmoid(self.temperature * (function_samples + self.bias))
        
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
        function_samples = function_dist.rsample(torch.Size([self.num_samples]))
        probs = torch.sigmoid(self.temperature * (function_samples + self.bias))
        
        # Average probabilities over samples
        marginal_probs = probs.mean(0)
        
        # Return Bernoulli with these average probabilities
        return torch.distributions.Bernoulli(probs=marginal_probs)
