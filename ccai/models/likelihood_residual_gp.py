import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from typing import Optional, Tuple, Union, Dict, List, Callable
from gpytorch.optim import NGD

class VariationalGP(ApproximateGP):
    """
    Variational Gaussian Process model that predicts a scalar value from a state vector.
    
    This implementation uses a CholeskyVariationalDistribution with VariationalStrategy for
    efficient sparse approximation. It's designed to work with high-dimensional inputs
    by providing flexible kernel choices and inducing point selection.
    
    Attributes:
        input_dim (int): Dimensionality of the input state
        num_inducing (int): Number of inducing points
        inducing_points (torch.Tensor): Locations of inducing points
        mean_module (gpytorch.means.Mean): GP mean function
        covar_module (gpytorch.kernels.Kernel): GP kernel/covariance function
    """
    
    def __init__(
        self,
        state_dim: int,
        num_inducing: int = 128,
        inducing_points: Optional[torch.Tensor] = None,
        mean_type: str = "zero",
        kernel_type: str = "matern32",
        learn_inducing_locations: bool = True,
        use_ard: bool = True,
        lengthscale_prior: Optional[gpytorch.priors.Prior] = None,
        outputscale_prior: Optional[gpytorch.priors.Prior] = None,
    ):
        """
        Initialize the Variational Gaussian Process model.
        
        Args:
            state_dim: Dimension of input state vector
            num_inducing: Number of inducing points to use
            inducing_points: Optional tensor of inducing point locations. If None, 
                            randomly initialized from N(0,1)
            mean_type: Type of mean function ('zero', 'constant')
            kernel_type: Type of kernel ('rbf', 'matern12', 'matern32', 'matern52')
            learn_inducing_locations: Whether to optimize inducing point locations
            use_ard: Whether to use Automatic Relevance Determination (separate lengthscales)
            lengthscale_prior: Prior for kernel lengthscale
            outputscale_prior: Prior for kernel outputscale
        """
        self.input_dim = state_dim
        self.num_inducing = num_inducing
        
        # Initialize inducing points if not provided
        if inducing_points is None:
            inducing_points = torch.randn(num_inducing, state_dim)
        
        # Set up variational distribution and strategy
        variational_distribution = CholeskyVariationalDistribution(num_inducing)
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, 
            learn_inducing_locations=learn_inducing_locations
        )
        super().__init__(variational_strategy)
        
        # Set up mean function
        if mean_type == "zero":
            self.mean_module = gpytorch.means.ZeroMean()
        else:  # Default to constant mean
            self.mean_module = gpytorch.means.ConstantMean()
        
        # Set up covariance function (kernel)
        ard_num_dims = state_dim if use_ard else None
        
        if kernel_type == "matern12":
            base_kernel = gpytorch.kernels.MaternKernel(
                nu=0.5, ard_num_dims=ard_num_dims, lengthscale_prior=lengthscale_prior
            )
        elif kernel_type == "matern32":
            base_kernel = gpytorch.kernels.MaternKernel(
                nu=1.5, ard_num_dims=ard_num_dims, lengthscale_prior=lengthscale_prior
            )
        elif kernel_type == "matern52":
            base_kernel = gpytorch.kernels.MaternKernel(
                nu=2.5, ard_num_dims=ard_num_dims, lengthscale_prior=lengthscale_prior
            )
        else:  # Default to RBF kernel
            base_kernel = gpytorch.kernels.RBFKernel(
                ard_num_dims=ard_num_dims, lengthscale_prior=lengthscale_prior
            )
        
        # Scale the kernel
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel, outputscale_prior=outputscale_prior
        )
    
    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        Forward pass through the Gaussian Process.
        
        Args:
            x: Input tensor of shape [batch_size, state_dim]
            
        Returns:
            MultivariateNormal distribution representing GP predictions
        """


        # Normalize
        x = (x - self.x_mean[:x.shape[-1]]) / self.x_std[:x.shape[-1]]

        # Apply mean and covariance functions
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        # Return distribution
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with the GP model (in eval mode).
        
        Args:
            x: Input tensor of shape [batch_size, state_dim]
            
        Returns:
            Tuple of (mean, variance) tensors
        """
        self.eval()
        output = self(x)
        return output.mean, output.variance
    
    def get_prediction_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get prediction with uncertainty by drawing multiple samples.
        
        Args:
            x: Input tensor of shape [batch_size, state_dim]
            n_samples: Number of samples to draw from the predictive distribution
            
        Returns:
            Tuple of (mean, variance, samples) tensors
        """
        self.eval()
        with torch.no_grad():
            output = self(x)
            samples = output.rsample(torch.Size([n_samples]))
            mean = output.mean
            variance = output.variance
        
        return mean, variance, samples
    
    def save(self, filepath: str) -> None:
        """
        Save the GP model to a file.
        
        Args:
            filepath: Path to save the model
        """
        state_dict = {
            'model_state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'num_inducing': self.num_inducing,
        }
        torch.save(state_dict, filepath)
    
    @classmethod
    def load(cls, filepath: str, **kwargs) -> 'VariationalGP':
        """
        Load a GP model from a file.
        
        Args:
            filepath: Path to the saved model
            **kwargs: Additional arguments to pass to the constructor
            
        Returns:
            Loaded VariationalGP model
        """
        state_dict = torch.load(filepath)
        model = cls(
            state_dim=state_dict['input_dim'],
            num_inducing=state_dict['num_inducing'],
            **kwargs
        )
        model.load_state_dict(state_dict['model_state_dict'])
        return model


class LikelihoodResidualGP:
    """
    Wrapper class for a Gaussian Process model with a likelihood function.
    Designed for learning residuals on top of a nominal model.
    
    Attributes:
        gp_model (VariationalGP): The underlying Gaussian Process model
        likelihood (gpytorch.likelihoods.Likelihood): Observation likelihood
    """
    
    def __init__(
        self,
        state_dim: int,
        num_inducing: int = 128,
        likelihood: Optional[gpytorch.likelihoods.Likelihood] = None,
        noise_constraint: Optional[gpytorch.constraints.Interval] = None,
        **gp_kwargs
    ):
        """
        Initialize the GP model with a likelihood function.
        
        Args:
            state_dim: Dimension of input state vector
            num_inducing: Number of inducing points to use
            likelihood: Observation likelihood. If None, uses GaussianLikelihood
            noise_constraint: Constraint on the noise parameter
            **gp_kwargs: Additional arguments to pass to VariationalGP constructor
        """
        self.gp_model = VariationalGP(state_dim=state_dim, num_inducing=num_inducing, **gp_kwargs)
        
        # Set up likelihood
        if likelihood is None:
            if noise_constraint is None:
                noise_constraint = gpytorch.constraints.GreaterThan(1e-4)
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_constraint)
        else:
            self.likelihood = likelihood
    
    def to(self, device: torch.device) -> 'LikelihoodResidualGP':
        """
        Move the GP model and likelihood to the specified device.
        
        Args:
            device: Device to move the model to
            
        Returns:
            Self for chaining
        """
        self.gp_model = self.gp_model.to(device)
        self.likelihood = self.likelihood.to(device)

        return self

    def set_norm_constants(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """
        Set normalization constants for the target values.
        
        Args:
            mean: Mean of the target values
            std: Standard deviation of the target values
        """
        self.gp_model.x_mean = mean
        self.gp_model.x_std = std

    def train_model(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        n_iterations: int = 1000,
        learning_rate: float = 0.01,
        ngd_lr: float = 0.1,  # Natural gradient learning rate
        use_natural_gradient: bool = True,
        optimizer_cls: Callable = torch.optim.Adam,
        verbose: bool = True,
        log_interval: int = 100,
    ) -> List[float]:
        """
        Train the GP model using variational inference with natural gradient descent.
        
        Args:
            train_x: Training input data [n_samples, state_dim]
            train_y: Training targets [n_samples]
            n_iterations: Number of training iterations
            learning_rate: Optimizer learning rate for hyperparameters
            ngd_lr: Learning rate for natural gradient descent (for variational parameters)
            use_natural_gradient: Whether to use natural gradient descent for variational params
            optimizer_cls: Optimizer class for hyperparameters
            verbose: Whether to print progress
            log_interval: How often to print progress
            
        Returns:
            List of loss values during training
        """
        # Set models to training mode
        self.gp_model.train()
        self.likelihood.train()
        
        # Define objective function
        mll = gpytorch.mlls.PredictiveLogLikelihood(
            self.likelihood, self.gp_model, num_data=train_x.size(0)
        )
        
        # Separate variational parameters from other parameters
        variational_params = []
        hyper_params = []
        
        for name, param in self.gp_model.named_parameters():
            if 'variational' in name:
                variational_params.append(param)
            else:
                hyper_params.append(param)
        
        # Add likelihood parameters to hyperparameters
        hyper_params.extend(self.likelihood.parameters())
        
        # Initialize optimizers
        if use_natural_gradient and len(variational_params) > 0:
            # Use NGD for variational parameters and standard optimizer for hyperparameters
            var_optimizer = NGD(variational_params, num_data=train_x.size(0), lr=ngd_lr)
            hyper_optimizer = optimizer_cls(hyper_params, lr=learning_rate)
        else:
            # Use standard optimizer for all parameters
            var_optimizer = None
            hyper_optimizer = optimizer_cls([
                {'params': self.gp_model.parameters()},
                {'params': self.likelihood.parameters()},
            ], lr=learning_rate)
        
        # Training loop
        losses = []
        for i in range(n_iterations):
            # Zero out gradients
            if var_optimizer is not None:
                var_optimizer.zero_grad()
            hyper_optimizer.zero_grad()
            
            # Get output from model
            output = self.gp_model(train_x)
            
            # Calculate loss
            loss = -mll(output, train_y)
            losses.append(loss.item())
            
            # Backpropagate
            loss.backward()
            
            # Update parameters
            if var_optimizer is not None:
                var_optimizer.step()
            hyper_optimizer.step()
            
            # Log progress
            if verbose and (i == 0 or (i + 1) % log_interval == 0 or i == n_iterations - 1):
                print(f"Iteration {i+1}/{n_iterations} - Loss: {loss.item():.4f}")
        
        return losses
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with the trained model.
        
        Args:
            x: Input tensor of shape [batch_size, state_dim]
            
        Returns:
            Tuple of (mean, variance) tensors
        """
        self.gp_model.eval()
        self.likelihood.eval()
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        # Get GP output distribution
        output = self.gp_model(x)
        
        # Get predictive distribution from likelihood
        pred_dist = self.likelihood(output)
        
        return pred_dist.mean, pred_dist.variance
    
    def forward_for_policy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GP for policy evaluation, ensuring gradient flow.
        Unlike predict(), this doesn't use @torch.no_grad() so it can be used in PPO.
        
        Args:
            x: Input tensor of shape [batch_size, state_dim]
            
        Returns:
            Mean predictions with gradient information preserved
        """
        # Run the model in train mode to preserve gradients
        self.gp_model.train()
        self.likelihood.train()
        
        # Get GP distribution
        output = self.gp_model(x)
        
        # Get predictive distribution
        pred_dist = self.likelihood(output)
        
        # Return mean predictions (with gradients)
        return pred_dist.mean
    
    def save(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        state_dict = {
            'gp_model_state_dict': self.gp_model.state_dict(),
            'likelihood_state_dict': self.likelihood.state_dict(),
            'input_dim': self.gp_model.input_dim,
            'num_inducing': self.gp_model.num_inducing,
        }
        torch.save(state_dict, filepath)
    
    @classmethod
    def load(cls, filepath: str, **kwargs) -> 'LikelihoodResidualGP':
        """
        Load a model from a file.
        
        Args:
            filepath: Path to the saved model
            **kwargs: Additional arguments to pass to the constructor
            
        Returns:
            Loaded LikelihoodResidualGP model
        """
        state_dict = torch.load(filepath)
        model = cls(
            state_dim=state_dict['input_dim'],
            num_inducing=state_dict['num_inducing'],
            **kwargs
        )
        model.gp_model.load_state_dict(state_dict['gp_model_state_dict'])
        model.likelihood.load_state_dict(state_dict['likelihood_state_dict'])
        return model
