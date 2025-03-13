import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import NaturalVariationalDistribution, VariationalStrategy
from gpytorch.distributions import MultivariateNormal
from typing import Optional, Tuple, Union, Dict, List, Callable, Any
from gpytorch.optim import NGD
from gpytorch.settings import fast_pred_var, fast_pred_samples, cg_tolerance, use_toeplitz, cholesky_jitter

class VariationalGP(ApproximateGP):
    """
    Variational Gaussian Process model with optimized prediction speed.
    
    This implementation uses a NaturalVariationalDistribution with WhitenedVariationalStrategy
    for efficient and numerically stable sparse approximation. Also includes fast prediction
    methods for improved inference speed.
    
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
        use_whitening: bool = True,  # Added whitening option
        lengthscale_prior: Optional[gpytorch.priors.Prior] = None,
        outputscale_prior: Optional[gpytorch.priors.Prior] = None,
    ):
        """
        Initialize the Variational Gaussian Process model.
        
        Args:
            state_dim: Dimension of input state vector
            num_inducing: Number of inducing points to use
            inducing_points: Optional tensor of inducing point locations
            mean_type: Type of mean function ('zero', 'constant')
            kernel_type: Type of kernel ('rbf', 'matern12', 'matern32', 'matern52')
            learn_inducing_locations: Whether to optimize inducing point locations
            use_ard: Whether to use Automatic Relevance Determination
            use_whitening: Whether to use whitened parameterization for numerical stability
            lengthscale_prior: Prior for kernel lengthscale
            outputscale_prior: Prior for kernel outputscale
        """
        self.input_dim = state_dim
        self.num_inducing = num_inducing
        
        # Initialize inducing points if not provided
        if inducing_points is None:
            inducing_points = torch.randn(num_inducing, state_dim)
        
        # Set up variational distribution and strategy
        variational_distribution = NaturalVariationalDistribution(num_inducing)
        
        # Use whitened variational strategy for better numerical stability
        if use_whitening:
            variational_strategy = VariationalStrategy(
                self, inducing_points, variational_distribution, 
                learn_inducing_locations=learn_inducing_locations
            )
        else:
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
        
        # Cache for faster predictions
        self._has_updated_cached_kernel = False
    
    def forward(self, x: torch.Tensor, base_likelihood: torch.Tensor = 0) -> gpytorch.distributions.MultivariateNormal:
        """
        Forward pass through the Gaussian Process.
        
        Args:
            x: Input tensor of shape [batch_size, state_dim]
            
        Returns:
            MultivariateNormal distribution representing GP predictions
        """
        # Apply mean and covariance functions
        mean_x = self.mean_module(x) + base_likelihood
        covar_x = self.covar_module(x)
        
        # Return distribution
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    @torch.no_grad()
    def predict(self, x: torch.Tensor, base_likelihood: torch.Tensor = 0, fast_mode: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with the GP model (in eval mode) using fast_pred_var.
        
        Args:
            x: Input tensor of shape [batch_size, state_dim]
            fast_mode: Whether to use GPyTorch fast prediction mode
            
        Returns:
            Tuple of (mean, variance) tensors
        """
        x = (x - self.x_mean[:x.shape[-1]]) / self.x_std[:x.shape[-1]]
        self.eval()
        
        # Use GPyTorch's fast predictive variance setting
        with fast_pred_var() if fast_mode else torch.no_grad():
            # Further optimize with higher CG tolerance for faster matrix solves
            with cg_tolerance(1e-3):
                output = self(x, base_likelihood=base_likelihood)
        
        return output.mean, output.variance
    
    def predict_with_grad(self, x: torch.Tensor, base_likelihood: torch.Tensor = 0, fast_mode: bool = True, normalize=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with the GP model, preserving gradients.
        Uses fast_pred_var for efficient variance computation.
        
        Args:
            x: Input tensor of shape [batch_size, state_dim]
            fast_mode: Whether to use GPyTorch fast prediction mode

        Returns:
            Tuple of (mean, variance) tensors
        """
        if normalize:
            x = (x - self.x_mean[:x.shape[-1]]) / self.x_std[:x.shape[-1]]
        self.eval()
        
        # Use fast_pred_var but keep gradients
        with fast_pred_var() if fast_mode else torch.enable_grad():
            # Higher CG tolerance for faster matrix solves
            with cg_tolerance(1e-3):
                output = self(x, base_likelihood=base_likelihood)
        
        return output.mean, output.variance
    
    def get_prediction_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 10, fast_mode: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get prediction with uncertainty by drawing multiple samples.
        Uses fast_pred_samples for efficient sample generation.
        
        Args:
            x: Input tensor of shape [batch_size, state_dim]
            n_samples: Number of samples to draw from the predictive distribution
            fast_mode: Whether to use GPyTorch fast prediction mode
            
        Returns:
            Tuple of (mean, variance, samples) tensors
        """
        x = (x - self.x_mean[:x.shape[-1]]) / self.x_std[:x.shape[-1]]
        self.eval()
        
        with torch.no_grad(), fast_pred_var(), fast_pred_samples() if fast_mode else torch.no_grad():
            # Higher CG tolerance for faster matrix solves
            with cg_tolerance(1e-3):
                output = self(x)
                samples = output.rsample(torch.Size([n_samples]))
                mean = output.mean
                variance = output.variance
        
        return mean, variance, samples

    def cache_kernel(self) -> None:
        """
        Pre-compute and cache kernel matrices for faster inference.
        This is useful when making repeated predictions with the same model.
        """
        with torch.no_grad():
            # Force kernel evaluation to create cache
            inducing_points = self.variational_strategy.inducing_points
            _ = self.covar_module(inducing_points)
            self._has_updated_cached_kernel = True
    
    def clear_cache(self) -> None:
        """Clear any cached computations to free memory."""
        self.covar_module.clear_cache()
        self._has_updated_cached_kernel = False


class LikelihoodResidualGP:
    """
    Wrapper class for a Gaussian Process model with a likelihood function.
    Enhanced with fast prediction capabilities.
    
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
        use_whitening: bool = True,  # Added whitening option
        **gp_kwargs
    ):
        """
        Initialize the GP model with a likelihood function.
        
        Args:
            state_dim: Dimension of input state vector
            num_inducing: Number of inducing points to use
            likelihood: Observation likelihood. If None, uses GaussianLikelihood
            noise_constraint: Constraint on the noise parameter
            use_whitening: Whether to use whitened variational strategy
            **gp_kwargs: Additional arguments to pass to VariationalGP constructor
        """
        # Pass use_whitening to the VariationalGP constructor
        self.gp_model = VariationalGP(
            state_dim=state_dim, 
            num_inducing=num_inducing, 
            use_whitening=use_whitening,
            **gp_kwargs
        )
        
        # Set up likelihood
        if likelihood is None:
            if noise_constraint is None:
                noise_constraint = gpytorch.constraints.GreaterThan(1e-4)
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=noise_constraint)
        else:
            self.likelihood = likelihood
            
        # Cache flag for repeated predictions
        self.prediction_cache_enabled = False
    
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
        batch_size: Optional[int] = None,  # Added batch training option
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
            batch_size: If not None, use mini-batch training with this batch size
            
        Returns:
            List of loss values during training
        """
        # Set models to training mode
        self.gp_model.train()
        self.likelihood.train()
        
        # Clear any caches before training
        self.gp_model.clear_cache()
        
        # Use full dataset size if batch_size is None
        full_dataset_size = train_x.size(0)
        batch_size = batch_size or full_dataset_size
        
        # Define objective function
        mll = gpytorch.mlls.PredictiveLogLikelihood(
            self.likelihood, self.gp_model, num_data=full_dataset_size
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
            var_optimizer = NGD(variational_params, num_data=full_dataset_size, lr=ngd_lr)
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
            # Use mini-batches if batch_size < full_dataset_size
            if batch_size < full_dataset_size:
                # Randomly select batch
                batch_idx = torch.randperm(full_dataset_size)[:batch_size]
                x_batch = train_x[batch_idx]
                y_batch = train_y[batch_idx]
                
                # Adjust MLL for batch
                batch_mll = gpytorch.mlls.PredictiveLogLikelihood(
                    self.likelihood, self.gp_model, num_data=full_dataset_size
                )
            else:
                # Use full dataset
                x_batch = train_x
                y_batch = train_y
                batch_mll = mll
            
            # Zero out gradients
            if var_optimizer is not None:
                var_optimizer.zero_grad()
            hyper_optimizer.zero_grad()
            
            # Get output from model
            output = self.gp_model(x_batch)
            
            # Calculate loss
            loss = -batch_mll(output, y_batch)
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
    def predict(self, x: torch.Tensor, base_likelihood = 0, fast_mode: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with the trained model using fast prediction settings.
        
        Args:
            x: Input tensor of shape [batch_size, state_dim]
            fast_mode: Whether to use GPyTorch fast prediction optimizations
            
        Returns:
            Tuple of (mean, variance) tensors
        """
        x = (x - self.gp_model.x_mean[:x.shape[-1]]) / self.gp_model.x_std[:x.shape[-1]]
        self.gp_model.eval()
        self.likelihood.eval()
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # Enable caching if we're going to make many predictions
        if not self.gp_model._has_updated_cached_kernel and self.prediction_cache_enabled:
            self.gp_model.cache_kernel()
        
        # Fix the with statement syntax
        if fast_mode:
            with fast_pred_var(), cg_tolerance(1e-3):
                # Get GP output distribution
                output = self.gp_model(x, base_likelihood=base_likelihood)
                # Get predictive distribution from likelihood
                pred_dist = self.likelihood(output)
        else:
            with torch.no_grad():
                output = self.gp_model(x, base_likelihood=base_likelihood)
                pred_dist = self.likelihood(output)
        
        return pred_dist, output.mean, output.variance
    
    def predict_with_grad(self, x: torch.Tensor, base_likelihood: torch.Tensor = 0, fast_mode: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with the trained model, preserving gradients.
        Uses GPyTorch fast prediction optimizations.
        
        Args:
            x: Input tensor of shape [batch_size, state_dim]
            fast_mode: Whether to use GPyTorch fast prediction optimizations
            
        Returns:
            Tuple of (mean, variance) tensors
        """
        x = (x - self.gp_model.x_mean[:x.shape[-1]]) / self.gp_model.x_std[:x.shape[-1]]
        self.gp_model.eval()
        self.likelihood.eval()
        
        # Fix the with statement syntax
        if fast_mode:
            with fast_pred_var(), cg_tolerance(1e-3):
                # Get GP output distribution
                output = self.gp_model(x, base_likelihood=base_likelihood)
                
                # Get predictive distribution from likelihood
                pred_dist = self.likelihood(output)
        else:
            with torch.enable_grad():
                output = self.gp_model(x, base_likelihood=base_likelihood)
                pred_dist = self.likelihood(output)
        
        return pred_dist, output.mean, output.variance
    
    def get_prediction_samples(
        self, x: torch.Tensor, n_samples: int = 10, fast_mode: bool = True
    ) -> torch.Tensor:
        """
        Get samples from the predictive distribution efficiently.
        
        Args:
            x: Input tensor of shape [batch_size, state_dim]
            n_samples: Number of samples to draw
            fast_mode: Whether to use fast prediction settings
            
        Returns:
            Samples tensor of shape [n_samples, batch_size, 1]
        """
        x = (x - self.gp_model.x_mean[:x.shape[-1]]) / self.gp_model.x_std[:x.shape[-1]]
        self.gp_model.eval()
        self.likelihood.eval()
        
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # Fix the with statement syntax
        with torch.no_grad():
            if fast_mode:
                with fast_pred_var(), fast_pred_samples(), cg_tolerance(1e-3):
                    # Get GP output distribution
                    output = self.gp_model(x)
                    
                    # Get samples from likelihood
                    pred_dist = self.likelihood(output)
                    samples = pred_dist.sample(torch.Size([n_samples]))
            else:
                output = self.gp_model(x)
                pred_dist = self.likelihood(output)
                samples = pred_dist.sample(torch.Size([n_samples]))
                
        return samples
    
    def enable_prediction_caching(self, enabled: bool = True) -> None:
        """
        Enable or disable kernel caching for faster repeated predictions.
        
        Args:
            enabled: Whether to enable caching
        """
        self.prediction_cache_enabled = enabled
        if enabled and not self.gp_model._has_updated_cached_kernel:
            self.gp_model.cache_kernel()
        elif not enabled:
            self.gp_model.clear_cache()

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
