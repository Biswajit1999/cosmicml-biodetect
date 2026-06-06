"""
Bayesian neural network layers for uncertainty quantification.

Implements:
- Bayesian linear layers with learned weight distributions
- MC Dropout for epistemic uncertainty
- Evidence Lower Bound (ELBO) loss
- Aleatoric and epistemic uncertainty estimation

Theory:
    In Bayesian neural networks, weights are distributions rather than
    point estimates. This naturally quantifies uncertainty.

    p(y|x) = ∫ p(y|x,w) p(w) dw

    Approximated via:
    1. Variational inference (learnable distributions)
    2. MC Dropout (stochastic forward passes)
    3. Ensemble of predictions

References:
    Blundell, C., Cornebise, J., Kavukcuoglu, K., & Welling, M. (2015).
    Weight uncertainty in neural networks. ICML.

    Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation:
    Representing model uncertainty in deep learning. ICML.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class BayesianLinear(nn.Module):
    """
    Bayesian linear layer with weight uncertainty.

    Instead of fixed weights w, we learn:
    - Mean: μ_w
    - Log variance: log(σ_w²)

    Weights are sampled: w ~ N(μ_w, σ_w²)

    This provides uncertainty in the predictions.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_scale: float = 1.0,
    ):
        """
        Initialize Bayesian linear layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            prior_scale: Scale of weight prior (for regularization)
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.prior_scale = prior_scale

        # Mean of weight posterior
        self.weight_mean = nn.Parameter(
            torch.randn(out_features, in_features) / math.sqrt(in_features)
        )

        # Log variance of weight posterior
        self.weight_logvar = nn.Parameter(
            torch.ones(out_features, in_features) * -5.0  # Start with small variance
        )

        # Bias parameters
        self.bias_mean = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.ones(out_features) * -5.0)

        # Register prior for KL divergence
        self.register_buffer('prior_mean', torch.zeros(out_features, in_features))
        self.register_buffer('prior_logvar', torch.ones(out_features, in_features) * math.log(prior_scale ** 2))

    def forward(
        self,
        x: torch.Tensor,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with weight sampling.

        Args:
            x: Input tensor [batch, in_features]
            sample: If True, sample weights; if False, use mean

        Returns:
            Tuple of (output, kl_divergence)
        """
        if sample:
            # Sample weights: w = μ + σ * ε where ε ~ N(0,1)
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight_eps = torch.randn_like(self.weight_mean)
            weight = self.weight_mean + weight_std * weight_eps

            bias_std = torch.exp(0.5 * self.bias_logvar)
            bias_eps = torch.randn_like(self.bias_mean)
            bias = self.bias_mean + bias_std * bias_eps
        else:
            # Use mean weights
            weight = self.weight_mean
            bias = self.bias_mean

        # Compute output
        output = F.linear(x, weight, bias)

        # Compute KL divergence for this layer
        # KL[q(w)||p(w)] = Σ [log(σ_p/σ_q) + (σ_q² + (μ_q - μ_p)²)/(2σ_p²) - 1/2]
        kl_weight = self._kl_divergence(
            self.weight_mean,
            self.weight_logvar,
            self.prior_mean,
            self.prior_logvar,
        )

        kl_bias = self._kl_divergence(
            self.bias_mean.unsqueeze(0),
            self.bias_logvar.unsqueeze(0),
            torch.zeros_like(self.bias_mean).unsqueeze(0),
            torch.zeros_like(self.bias_logvar).unsqueeze(0),
        )

        kl = kl_weight + kl_bias

        return output, kl

    @staticmethod
    def _kl_divergence(
        mean_q: torch.Tensor,
        logvar_q: torch.Tensor,
        mean_p: torch.Tensor,
        logvar_p: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute KL divergence between two Gaussians.

        KL[q(w)||p(w)] = 0.5 * Σ [log(σ_p²/σ_q²) + (σ_q² + (μ_q-μ_p)²)/σ_p² - 1]

        Args:
            mean_q, logvar_q: Parameters of posterior q(w)
            mean_p, logvar_p: Parameters of prior p(w)

        Returns:
            KL divergence (scalar)
        """
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)

        kl = 0.5 * torch.sum(
            logvar_p - logvar_q
            + (var_q + (mean_q - mean_p) ** 2) / var_p
            - 1.0
        )

        return kl


class BayesianMLPBlock(nn.Module):
    """
    Bayesian multilayer perceptron block.

    Combines Bayesian linear layers with uncertainty-aware activations.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: Optional[int] = None,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
    ):
        """
        Initialize Bayesian MLP block.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            hidden_features: Hidden layer dimension
            num_layers: Number of layers
            dropout_rate: Dropout rate for uncertainty
        """
        super().__init__()

        if hidden_features is None:
            hidden_features = (in_features + out_features) // 2

        layers = []
        prev_features = in_features

        for i in range(num_layers):
            current_out = hidden_features if i < num_layers - 1 else out_features
            layers.append(BayesianLinear(prev_features, current_out))

            if i < num_layers - 1:
                layers.append(nn.BatchNorm1d(current_out))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))

            prev_features = current_out

        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        x: torch.Tensor,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through Bayesian MLP.

        Args:
            x: Input tensor
            sample: Whether to sample weights

        Returns:
            Tuple of (output, total_kl)
        """
        total_kl = 0.0

        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                x, kl = layer(x, sample=sample)
                total_kl = total_kl + kl
            else:
                x = layer(x)

        return x, total_kl


class BayesianUncertaintyEstimator(nn.Module):
    """
    Estimate aleatoric and epistemic uncertainty from predictions.

    Aleatoric uncertainty (data uncertainty):
    - Comes from noise in observations
    - Irreducible - can't improve with more data
    - Estimated from prediction variance in output

    Epistemic uncertainty (model uncertainty):
    - Comes from limited training data
    - Reducible - improves with more data
    - Estimated from MC Dropout or ensemble variance
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_mc_samples: int = 10,
    ):
        """
        Initialize uncertainty estimator.

        Args:
            input_dim: Input dimension
            output_dim: Output dimension (composition)
            num_mc_samples: Number of MC samples for epistemic uncertainty
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_mc_samples = num_mc_samples

        # Mean predictor (deterministic)
        self.mean_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim),
        )

        # Aleatoric uncertainty (log variance) predictor
        self.aleatoric_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softplus(),  # Ensure positive variance
        )

        # Epistemic uncertainty network (with MC Dropout)
        self.epistemic_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # Higher dropout for uncertainty
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Estimate mean and uncertainties.

        Args:
            features: Input features

        Returns:
            Tuple of (mean, aleatoric_std, epistemic_std)
        """
        batch_size = features.shape[0]

        # Mean prediction
        mean = self.mean_net(features)

        # Aleatoric uncertainty (data noise)
        aleatoric_var = self.aleatoric_net(features)
        aleatoric_std = torch.sqrt(aleatoric_var + 1e-8)

        # Epistemic uncertainty via MC Dropout
        self.epistemic_net.train()  # Enable dropout during inference
        epistemic_samples = []

        for _ in range(self.num_mc_samples):
            sample = self.epistemic_net(features)
            epistemic_samples.append(sample)

        self.epistemic_net.eval()  # Return to eval mode

        # Epistemic std = variance across MC samples
        epistemic_samples = torch.stack(epistemic_samples)  # [mc_samples, batch, output]
        epistemic_std = epistemic_samples.std(dim=0)

        return mean, aleatoric_std, epistemic_std

    def total_uncertainty(
        self,
        mean: torch.Tensor,
        aleatoric_std: torch.Tensor,
        epistemic_std: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute total uncertainty.

        Total variance = aleatoric variance + epistemic variance

        Args:
            mean: Mean prediction
            aleatoric_std: Aleatoric standard deviation
            epistemic_std: Epistemic standard deviation

        Returns:
            Total standard deviation
        """
        total_var = (aleatoric_std ** 2) + (epistemic_std ** 2)
        total_std = torch.sqrt(total_var + 1e-8)
        return total_std


class ELBOLoss(nn.Module):
    """
    Evidence Lower Bound (ELBO) loss for Bayesian neural networks.

    ELBO = E_q[log p(y|x,w)] - KL[q(w)||p(w)]
         ≈ 1/N Σ log p(y|x,w_sampled) - KL/N

    where:
    - First term: data fit (likelihood)
    - Second term: regularization (prior penalty)

    This balances data fitting with staying close to the prior.
    """

    def __init__(
        self,
        num_batches: int,
        kl_weight: float = 1.0,
    ):
        """
        Initialize ELBO loss.

        Args:
            num_batches: Total number of batches in epoch
            kl_weight: Weight for KL divergence term
        """
        super().__init__()
        self.num_batches = num_batches
        self.kl_weight = kl_weight

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        kl_divergence: torch.Tensor,
        reduction: str = 'mean',
    ) -> torch.Tensor:
        """
        Compute ELBO loss.

        Args:
            predictions: Model predictions
            targets: Target values
            kl_divergence: KL divergence from Bayesian layer
            reduction: 'mean' or 'sum'

        Returns:
            ELBO loss
        """
        # Data fitting term (negative log likelihood)
        nll = F.mse_loss(predictions, targets, reduction=reduction)

        # Regularization term (KL divergence weighted)
        # Scale KL by 1/num_batches so it averages out over epoch
        kl_term = (self.kl_weight / self.num_batches) * kl_divergence

        # ELBO = -NLL - KL (we minimize this)
        loss = nll + kl_term

        return loss


class UncertaintyCalibration(nn.Module):
    """
    Calibrate uncertainty estimates to match prediction errors.

    Uncertainty should correlate with actual errors:
    - High confidence (low uncertainty) on correct predictions
    - Low confidence (high uncertainty) on incorrect predictions

    Calibration via temperature scaling or Platt scaling.
    """

    def __init__(self):
        """Initialize calibration."""
        super().__init__()
        # Temperature parameter (initially 1.0)
        self.temperature = nn.Parameter(torch.ones(1))

    def calibrate(
        self,
        uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply temperature scaling to calibrate uncertainty.

        Args:
            uncertainty: Raw uncertainty estimates

        Returns:
            Calibrated uncertainty
        """
        # Scale by learned temperature
        calibrated = uncertainty / (self.temperature + 1e-8)
        return calibrated

    def compute_calibration_error(
        self,
        uncertainties: torch.Tensor,
        errors: torch.Tensor,
        num_bins: int = 10,
    ) -> torch.Tensor:
        """
        Compute Expected Calibration Error (ECE).

        ECE measures how well uncertainty correlates with error.
        Calibration is perfect when ECE = 0.

        Args:
            uncertainties: Uncertainty estimates
            errors: Absolute prediction errors
            num_bins: Number of bins for ECE

        Returns:
            ECE value
        """
        # Normalize uncertainties to [0, 1]
        unc_min = uncertainties.min()
        unc_max = uncertainties.max()
        unc_norm = (uncertainties - unc_min) / (unc_max - unc_min + 1e-8)

        # Bin uncertainties
        bin_edges = torch.linspace(0, 1, num_bins + 1, device=uncertainties.device)
        bin_indices = torch.searchsorted(bin_edges, unc_norm.detach(), right=False)

        # Compute calibration error per bin
        ece = 0.0
        for b in range(1, num_bins + 1):
            mask = bin_indices == b
            if mask.sum() > 0:
                mean_unc = uncertainties[mask].mean()
                mean_error = errors[mask].mean()
                bin_error = torch.abs(mean_unc - mean_error)
                ece = ece + (mask.sum() / len(uncertainties)) * bin_error

        return ece
