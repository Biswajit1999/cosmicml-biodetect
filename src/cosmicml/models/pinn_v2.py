"""
Physics-Informed Neural Network (PINN) v2 - Enhanced with Real Physics

Major improvements over v1:
1. Real Gibbs free energy minimization loss
2. Chemical equilibrium constraints
3. Better loss functions (log-scale for rare species)
4. Proper layer architecture (no redundant softmax)
5. Temperature-dependent physics

References:
    Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks.
    Atkins, P. W., & De Paula, J. (2019). Physical Chemistry (11th ed.).
    NIST Chemistry WebBook.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional
import sys
from pathlib import Path

# Import thermodynamics modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from physics.thermodynamics import GibbsFreeEnergyCalculator, ChemicalConstraint


class PINNv2(nn.Module):
    """
    Physics-Informed Neural Network with Real Physics Constraints.

    Combines neural network inference with:
    - Gibbs free energy minimization
    - Chemical equilibrium constraints
    - Temperature-dependent thermodynamics
    - Proper scale-dependent loss functions

    Architecture:
        Input (512 wavelength bins)
        → Encoder [512→256→128→64]
        → Latent space (64 dims)
        → Decoder [64→128→256→512]
        → Output layer [512→10 species]
        → Softmax normalization
        → Physics constraint verification
    """

    def __init__(
        self,
        input_dim: int = 512,
        latent_dim: int = 64,
        output_dim: int = 10,
        hidden_layers: Optional[List[int]] = None,
        physics_weight: float = 1.0,
        gibbs_weight: float = 0.5,
        equilibrium_weight: float = 0.3,
    ):
        """
        Initialize PINN v2 model.

        Args:
            input_dim: Spectrum resolution (wavelength bins)
            latent_dim: Latent space dimension
            output_dim: Number of atmospheric species (10)
            hidden_layers: Encoder/decoder layer sizes
            physics_weight: Weight for physics loss component
            gibbs_weight: Weight for Gibbs free energy loss
            equilibrium_weight: Weight for equilibrium constraint loss
        """
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.physics_weight = physics_weight
        self.gibbs_weight = gibbs_weight
        self.equilibrium_weight = equilibrium_weight

        if hidden_layers is None:
            hidden_layers = [256, 128, 64]

        # ===== ENCODER: Spectrum → Latent Space =====
        # Extract features from spectrum (512 wavelength bins)
        encoder_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(0.2))  # Slightly higher dropout
            prev_dim = hidden_dim

        # Final layer to latent space
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # ===== DECODER: Latent Space → Composition =====
        # Reconstruct composition (10 species)
        decoder_layers = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim

        # Output logits for composition
        # NO softplus here - use log-softmax directly
        decoder_layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # ===== PHYSICS CONSTRAINTS =====
        # Initialize thermodynamic calculators
        self.gibbs_calculator = GibbsFreeEnergyCalculator()
        self.chem_constraint = ChemicalConstraint()

        # Temperature predictor (auxiliary head)
        # From spectrum shape, predict temperature
        self.temp_predictor = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # Ensures positive temperature
        )

    def forward(self, spectrum: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through PINN.

        Args:
            spectrum: [batch, 512] wavelength-dependent spectrum

        Returns:
            composition: [batch, 10] normalized composition (sum=1, all in [0,1])
        """
        # Encode spectrum to latent space
        latent = self.encoder(spectrum)

        # Decode to composition logits
        composition_logits = self.decoder(latent)

        # Apply log-softmax for numerical stability
        # This automatically ensures sum=1 and all values in [0,1]
        composition = torch.softmax(composition_logits, dim=1)

        return composition

    def predict_temperature(self, spectrum: torch.Tensor) -> torch.Tensor:
        """
        Predict atmospheric temperature from spectrum shape.

        Temperature affects:
        - Molecular absorption cross-sections
        - Chemical equilibrium
        - Atmospheric scale height

        Args:
            spectrum: [batch, 512] spectrum

        Returns:
            temperature: [batch, 1] in Kelvin
        """
        latent = self.encoder(spectrum)
        # Add offset to Softplus output to get realistic range (200-1000 K)
        temp = self.temp_predictor(latent) * 750.0 + 250.0
        return temp

    def compute_abundance_loss(self, composition: torch.Tensor) -> torch.Tensor:
        """
        Loss for abundance constraints.

        Enforces:
        1. All abundances in [0, 1] (automatic via softmax)
        2. Sum of abundances = 1 (automatic via softmax)
        3. Reasonable distributions (penalize flat/uniform)

        Args:
            composition: [batch, 10] predicted composition

        Returns:
            Loss scalar
        """
        # Check sum = 1 (should be automatic, but verify)
        abundance_sum = torch.sum(composition, dim=1)
        sum_loss = torch.mean((abundance_sum - 1.0) ** 2)

        # Entropy penalty (encourage peaked distributions, not uniform)
        # High entropy = flat distribution = unphysical
        epsilon = 1e-10
        entropy = -torch.sum(composition * torch.log(composition + epsilon), dim=1)
        entropy_loss = torch.mean(torch.relu(entropy - 2.0))  # Penalize if entropy > 2 nats

        return sum_loss + 0.05 * entropy_loss

    def compute_scale_dependent_loss(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Loss that accounts for different scales of composition values.

        Common species (N2, O2, etc): MAE loss
        Rare species (O3, CH4, etc): Relative error (log) loss

        This prevents rare species from being ignored since they
        have smaller absolute values.

        Args:
            predicted: [batch, 10] predicted composition
            target: [batch, 10] target composition

        Returns:
            Loss scalar
        """
        # Clip to prevent log(0)
        predicted_clipped = torch.clamp(predicted, min=1e-10)
        target_clipped = torch.clamp(target, min=1e-10)

        # For each species, determine if it's rare or common
        mean_abundance = target_clipped.mean(dim=0)

        # Rare species: x_i < 0.01 → use relative error
        # Common species: x_i ≥ 0.01 → use absolute error
        loss = torch.zeros_like(predicted)

        for i in range(self.output_dim):
            if mean_abundance[i] < 0.01:
                # Rare species: log-scale loss
                # |log(pred/true)| = |log(pred) - log(true)|
                loss[:, i] = torch.abs(torch.log(predicted_clipped[:, i] / target_clipped[:, i]))
            else:
                # Common species: absolute loss
                loss[:, i] = torch.abs(predicted[:, i] - target[:, i])

        return torch.mean(loss)

    def compute_gibbs_loss(
        self,
        composition: torch.Tensor,
        temperature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Gibbs free energy minimization loss.

        Predicted compositions should minimize free energy at the given T.
        This is a thermodynamic constraint that pushes predictions toward
        thermodynamically stable compositions.

        Args:
            composition: [batch, 10] predicted composition
            temperature: [batch] or scalar temperature in K

        Returns:
            Loss for out-of-equilibrium compositions
        """
        if temperature is None:
            temperature = 288.0  # Earth-like default

        # Compute loss for each composition
        if isinstance(temperature, torch.Tensor) and temperature.shape:
            # Use first temperature value (or take mean if batched)
            T_val = temperature[0].item() if temperature.dim() > 0 else temperature.item()
        else:
            T_val = float(temperature)

        gibbs_loss = self.gibbs_calculator.gibbs_minimization_loss(composition, T_val)

        return torch.mean(gibbs_loss)

    def compute_equilibrium_loss(
        self,
        composition: torch.Tensor,
        temperature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Chemical equilibrium constraint loss.

        For key reactions, check if reaction quotient Q ≈ K_eq(T).
        Penalizes compositions that violate chemical equilibrium.

        Args:
            composition: [batch, 10] predicted composition
            temperature: [batch] or scalar temperature in K

        Returns:
            Loss for reactions out of equilibrium
        """
        if temperature is None:
            temperature = 288.0

        if isinstance(temperature, torch.Tensor) and temperature.shape:
            T_val = temperature[0].item() if temperature.dim() > 0 else temperature.item()
        else:
            T_val = float(temperature)

        eq_loss = self.chem_constraint.equilibrium_loss(composition, T_val, weight=1.0)

        return torch.mean(eq_loss)

    def compute_physics_loss(
        self,
        composition: torch.Tensor,
        temperature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Total physics-informed loss combining all constraints.

        L_physics = L_abundance + λ_gibbs * L_gibbs + λ_eq * L_equilibrium

        Args:
            composition: [batch, 10] predicted composition
            temperature: Atmospheric temperature in K

        Returns:
            Weighted sum of all physics losses
        """
        abundance_loss = self.compute_abundance_loss(composition)
        gibbs_loss = self.compute_gibbs_loss(composition, temperature)
        equilibrium_loss = self.compute_equilibrium_loss(composition, temperature)

        total_physics_loss = (
            abundance_loss
            + self.gibbs_weight * gibbs_loss
            + self.equilibrium_weight * equilibrium_loss
        )

        return total_physics_loss

    def compute_loss(
        self,
        spectrum: torch.Tensor,
        target_composition: torch.Tensor,
        target_temperature: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute total loss = data loss + physics loss.

        L_total = L_data + λ * L_physics

        where:
        - L_data: Scale-dependent loss (log for rare, MSE for common)
        - L_physics: Gibbs + equilibrium + abundance constraints

        Args:
            spectrum: [batch, 512] input spectrum
            target_composition: [batch, 10] target composition
            target_temperature: [batch] or scalar temperature (optional)

        Returns:
            Tuple of (total_loss, data_loss, physics_loss)
        """
        # Forward pass
        predicted_composition = self.forward(spectrum)
        predicted_temperature = self.predict_temperature(spectrum)

        # Data loss: scale-aware MSE/relative loss
        data_loss = self.compute_scale_dependent_loss(predicted_composition, target_composition)

        # Use predicted temperature if target not given
        if target_temperature is None:
            target_temperature = predicted_temperature

        # Physics loss: thermodynamic constraints
        physics_loss = self.compute_physics_loss(predicted_composition, target_temperature)

        # Total loss
        total_loss = data_loss + self.physics_weight * physics_loss

        return total_loss, data_loss, physics_loss

    def predict_with_uncertainty(
        self,
        spectrum: torch.Tensor,
        n_samples: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict composition with uncertainty using MC Dropout.

        Args:
            spectrum: [batch, 512] input spectrum
            n_samples: Number of stochastic samples

        Returns:
            Tuple of (mean_composition, std_composition)
        """
        # Enable dropout during inference
        self.train()

        predictions = []
        for _ in range(n_samples):
            pred = self.forward(spectrum)
            predictions.append(pred)

        self.eval()

        predictions = torch.stack(predictions)
        mean = torch.mean(predictions, dim=0)
        std = torch.std(predictions, dim=0)

        return mean, std
