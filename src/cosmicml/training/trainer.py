"""
Advanced Training Orchestrator for PINN v3

Implements:
- Curriculum learning schedule
- Advanced optimizers (AdamW with weight decay)
- Learning rate scheduling (cosine annealing, warmup)
- Multi-task loss balancing
- Checkpoint management
- Training monitoring and logging

References:
    Loshchilov & Hutter (2019) - Decoupled Weight Decay Regularization
    Loshchilov & Hutter (2016) - SGDR: Stochastic Gradient Descent with Warm Restarts
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, ChainedScheduler
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import json
from datetime import datetime
import numpy as np


class AdvancedOptimizer:
    """
    Wrapper for advanced optimizer with learning rate scheduling.

    Uses AdamW (decoupled weight decay) with:
    - Linear warmup (first 10% of epoch)
    - Cosine annealing with warm restarts
    - Optional gradient clipping
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        gradient_clip: Optional[float] = 1.0,
    ):
        """
        Initialize optimizer.

        Args:
            model: Neural network model
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            warmup_epochs: Linear warmup epochs
            total_epochs: Total training epochs
            gradient_clip: Max gradient norm (None to disable)
        """
        self.model = model
        self.learning_rate = learning_rate
        self.gradient_clip = gradient_clip

        # AdamW optimizer (decoupled weight decay)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            amsgrad=True,  # Use AMSGrad variant for stability
        )

        # Learning rate scheduling
        # 1. Linear warmup for first warmup_epochs
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=warmup_epochs,
        )

        # 2. Cosine annealing with restarts
        cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Period of first restart
            T_mult=2,  # Restart period multiplier
            eta_min=1e-6,  # Minimum learning rate
        )

        # Chain the schedulers
        self.scheduler = ChainedScheduler([warmup_scheduler, cosine_scheduler])

    def step(self, loss: torch.Tensor) -> None:
        """
        Optimizer step.

        Args:
            loss: Loss tensor (used only if needed)
        """
        # Gradient clipping
        if self.gradient_clip is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

        self.optimizer.step()
        self.scheduler.step()

    def zero_grad(self) -> None:
        """Zero out gradients."""
        self.optimizer.zero_grad()

    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']


class MultiTaskLossBalancer:
    """
    Dynamically balance multiple loss terms during training.

    Uses task weighting that adapts based on loss magnitudes.
    Prevents one task from dominating training.

    Theory:
        Weighted loss: L = Σ_i (λ_i / σ_i²) L_i + log(σ_i)

    where σ_i is learned task uncertainty weight.
    """

    def __init__(self, num_tasks: int = 4, initial_weights: Optional[List[float]] = None):
        """
        Initialize loss balancer.

        Args:
            num_tasks: Number of loss terms
            initial_weights: Initial task weights
        """
        self.num_tasks = num_tasks

        if initial_weights is None:
            initial_weights = [1.0] * num_tasks

        # Log standard deviations (learnable task uncertainties)
        self.log_sigma = nn.Parameter(torch.zeros(num_tasks))

        self.loss_history = [[] for _ in range(num_tasks)]

    def compute_weighted_loss(
        self,
        loss_dict: Dict[str, torch.Tensor],
        loss_names: List[str],
    ) -> torch.Tensor:
        """
        Compute weighted loss with automatic balancing.

        Args:
            loss_dict: Dictionary of losses
            loss_names: Names of losses in order

        Returns:
            Balanced total loss
        """
        weighted_loss = 0.0

        for i, name in enumerate(loss_names):
            if name not in loss_dict:
                continue

            loss_i = loss_dict[name]

            # Precision weight: exp(-log_sigma_i²)
            precision = torch.exp(-self.log_sigma[i])

            # Weighted loss: precision * loss + log_sigma
            weighted_loss = weighted_loss + precision * loss_i + self.log_sigma[i]

            # Record loss
            self.loss_history[i].append(loss_i.item())

        return weighted_loss

    def get_task_weights(self) -> torch.Tensor:
        """Get normalized task weights."""
        weights = torch.exp(-self.log_sigma)
        return weights / weights.sum()

    def get_loss_statistics(self) -> Dict:
        """Get loss statistics."""
        stats = {}
        for i, history in enumerate(self.loss_history):
            if history:
                stats[f'task_{i}_mean'] = np.mean(history[-100:])  # Last 100
                stats[f'task_{i}_std'] = np.std(history[-100:])
        return stats


class TrainingManager:
    """
    Complete training orchestration.

    Manages:
    - Data loading with curriculum
    - Forward pass + loss computation
    - Backward pass + optimization
    - Checkpointing
    - Metrics tracking
    - Training state management
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        device: str = "cpu",
    ):
        """
        Initialize training manager.

        Args:
            model: PINN v3 model
            learning_rate: Initial learning rate
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.device = device

        # Optimizer
        self.optimizer_manager = AdvancedOptimizer(
            model,
            learning_rate=learning_rate,
            warmup_epochs=10,
            total_epochs=100,
        )

        # Loss balancer
        self.loss_balancer = MultiTaskLossBalancer(
            num_tasks=4,
            initial_weights=[1.0, 0.1, 1.0, 0.05],  # data, temp, physics, aleatoric
        )

        # Metrics tracking
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
        }

        # Checkpoint management
        self.best_val_loss = float('inf')
        self.checkpoint_dir = Path('models/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def train_step(
        self,
        spectra: torch.Tensor,
        target_composition: torch.Tensor,
        target_temperature: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Single training step.

        Args:
            spectra: [batch, 512] input spectrum
            target_composition: [batch, 10] target composition
            target_temperature: [batch] target temperature

        Returns:
            Loss dictionary
        """
        # Forward pass
        spectra = spectra.to(self.device)
        target_composition = target_composition.to(self.device)

        outputs = self.model(spectra)

        # Compute losses
        loss_dict = {}

        # Data loss
        data_loss = self.model._scale_dependent_loss(
            outputs['composition'],
            target_composition,
        )
        loss_dict['data'] = data_loss

        # Temperature loss
        if target_temperature is not None:
            target_temperature = target_temperature.to(self.device)
            temp_loss = torch.mean((outputs['temperature'] - target_temperature) ** 2)
            loss_dict['temperature'] = temp_loss

        # Physics loss
        if target_temperature is not None:
            physics_loss = self.model.compute_physics_loss(
                outputs['composition'],
                target_temperature,
            )
        else:
            physics_loss = self.model.compute_physics_loss(
                outputs['composition'],
                outputs['temperature'],
            )
        loss_dict['physics'] = physics_loss

        # Aleatoric uncertainty loss
        prediction_error = torch.abs(outputs['composition'] - target_composition)
        aleatoric_loss = torch.mean(
            (outputs['aleatoric_uncertainty'] - prediction_error) ** 2
        )
        loss_dict['aleatoric'] = aleatoric_loss

        # Balanced total loss
        total_loss = self.loss_balancer.compute_weighted_loss(
            loss_dict,
            ['data', 'temperature', 'physics', 'aleatoric'],
        )

        # Backward pass
        self.optimizer_manager.zero_grad()
        total_loss.backward()
        self.optimizer_manager.step()

        # Return metrics
        return {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'temperature_loss': loss_dict.get('temperature', 0).item(),
            'physics_loss': physics_loss.item(),
            'aleatoric_loss': aleatoric_loss.item(),
            'learning_rate': self.optimizer_manager.get_learning_rate(),
        }

    @torch.no_grad()
    def validate(
        self,
        spectra: torch.Tensor,
        target_composition: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Validation step.

        Args:
            spectra: [batch, 512]
            target_composition: [batch, 10]

        Returns:
            Validation metrics
        """
        self.model.eval()

        spectra = spectra.to(self.device)
        target_composition = target_composition.to(self.device)

        outputs = self.model(spectra)

        # Compute R²
        ss_res = torch.sum((outputs['composition'] - target_composition) ** 2)
        ss_tot = torch.sum((target_composition - target_composition.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        # Compute MAE per species
        mae = torch.mean(torch.abs(outputs['composition'] - target_composition), dim=0)

        self.model.train()

        return {
            'val_r2': r2.item(),
            'val_mae_mean': mae.mean().item(),
            'val_mae_rare': mae[6:].mean().item(),  # O3, NH3, NO, H2S
        }

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """
        Save model checkpoint.

        Args:
            epoch: Epoch number
            val_loss: Validation loss
        """
        checkpoint = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer_manager.optimizer.state_dict(),
            'scheduler_state': self.optimizer_manager.scheduler.state_dict(),
            'val_loss': val_loss,
            'timestamp': datetime.now().isoformat(),
        }

        # Save best model
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, path)

        # Save periodic checkpoint
        if epoch % 10 == 0:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint

        Returns:
            Epoch number
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer_manager.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.optimizer_manager.scheduler.load_state_dict(checkpoint['scheduler_state'])

        return checkpoint['epoch']
