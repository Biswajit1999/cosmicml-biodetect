#!/usr/bin/env python
"""
Train Physics-Informed Neural Network for exoplanet biosignature detection.

Implements full training pipeline with:
- PyTorch Lightning for distributed training
- TensorBoard logging
- Early stopping and checkpointing
- Learning rate scheduling
- Mixed precision training

References:
    Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
    arXiv preprint arXiv:1412.6980.

    Smith, L. N. (2018). A disciplined approach to neural network hyper-parameters:
    Part 1--learning rate, batch size, momentum, and weight decay. arXiv preprint
    arXiv:1803.09820.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import yaml
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cosmicml.models import PINN


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PINNLightningModule(pl.LightningModule):
    """PyTorch Lightning module for PINN training."""

    def __init__(
        self,
        model: PINN,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_type: str = "cosine",
        physics_weight: float = 1.0,
    ):
        """
        Initialize Lightning module.

        Args:
            model: PINN model
            learning_rate: Initial learning rate
            weight_decay: L2 regularization
            scheduler_type: Learning rate scheduler type
            physics_weight: Weight for physics loss
        """
        super().__init__()

        self.pinn_model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.physics_weight = physics_weight

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_data_losses = []
        self.train_physics_losses = []

    def forward(self, x):
        """Forward pass."""
        return self.pinn_model(x)

    def training_step(self, batch, batch_idx):
        """Single training step."""
        x, y = batch

        # Compute loss
        total_loss, data_loss, physics_loss = self.pinn_model.compute_loss(x, y)

        # Log metrics
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_data_loss", data_loss)
        self.log("train_physics_loss", physics_loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Single validation step."""
        x, y = batch

        # Forward pass
        predictions = self.pinn_model(x)

        # Compute MSE
        val_loss = nn.MSELoss()(predictions, y)

        self.log("val_loss", val_loss, prog_bar=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        """Single test step."""
        x, y = batch

        predictions = self.pinn_model(x)
        test_loss = nn.MSELoss()(predictions, y)

        return test_loss

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if self.scheduler_type == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=1e-6,
            )
        elif self.scheduler_type == "step":
            scheduler = StepLR(
                optimizer,
                step_size=10,
                gamma=0.5,
            )
        else:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


def load_data(
    data_file: str,
    train_split: float = 0.8,
    val_split: float = 0.1,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load training data from HDF5 file.

    Args:
        data_file: Path to HDF5 file
        train_split: Training set fraction
        val_split: Validation set fraction
        batch_size: Batch size
        num_workers: Number of data loader workers

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    logger.info(f"Loading data from {data_file}")

    with h5py.File(data_file, "r") as f:
        spectra = torch.from_numpy(np.array(f["spectra"])).float()
        compositions = torch.from_numpy(np.array(f["compositions"])).float()

        # Get metadata
        metadata_str = f.attrs.get("metadata", "{}")
        if isinstance(metadata_str, bytes):
            metadata_str = metadata_str.decode("utf-8")
        metadata = json.loads(metadata_str)

    logger.info(f"Loaded {len(spectra)} spectra")
    logger.info(f"Spectrum dimension: {spectra.shape[1]}")
    logger.info(f"Composition dimension: {compositions.shape[1]}")

    # Create dataset
    dataset = TensorDataset(spectra, compositions)

    # Split into train/val/test
    n_total = len(dataset)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    n_test = n_total - n_train - n_val

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    logger.info(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")

    return train_loader, val_loader, test_loader, metadata


def load_config(config_file: str) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_file: Path to YAML config file

    Returns:
        Configuration dictionary
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Train PINN for exoplanet biosignature detection"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/gpu.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/simulated/",
        help="Path to synthetic data directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (overrides config)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Enable mixed precision training",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override with command line arguments
    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate

    logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    # Setup device
    device = torch.device(config["device"])
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    data_dir = Path(args.data_dir)
    data_file = data_dir / "synthetic_atmospheres.h5"

    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        logger.info("Run: python scripts/generate_synthetic_data.py")
        return

    train_loader, val_loader, test_loader, metadata = load_data(
        str(data_file),
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    # Create model
    logger.info("Creating PINN model...")
    model = PINN(
        input_dim=config["data"]["input_dim"],
        latent_dim=config["data"]["latent_dim"],
        output_dim=config["data"]["output_dim"],
        hidden_layers=config["model"]["hidden_layers"],
        physics_weight=config["model"]["physics_weight"],
    )

    logger.info(f"Model: {model}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create Lightning module
    lightning_module = PINNLightningModule(
        model=model,
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        scheduler_type=config["training"]["scheduler"],
        physics_weight=config["model"]["physics_weight"],
    )

    # Setup logger
    tb_logger = TensorBoardLogger(
        str(output_dir),
        name="logs",
        version=0,
    )

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir),
        filename="pinn-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        verbose=True,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=config["validation"]["patience"],
        verbose=True,
        mode="min",
    )

    # Setup trainer
    precision = "16-mixed" if args.mixed_precision else "32"

    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        accelerator=config["device"],
        devices=1,
        precision=precision,
        log_every_n_steps=config["logging"]["log_interval"],
        gradient_clip_val=1.0,
        enable_progress_bar=True,
    )

    # Train model
    logger.info("Starting training...")
    trainer.fit(
        lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Test model
    logger.info("Testing model...")
    test_results = trainer.test(
        lightning_module,
        dataloaders=test_loader,
    )

    logger.info(f"Test results: {test_results}")

    # Save final model
    final_model_path = output_dir / "pinn_final.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved to {final_model_path}")

    # Save configuration
    config_save_path = output_dir / "training_config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f)
    logger.info(f"Configuration saved to {config_save_path}")

    logger.info("✓ Training complete!")


if __name__ == "__main__":
    main()
