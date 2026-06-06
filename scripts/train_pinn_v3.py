#!/usr/bin/env python
"""
Complete training pipeline for PINN v3.

Implements:
- Phase 2 enhanced radiative transfer for data generation
- Curriculum learning (easy → hard)
- Advanced optimization with learning rate scheduling
- Multi-task loss balancing
- Comprehensive monitoring and checkpointing

Usage:
    python scripts/train_pinn_v3.py --config configs/cpu.yaml --epochs 100 --batch-size 32

References:
    All phases 1-4 combined into single training loop.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import argparse
import yaml
from tqdm import tqdm
import json
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cosmicml.models.pinn_v3 import PINNv3
from cosmicml.training import (
    EnhancedDataGenerator,
    CurriculumSchedule,
    TrainingManager,
)


def load_config(config_path: str) -> dict:
    """Load training configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train PINN v3 model")
    parser.add_argument('--config', type=str, default='configs/cpu.yaml',
                       help='Training configuration file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--data-dir', type=str, default='data/simulated/',
                       help='Data directory')
    parser.add_argument('--output-dir', type=str, default='models/',
                       help='Output directory')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    device = config['device']

    print(f"\n{'='*70}")
    print("PINN v3 TRAINING PIPELINE")
    print(f"{'='*70}\n")

    print(f"Configuration: {args.config}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")

    # ===== STEP 1: DATA GENERATION =====
    print(f"\n{'='*70}")
    print("STEP 1: GENERATING SYNTHETIC DATA (Phase 2 Physics)")
    print(f"{'='*70}\n")

    data_generator = EnhancedDataGenerator(
        num_atmospheres=5000,  # Smaller for faster training
        use_enhanced_rt=True,
    )

    print("Generating training set (4000 samples)...")
    train_spectra, train_comp, train_temps, train_press = data_generator.generate_batch(4000)

    print("Generating validation set (1000 samples)...")
    val_spectra, val_comp, val_temps, val_press = data_generator.generate_batch(1000)

    train_spectra = torch.from_numpy(train_spectra).float()
    train_comp = torch.from_numpy(train_comp).float()
    train_temps = torch.from_numpy(train_temps).float()

    val_spectra = torch.from_numpy(val_spectra).float()
    val_comp = torch.from_numpy(val_comp).float()
    val_temps = torch.from_numpy(val_temps).float()

    print(f"✓ Training set: {train_spectra.shape}")
    print(f"✓ Validation set: {val_spectra.shape}")

    # ===== STEP 2: MODEL AND TRAINING SETUP =====
    print(f"\n{'='*70}")
    print("STEP 2: INITIALIZING MODEL AND TRAINING")
    print(f"{'='*70}\n")

    # Model
    model = PINNv3(
        input_dim=512,
        output_dim=10,
        num_attention_heads=4,
        num_mc_samples=10,
        physics_weight=1.0,
        gibbs_weight=0.5,
        equilibrium_weight=0.3,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")

    # Training manager
    trainer = TrainingManager(
        model,
        learning_rate=args.learning_rate,
        device=device,
    )

    # Data loaders
    train_dataset = TensorDataset(train_spectra, train_comp, train_temps)
    val_dataset = TensorDataset(val_spectra, val_comp, val_temps)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Curriculum schedule
    curriculum = CurriculumSchedule()

    # ===== STEP 3: TRAINING LOOP =====
    print(f"\n{'='*70}")
    print("STEP 3: TRAINING")
    print(f"{'='*70}\n")

    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)

    training_log = {
        'start_time': datetime.now().isoformat(),
        'config': args.config,
        'epochs': [],
    }

    for epoch in range(start_epoch, args.epochs):
        # Get curriculum stage
        stage = curriculum.get_current_stage()

        print(f"\nEpoch {epoch+1}/{args.epochs} - {stage['name']}")
        print(f"Learning rate: {trainer.optimizer_manager.get_learning_rate():.2e}")

        # Training
        model.train()
        train_metrics = {
            'total_loss': [],
            'data_loss': [],
            'physics_loss': [],
        }

        pbar = tqdm(train_loader, desc="Training")
        for spectra, compositions, temperatures in pbar:
            spectra = spectra.to(device)
            compositions = compositions.to(device)
            temperatures = temperatures.to(device)

            # Apply curriculum mask
            masked_comp = curriculum.apply_mask_to_composition(
                compositions,
                stage['species_mask'],
            )

            metrics = trainer.train_step(spectra, masked_comp, temperatures)

            for key, value in metrics.items():
                if key in train_metrics and key != 'learning_rate':
                    train_metrics[key].append(value)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{metrics['total_loss']:.4f}",
                'lr': f"{metrics['learning_rate']:.2e}",
            })

        # Validation
        model.eval()
        val_losses = []
        val_r2_scores = []

        with torch.no_grad():
            for spectra, compositions, temperatures in val_loader:
                spectra = spectra.to(device)
                compositions = compositions.to(device)

                outputs = model(spectra)
                val_loss = torch.mean((outputs['composition'] - compositions) ** 2)
                val_losses.append(val_loss.item())

                # R² score
                ss_res = torch.sum((outputs['composition'] - compositions) ** 2)
                ss_tot = torch.sum((compositions - compositions.mean()) ** 2)
                r2 = 1 - (ss_res / ss_tot)
                val_r2_scores.append(r2.item())

        val_loss_mean = np.mean(val_losses)
        val_r2_mean = np.mean(val_r2_scores)

        # Save checkpoint
        trainer.save_checkpoint(epoch, val_loss_mean)

        # Log epoch
        epoch_log = {
            'epoch': epoch + 1,
            'stage': stage['name'],
            'train_loss': np.mean(train_metrics['total_loss']),
            'val_loss': val_loss_mean,
            'val_r2': val_r2_mean,
            'learning_rate': trainer.optimizer_manager.get_learning_rate(),
        }
        training_log['epochs'].append(epoch_log)

        print(f"Train Loss: {epoch_log['train_loss']:.4f} | Val Loss: {val_loss_mean:.4f} | R²: {val_r2_mean:.4f}")

        # Advance curriculum if specified
        if (epoch + 1) % stage['epochs'] == 0:
            if curriculum.advance_stage():
                print(f"→ Advanced to next curriculum stage")

    # ===== STEP 4: FINALIZATION =====
    print(f"\n{'='*70}")
    print("STEP 4: TRAINING COMPLETE")
    print(f"{'='*70}\n")

    # Save training log
    training_log['end_time'] = datetime.now().isoformat()

    log_path = Path(args.output_dir) / 'training_log.json'
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)

    print(f"✓ Training log saved to {log_path}")
    print(f"✓ Best model saved to models/checkpoints/best_model.pt")

    # Final validation
    model.eval()
    with torch.no_grad():
        final_outputs = model(val_spectra.to(device))
        final_r2 = 1 - (torch.sum((final_outputs['composition'] - val_comp.to(device)) ** 2) /
                       torch.sum((val_comp.to(device) - val_comp.to(device).mean()) ** 2))

    print(f"\nFinal Validation R²: {final_r2.item():.4f}")
    print(f"Training completed successfully! 🎉\n")


if __name__ == '__main__':
    main()
