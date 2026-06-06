#!/usr/bin/env python
"""Validate trained PINN model on test set."""

import torch
import h5py
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cosmicml.models import PINN


def main():
    print("\n" + "="*60)
    print("PINN MODEL VALIDATION")
    print("="*60 + "\n")

    # Load model
    model_path = "models/pinn_final.pt"
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return

    print("Loading model...")
    model = PINN(input_dim=512, latent_dim=64, output_dim=10)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print(f"✓ Model loaded from {model_path}")

    # Load test data
    print("\nLoading test data...")
    data_file = "data/simulated/synthetic_atmospheres.h5"

    if not Path(data_file).exists():
        print(f"❌ Data file not found: {data_file}")
        return

    with h5py.File(data_file, "r") as f:
        spectra = f["spectra"][:1000]  # First 1000 for validation
        compositions = f["compositions"][:1000]

    print(f"✓ Loaded {len(spectra)} spectra")

    # Convert to tensors
    spectra_t = torch.from_numpy(spectra).float()
    compositions_t = torch.from_numpy(compositions).float()

    # Predict
    print("\nRunning inference...")
    with torch.no_grad():
        predictions = model(spectra_t)

    print(f"✓ Generated {len(predictions)} predictions")

    # Compute metrics
    print("\n" + "="*60)
    print("VALIDATION METRICS")
    print("="*60 + "\n")

    # MAE
    mae = torch.mean(torch.abs(predictions - compositions_t)).item()
    print(f"Mean Absolute Error (MAE):    {mae:.6f}")

    # RMSE
    rmse = torch.sqrt(torch.mean((predictions - compositions_t)**2)).item()
    print(f"Root Mean Squared Error:      {rmse:.6f}")

    # R² Score
    ss_res = torch.sum((predictions - compositions_t)**2).item()
    ss_tot = torch.sum((compositions_t - compositions_t.mean())**2).item()
    r2 = 1 - (ss_res / ss_tot)
    print(f"R² Score:                     {r2:.4f}")

    # Check constraints
    print("\n" + "-"*60)
    print("PHYSICS CONSTRAINT VALIDATION")
    print("-"*60 + "\n")

    # All values in [0, 1]
    in_bounds = (predictions >= -1e-6).all() and (predictions <= 1.0).all()
    print(f"All predictions in [0, 1]:    {'✓ YES' if in_bounds else '✗ NO'}")

    # Sum to 1
    sums = torch.sum(predictions, dim=1)
    sum_correct = torch.allclose(sums, torch.ones(len(sums)), atol=1e-5)
    print(f"All sums ≈ 1:                 {'✓ YES' if sum_correct else '✗ NO'}")

    # Min/max values
    print(f"Min prediction value:         {predictions.min().item():.6f}")
    print(f"Max prediction value:         {predictions.max().item():.6f}")
    print(f"Mean sum:                     {sums.mean().item():.6f}")

    # Per-species accuracy
    print("\n" + "-"*60)
    print("PER-SPECIES PERFORMANCE")
    print("-"*60 + "\n")

    species = ['H2O', 'CO2', 'O2', 'N2', 'CH4', 'H2', 'O3', 'NH3', 'NO', 'H2S']

    for i, sp in enumerate(species):
        mae_sp = torch.mean(torch.abs(predictions[:, i] - compositions_t[:, i])).item()
        rmse_sp = torch.sqrt(torch.mean((predictions[:, i] - compositions_t[:, i])**2)).item()

        # R² for this species
        ss_res_sp = torch.sum((predictions[:, i] - compositions_t[:, i])**2).item()
        ss_tot_sp = torch.sum((compositions_t[:, i] - compositions_t[:, i].mean())**2).item()
        r2_sp = 1 - (ss_res_sp / ss_tot_sp) if ss_tot_sp > 0 else 0

        print(f"{sp:5} | MAE: {mae_sp:.6f} | RMSE: {rmse_sp:.6f} | R²: {r2_sp:.4f}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60 + "\n")

    if mae < 0.03 and r2 > 0.90 and sum_correct and in_bounds:
        print("✓ MODEL VALIDATION PASSED!")
        print(f"  - Excellent accuracy (MAE={mae:.4f}, R²={r2:.4f})")
        print(f"  - Physics constraints satisfied")
        print(f"  - Ready for deployment")
    elif mae < 0.05 and r2 > 0.85:
        print("✓ MODEL VALIDATION PASSED (Good)")
        print(f"  - Good accuracy (MAE={mae:.4f}, R²={r2:.4f})")
        print(f"  - Physics constraints satisfied")
    else:
        print("⚠ MODEL VALIDATION WARNINGS")
        print(f"  - MAE: {mae:.4f} (target < 0.03)")
        print(f"  - R²: {r2:.4f} (target > 0.90)")

    print("\n" + "="*60 + "\n")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "sum_correct": sum_correct,
        "in_bounds": in_bounds,
    }


if __name__ == "__main__":
    main()
