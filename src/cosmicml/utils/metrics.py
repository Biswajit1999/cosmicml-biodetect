"""Evaluation metrics for biosignature detection."""

import numpy as np
from typing import Dict


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Args:
        predictions: Model predictions
        targets: Ground truth values

    Returns:
        Dict with MSE, MAE, R2 score
    """
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    return {"mse": mse, "mae": mae, "r2": r2}


def calculate_uncertainties(
    posterior_samples: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Calculate credible intervals and uncertainties from posterior samples.

    Args:
        posterior_samples: MCMC posterior samples

    Returns:
        Dict with means, stds, credible intervals
    """
    return {
        "mean": np.mean(posterior_samples, axis=0),
        "std": np.std(posterior_samples, axis=0),
        "lower_credible": np.percentile(posterior_samples, 2.5, axis=0),
        "upper_credible": np.percentile(posterior_samples, 97.5, axis=0),
    }
