"""Deterministic synthetic spectral-amplitude recovery benchmark.

This module deliberately tests an analytic template emulator. It does not
model line-by-line radiative transfer and cannot establish biosignature
detection capability on observed spectra.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SPECIES = ("H2O", "CO2", "O2", "O3", "CH4")


@dataclass(frozen=True)
class BenchmarkData:
    wavelengths_um: np.ndarray
    spectra: np.ndarray
    amplitudes: np.ndarray
    family_ids: np.ndarray


def _gaussian(wavelengths: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((wavelengths - center) / width) ** 2)


def molecular_templates(wavelengths: np.ndarray) -> np.ndarray:
    """Return normalized analytic feature templates, one row per species."""

    bands = {
        "H2O": ((1.15, 0.09, 0.55), (1.40, 0.12, 1.0), (1.90, 0.15, 0.75)),
        "CO2": ((2.00, 0.10, 0.30), (2.70, 0.14, 0.55), (4.30, 0.13, 1.0)),
        "O2": ((0.69, 0.025, 0.40), (0.76, 0.018, 1.0), (1.27, 0.035, 0.35)),
        "O3": ((0.60, 0.11, 1.0), (4.75, 0.10, 0.22)),
        "CH4": ((1.65, 0.07, 0.45), (2.30, 0.11, 0.75), (3.30, 0.12, 1.0)),
    }
    rows = []
    for species in SPECIES:
        template = sum(
            weight * _gaussian(wavelengths, center, width)
            for center, width, weight in bands[species]
        )
        rows.append(template / template.max())
    return np.asarray(rows)


def generate_families(
    n_families: int = 360,
    replicates: int = 4,
    seed: int = 20260927,
    noise_sigma: float = 6e-5,
) -> BenchmarkData:
    """Generate family-averaged spectra from a declared analytic emulator."""

    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(0.5, 5.0, 256)
    templates = molecular_templates(wavelengths)
    amplitudes = rng.beta(1.4, 3.0, size=(n_families, len(SPECIES)))
    # Deliberately include correlated atmospheric archetypes.
    archetype = np.arange(n_families) % 5
    amplitudes[archetype == 0, 0] = rng.uniform(0.55, 1.0, (archetype == 0).sum())
    amplitudes[archetype == 1, 1] = rng.uniform(0.55, 1.0, (archetype == 1).sum())
    amplitudes[archetype == 2, 2:4] = rng.uniform(0.35, 0.9, ((archetype == 2).sum(), 2))
    amplitudes[archetype == 3, 4] = rng.uniform(0.55, 1.0, (archetype == 3).sum())

    spectra = []
    x = (wavelengths - wavelengths.mean()) / np.ptp(wavelengths)
    for family in range(n_families):
        continuum = (
            rng.normal(0.010, 2e-4)
            + rng.normal(0, 1.8e-4) * x
            + rng.normal(0, 1.0e-4) * (x**2 - np.mean(x**2))
        )
        noiseless = continuum + 8e-4 * amplitudes[family] @ templates
        realizations = noiseless + rng.normal(
            0, noise_sigma, size=(replicates, wavelengths.size)
        )
        spectra.append(realizations.mean(axis=0))

    return BenchmarkData(
        wavelengths_um=wavelengths,
        spectra=np.asarray(spectra),
        amplitudes=amplitudes,
        family_ids=np.arange(n_families),
    )


def split_family_ids(family_ids: np.ndarray, seed: int = 4107) -> dict[str, np.ndarray]:
    """Create disjoint 50/20/15/15 train/tune/calibration/test splits."""

    ids = np.asarray(family_ids)
    shuffled = np.random.default_rng(seed).permutation(ids)
    n = len(shuffled)
    cuts = (round(0.50 * n), round(0.70 * n), round(0.85 * n))
    return {
        "train": shuffled[: cuts[0]],
        "tune": shuffled[cuts[0] : cuts[1]],
        "calibration": shuffled[cuts[1] : cuts[2]],
        "test": shuffled[cuts[2] :],
    }


def _conformal_quantile(residuals: np.ndarray, coverage: float) -> np.ndarray:
    n = residuals.shape[0]
    rank = min(int(np.ceil((n + 1) * coverage)), n)
    return np.sort(residuals, axis=0)[rank - 1]


def run_benchmark(
    seed: int = 20260927,
    noise_sigma: float = 6e-5,
) -> tuple[dict[str, Any], BenchmarkData]:
    """Fit, tune, calibrate, and evaluate a ridge inverse model."""

    data = generate_families(seed=seed, noise_sigma=noise_sigma)
    splits = split_family_ids(data.family_ids)
    alphas = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    tune_scores: dict[float, float] = {}
    for alpha in alphas:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(data.spectra[splits["train"]], data.amplitudes[splits["train"]])
        pred = np.clip(model.predict(data.spectra[splits["tune"]]), 0, 1)
        tune_scores[alpha] = float(mean_absolute_error(data.amplitudes[splits["tune"]], pred))

    selected_alpha = min(tune_scores, key=tune_scores.get)
    fit_ids = np.concatenate([splits["train"], splits["tune"]])
    model = make_pipeline(StandardScaler(), Ridge(alpha=selected_alpha))
    model.fit(data.spectra[fit_ids], data.amplitudes[fit_ids])

    calibration_pred = np.clip(model.predict(data.spectra[splits["calibration"]]), 0, 1)
    q90 = _conformal_quantile(
        np.abs(calibration_pred - data.amplitudes[splits["calibration"]]), 0.90
    )
    test_truth = data.amplitudes[splits["test"]]
    test_pred = np.clip(model.predict(data.spectra[splits["test"]]), 0, 1)
    lower, upper = np.clip(test_pred - q90, 0, 1), np.clip(test_pred + q90, 0, 1)

    mean_baseline = data.amplitudes[fit_ids].mean(axis=0)
    baseline_pred = np.repeat(mean_baseline[None, :], len(test_truth), axis=0)
    rng = np.random.default_rng(seed + 1)
    permuted_model = make_pipeline(StandardScaler(), Ridge(alpha=selected_alpha))
    permuted_model.fit(data.spectra[fit_ids], rng.permutation(data.amplitudes[fit_ids]))
    permuted_pred = np.clip(permuted_model.predict(data.spectra[splits["test"]]), 0, 1)

    species_metrics = []
    for idx, species in enumerate(SPECIES):
        species_metrics.append(
            {
                "species": species,
                "test_mae": float(mean_absolute_error(test_truth[:, idx], test_pred[:, idx])),
                "test_r2": float(r2_score(test_truth[:, idx], test_pred[:, idx])),
                "mean_baseline_mae": float(
                    mean_absolute_error(test_truth[:, idx], baseline_pred[:, idx])
                ),
                "permuted_label_mae": float(
                    mean_absolute_error(test_truth[:, idx], permuted_pred[:, idx])
                ),
                "conformal_half_width": float(q90[idx]),
                "test_interval_coverage": float(
                    np.mean((test_truth[:, idx] >= lower[:, idx]) & (test_truth[:, idx] <= upper[:, idx]))
                ),
            }
        )

    result = {
        "benchmark": "analytic template-amplitude recovery",
        "scope": "synthetic emulator only; not observed-data biosignature detection",
        "seed": seed,
        "n_families": len(data.family_ids),
        "replicates_averaged_per_family": 4,
        "per_replicate_noise_sigma": noise_sigma,
        "split_counts": {name: len(ids) for name, ids in splits.items()},
        "split_overlap_count": int(
            sum(
                len(set(splits[a]).intersection(splits[b]))
                for i, a in enumerate(splits)
                for b in list(splits)[i + 1 :]
            )
        ),
        "selected_ridge_alpha": selected_alpha,
        "tune_mae_by_alpha": {str(k): v for k, v in tune_scores.items()},
        "interval_target_coverage": 0.90,
        "species_metrics": species_metrics,
        "macro_test_mae": float(mean_absolute_error(test_truth, test_pred)),
        "macro_mean_baseline_mae": float(mean_absolute_error(test_truth, baseline_pred)),
        "macro_permuted_label_mae": float(mean_absolute_error(test_truth, permuted_pred)),
        "macro_test_interval_coverage": float(
            np.mean((test_truth >= lower) & (test_truth <= upper))
        ),
    }
    return result, data
