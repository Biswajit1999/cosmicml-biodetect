"""Visualization utilities for spectral analysis."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional


def plot_spectrum(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    uncertainties: Optional[np.ndarray] = None,
    title: str = "Transmission Spectrum",
    save_path: Optional[str] = None,
):
    """
    Plot transmission spectrum with uncertainties.

    Args:
        wavelengths: Wavelength array (micrometers)
        spectrum: Transit depth array
        uncertainties: Optional uncertainty array
        title: Plot title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(wavelengths, spectrum, "b-", linewidth=2, label="Spectrum")

    if uncertainties is not None:
        ax.fill_between(
            wavelengths,
            spectrum - uncertainties,
            spectrum + uncertainties,
            alpha=0.3,
        )

    ax.set_xlabel("Wavelength (μm)")
    ax.set_ylabel("Transit Depth")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax


def plot_atmospheric_composition(
    species: list,
    composition: np.ndarray,
    save_path: Optional[str] = None,
):
    """
    Plot atmospheric composition as bar chart.

    Args:
        species: List of species names
        composition: Mixing ratio array
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    indices = np.arange(len(species))
    ax.bar(indices, composition)
    ax.set_xticks(indices)
    ax.set_xticklabels(species, rotation=45, ha="right")
    ax.set_ylabel("Mixing Ratio")
    ax.set_title("Atmospheric Composition")
    ax.set_yscale("log")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig, ax
