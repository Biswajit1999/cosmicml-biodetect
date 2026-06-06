"""Data loading utilities."""

import numpy as np
from typing import Tuple, Dict


class DataLoader:
    """Loads and manages spectral datasets for training and inference."""

    def __init__(self, data_dir: str):
        """
        Initialize data loader.

        Args:
            data_dir: Path to data directory
        """
        self.data_dir = data_dir

    def load_synthetic_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load synthetic spectra and compositions.

        Returns:
            Tuple of (spectra, compositions)
        """
        # Placeholder for HDF5 data loading
        return np.zeros((1000, 512)), np.zeros((1000, 32))

    def load_jwst_data(self, planet_name: str) -> Dict:
        """
        Load JWST observation data for a specific exoplanet.

        Args:
            planet_name: Name of exoplanet

        Returns:
            Dict with spectrum, uncertainties, metadata
        """
        return {
            "wavelengths": np.linspace(0.3, 5.0, 512),
            "spectrum": np.ones(512),
            "uncertainties": np.ones(512) * 0.0001,
            "metadata": {},
        }
