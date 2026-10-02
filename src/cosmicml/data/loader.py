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
        raise NotImplementedError(
            "No synthetic dataset is bundled. Use cosmicml.benchmark for the "
            "validated emulator workflow or implement an explicit HDF5 schema."
        )

    def load_jwst_data(self, planet_name: str) -> Dict:
        """
        Load JWST observation data for a specific exoplanet.

        Args:
            planet_name: Name of exoplanet

        Returns:
            Dict with spectrum, uncertainties, metadata
        """
        raise NotImplementedError(
            "Observed JWST ingestion is not implemented; this method must not "
            "return fabricated arrays."
        )
