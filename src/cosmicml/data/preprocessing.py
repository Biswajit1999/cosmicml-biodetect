"""Data preprocessing and normalization."""

import numpy as np
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """Normalizes and preprocesses spectral data."""

    def __init__(self):
        """Initialize preprocessor."""
        self.spectrum_scaler = StandardScaler()
        self.composition_scaler = StandardScaler()

    def normalize_spectrum(self, spectra: np.ndarray) -> np.ndarray:
        """Normalize spectral data to zero mean, unit variance."""
        return self.spectrum_scaler.fit_transform(spectra)

    def normalize_composition(self, compositions: np.ndarray) -> np.ndarray:
        """Normalize composition data."""
        return self.composition_scaler.fit_transform(compositions)

    def denormalize_spectrum(self, normalized: np.ndarray) -> np.ndarray:
        """Reverse spectrum normalization."""
        return self.spectrum_scaler.inverse_transform(normalized)

    def denormalize_composition(self, normalized: np.ndarray) -> np.ndarray:
        """Reverse composition normalization."""
        return self.composition_scaler.inverse_transform(normalized)
