"""JWST data pipeline for real exoplanet observations."""

import numpy as np
from typing import Dict


class JWSTDataPipeline:
    """
    Processes JWST transmission spectroscopy data.

    Handles wavelength calibration, systematic removal, and uncertainty quantification.
    """

    def __init__(self):
        """Initialize JWST data pipeline."""
        pass

    def load_jwst_spectrum(self, fits_file: str) -> Dict:
        """
        Load JWST FITS file.

        Args:
            fits_file: Path to JWST data FITS file

        Returns:
            Dict with wavelengths, flux, uncertainties
        """
        raise NotImplementedError(
            "JWST FITS ingestion and provenance validation are not implemented. "
            "No synthetic spectrum is substituted for an observation."
        )

    def calibrate_wavelengths(self, wavelengths: np.ndarray) -> np.ndarray:
        """Calibrate wavelength scale."""
        raise NotImplementedError("Wavelength calibration is not implemented.")

    def remove_systematics(
        self,
        spectrum: np.ndarray,
        uncertainties: np.ndarray,
    ) -> np.ndarray:
        """Remove instrumental systematics using GP or polynomial fitting."""
        raise NotImplementedError("Instrument-systematics removal is not implemented.")
