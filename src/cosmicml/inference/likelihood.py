"""Likelihood functions for spectroscopic data."""

import numpy as np
from typing import Tuple


class SpectraLikelihood:
    """
    Computes likelihood of observed spectrum given atmospheric model.

    Implements Gaussian likelihood with noise characterization.
    """

    def __init__(self, wavelengths: np.ndarray):
        """
        Initialize likelihood function.

        Args:
            wavelengths: Array of wavelengths
        """
        self.wavelengths = wavelengths

    def compute_likelihood(
        self,
        observed_spectrum: np.ndarray,
        predicted_spectrum: np.ndarray,
        uncertainties: np.ndarray,
    ) -> float:
        """
        Compute Gaussian likelihood.

        Args:
            observed_spectrum: Observed transmission spectrum
            predicted_spectrum: Model prediction
            uncertainties: Measurement uncertainties

        Returns:
            Log-likelihood value
        """
        residuals = observed_spectrum - predicted_spectrum
        chi2 = np.sum((residuals / uncertainties) ** 2)
        log_likelihood = -0.5 * chi2

        return log_likelihood
