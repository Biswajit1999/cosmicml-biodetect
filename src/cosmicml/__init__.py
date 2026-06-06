"""
CosmicML-Biodetect: Physics-Informed Neural Networks for Exoplanet Biosignature Detection

A research framework combining PINNs with Bayesian inference to detect signs of life
in exoplanet atmospheres through transmission spectroscopy analysis.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from . import atmosphere, models, inference, data, utils

__all__ = ["atmosphere", "models", "inference", "data", "utils"]
