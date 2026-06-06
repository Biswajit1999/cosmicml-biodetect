"""
Atmospheric simulation and modeling.

Generates synthetic exoplanet spectra by simulating atmospheric composition,
radiative transfer, and transmission spectroscopy under various planetary conditions.
"""

from .simulator import AtmosphereSimulator
from .chemistry import ChemistryEngine
from .radiative_transfer import RadioativeTransferCalculator

__all__ = ["AtmosphereSimulator", "ChemistryEngine", "RadioativeTransferCalculator"]
