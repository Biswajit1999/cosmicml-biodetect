"""
Bayesian inference pipeline for biosignature detection.

Uses MCMC and variational inference to estimate atmospheric composition
and compute posterior probabilities for biosignatures.
"""

from .bayesian import BayesianInference
from .likelihood import SpectraLikelihood

__all__ = ["BayesianInference", "SpectraLikelihood"]
