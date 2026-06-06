"""
Radiative transfer calculations for transmission spectroscopy.

Implements solutions to the radiative transfer equation for exoplanet atmospheres.

References:
    Lecavelier Des Etangs, A., Pont, F., Vidal-Madjar, A., & Sing, D. (2008). Atmospheric
    escape from hot Jupiters: Hydrogen high-resolution transmission spectroscopy. Astronomy &
    Astrophysics, 481(2), L83-L86.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.integrate import odeint, cumulative_trapezoid


class RadioativeTransferCalculator:
    """
    Solves radiative transfer equation for transmission spectroscopy.

    The radiative transfer equation in plane-parallel geometry for a non-scattering atmosphere:

    dI_nu/dz = -k_nu * I_nu + j_nu

    where:
        I_nu = intensity at frequency nu
        k_nu = mass extinction coefficient
        j_nu = emission coefficient

    For transmission spectroscopy (stellar light passing through atmosphere):
        I_nu ~ exp(-tau_nu) where tau_nu = optical depth
    """

    def __init__(self, wavelengths: np.ndarray):
        """
        Initialize radiative transfer calculator.

        Args:
            wavelengths: Wavelength array in micrometers
        """
        self.wavelengths = wavelengths

    def compute_optical_depth(
        self,
        wavelength: float,
        composition: Dict[str, float],
        altitude_profile: np.ndarray,
        density_profile: np.ndarray,
        temperature_profile: np.ndarray,
        cross_section_func,
    ) -> float:
        """
        Compute total optical depth for a given wavelength.

        tau = integral of kappa(lambda) * rho(z) dz

        Args:
            wavelength: Wavelength in micrometers
            composition: Atmospheric composition {species: mixing_ratio}
            altitude_profile: Altitude in km
            density_profile: Number density in m^-3
            temperature_profile: Temperature in K
            cross_section_func: Function to get cross-sections

        Returns:
            Total optical depth (dimensionless)
        """
        tau_total = 0.0

        for species, mixing_ratio in composition.items():
            if mixing_ratio < 1e-12:
                continue

            # Get cross-section for this species and wavelength
            sigma = cross_section_func(species, wavelength, temperature_profile.mean())

            if sigma is None or sigma < 1e-25:
                continue

            # Number density of this species
            n_species = density_profile * mixing_ratio  # m^-3
            n_species *= 1e-6  # Convert to cm^-3

            # Optical depth integral: tau = integral(n * sigma * dz)
            integrand = n_species * sigma
            dz = np.gradient(altitude_profile) * 1e5  # Convert to cm

            tau_species = np.sum(integrand * dz)
            tau_total += tau_species

        return np.clip(tau_total, 0, 100)

    def compute_transit_depth(
        self,
        optical_depth: np.ndarray,
        scale_height: float,
        planet_radius_m: float,
        star_radius_m: float,
    ) -> np.ndarray:
        """
        Compute transmission spectrum transit depth from optical depth.

        Uses formula from Seager et al. (2005):

        d(lambda) = (R_p + H * ln(1 + tau))^2 - R_p^2 / R_star^2

        Args:
            optical_depth: Optical depth array
            scale_height: Atmospheric scale height in meters
            planet_radius_m: Planet radius in meters
            star_radius_m: Star radius in meters

        Returns:
            Transit depth array (dimensionless)
        """
        # Effective scale height (wavelength-dependent)
        H_eff = scale_height * np.log(1 + optical_depth)

        # Transit depth formula
        transit_depth = ((planet_radius_m + H_eff)**2 - planet_radius_m**2) / (star_radius_m**2)

        return np.clip(transit_depth, 0, 0.1)

    def compute_transmission_spectrum(
        self,
        wavelengths: np.ndarray,
        composition: Dict[str, float],
        altitude_profile: np.ndarray,
        density_profile: np.ndarray,
        temperature_profile: np.ndarray,
        scale_height: float,
        planet_radius: float,
        star_radius: float,
        cross_section_func,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute full transmission spectrum.

        Args:
            wavelengths: Wavelength array in micrometers
            composition: Atmospheric composition
            altitude_profile: Altitude profile in km
            density_profile: Number density profile in m^-3
            temperature_profile: Temperature profile in K
            scale_height: Atmospheric scale height in m
            planet_radius: Planet radius in m
            star_radius: Star radius in m
            cross_section_func: Function to get cross-sections

        Returns:
            Tuple of (wavelengths, transit_depths)
        """
        transit_depths = np.zeros_like(wavelengths)

        for i, wavelength in enumerate(wavelengths):
            tau = self.compute_optical_depth(
                wavelength,
                composition,
                altitude_profile,
                density_profile,
                temperature_profile,
                cross_section_func,
            )

            transit_depth = self.compute_transit_depth(
                tau,
                scale_height,
                planet_radius,
                star_radius,
            )

            transit_depths[i] = transit_depth

        return wavelengths, transit_depths
