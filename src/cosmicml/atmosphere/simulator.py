"""
Exoplanet atmosphere simulator with radiative transfer physics.

Generates synthetic transmission spectra using:
- Radiative transfer equation solver
- Molecular absorption cross-sections
- Chemical species profiles
- Temperature/pressure models

References:
    Seager, S., Turner, E. L., Schafer, E., & Ford, E. B. (2005). Vegetation's Red Edge: A
    Possible Spectroscopic Biosignature of Extraterrestrial Plants. Astrobiology, 5(2), 372-390.

    Madhusudhan, N., Amin, M. A., & Kennedy, G. M. (2014). Architecture and Fate of Planetary
    Systems. Monthly Notices of the Royal Astronomical Society, 445(2), 1561-1598.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid


@dataclass
class PlanetaryConfig:
    """Configuration for planetary atmosphere simulation."""

    planet_radius: float  # Earth radii
    planet_mass: float  # Earth masses
    star_temp: float  # Kelvin
    orbital_period: float  # Days
    equilibrium_temp: float  # Kelvin
    surface_gravity: float  # m/s^2
    stellar_radius: float = 1.0  # Solar radii


# Molecular cross-section data (simplified from HITRAN)
# Units: cm^2 (cross-section at 0.1 bar, 250 K)
CROSS_SECTIONS = {
    "H2O": {
        "wavelengths": np.array([0.3, 0.5, 0.8, 1.1, 1.4, 1.9, 2.7, 4.3, 5.0]),
        "sigmas": np.array([1e-19, 2e-20, 5e-21, 1e-20, 5e-20, 2e-19, 1e-19, 5e-20, 1e-20]),
    },
    "CO2": {
        "wavelengths": np.array([0.3, 0.5, 0.8, 1.1, 1.4, 1.9, 2.7, 4.3, 5.0]),
        "sigmas": np.array([5e-20, 1e-20, 2e-21, 5e-21, 1e-21, 5e-21, 1e-20, 5e-19, 1e-18]),
    },
    "O2": {
        "wavelengths": np.array([0.3, 0.5, 0.8, 1.1, 1.4, 1.9, 2.7, 4.3, 5.0]),
        "sigmas": np.array([1e-20, 5e-21, 1e-21, 5e-22, 1e-22, 1e-22, 1e-22, 1e-21, 1e-20]),
    },
    "O3": {
        "wavelengths": np.array([0.3, 0.5, 0.8, 1.1, 1.4, 1.9, 2.7, 4.3, 5.0]),
        "sigmas": np.array([1e-17, 5e-18, 1e-18, 1e-19, 5e-20, 1e-20, 1e-20, 1e-21, 1e-21]),
    },
    "CH4": {
        "wavelengths": np.array([0.3, 0.5, 0.8, 1.1, 1.4, 1.9, 2.7, 4.3, 5.0]),
        "sigmas": np.array([1e-20, 5e-21, 1e-21, 5e-22, 1e-21, 2e-20, 1e-19, 1e-19, 5e-20]),
    },
    "N2": {
        "wavelengths": np.array([0.3, 0.5, 0.8, 1.1, 1.4, 1.9, 2.7, 4.3, 5.0]),
        "sigmas": np.array([1e-21, 5e-22, 1e-22, 5e-23, 1e-23, 1e-23, 1e-23, 1e-22, 1e-21]),
    },
}


class AtmosphereSimulator:
    """
    Simulates exoplanet atmospheres and generates transmission spectra.

    Implements the radiative transfer equation for transmission spectroscopy:

    tau(lambda) = integral of n_i(z) * sigma_i(lambda) dz

    Transit depth: d(lambda) = (R_p + H_eff(lambda))^2 - R_p^2 / R_star^2

    where H_eff is the effective scale height dependent on wavelength-dependent
    absorption (Seager et al., 2005).
    """

    def __init__(self, config: PlanetaryConfig, wavelength_range: Tuple[float, float] = (0.3, 5.0),
                 n_wavelengths: int = 512):
        """
        Initialize the atmosphere simulator.

        Args:
            config: PlanetaryConfig object with physical parameters
            wavelength_range: (min_wavelength, max_wavelength) in micrometers
            n_wavelengths: Number of wavelength points for spectrum
        """
        self.config = config
        self.wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_wavelengths)

        # Compute scale height in km
        k_B = 1.38e-23  # J/K
        self.mean_molecular_mass = 28 * 1.66e-27  # kg (average for N2/O2 like atmosphere)
        self.scale_height = k_B * config.equilibrium_temp / \
                           (self.mean_molecular_mass * config.surface_gravity)
        self.scale_height /= 1000  # Convert to km

    def simulate_atmosphere(
        self,
        composition: Dict[str, float],
        n_layers: int = 100,
        temperature_profile: Optional[str] = "isothermal",
    ) -> Dict[str, np.ndarray]:
        """
        Simulate atmospheric properties at different altitudes.

        Args:
            composition: Dict of {gas_name: mixing_ratio} where mixing ratios sum to 1
            n_layers: Number of altitude layers to simulate
            temperature_profile: Type of temperature profile ('isothermal', 'linear', 'stratified')

        Returns:
            Dict with temperature, pressure, density, and composition profiles
        """
        # Validate composition
        total_mixing = sum(composition.values())
        if not np.isclose(total_mixing, 1.0, atol=0.01):
            # Normalize to sum to 1
            composition = {k: v / total_mixing for k, v in composition.items()}

        altitudes = np.linspace(0, 150, n_layers)  # km
        temperatures = self._compute_temperature_profile(altitudes, temperature_profile)
        pressures = self._compute_pressure_profile(altitudes, temperatures)
        densities = self._compute_density_profile(pressures, temperatures)

        return {
            "altitudes": altitudes,
            "temperatures": temperatures,
            "pressures": pressures,
            "densities": densities,
            "composition": composition,
            "scale_height": self.scale_height,
        }

    def generate_spectrum(
        self,
        atmosphere: Dict[str, np.ndarray],
        noise_level: float = 1e-4,
        add_noise: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate transmission spectrum from atmospheric properties.

        Uses radiative transfer equation to compute optical depth and transit depth.

        Args:
            atmosphere: Dict from simulate_atmosphere()
            noise_level: Relative noise level (as fraction of signal)
            add_noise: Whether to add realistic instrumental noise

        Returns:
            Tuple of (wavelengths, transit_depths, uncertainties)
        """
        transit_depths = np.zeros_like(self.wavelengths)

        for i, wavelength in enumerate(self.wavelengths):
            # Compute optical depth
            tau = self._compute_optical_depth(
                wavelength,
                atmosphere["composition"],
                atmosphere["densities"],
                atmosphere["altitudes"],
                atmosphere["temperatures"],
            )

            # Compute transit depth from optical depth
            # Using formula from Seager et al. (2005)
            transit_depth = self._optical_depth_to_transit_depth(tau)
            transit_depths[i] = transit_depth

        # Add realistic noise
        uncertainties = np.ones_like(self.wavelengths) * noise_level * np.mean(transit_depths)

        if add_noise:
            noise = np.random.normal(0, uncertainties, len(self.wavelengths))
            transit_depths += noise

        return self.wavelengths, transit_depths, uncertainties

    def _compute_temperature_profile(
        self,
        altitudes: np.ndarray,
        profile_type: str = "isothermal",
    ) -> np.ndarray:
        """
        Compute temperature as function of altitude.

        Args:
            altitudes: Altitude array in km
            profile_type: Type of profile to compute

        Returns:
            Temperature array in Kelvin
        """
        if profile_type == "isothermal":
            return np.ones_like(altitudes) * self.config.equilibrium_temp

        elif profile_type == "linear":
            # Temperature decreases linearly with altitude
            T_top = self.config.equilibrium_temp * 0.5
            return self.config.equilibrium_temp - (self.config.equilibrium_temp - T_top) * (altitudes / 150)

        elif profile_type == "stratified":
            # Stratified model: isothermal troposphere + temperature inversion in stratosphere
            troposphere_height = 50  # km
            T_trop = self.config.equilibrium_temp
            T_strat = self.config.equilibrium_temp * 1.2

            temps = np.where(
                altitudes <= troposphere_height,
                T_trop,
                T_trop + (T_strat - T_trop) * (altitudes - troposphere_height) / (150 - troposphere_height)
            )
            return temps

        else:
            raise ValueError(f"Unknown profile type: {profile_type}")

    def _compute_pressure_profile(
        self,
        altitudes: np.ndarray,
        temperatures: np.ndarray,
    ) -> np.ndarray:
        """
        Compute pressure as function of altitude using hydrostatic equation.

        Uses barometric formula with variable scale height:
        dP/dz = -rho * g = -P * M * g / (R * T)

        Args:
            altitudes: Altitude array in km
            temperatures: Temperature array in K

        Returns:
            Pressure array in Pa
        """
        P0 = 1e5  # Reference pressure at surface (Pa)
        g = self.config.surface_gravity  # m/s^2
        M = self.mean_molecular_mass  # kg/mol (average)
        R_gas = 8.314  # J/(mol*K)

        # Use simplified exponential model with variable scale height
        pressures = np.zeros_like(altitudes)
        pressures[0] = P0

        for i in range(1, len(altitudes)):
            dz = (altitudes[i] - altitudes[i - 1]) * 1000  # Convert to meters
            T_mean = (temperatures[i] + temperatures[i - 1]) / 2

            # Local scale height
            H_local = R_gas * T_mean / (M * g)

            # Pressure exponential decrease
            pressures[i] = pressures[i - 1] * np.exp(-dz / H_local)

        return pressures

    def _compute_density_profile(
        self,
        pressures: np.ndarray,
        temperatures: np.ndarray,
    ) -> np.ndarray:
        """
        Compute number density using ideal gas law.

        n = P / (k_B * T)

        Args:
            pressures: Pressure array in Pa
            temperatures: Temperature array in K

        Returns:
            Number density array in m^-3
        """
        k_B = 1.38e-23  # Boltzmann constant in J/K
        return pressures / (k_B * temperatures)

    def _compute_optical_depth(
        self,
        wavelength: float,
        composition: Dict[str, float],
        densities: np.ndarray,
        altitudes: np.ndarray,
        temperatures: np.ndarray,
    ) -> float:
        """
        Compute optical depth using radiative transfer equation.

        tau = integral of n_i(z) * sigma_i(lambda) dz

        Args:
            wavelength: Wavelength in micrometers
            composition: Atmospheric composition
            densities: Number density profile in m^-3
            altitudes: Altitude profile in km
            temperatures: Temperature profile in K

        Returns:
            Optical depth (dimensionless)
        """
        tau = 0.0

        # Sum contribution from each species
        for species, mixing_ratio in composition.items():
            if mixing_ratio < 1e-10:
                continue

            if species not in CROSS_SECTIONS:
                continue

            # Interpolate cross-section at this wavelength
            cross_section_data = CROSS_SECTIONS[species]
            sigma_interp = interp1d(
                cross_section_data["wavelengths"],
                cross_section_data["sigmas"],
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
            sigma = sigma_interp(wavelength)
            sigma = np.clip(sigma, 1e-25, 1e-15)  # Prevent unphysical values

            # Temperature-dependent cross-section scaling
            # Assuming cross-section scales with T^-0.5 (approximate)
            T_ref = 250  # Reference temperature
            sigma *= np.sqrt(T_ref / temperatures.mean())

            # Compute absorption from this species
            species_density = densities * mixing_ratio  # m^-3
            species_density *= 1e4  # Convert to cm^-3

            # Optical depth contribution: tau = integral(n * sigma * dz)
            integrand = species_density * sigma
            tau_species = trapezoid(integrand, altitudes * 1e5)  # Convert altitudes to cm

            tau += tau_species

        return np.clip(tau, 0, 50)  # Clip to reasonable values

    def _optical_depth_to_transit_depth(self, tau: float) -> float:
        """
        Convert optical depth to transmission spectrum transit depth.

        Uses the formula from Seager et al. (2005):
        d = (R_p + H_eff)^2 - R_p^2 / R_star^2

        where H_eff = H * sqrt(tau / (2*pi)) for optically thin limit

        Args:
            tau: Optical depth (dimensionless)

        Returns:
            Transit depth (dimensionless)
        """
        # Convert planet radius from Earth radii to meters
        R_p_m = self.config.planet_radius * 6.371e6  # meters
        R_star_m = self.config.stellar_radius * 6.96e8  # meters

        # Effective scale height contribution to transit depth
        # In the optically thick limit: H_eff = H * sqrt(tau)
        if tau > 0.1:
            H_eff = self.scale_height * 1000 * np.sqrt(tau)  # Convert to meters
        else:
            H_eff = self.scale_height * 1000 * tau / 2  # Optically thin limit

        # Transit depth formula
        transit_depth = ((R_p_m + H_eff)**2 - R_p_m**2) / (R_star_m**2)

        return np.clip(transit_depth, 0, 0.1)  # Realistic range
