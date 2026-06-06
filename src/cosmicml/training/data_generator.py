"""
Enhanced Data Generator for PINN Training

Uses Phase 2 radiative transfer physics to create realistic synthetic data
with proper temperature/pressure dependence, scattering, and other effects.

This replaces the simple synthetic data with physics-based generation.

References:
    Kitzmann et al. (2011) - Radiative transfer in exoplanet atmospheres
    Burrows et al. (1997) - Non-gray atmospheres
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path
import sys
import h5py
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from physics.radiative_transfer import EnhancedRadiativeTransfer
from atmosphere.simulator import AtmosphereSimulator, PlanetaryConfig
from atmosphere.chemistry import ChemistryEngine


class EnhancedDataGenerator:
    """
    Generate realistic synthetic spectra using physics models.

    Improvements over v1:
    - Uses Phase 2 enhanced radiative transfer
    - Temperature/pressure dependent cross-sections
    - Rayleigh scattering included
    - Realistic line broadening
    - Collision-induced absorption
    - Physically consistent compositions
    """

    def __init__(
        self,
        num_atmospheres: int = 10000,
        wavelength_range: Tuple[float, float] = (0.3, 5.0),
        n_wavelengths: int = 512,
        use_enhanced_rt: bool = True,
    ):
        """
        Initialize data generator.

        Args:
            num_atmospheres: Number of synthetic atmospheres
            wavelength_range: Wavelength range (micrometers)
            n_wavelengths: Number of wavelength points
            use_enhanced_rt: Use Phase 2 enhanced radiative transfer
        """
        self.num_atmospheres = num_atmospheres
        self.wavelength_range = wavelength_range
        self.n_wavelengths = n_wavelengths
        self.use_enhanced_rt = use_enhanced_rt

        # Initialize physics engines
        self.radiative_transfer = EnhancedRadiativeTransfer() if use_enhanced_rt else None
        self.chemistry_engine = ChemistryEngine()
        self.wavelengths = np.linspace(wavelength_range[0], wavelength_range[1], n_wavelengths)

        # Planetary parameters for diversity
        self.stellar_temps = [3500, 4500, 5778, 6500, 7500]  # K-F-G-A dwarfs
        self.planet_radii = np.linspace(0.5, 10, 20)  # Earth radii
        self.equilibrium_temps = np.linspace(100, 1500, 50)  # K

    def generate_diverse_compositions(self, num: int) -> np.ndarray:
        """
        Generate diverse atmospheric compositions.

        Uses 5 scenarios + random variations:
        1. Earth-like (N2/O2 dominated)
        2. Venus-like (CO2 dominated)
        3. H2/He dominated (mini-Neptune)
        4. O3-rich (high photochemistry)
        5. CH4-rich (biosignature scenario)
        6. Random variations

        Args:
            num: Number of compositions

        Returns:
            [num, 10] composition array
        """
        compositions = []

        # Scenario distributions
        scenarios = [
            # Earth-like
            {'N2': 0.78, 'O2': 0.21, 'Ar': 0.01},
            # Venus-like
            {'CO2': 0.96, 'N2': 0.03, 'O2': 0.0001},
            # H2/He world
            {'H2': 0.7, 'He': 0.2, 'H2O': 0.1},
            # O3-rich (photochemistry)
            {'N2': 0.5, 'O2': 0.3, 'O3': 0.15, 'H2O': 0.05},
            # CH4-rich (biosignature)
            {'N2': 0.7, 'O2': 0.15, 'CH4': 0.1, 'H2O': 0.05},
        ]

        # Generate compositions
        species_list = ['H2O', 'CO2', 'O2', 'N2', 'CH4', 'H2', 'O3', 'NH3', 'NO', 'H2S']

        for i in range(num):
            # Pick scenario
            scenario_idx = i % len(scenarios)
            scenario = scenarios[scenario_idx].copy()

            # Add variations
            for species, abundance in scenario.items():
                variation = np.random.normal(1.0, 0.2)
                scenario[species] = max(1e-6, abundance * variation)

            # Fill missing species with trace amounts
            total = sum(scenario.values())
            for species in species_list:
                if species not in scenario:
                    scenario[species] = np.random.uniform(1e-6, 1e-4)

            # Normalize
            total = sum(scenario.values())
            composition = np.array([scenario.get(sp, 1e-8) / total for sp in species_list])

            compositions.append(composition)

        return np.array(compositions)

    def generate_realistic_spectrum(
        self,
        composition: Dict[str, float],
        temperature: float,
        pressure: float,
        star_temp: float = 5778,
    ) -> np.ndarray:
        """
        Generate realistic spectrum using physics models.

        Args:
            composition: {species: mixing_ratio}
            temperature: Atmospheric temperature (K)
            pressure: Surface pressure (bar)
            star_temp: Stellar temperature (K)

        Returns:
            Normalized transit depth spectrum
        """
        if not self.use_enhanced_rt:
            # Fall back to simple model
            return self._simple_spectrum(composition, temperature)

        try:
            # Use enhanced radiative transfer
            spectrum = np.zeros(len(self.wavelengths))

            # Create density profile
            n_layers = 50
            altitudes = np.linspace(0, 150, n_layers)  # km
            temperatures = np.ones(n_layers) * temperature

            # Simple pressure profile (exponential)
            scale_height = 8.5  # km (for Earth-like)
            pressures = np.array([
                pressure * np.exp(-alt / scale_height)
                for alt in altitudes
            ])

            # Number density profile
            k_B = 1.38e-23
            densities = pressures * 1e5 / (k_B * temperatures)

            # Compute spectrum
            for i, wavelength in enumerate(self.wavelengths):
                tau_total, tau_abs, tau_scatter = self.radiative_transfer.compute_optical_depth(
                    wavelength,
                    composition,
                    densities,
                    altitudes,
                    temperatures,
                    pressures * 1e5,
                )

                # Transit depth from optical depth
                R_p = 1.0  # Earth radii
                R_star = 1.0
                H = scale_height * 1000  # meters

                if tau_total > 0.1:
                    H_eff = H * np.sqrt(tau_total)
                else:
                    H_eff = H * tau_total / 2

                R_p_m = R_p * 6.371e6
                R_star_m = R_star * 6.96e8

                transit_depth = ((R_p_m + H_eff) ** 2 - R_p_m ** 2) / (R_star_m ** 2)
                spectrum[i] = np.clip(transit_depth, 0, 0.1)

            return spectrum

        except Exception as e:
            # Fall back to simple spectrum if enhanced RT fails
            print(f"Warning: Enhanced RT failed ({e}), using simple spectrum")
            return self._simple_spectrum(composition, temperature)

    def _simple_spectrum(
        self,
        composition: Dict[str, float],
        temperature: float,
    ) -> np.ndarray:
        """
        Simple backup spectrum generation.

        Args:
            composition: {species: mixing_ratio}
            temperature: Temperature (K)

        Returns:
            Spectrum array
        """
        # Base spectrum
        spectrum = np.ones(len(self.wavelengths)) * 1e-4

        # Add absorption features
        species_features = {
            'H2O': (1.4, 0.3),  # (center μm, width μm)
            'CO2': (4.3, 0.4),
            'O2': (0.76, 0.1),
            'O3': (0.6, 0.1),
            'CH4': (2.3, 0.2),
        }

        for species, (center, width) in species_features.items():
            if species in composition and composition[species] > 1e-6:
                # Gaussian absorption feature
                feature = composition[species] * np.exp(
                    -0.5 * ((self.wavelengths - center) / width) ** 2
                )
                spectrum += feature

        # Temperature dependence: hotter → weaker features
                spectrum *= (300 / temperature) ** 0.5

        return np.clip(spectrum, 1e-6, 0.1)

    def generate_batch(
        self,
        batch_size: int,
        temperature_range: Tuple[float, float] = (200, 500),
        pressure_range: Tuple[float, float] = (0.1, 10),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a batch of synthetic data.

        Args:
            batch_size: Number of atmospheres in batch
            temperature_range: Temperature range (K)
            pressure_range: Pressure range (bar)

        Returns:
            Tuple of (spectra, compositions, temperatures, pressures)
        """
        spectra = []
        compositions = []
        temperatures = []
        pressures = []

        # Generate compositions
        comp_array = self.generate_diverse_compositions(batch_size)
        species_list = ['H2O', 'CO2', 'O2', 'N2', 'CH4', 'H2', 'O3', 'NH3', 'NO', 'H2S']

        for i in range(batch_size):
            # Temperature and pressure
            T = np.random.uniform(*temperature_range)
            P = np.random.uniform(*pressure_range)

            temperatures.append(T)
            pressures.append(P)

            # Composition
            composition = {
                sp: comp_array[i, j]
                for j, sp in enumerate(species_list)
            }

            # Spectrum
            spectrum = self.generate_realistic_spectrum(
                composition,
                T,
                P,
            )

            spectra.append(spectrum)
            compositions.append(comp_array[i])

        return (
            np.array(spectra),
            np.array(compositions),
            np.array(temperatures),
            np.array(pressures),
        )

    def generate_full_dataset(
        self,
        output_path: str,
        batch_size: int = 100,
        verbose: bool = True,
    ) -> None:
        """
        Generate complete dataset and save to HDF5.

        Args:
            output_path: HDF5 file path
            batch_size: Batch size for generation
            verbose: Show progress bar
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        num_batches = (self.num_atmospheres + batch_size - 1) // batch_size

        with h5py.File(output_path, 'w') as f:
            # Create datasets
            f.create_dataset(
                'spectra',
                shape=(self.num_atmospheres, self.n_wavelengths),
                dtype=np.float32,
            )
            f.create_dataset(
                'compositions',
                shape=(self.num_atmospheres, 10),
                dtype=np.float32,
            )
            f.create_dataset(
                'temperatures',
                shape=(self.num_atmospheres,),
                dtype=np.float32,
            )
            f.create_dataset(
                'pressures',
                shape=(self.num_atmospheres,),
                dtype=np.float32,
            )

            # Metadata
            f.attrs['wavelengths'] = self.wavelengths
            f.attrs['num_atmospheres'] = self.num_atmospheres
            f.attrs['species'] = ['H2O', 'CO2', 'O2', 'N2', 'CH4', 'H2', 'O3', 'NH3', 'NO', 'H2S']

            # Generate in batches
            iterator = tqdm(range(num_batches), disable=not verbose, desc="Generating data")

            for batch_idx in iterator:
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, self.num_atmospheres)
                current_batch_size = end_idx - start_idx

                spectra, compositions, temperatures, pressures = self.generate_batch(
                    current_batch_size
                )

                f['spectra'][start_idx:end_idx] = spectra
                f['compositions'][start_idx:end_idx] = compositions
                f['temperatures'][start_idx:end_idx] = temperatures
                f['pressures'][start_idx:end_idx] = pressures

                iterator.set_postfix({
                    'idx': end_idx,
                    'temp_mean': f"{temperatures.mean():.0f}K",
                })


class CurriculumSchedule:
    """
    Curriculum learning schedule for progressive training difficulty.

    Strategy:
    Stage 1: Easy - Large abundant species only
    Stage 2: Medium - Add medium-abundance species
    Stage 3: Hard - All species including trace
    Stage 4: Full - All species with temperature/pressure variations
    """

    def __init__(self):
        """Initialize curriculum schedule."""
        self.current_stage = 0
        self.stages = [
            {
                'name': 'Abundant Species Only',
                'species_mask': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],  # N2, O2, CO2, H2O
                'temperature_range': (288, 288),  # Fixed
                'learning_rate_scale': 1.0,
                'epochs': 10,
            },
            {
                'name': 'Add Medium Species',
                'species_mask': [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],  # +CH4, H2
                'temperature_range': (200, 400),  # Varying
                'learning_rate_scale': 0.5,
                'epochs': 10,
            },
            {
                'name': 'Add Trace Species',
                'species_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # All species
                'temperature_range': (100, 500),
                'learning_rate_scale': 0.2,
                'epochs': 10,
            },
            {
                'name': 'Full Training',
                'species_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                'temperature_range': (100, 1500),
                'learning_rate_scale': 0.1,
                'epochs': 20,
            },
        ]

    def get_current_stage(self) -> Dict:
        """Get current curriculum stage."""
        return self.stages[self.current_stage].copy()

    def advance_stage(self) -> bool:
        """
        Advance to next stage.

        Returns:
            True if advanced, False if at final stage
        """
        if self.current_stage < len(self.stages) - 1:
            self.current_stage += 1
            return True
        return False

    def apply_mask_to_composition(
        self,
        composition: torch.Tensor,
        mask: List[int],
    ) -> torch.Tensor:
        """
        Apply species mask to composition.

        Args:
            composition: [batch, 10] composition
            mask: [10] binary mask

        Returns:
            Masked composition (masked species set to near-zero)
        """
        masked = composition.clone()
        for i, include in enumerate(mask):
            if not include:
                masked[:, i] = 1e-8

        # Renormalize
        masked = masked / (masked.sum(dim=1, keepdim=True) + 1e-8)

        return masked
