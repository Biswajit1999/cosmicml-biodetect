#!/usr/bin/env python
"""
Generate synthetic exoplanet atmosphere dataset.

Creates 10,000+ realistic exoplanet atmospheres with transmission spectra
using the atmospheric simulator.

Generates diverse scenarios:
- Multiple stellar types (K, F, G, A dwarfs)
- Range of planet sizes (0.5-10 Earth radii)
- Varying atmospheric compositions (habitable to extreme)
- Different equilibrium temperatures

Output: HDF5 files with spectra and metadata

References:
    Kaltenegger, L., & Sasselov, D. (2011). Exploring the habitable zone for
    Kepler transiting planets. The Astrophysical Journal Letters, 736(2), L25.

    Kopparapu, R. K., Ramirez, R. M., Kasting, J. F., et al. (2013). Habitable
    zones around main-sequence stars: new estimates. The Astrophysical Journal, 765(2), 131.
"""

import argparse
import numpy as np
import h5py
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List
from tqdm import tqdm
from multiprocessing import Pool
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cosmicml.atmosphere.simulator import AtmosphereSimulator, PlanetaryConfig


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AtmosphereGenerator:
    """Generates diverse synthetic exoplanet atmospheres."""

    # Habitable zone boundaries (Kopparapu et al., 2013)
    STAR_PROPERTIES = {
        "K": {"Teff": 4700, "radius": 0.60},  # K dwarf
        "G": {"Teff": 5778, "radius": 1.00},  # G dwarf (Sun-like)
        "F": {"Teff": 6350, "radius": 1.25},  # F dwarf
        "A": {"Teff": 8000, "radius": 1.40},  # A dwarf
    }

    def __init__(self, seed: int = 42):
        """Initialize atmosphere generator."""
        np.random.seed(seed)
        self.seed = seed

    def generate_planetary_config(self) -> Tuple[PlanetaryConfig, Dict]:
        """
        Generate random planetary configuration.

        Returns:
            Tuple of (PlanetaryConfig, metadata_dict)
        """
        # Random stellar type
        star_type = np.random.choice(["K", "G", "F", "A"])
        star_props = self.STAR_PROPERTIES[star_type]

        # Planet size (Earth radii)
        planet_radius = 10 ** np.random.uniform(np.log10(0.5), np.log10(10))

        # Planet mass (Earth masses) - approximated from radius
        # Using M ~ R^1.5 power law
        planet_mass = planet_radius ** 1.5

        # Orbital distance (AU) - random for now
        orbital_distance = 10 ** np.random.uniform(-1, 2)  # 0.1 to 100 AU

        # Equilibrium temperature
        # T_eq = T_star * sqrt(R_star / (2 * a))
        T_eq = star_props["Teff"] * np.sqrt(star_props["radius"] / (2 * orbital_distance))
        T_eq = np.clip(T_eq, 100, 1500)  # Reasonable bounds

        # Surface gravity (m/s^2)
        # g = G * M / R^2, use simplified scaling
        R_earth = 6.371e6  # m
        M_earth = 5.972e24  # kg
        G = 6.674e-11

        planet_radius_m = planet_radius * R_earth
        planet_mass_kg = planet_mass * M_earth
        surface_gravity = G * planet_mass_kg / (planet_radius_m ** 2)

        config = PlanetaryConfig(
            planet_radius=planet_radius,
            planet_mass=planet_mass,
            star_temp=star_props["Teff"],
            orbital_period=np.sqrt(orbital_distance ** 3),  # Kepler's 3rd law (years)
            equilibrium_temp=T_eq,
            surface_gravity=surface_gravity,
            stellar_radius=star_props["radius"],
        )

        metadata = {
            "star_type": star_type,
            "star_teff": star_props["Teff"],
            "star_radius": star_props["radius"],
            "planet_radius_earth": float(planet_radius),
            "planet_mass_earth": float(planet_mass),
            "equilibrium_temp": float(T_eq),
            "orbital_distance_au": float(orbital_distance),
            "surface_gravity": float(surface_gravity),
        }

        return config, metadata

    def generate_atmospheric_composition(
        self,
        equilibrium_temp: float,
    ) -> Dict[str, float]:
        """
        Generate atmospheric composition based on equilibrium temperature.

        Simulates diverse scenarios:
        - Earth-like (O2/N2 dominated)
        - Venus-like (CO2 dominated)
        - Extreme (H2-rich, CH4-rich, etc.)

        Args:
            equilibrium_temp: Equilibrium temperature in K

        Returns:
            Dict of {species: mixing_ratio}
        """
        composition_type = np.random.choice([
            "earth-like",
            "Venus-like",
            "reducing",
            "oxygen-rich",
            "methane-rich",
        ])

        composition = {}

        if composition_type == "earth-like":
            # Earth-like: N2/O2 dominated
            composition["N2"] = 0.78 + np.random.normal(0, 0.05)
            composition["O2"] = 0.21 + np.random.normal(0, 0.05)
            composition["H2O"] = 0.01 + np.random.normal(0, 0.005)
            composition["CO2"] = 0.0004 + np.random.normal(0, 0.0002)

        elif composition_type == "Venus-like":
            # Venus-like: CO2 dominated
            composition["CO2"] = 0.96 + np.random.normal(0, 0.02)
            composition["N2"] = 0.03 + np.random.normal(0, 0.01)
            composition["H2O"] = 0.01 + np.random.normal(0, 0.005)

        elif composition_type == "reducing":
            # Reducing: H2/CH4 dominated
            composition["H2"] = 0.7 + np.random.normal(0, 0.1)
            composition["CH4"] = 0.2 + np.random.normal(0, 0.05)
            composition["H2O"] = 0.1 + np.random.normal(0, 0.05)

        elif composition_type == "oxygen-rich":
            # Oxygen-rich: O2/O3 rich
            composition["N2"] = 0.7 + np.random.normal(0, 0.1)
            composition["O2"] = 0.25 + np.random.normal(0, 0.1)
            composition["O3"] = 0.01 + np.random.normal(0, 0.005)
            composition["H2O"] = 0.04 + np.random.normal(0, 0.02)

        elif composition_type == "methane-rich":
            # Methane-rich: CH4 dominated
            composition["CH4"] = 0.5 + np.random.normal(0, 0.1)
            composition["N2"] = 0.4 + np.random.normal(0, 0.1)
            composition["H2O"] = 0.1 + np.random.normal(0, 0.05)

        # Normalize to sum to 1
        total = sum(composition.values())
        composition = {k: max(v / total, 1e-6) for k, v in composition.items()}
        total = sum(composition.values())
        composition = {k: v / total for k, v in composition.items()}

        return composition

    def generate_spectrum(
        self,
        config: PlanetaryConfig,
        composition: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate transmission spectrum for given configuration.

        Args:
            config: Planetary configuration
            composition: Atmospheric composition

        Returns:
            Tuple of (wavelengths, transit_depths, metadata)
        """
        try:
            simulator = AtmosphereSimulator(config)

            atmosphere = simulator.simulate_atmosphere(
                composition,
                n_layers=100,
                temperature_profile="stratified",
            )

            wavelengths, transit_depths, uncertainties = simulator.generate_spectrum(
                atmosphere,
                noise_level=1e-4,
                add_noise=True,
            )

            metadata = {
                "composition": composition,
                "noise_level": 1e-4,
            }

            return wavelengths, transit_depths, metadata

        except Exception as e:
            logger.error(f"Error generating spectrum: {e}")
            return None, None, None

    def generate_dataset(
        self,
        n_samples: int = 1000,
        output_file: str = "synthetic_atmospheres.h5",
    ):
        """
        Generate full synthetic dataset.

        Args:
            n_samples: Number of atmospheres to generate
            output_file: Output HDF5 filename
        """
        logger.info(f"Generating {n_samples} synthetic atmospheres...")

        # Pre-generate all configurations and compositions
        configs = []
        compositions = []
        metadata_list = []

        for _ in range(n_samples):
            config, meta = self.generate_planetary_config()
            composition = self.generate_atmospheric_composition(config.equilibrium_temp)

            configs.append(config)
            compositions.append(composition)
            metadata_list.append(meta)

        # Generate spectra (can be parallelized)
        spectra_list = []
        uncertainties_list = []
        valid_indices = []

        for i, (config, composition) in tqdm(
            enumerate(zip(configs, compositions)),
            total=n_samples,
            desc="Generating spectra"
        ):
            wavelengths, transit_depths, meta = self.generate_spectrum(config, composition)

            if transit_depths is not None:
                spectra_list.append(transit_depths)
                if len(spectra_list) == 1:
                    wavelengths_array = wavelengths
                valid_indices.append(i)

        logger.info(f"Successfully generated {len(spectra_list)} valid spectra")

        # Save to HDF5
        with h5py.File(output_file, "w") as f:
            # Spectra dataset
            f.create_dataset(
                "spectra",
                data=np.array(spectra_list),
                compression="gzip",
                compression_opts=4,
            )

            # Wavelengths
            f.create_dataset("wavelengths", data=wavelengths_array)

            # Compositions
            n_valid = len(spectra_list)
            composition_array = np.zeros((n_valid, 10))  # 10 major species

            species_list = [
                "H2O", "CO2", "O2", "N2", "CH4", "H2",
                "O3", "NH3", "NO", "H2S"
            ]

            for i, idx in enumerate(valid_indices):
                for j, species in enumerate(species_list):
                    composition_array[i, j] = compositions[idx].get(species, 0.0)

            f.create_dataset("compositions", data=composition_array)

            # Metadata
            metadata_array = np.zeros((n_valid, 6))  # 6 metadata fields
            for i, idx in enumerate(valid_indices):
                meta = metadata_list[idx]
                metadata_array[i] = [
                    meta["planet_radius_earth"],
                    meta["planet_mass_earth"],
                    meta["equilibrium_temp"],
                    meta["surface_gravity"],
                    meta["star_teff"],
                    meta["star_radius"],
                ]

            f.create_dataset("metadata", data=metadata_array)

            # Store as JSON for better readability
            metadata_json = json.dumps({
                "species_labels": species_list,
                "metadata_fields": [
                    "planet_radius_earth",
                    "planet_mass_earth",
                    "equilibrium_temp",
                    "surface_gravity",
                    "star_teff",
                    "star_radius",
                ],
                "n_samples": n_valid,
                "wavelength_range_um": [float(wavelengths_array[0]), float(wavelengths_array[-1])],
            })
            f.attrs["metadata"] = metadata_json

        logger.info(f"Dataset saved to {output_file}")
        logger.info(f"  - {len(spectra_list)} spectra")
        logger.info(f"  - Wavelength range: {wavelengths_array[0]:.2f}-{wavelengths_array[-1]:.2f} μm")
        logger.info(f"  - {len(species_list)} species tracked")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic exoplanet atmosphere dataset"
    )
    parser.add_argument(
        "--num_atmospheres",
        type=int,
        default=1000,
        help="Number of synthetic atmospheres to generate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/simulated/",
        help="Output directory for HDF5 files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate dataset
    generator = AtmosphereGenerator(seed=args.seed)
    output_file = output_dir / "synthetic_atmospheres.h5"

    generator.generate_dataset(
        n_samples=args.num_atmospheres,
        output_file=str(output_file),
    )

    logger.info("✓ Dataset generation complete!")


if __name__ == "__main__":
    main()
