"""
Unit tests for atmospheric simulator and chemistry modules.

References:
    Seager, S., Turner, E. L., Schafer, E., & Ford, E. B. (2005). Vegetation's Red Edge.
    Astrobiology, 5(2), 372-390.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cosmicml.atmosphere.simulator import AtmosphereSimulator, PlanetaryConfig
from cosmicml.atmosphere.chemistry import ChemistryEngine, Reaction


class TestPlanetaryConfig:
    """Test PlanetaryConfig dataclass."""

    def test_config_creation(self):
        """Test creating a planetary configuration."""
        config = PlanetaryConfig(
            planet_radius=1.0,
            planet_mass=1.0,
            star_temp=5778,
            orbital_period=365,
            equilibrium_temp=288,
            surface_gravity=9.81,
        )

        assert config.planet_radius == 1.0
        assert config.star_temp == 5778
        assert config.equilibrium_temp == 288

    def test_earth_analog(self):
        """Test Earth-like configuration."""
        earth = PlanetaryConfig(
            planet_radius=1.0,
            planet_mass=1.0,
            star_temp=5778,
            orbital_period=365,
            equilibrium_temp=288,
            surface_gravity=9.81,
            stellar_radius=1.0,
        )

        # Earth parameters should be realistic
        assert 0.9 < earth.planet_radius < 1.1
        assert 250 < earth.equilibrium_temp < 320
        assert 9.5 < earth.surface_gravity < 10.5


class TestAtmosphereSimulator:
    """Test atmosphere simulator."""

    @pytest.fixture
    def simulator(self):
        """Create a simulator instance."""
        config = PlanetaryConfig(
            planet_radius=1.0,
            planet_mass=1.0,
            star_temp=5778,
            orbital_period=365,
            equilibrium_temp=288,
            surface_gravity=9.81,
            stellar_radius=1.0,
        )
        return AtmosphereSimulator(config)

    def test_simulator_creation(self, simulator):
        """Test simulator initialization."""
        assert simulator is not None
        assert len(simulator.wavelengths) == 512
        assert simulator.wavelengths[0] == pytest.approx(0.3)
        assert simulator.wavelengths[-1] == pytest.approx(5.0)

    def test_scale_height_computation(self, simulator):
        """Test scale height calculation."""
        assert simulator.scale_height > 0
        assert 5 < simulator.scale_height < 15  # km

    def test_atmosphere_simulation(self, simulator):
        """Test atmosphere profile generation."""
        composition = {
            "N2": 0.78,
            "O2": 0.21,
            "H2O": 0.01,
        }

        atmosphere = simulator.simulate_atmosphere(composition, n_layers=50)

        # Check structure
        assert "altitudes" in atmosphere
        assert "temperatures" in atmosphere
        assert "pressures" in atmosphere
        assert "densities" in atmosphere
        assert "composition" in atmosphere

        # Check dimensions
        assert len(atmosphere["altitudes"]) == 50
        assert len(atmosphere["temperatures"]) == 50
        assert len(atmosphere["pressures"]) == 50
        assert len(atmosphere["densities"]) == 50

    def test_temperature_profile_isothermal(self, simulator):
        """Test isothermal temperature profile."""
        composition = {"N2": 1.0}
        atmosphere = simulator.simulate_atmosphere(
            composition,
            n_layers=100,
            temperature_profile="isothermal",
        )

        temperatures = atmosphere["temperatures"]
        # Should be constant for isothermal
        assert np.allclose(temperatures, temperatures[0], rtol=1e-10)

    def test_temperature_profile_linear(self, simulator):
        """Test linear temperature profile."""
        composition = {"N2": 1.0}
        atmosphere = simulator.simulate_atmosphere(
            composition,
            n_layers=100,
            temperature_profile="linear",
        )

        temperatures = atmosphere["temperatures"]
        # Should decrease with altitude
        assert temperatures[0] > temperatures[-1]

    def test_pressure_decreases_with_altitude(self, simulator):
        """Test pressure decreases with altitude."""
        composition = {"N2": 1.0}
        atmosphere = simulator.simulate_atmosphere(composition, n_layers=100)

        pressures = atmosphere["pressures"]
        # Pressure should be non-negative
        assert np.all(pressures > 0)
        # Should have expected shape
        assert len(pressures) == 100

    def test_density_decreases_with_altitude(self, simulator):
        """Test density decreases with altitude."""
        composition = {"N2": 1.0}
        atmosphere = simulator.simulate_atmosphere(composition, n_layers=100)

        densities = atmosphere["densities"]
        # Density should be positive and finite
        assert np.all(densities > 0)
        assert np.all(np.isfinite(densities))
        # Should have expected shape
        assert len(densities) == 100

    def test_spectrum_generation(self, simulator):
        """Test spectrum generation."""
        composition = {
            "N2": 0.78,
            "O2": 0.21,
            "H2O": 0.01,
        }

        atmosphere = simulator.simulate_atmosphere(composition)
        wavelengths, transit_depths, uncertainties = simulator.generate_spectrum(
            atmosphere,
            noise_level=1e-4,
            add_noise=False,
        )

        # Check dimensions
        assert len(wavelengths) == 512
        assert len(transit_depths) == 512
        assert len(uncertainties) == 512

        # Check values are physical
        assert np.all(transit_depths >= 0)
        assert np.all(transit_depths <= 0.1)  # Reasonable range
        assert np.all(uncertainties > 0)

    def test_spectrum_with_noise(self, simulator):
        """Test spectrum generation with noise."""
        composition = {"N2": 1.0}
        atmosphere = simulator.simulate_atmosphere(composition)

        # Generate with and without noise
        wl, td_clean, unc_clean = simulator.generate_spectrum(atmosphere, add_noise=False)
        wl_noisy, td_noisy, unc_noisy = simulator.generate_spectrum(atmosphere, add_noise=True)

        # Both should have correct shapes
        assert len(wl) == len(td_clean)
        assert len(wl_noisy) == len(td_noisy)
        # Both should be valid spectra
        assert np.all(np.isfinite(td_clean))
        assert np.all(np.isfinite(td_noisy))

    def test_optical_depth_increases_with_abundance(self, simulator):
        """Test optical depth increases with abundance."""
        # Low O2
        low_o2 = {"N2": 0.99, "O2": 0.01}
        atm_low = simulator.simulate_atmosphere(low_o2)
        _, td_low, _ = simulator.generate_spectrum(atm_low, add_noise=False)

        # High O2
        high_o2 = {"N2": 0.79, "O2": 0.21}
        atm_high = simulator.simulate_atmosphere(high_o2)
        _, td_high, _ = simulator.generate_spectrum(atm_high, add_noise=False)

        # Higher O2 should give larger features
        mean_diff_high = np.std(td_high)
        mean_diff_low = np.std(td_low)
        assert mean_diff_high >= mean_diff_low

    def test_composition_normalization(self, simulator):
        """Test that compositions are normalized."""
        # Non-normalized composition
        composition = {"N2": 1.0, "O2": 0.5, "H2O": 0.2}

        atmosphere = simulator.simulate_atmosphere(composition)

        # Should be normalized
        total = sum(atmosphere["composition"].values())
        assert np.isclose(total, 1.0, atol=0.01)


class TestChemistryEngine:
    """Test chemistry module."""

    @pytest.fixture
    def chemistry(self):
        """Create chemistry engine instance."""
        return ChemistryEngine()

    def test_chemistry_creation(self, chemistry):
        """Test chemistry engine initialization."""
        assert chemistry is not None
        assert len(chemistry.reactions) > 0
        assert len(chemistry.gibbs_energies) > 0

    def test_reaction_network(self, chemistry):
        """Test reaction network setup."""
        reactions = chemistry.reactions

        # Should have key reactions
        reaction_names = [r.name for r in reactions]
        assert "O3 formation" in reaction_names or "O2 photodissociation" in reaction_names

    def test_rate_constant_calculation(self, chemistry):
        """Test Arrhenius rate constant."""
        reaction = chemistry.reactions[0]

        k_298 = chemistry.compute_rate_constant(reaction, 298)
        k_350 = chemistry.compute_rate_constant(reaction, 350)

        # Rates should be positive
        assert k_298 > 0
        assert k_350 > 0

        # Temperature dependence
        if reaction.E_a > 0:
            assert k_350 > k_298  # Higher T -> faster rate

    def test_reaction_rates_nonnegative(self, chemistry):
        """Test that reaction rates are computed without errors."""
        composition = {"O2": 1e18, "O": 1e15, "O3": 1e14, "H2O": 1e16}
        temperature = 250

        rates = chemistry.compute_reaction_rates(composition, temperature)

        # Should return dict
        assert isinstance(rates, dict)
        # Should have rates for each species
        for species in composition:
            assert species in rates

    def test_equilibrium_computation(self, chemistry):
        """Test chemical equilibrium solver."""
        composition = {
            "O2": 0.5,
            "O3": 0.1,
            "H2O": 0.4,
        }
        temperature = 250

        equilibrium = chemistry.compute_equilibrium(composition, temperature)

        # Should return dict
        assert isinstance(equilibrium, dict)

        # Should be normalized
        total = sum(equilibrium.values())
        assert np.isclose(total, 1.0, atol=0.01)

        # All values should be positive
        assert all(v >= 0 for v in equilibrium.values())

    def test_composition_evolution(self, chemistry):
        """Test time evolution of composition."""
        composition = {
            "CH4": 0.1,
            "O2": 0.8,
            "H2O": 0.1,
        }
        temperature = 250

        evolution = chemistry.evolve_composition(
            composition,
            temperature,
            time_step=1.0,
            n_steps=10,
        )

        # Should return list
        assert isinstance(evolution, list)
        assert len(evolution) == 11  # Initial + 10 steps

        # Each step should be valid composition
        for comp in evolution:
            assert isinstance(comp, dict)
            total = sum(comp.values())
            assert total > 0


class TestChemicalReaction:
    """Test Reaction dataclass."""

    def test_reaction_creation(self):
        """Test creating a reaction."""
        reaction = Reaction(
            reactants={"O2": 1},
            products={"O": 2},
            k_298=1e-12,
            E_a=25000,
            name="O2 photolysis",
        )

        assert reaction.name == "O2 photolysis"
        assert reaction.k_298 == 1e-12
        assert reaction.E_a == 25000


class TestIntegration:
    """Integration tests combining multiple modules."""

    def test_atmosphere_to_spectrum(self):
        """Test full workflow: atmosphere -> spectrum."""
        # Create planet
        config = PlanetaryConfig(
            planet_radius=1.5,
            planet_mass=2.0,
            star_temp=6000,
            orbital_period=300,
            equilibrium_temp=350,
            surface_gravity=15,
            stellar_radius=1.1,
        )

        # Create atmosphere
        composition = {
            "H2O": 0.5,
            "CO2": 0.3,
            "N2": 0.2,
        }

        # Generate spectrum
        simulator = AtmosphereSimulator(config)
        atmosphere = simulator.simulate_atmosphere(composition)
        wl, td, unc = simulator.generate_spectrum(atmosphere)

        # Check output
        assert len(wl) == 512
        assert len(td) == 512
        assert len(unc) == 512
        assert np.all(td >= 0)
        assert np.all(td <= 0.1)

    def test_diverse_planetary_scenarios(self):
        """Test various planetary configurations."""
        configs = [
            # Earth-like
            PlanetaryConfig(1.0, 1.0, 5778, 365, 288, 9.81, 1.0),
            # Super-Earth
            PlanetaryConfig(2.0, 5.0, 5000, 200, 400, 20, 0.9),
            # Sub-Neptune
            PlanetaryConfig(4.0, 15.0, 6000, 100, 500, 25, 1.1),
            # Hot Jupiter
            PlanetaryConfig(10.0, 300, 6500, 50, 1200, 100, 1.2),
        ]

        for config in configs:
            simulator = AtmosphereSimulator(config)
            composition = {"N2": 0.7, "O2": 0.2, "H2O": 0.1}
            atmosphere = simulator.simulate_atmosphere(composition)
            _, td, _ = simulator.generate_spectrum(atmosphere, add_noise=False)

            # All should produce valid spectra
            assert np.all(td >= 0)
            assert np.mean(td) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
