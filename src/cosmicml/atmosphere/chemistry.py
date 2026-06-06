"""
Atmospheric chemistry engine for reaction kinetics and equilibrium.

Implements:
- Chemical reaction networks
- Equilibrium calculations using Gibbs minimization
- Temperature-dependent reaction rates
- Photochemistry for UV-driven reactions

References:
    Burrows, A., & Orton, G. S. (2010). Future prospects for spectroscopic studies of
    transiting exoplanet atmospheres. The Astrophysical Journal, 722(2), 1219.

    Liang, M. C., Heays, A. N., Lewis, B. R., Gibson, S. T., & Yung, Y. L. (2013).
    Wavelength-resolved photochemistry of the ozone and oxygen in the stratosphere and
    mesosphere. The Astrophysical Journal, 661(1), L73.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass
class Reaction:
    """Represents a chemical reaction."""

    reactants: Dict[str, int]  # {species: stoichiometric_coefficient}
    products: Dict[str, int]
    k_298: float  # Rate constant at 298 K (cm^3 s^-1)
    E_a: float  # Activation energy (K)
    name: str


class ChemistryEngine:
    """
    Simulates chemical reactions and equilibrium in exoplanet atmospheres.

    For biosignature detection, key reactions include:
    - O2 photochemistry
    - CH4 oxidation
    - O3 formation/destruction
    - NOx chemistry
    """

    def __init__(self):
        """Initialize chemistry engine with reaction network."""
        self.reactions = self._setup_reaction_network()
        self.gibbs_energies = self._setup_gibbs_energies()

    def _setup_reaction_network(self) -> List[Reaction]:
        """
        Setup the atmospheric reaction network for O2/O3/CH4 chemistry.

        Based on networks from Burrows & Orton (2010).

        Returns:
            List of Reaction objects
        """
        reactions = [
            # O2/O3 chemistry (Chapman mechanism)
            Reaction(
                reactants={"O2": 1},
                products={"O": 2},
                k_298=1e-12,
                E_a=25000,
                name="O2 photodissociation",
            ),
            Reaction(
                reactants={"O": 1, "O2": 1},
                products={"O3": 1},
                k_298=6e-34,
                E_a=0,
                name="O3 formation",
            ),
            Reaction(
                reactants={"O3": 1, "O": 1},
                products={"O2": 2},
                k_298=8e-12,
                E_a=2060,
                name="O3 destruction",
            ),
            # CH4 chemistry
            Reaction(
                reactants={"CH4": 1, "OH": 1},
                products={"CH3": 1, "H2O": 1},
                k_298=2.45e-12,
                E_a=1775,
                name="CH4 oxidation",
            ),
            Reaction(
                reactants={"CH3": 1, "O2": 1},
                products={"CH2O": 1, "HO2": 1},
                k_298=2.3e-12,
                E_a=0,
                name="CH3 oxidation",
            ),
            # OH chemistry
            Reaction(
                reactants={"H": 1, "O2": 1},
                products={"OH": 1, "O": 1},
                k_298=1.4e-10,
                E_a=0,
                name="OH formation",
            ),
        ]
        return reactions

    def _setup_gibbs_energies(self) -> Dict[str, Tuple[float, float]]:
        """
        Setup standard Gibbs free energies of formation.

        Data from NIST database (J/mol at 298 K).
        Tuple format: (H_f, S_f) - enthalpy and entropy of formation

        Returns:
            Dict mapping species to (enthalpy, entropy) tuples
        """
        return {
            "O2": (-0.0, 205.0),  # Reference state
            "H2O": (-241.8e3, 188.7),
            "CO2": (-393.5e3, 213.7),
            "O3": (142.7e3, 238.9),
            "CH4": (-74.8e3, 186.2),
            "N2": (0.0, 191.6),  # Reference state
            "NO": (90.25e3, 210.7),
            "NO2": (33.18e3, 240.0),
            "OH": (39.35e3, 183.7),
            "O": (249.2e3, 161.0),
            "H": (218.0e3, 114.6),
        }

    def compute_rate_constant(
        self,
        reaction: Reaction,
        temperature: float,
    ) -> float:
        """
        Compute temperature-dependent rate constant using Arrhenius equation.

        k(T) = k_298 * (T/298)^n * exp(E_a/k_B * (1/298 - 1/T))

        Args:
            reaction: Reaction object
            temperature: Temperature in Kelvin

        Returns:
            Rate constant in cm^3 s^-1
        """
        k_B = 1.38e-23  # J/K

        # Simple Arrhenius form (ignoring temperature exponent for simplicity)
        exponent = reaction.E_a / (k_B * 8.314) * (1/298 - 1/temperature)
        exponent = np.clip(exponent, -100, 100)  # Prevent overflow

        k_T = reaction.k_298 * np.exp(exponent)

        return max(k_T, 1e-30)  # Prevent negative rates

    def compute_reaction_rates(
        self,
        composition: Dict[str, float],
        temperature: float,
    ) -> Dict[str, float]:
        """
        Compute reaction rates and net production rates for each species.

        Args:
            composition: Atmospheric composition (number densities in cm^-3)
            temperature: Temperature in Kelvin

        Returns:
            Dict of {species: d[species]/dt} (cm^-3 s^-1)
        """
        rates = {species: 0.0 for species in composition}

        for reaction in self.reactions:
            k = self.compute_rate_constant(reaction, temperature)

            # Compute reaction rate: rate = k * product(n_i^nu_i)
            reaction_rate = k
            for reactant, stoich in reaction.reactants.items():
                if reactant not in composition:
                    continue
                reaction_rate *= composition[reactant] ** stoich

            # Add/subtract contributions to each species
            for reactant, stoich in reaction.reactants.items():
                if reactant in rates:
                    rates[reactant] -= stoich * reaction_rate

            for product, stoich in reaction.products.items():
                if product in rates:
                    rates[product] += stoich * reaction_rate

        return rates

    def compute_equilibrium(
        self,
        composition: Dict[str, float],
        temperature: float,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> Dict[str, float]:
        """
        Compute chemical equilibrium composition using Gibbs minimization.

        Minimizes Gibbs free energy: G = sum_i(mu_i * n_i)

        where mu_i is chemical potential of species i.

        Args:
            composition: Initial atmospheric composition
            temperature: Temperature in Kelvin
            max_iterations: Maximum iterations for minimization
            tolerance: Convergence tolerance

        Returns:
            Equilibrium composition
        """
        # Normalize initial composition
        total = sum(composition.values())
        x0 = np.array([composition.get(species, 1e-10) / total
                       for species in sorted(composition.keys())])
        x0 = np.maximum(x0, 1e-10)  # Prevent zero abundances

        species_list = sorted(composition.keys())

        def gibbs_energy_func(x):
            """Objective: Gibbs free energy (to minimize)."""
            x = np.maximum(x, 1e-15)  # Prevent negative values
            energy = 0.0

            for i, species in enumerate(species_list):
                if species not in self.gibbs_energies:
                    continue

                h_f, s_f = self.gibbs_energies[species]
                mu_i = h_f - temperature * s_f

                energy += mu_i * x[i]

            return energy

        def constraint_sum(x):
            """Constraint: compositions must sum to 1."""
            return np.sum(x) - 1.0

        # Minimize Gibbs energy
        result = minimize(
            gibbs_energy_func,
            x0,
            method="SLSQP",
            bounds=[(1e-10, 1.0) for _ in x0],
            constraints={"type": "eq", "fun": constraint_sum},
            options={"maxiter": max_iterations, "ftol": tolerance},
        )

        # Normalize result
        x_eq = np.maximum(result.x, 1e-15)
        x_eq /= np.sum(x_eq)

        # Convert back to dictionary
        equilibrium = {
            species: x_eq[i]
            for i, species in enumerate(species_list)
        }

        return equilibrium

    def evolve_composition(
        self,
        composition: Dict[str, float],
        temperature: float,
        time_step: float = 1.0,
        n_steps: int = 100,
    ) -> List[Dict[str, float]]:
        """
        Evolve composition forward in time using reaction rates.

        Uses simple Euler method for time integration.

        Args:
            composition: Initial composition
            temperature: Temperature in Kelvin
            time_step: Time step in seconds
            n_steps: Number of integration steps

        Returns:
            List of composition dicts at each time step
        """
        compositions = [composition.copy()]
        current = composition.copy()

        for _ in range(n_steps):
            rates = self.compute_reaction_rates(current, temperature)

            # Euler integration: x_{n+1} = x_n + f(x_n) * dt
            for species in current:
                current[species] += rates.get(species, 0.0) * time_step
                current[species] = max(current[species], 1e-15)

            compositions.append(current.copy())

        return compositions
