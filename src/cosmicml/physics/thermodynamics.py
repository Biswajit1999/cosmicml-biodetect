"""
Thermodynamic calculations for atmospheric chemistry.

Implements:
- Gibbs free energy minimization
- Temperature-dependent thermodynamic properties
- Chemical equilibrium constants
- Constraint enforcement for physical systems

References:
    Atkins, P. W., & De Paula, J. (2019). Physical Chemistry (11th ed.).
    NIST Chemistry WebBook. https://webbook.nist.gov/
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


class ThermodynamicDatabase:
    """
    Standard thermodynamic properties for atmospheric species.

    Data from NIST at T=298.15K: (ΔH_f°, ΔS°, C_p)
    Units: J/mol for enthalpy/Gibbs, J/(mol·K) for entropy/heat capacity

    References:
        NIST Standard Reference Database (https://webbook.nist.gov/)
    """

    # Standard molar enthalpy of formation (J/mol) at 298.15 K
    # Entropy of formation (J/mol·K) at 298.15 K
    # Heat capacity (J/mol·K) at 298.15 K
    SPECIES_DATA = {
        "H2O": {
            "H_f": -241820.0,      # Formation enthalpy
            "S_f": 188.7,          # Absolute entropy
            "C_p": 33.6,           # Heat capacity (gas)
            "molecular_weight": 18.01,
        },
        "O2": {
            "H_f": 0.0,            # Reference state
            "S_f": 205.2,
            "C_p": 29.4,
            "molecular_weight": 32.0,
        },
        "CO2": {
            "H_f": -393510.0,
            "S_f": 213.7,
            "C_p": 37.1,
            "molecular_weight": 44.01,
        },
        "O3": {
            "H_f": 142700.0,       # Endothermic formation!
            "S_f": 238.9,
            "C_p": 39.2,
            "molecular_weight": 48.0,
        },
        "CH4": {
            "H_f": -74850.0,
            "S_f": 186.2,
            "C_p": 35.3,
            "molecular_weight": 16.04,
        },
        "N2": {
            "H_f": 0.0,            # Reference state
            "S_f": 191.6,
            "C_p": 29.1,
            "molecular_weight": 28.01,
        },
        "H2": {
            "H_f": 0.0,            # Reference state
            "S_f": 130.7,
            "C_p": 28.8,
            "molecular_weight": 2.016,
        },
        "NO": {
            "H_f": 90250.0,
            "S_f": 210.7,
            "C_p": 29.9,
            "molecular_weight": 30.0,
        },
        "NH3": {
            "H_f": -45900.0,
            "S_f": 192.8,
            "C_p": 35.1,
            "molecular_weight": 17.03,
        },
        "H2S": {
            "H_f": -20630.0,
            "S_f": 205.8,
            "C_p": 34.2,
            "molecular_weight": 34.08,
        },
    }

    # Heat capacity temperature dependence: C_p(T) = C_p° + a·T + b·T²
    # Empirical NASA polynomials
    HEAT_CAPACITY_COEFFICIENTS = {
        "H2O": {"a": 0.003, "b": 1e-6},
        "O2": {"a": 0.0025, "b": 5e-7},
        "CO2": {"a": 0.0044, "b": 1.5e-6},
        "O3": {"a": 0.0035, "b": 8e-7},
        "CH4": {"a": 0.0034, "b": 1e-6},
        "N2": {"a": 0.0026, "b": 5e-7},
    }


class GibbsFreeEnergyCalculator(nn.Module):
    """
    Calculate Gibbs free energy for atmospheric compositions.

    Theory:
        G = H - TS = Σᵢ nᵢ(μᵢ° + RT ln(xᵢ))
        ΔG° = ΔH° - TΔS°

    Used for:
    - Predicting equilibrium compositions
    - Creating physics loss for PINN
    - Checking if predicted compositions are plausible
    """

    def __init__(self):
        """Initialize thermodynamic database."""
        super().__init__()
        self.db = ThermodynamicDatabase()
        self.R = 8.314  # Universal gas constant (J/(mol·K))

        # Register species names
        self.species_list = list(self.db.SPECIES_DATA.keys())
        self.n_species = len(self.species_list)

    def compute_gibbs_energy(
        self,
        composition: torch.Tensor,
        temperature: float = 288.0,
        pressure: float = 1e5,
    ) -> torch.Tensor:
        """
        Compute molar Gibbs free energy for a composition.

        Formula:
            G_m = Σᵢ xᵢ[ΔG°ᵢ(T) + RT ln(xᵢ)]

        where:
            ΔG°(T) = ΔH° - TΔS° + ∫Cₚ dT

        Args:
            composition: [batch, n_species] normalized mole fractions (sum=1)
            temperature: Temperature in Kelvin
            pressure: Pressure in Pa (for fugacity, not used in ideal gas limit)

        Returns:
            Gibbs energy [batch] in J/mol
        """
        batch_size = composition.shape[0]
        device = composition.device

        # Compute standard Gibbs energy at temperature T
        # ΔG° = ΔH° - TΔS°
        gibbs_energies = []

        for i, species in enumerate(self.species_list):
            data = self.db.SPECIES_DATA[species]
            H_f = data["H_f"]  # J/mol
            S_f = data["S_f"]  # J/(mol·K)

            # Account for temperature dependence
            # C_p(T) = C_p° + a·T + b·T²
            coeff = self.db.HEAT_CAPACITY_COEFFICIENTS.get(species, {"a": 0, "b": 0})

            # ΔH(T) ≈ H_f + C_p°(T-298)
            # ΔS(T) ≈ S_f + C_p ln(T/298)
            delta_T = temperature - 298.15
            H_T = H_f + data["C_p"] * delta_T  # Simplified
            S_T = S_f + data["C_p"] * np.log(temperature / 298.15)

            # ΔG° = ΔH° - TΔS°
            G_0 = H_T - temperature * S_T  # J/mol
            gibbs_energies.append(G_0)

        gibbs_energies = torch.tensor(gibbs_energies, dtype=composition.dtype, device=device)

        # Add ideal solution term: RT ln(xᵢ)
        # Avoid log(0) by clipping small values
        x_clipped = torch.clamp(composition, min=1e-10)
        ideal_term = self.R * temperature * torch.log(x_clipped)

        # Total molar Gibbs energy
        G_m_total = torch.sum(composition * (gibbs_energies + ideal_term), dim=1)

        return G_m_total

    def compute_equilibrium_constant(
        self,
        reactants: Dict[str, int],
        products: Dict[str, int],
        temperature: float = 288.0,
    ) -> float:
        """
        Compute equilibrium constant K_eq from standard Gibbs energies.

        Formula:
            K_eq = exp(-ΔG°_rxn / RT)

        where:
            ΔG°_rxn = Σ νᵢ ΔG°ᵢ(products) - Σ νⱼ ΔG°ⱼ(reactants)

        Args:
            reactants: Dict {species: stoichiometric_coefficient}
            products: Dict {species: stoichiometric_coefficient}
            temperature: Temperature in Kelvin

        Returns:
            Equilibrium constant K_eq (dimensionless for this reaction)
        """
        R = self.R
        delta_G_rxn = 0.0

        # Products contribution (positive)
        for species, coeff in products.items():
            if species in self.db.SPECIES_DATA:
                data = self.db.SPECIES_DATA[species]
                H_f = data["H_f"]
                S_f = data["S_f"]
                delta_T = temperature - 298.15
                H_T = H_f + data["C_p"] * delta_T
                S_T = S_f + data["C_p"] * np.log(temperature / 298.15)
                G_f = H_T - temperature * S_T
                delta_G_rxn += coeff * G_f

        # Reactants contribution (negative)
        for species, coeff in reactants.items():
            if species in self.db.SPECIES_DATA:
                data = self.db.SPECIES_DATA[species]
                H_f = data["H_f"]
                S_f = data["S_f"]
                delta_T = temperature - 298.15
                H_T = H_f + data["C_p"] * delta_T
                S_T = S_f + data["C_p"] * np.log(temperature / 298.15)
                G_f = H_T - temperature * S_T
                delta_G_rxn -= coeff * G_f

        # K_eq = exp(-ΔG°/RT)
        K_eq = np.exp(-delta_G_rxn / (R * temperature))

        return K_eq

    def gibbs_minimization_loss(
        self,
        composition: torch.Tensor,
        temperature: float = 288.0,
    ) -> torch.Tensor:
        """
        Compute loss that penalizes high Gibbs energy compositions.

        Lower Gibbs energy = more stable/likely composition.

        This is the physics constraint: the model should predict compositions
        that minimize Gibbs free energy at the given temperature.

        Args:
            composition: [batch, n_species] predicted composition
            temperature: Temperature in Kelvin

        Returns:
            Loss tensor [batch]
        """
        # Compute Gibbs energy for each composition
        G_m = self.compute_gibbs_energy(composition, temperature)

        # Penalize high energy states
        # Normalize by reference energy to make dimensionless
        G_ref = 100000  # Reference scale in J/mol
        normalized_G = G_m / G_ref

        # Loss: compositions with high G should be penalized
        # Using ReLU so only high-energy states are penalized
        loss = torch.nn.functional.relu(normalized_G - 0.5)  # Offset to allow some range

        return loss


class ChemicalConstraint(nn.Module):
    """
    Enforce that predicted compositions satisfy chemical equilibrium.

    Theory:
        For reaction: aA + bB ⇌ cC + dD
        K_eq = [C]^c [D]^d / [A]^a [B]^b
        Q = product of (mole fraction)^coefficient

        At equilibrium: Q = K_eq
        Out of equilibrium: |log(Q) - log(K_eq)| is loss
    """

    def __init__(self):
        """Initialize with common exoplanet chemistry reactions."""
        super().__init__()
        self.gibbs_calc = GibbsFreeEnergyCalculator()

        # Define key reactions for exoplanet atmospheres
        self.reactions = [
            # O2/O3 Chapman cycle
            {
                "name": "O3 formation",
                "reactants": {"O": 1, "O2": 1},
                "products": {"O3": 1},
            },
            {
                "name": "O3 destruction",
                "reactants": {"O3": 1, "O": 1},
                "products": {"O2": 2},
            },
            # H2O dissociation
            {
                "name": "H2O dissociation",
                "reactants": {"H2O": 1},
                "products": {"H2": 1, "O": 0.5},  # Note: O as radical
            },
            # CH4 oxidation
            {
                "name": "CH4 oxidation",
                "reactants": {"CH4": 1, "O2": 2},
                "products": {"CO2": 1, "H2O": 2},
            },
        ]

    def compute_reaction_quotient(
        self,
        composition: torch.Tensor,
        reaction: Dict,
    ) -> torch.Tensor:
        """
        Compute reaction quotient Q for a reaction and composition.

        Q = Π (xᵢ)^νᵢ where ν is stoichiometric coefficient

        Args:
            composition: [batch, n_species] mole fractions
            reaction: Dict with reactants and products

        Returns:
            Q value [batch]
        """
        batch_size = composition.shape[0]
        device = composition.device

        Q = torch.ones(batch_size, device=device)

        # Species list matches gibbs_calc
        species_list = self.gibbs_calc.species_list
        species_to_idx = {s: i for i, s in enumerate(species_list)}

        # Products (numerator)
        for species, coeff in reaction["products"].items():
            if species in species_to_idx:
                idx = species_to_idx[species]
                Q = Q * torch.pow(composition[:, idx], coeff)

        # Reactants (denominator)
        for species, coeff in reaction["reactants"].items():
            if species in species_to_idx:
                idx = species_to_idx[species]
                Q = Q / torch.pow(composition[:, idx], coeff)

        return Q

    def equilibrium_loss(
        self,
        composition: torch.Tensor,
        temperature: float = 288.0,
        weight: float = 0.1,
    ) -> torch.Tensor:
        """
        Loss for chemical equilibrium constraints.

        For each reaction, compute K_eq(T) and check if Q ≈ K_eq.

        Args:
            composition: [batch, n_species] predicted composition
            temperature: Temperature in Kelvin
            weight: Weight for this loss component

        Returns:
            Loss tensor [batch]
        """
        batch_size = composition.shape[0]
        device = composition.device

        total_loss = torch.zeros(batch_size, device=device)

        for reaction in self.reactions:
            # Compute K_eq
            K_eq = self.gibbs_calc.compute_equilibrium_constant(
                reaction["reactants"],
                reaction["products"],
                temperature,
            )
            K_eq_tensor = torch.tensor(K_eq, dtype=composition.dtype, device=device)

            # Compute Q for this composition
            Q = self.compute_reaction_quotient(composition, reaction)
            Q = torch.clamp(Q, min=1e-10)  # Avoid log(0)

            # Loss: |log(Q) - log(K_eq)|
            # Compositions not at equilibrium incur loss
            equilibrium_error = torch.abs(torch.log(Q) - torch.log(K_eq_tensor + 1e-10))

            # Only penalize if significantly out of equilibrium
            total_loss += torch.nn.functional.relu(equilibrium_error - 0.5)

        return weight * total_loss / len(self.reactions)
