"""
High-level inference API for PINN v3 model.

Provides simple interfaces for:
- Loading trained models
- Making predictions on spectra
- Uncertainty estimation
- Batch processing
- Result visualization

References:
    Model architecture: Phase 3 (PINN v3)
    Training pipeline: Phase 4
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List
from dataclasses import dataclass
import json


@dataclass
class PredictionResult:
    """Container for model prediction results."""

    composition: np.ndarray
    """[10] species abundances (sum=1)"""

    temperature: float
    """Atmospheric temperature (K)"""

    pressure: Optional[float] = None
    """Surface pressure (bar)"""

    uncertainty: Optional[np.ndarray] = None
    """[10] uncertainty bounds per species"""

    aleatoric_unc: Optional[np.ndarray] = None
    """[10] data noise uncertainty"""

    epistemic_unc: Optional[np.ndarray] = None
    """[10] model uncertainty"""

    attention_weights: Optional[Dict[str, np.ndarray]] = None
    """Attention head weights for interpretability"""

    def __str__(self) -> str:
        """String representation."""
        species = ['H2O', 'CO2', 'O2', 'N2', 'CH4', 'H2', 'O3', 'NH3', 'NO', 'H2S']

        lines = [
            "=" * 60,
            "SPECTRUM ANALYSIS RESULTS",
            "=" * 60,
            "",
            f"Temperature: {self.temperature:.1f} K",
        ]

        if self.pressure is not None:
            lines.append(f"Pressure: {self.pressure:.2f} bar")

        lines.extend([
            "",
            "Composition:",
            "-" * 60,
        ])

        # Sort by abundance
        indices = np.argsort(self.composition)[::-1]
        for idx in indices:
            abundance = self.composition[idx]
            spec_name = species[idx]
            unc_str = ""
            if self.uncertainty is not None:
                unc_str = f" ± {self.uncertainty[idx]:.1e}"

            lines.append(f"  {spec_name:6s}: {abundance:.3e}{unc_str}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'composition': self.composition.tolist(),
            'temperature': float(self.temperature),
            'pressure': float(self.pressure) if self.pressure else None,
            'uncertainty': self.uncertainty.tolist() if self.uncertainty is not None else None,
        }

    def save_json(self, filepath: str) -> None:
        """Save results to JSON file."""
        data = self.to_dict()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


class SpectrumPredictor:
    """
    High-level API for spectrum analysis with PINN v3.

    Usage:
        predictor = SpectrumPredictor('models/best_model.pt')
        spectrum = load_spectrum('data/transit.fits')
        result = predictor.predict(spectrum)
        print(result)
    """

    def __init__(
        self,
        model_path: str,
        device: str = 'cpu',
        mc_samples: int = 50,
    ):
        """
        Initialize predictor.

        Args:
            model_path: Path to trained PINN v3 checkpoint
            device: 'cpu' or 'cuda'
            mc_samples: Monte Carlo samples for uncertainty
        """
        self.device = device
        self.mc_samples = mc_samples

        # Load model
        self.model = self._load_model(model_path)
        self.model.to(device)
        self.model.eval()

        # Species list
        self.species = ['H2O', 'CO2', 'O2', 'N2', 'CH4', 'H2', 'O3', 'NH3', 'NO', 'H2S']

    def _load_model(self, model_path: str):
        """Load PINN v3 model from checkpoint."""
        from cosmicml.models.pinn_v3 import PINNv3

        checkpoint_path = Path(model_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Create model
        model = PINNv3(
            input_dim=512,
            output_dim=10,
            num_attention_heads=4,
            num_mc_samples=10,
        )

        # Load weights
        if 'model_state' in checkpoint:
            model.load_state_dict(checkpoint['model_state'])
        else:
            model.load_state_dict(checkpoint)

        return model

    def predict(
        self,
        spectrum: Union[np.ndarray, torch.Tensor],
        return_attention: bool = True,
    ) -> PredictionResult:
        """
        Predict atmospheric composition from spectrum.

        Args:
            spectrum: [512] normalized transit spectrum
            return_attention: Whether to return attention weights

        Returns:
            PredictionResult with composition, temperature, uncertainties
        """
        # Convert to tensor
        if isinstance(spectrum, np.ndarray):
            spec_tensor = torch.from_numpy(spectrum).float()
        else:
            spec_tensor = spectrum.float()

        # Ensure shape [1, 512]
        if spec_tensor.dim() == 1:
            spec_tensor = spec_tensor.unsqueeze(0)

        spec_tensor = spec_tensor.to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(spec_tensor)

        # Extract results
        composition = outputs['composition'][0].cpu().numpy()
        temperature = outputs['temperature'][0, 0].item()
        uncertainty = outputs['uncertainty'][0].cpu().numpy()

        # MC dropout uncertainty (optional)
        if hasattr(self.model, 'apply_mc_dropout'):
            epistemic_unc = self._estimate_epistemic_uncertainty(spec_tensor)
        else:
            epistemic_unc = None

        # Attention weights (optional)
        attention_weights = None
        if return_attention and 'attention_weights' in outputs:
            attention_weights = {
                f'head_{i}': w[0].cpu().numpy()
                for i, w in enumerate(outputs['attention_weights'])
            }

        return PredictionResult(
            composition=composition,
            temperature=temperature,
            uncertainty=uncertainty,
            epistemic_unc=epistemic_unc,
            attention_weights=attention_weights,
        )

    def predict_batch(
        self,
        spectra: Union[np.ndarray, torch.Tensor],
        batch_size: int = 32,
    ) -> List[PredictionResult]:
        """
        Predict on batch of spectra.

        Args:
            spectra: [N, 512] batch of spectra
            batch_size: Processing batch size

        Returns:
            List of PredictionResult objects
        """
        if isinstance(spectra, np.ndarray):
            spectra_tensor = torch.from_numpy(spectra).float()
        else:
            spectra_tensor = spectra.float()

        results = []

        for i in range(0, len(spectra_tensor), batch_size):
            batch = spectra_tensor[i:i+batch_size].to(self.device)

            with torch.no_grad():
                outputs = self.model(batch)

            for j in range(len(batch)):
                result = PredictionResult(
                    composition=outputs['composition'][j].cpu().numpy(),
                    temperature=outputs['temperature'][j, 0].item(),
                    uncertainty=outputs['uncertainty'][j].cpu().numpy(),
                )
                results.append(result)

        return results

    def _estimate_epistemic_uncertainty(self, spectrum_tensor: torch.Tensor) -> np.ndarray:
        """
        Estimate epistemic uncertainty using MC dropout.

        Args:
            spectrum_tensor: [batch, 512] spectrum

        Returns:
            [batch, 10] epistemic uncertainty
        """
        mc_predictions = []

        for _ in range(self.mc_samples):
            # Enable dropout
            self.model.train()
            with torch.no_grad():
                outputs = self.model(spectrum_tensor)
            mc_predictions.append(outputs['composition'])

        # Stack predictions
        mc_pred_stack = torch.stack(mc_predictions, dim=0)  # [mc_samples, batch, 10]

        # Epistemic uncertainty = variance across MC samples
        epistemic = torch.var(mc_pred_stack, dim=0).cpu().numpy()

        # Back to eval mode
        self.model.eval()

        return epistemic

    def get_species_importance(self, spectrum: np.ndarray) -> Dict[str, float]:
        """
        Estimate importance of each species for prediction.

        Uses attention weights and composition.

        Args:
            spectrum: [512] transit spectrum

        Returns:
            Dict mapping species name to importance score [0, 1]
        """
        spec_tensor = torch.from_numpy(spectrum).float().unsqueeze(0)
        spec_tensor = spec_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(spec_tensor)

        composition = outputs['composition'][0].cpu().numpy()

        # Importance = composition * attention variance
        # (species with high abundance and variable attention are important)
        importance_scores = composition.copy()

        # Normalize
        importance_scores = importance_scores / (importance_scores.sum() + 1e-8)

        return {
            species: float(score)
            for species, score in zip(self.species, importance_scores)
        }

    def detect_biosignatures(
        self,
        spectrum: np.ndarray,
        biosignature_species: Optional[List[str]] = None,
        threshold: float = 1e-4,
    ) -> Dict[str, bool]:
        """
        Check for biosignatures in spectrum.

        Args:
            spectrum: [512] transit spectrum
            biosignature_species: Species to check (default: ['O3', 'CH4', 'N2O'])
            threshold: Abundance threshold for detection

        Returns:
            Dict mapping species to detection (True/False)
        """
        if biosignature_species is None:
            biosignature_species = ['O3', 'CH4', 'NH3']

        result = self.predict(spectrum)

        detections = {}
        for species in biosignature_species:
            if species in self.species:
                idx = self.species.index(species)
                abundance = result.composition[idx]
                detection = abundance > threshold
                detections[species] = detection

        return detections

    def explain_prediction(
        self,
        spectrum: np.ndarray,
        top_k: int = 5,
    ) -> Dict:
        """
        Generate interpretable explanation for prediction.

        Args:
            spectrum: [512] transit spectrum
            top_k: Number of top species to explain

        Returns:
            Dict with explanations
        """
        result = self.predict(spectrum)

        # Top k species
        top_indices = np.argsort(result.composition)[::-1][:top_k]

        explanation = {
            'temperature': f"{result.temperature:.1f} K",
            'top_species': [
                {
                    'species': self.species[idx],
                    'abundance': float(result.composition[idx]),
                    'uncertainty': float(result.uncertainty[idx]) if result.uncertainty is not None else None,
                }
                for idx in top_indices
            ],
        }

        return explanation


# Convenience functions

def load_predictor(model_path: str, device: str = 'cpu') -> SpectrumPredictor:
    """Load a predictor from checkpoint path."""
    return SpectrumPredictor(model_path, device=device)


def predict_spectrum(
    spectrum: np.ndarray,
    model_path: str,
    device: str = 'cpu',
) -> PredictionResult:
    """Single-shot prediction on spectrum."""
    predictor = load_predictor(model_path, device=device)
    return predictor.predict(spectrum)
