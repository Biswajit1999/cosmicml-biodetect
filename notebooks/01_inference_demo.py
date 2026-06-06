"""
Inference Demo: Using CosmicML-Biodetect for Spectrum Analysis

This notebook demonstrates how to use the trained PINN v3 model
to analyze exoplanet spectra and detect biosignatures.

Sections:
1. Load pre-trained model
2. Generate example spectra
3. Make predictions
4. Analyze uncertainties
5. Detect biosignatures
6. Visualize attention weights
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cosmicml.inference.predictor import SpectrumPredictor
from cosmicml.training.data_generator import EnhancedDataGenerator


def main():
    print("\n" + "=" * 70)
    print("COSMICML-BIODETECT INFERENCE DEMO")
    print("=" * 70 + "\n")

    # ===== 1. LOAD MODEL =====
    print("Step 1: Loading trained PINN v3 model")
    print("-" * 70)

    model_path = "models/checkpoints/best_model.pt"
    try:
        predictor = SpectrumPredictor(
            model_path,
            device='cpu',
            mc_samples=50,
        )
        print(f"✓ Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"⚠ Model not found at {model_path}")
        print("  Please train the model first with: python scripts/train_pinn_v3.py")
        return

    # ===== 2. GENERATE EXAMPLE SPECTRA =====
    print("\nStep 2: Generating example spectra")
    print("-" * 70)

    generator = EnhancedDataGenerator(
        num_atmospheres=100,
        n_wavelengths=512,
    )

    # Generate a batch
    spectra, compositions, temps, press = generator.generate_batch(5)
    print(f"✓ Generated 5 example spectra")
    print(f"  Shape: {spectra.shape}")
    print(f"  Temperature range: {temps.min():.0f}K - {temps.max():.0f}K")
    print(f"  Pressure range: {press.min():.2f} - {press.max():.2f} bar")

    # ===== 3. MAKE PREDICTIONS =====
    print("\nStep 3: Making predictions")
    print("-" * 70)

    results = []
    for i, spectrum in enumerate(spectra):
        print(f"\nAnalyzing spectrum {i+1}/5...")
        result = predictor.predict(spectrum)
        results.append(result)

        # Print summary
        print(f"  Temperature: {result.temperature:.1f} K")
        print(f"  Top 3 species:")

        species = ['H2O', 'CO2', 'O2', 'N2', 'CH4', 'H2', 'O3', 'NH3', 'NO', 'H2S']
        comp_with_names = list(zip(species, result.composition))
        comp_with_names.sort(key=lambda x: x[1], reverse=True)

        for spec, abundance in comp_with_names[:3]:
            print(f"    {spec}: {abundance:.2e}")

    # ===== 4. ANALYZE UNCERTAINTIES =====
    print("\nStep 4: Uncertainty Analysis")
    print("-" * 70)

    result = results[0]
    print(f"\nSpectrum 1 uncertainty estimates:")
    print(f"  Mean uncertainty: {result.uncertainty.mean():.2e}")
    print(f"  Max uncertainty: {result.uncertainty.max():.2e}")
    print(f"  Min uncertainty: {result.uncertainty.min():.2e}")

    # Identify species with high uncertainty
    high_unc_indices = np.where(result.uncertainty > result.uncertainty.mean())[0]
    species_with_high_unc = [
        species[i] for i in high_unc_indices
        if i < len(species)
    ]
    print(f"  Species with above-average uncertainty: {species_with_high_unc}")

    # ===== 5. DETECT BIOSIGNATURES =====
    print("\nStep 5: Biosignature Detection")
    print("-" * 70)

    biosignatures = {
        'O3': 'Oxygen (photochemical, potential biosignature)',
        'CH4': 'Methane (potential biosignature)',
        'NH3': 'Ammonia (potential biosignature)',
    }

    for spec_name, desc in biosignatures.items():
        print(f"\nDetecting {spec_name}...")
        detections = [
            predictor.detect_biosignatures(spectrum, [spec_name])[spec_name]
            for spectrum in spectra
        ]
        count = sum(detections)
        print(f"  Detected in {count}/5 spectra")
        print(f"  ({desc})")

    # ===== 6. SPECIES IMPORTANCE =====
    print("\nStep 6: Species Importance Analysis")
    print("-" * 70)

    spectrum = spectra[0]
    importance = predictor.get_species_importance(spectrum)

    print(f"\nSpectrum 1 species importance:")
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for spec, score in sorted_importance[:5]:
        print(f"  {spec}: {score:.3f}")

    # ===== 7. INTERPRETABLE EXPLANATIONS =====
    print("\nStep 7: Prediction Explanations")
    print("-" * 70)

    for i in range(min(3, len(spectra))):
        print(f"\nSpectrum {i+1} explanation:")
        explanation = predictor.explain_prediction(spectra[i], top_k=4)
        print(f"  Temperature: {explanation['temperature']}")
        print(f"  Key species:")
        for item in explanation['top_species']:
            print(f"    {item['species']}: {item['abundance']:.2e}")

    # ===== 8. BATCH PROCESSING =====
    print("\nStep 8: Batch Processing")
    print("-" * 70)

    batch_results = predictor.predict_batch(spectra, batch_size=2)
    print(f"✓ Processed batch of 5 spectra")
    print(f"  Results: {len(batch_results)} predictions")

    # ===== SUMMARY =====
    print("\n" + "=" * 70)
    print("INFERENCE DEMO COMPLETE")
    print("=" * 70)

    print("\nKey Features Demonstrated:")
    print("✓ Model loading")
    print("✓ Single spectrum prediction")
    print("✓ Batch processing")
    print("✓ Uncertainty estimation")
    print("✓ Biosignature detection")
    print("✓ Species importance analysis")
    print("✓ Prediction explanation")

    print("\nNext Steps:")
    print("1. Train your own model: python scripts/train_pinn_v3.py")
    print("2. Analyze real exoplanet spectra")
    print("3. Generate publication figures")
    print("4. Deploy to cloud infrastructure")

    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()
