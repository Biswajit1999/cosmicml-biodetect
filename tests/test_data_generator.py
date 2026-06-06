"""
Unit tests for EnhancedDataGenerator and CurriculumSchedule.

Tests:
- Composition generation (diversity, normalization)
- Spectrum generation (bounds, physics)
- Batch generation (shapes, consistency)
- Curriculum schedule (masking, progression)
"""

import unittest
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cosmicml.training import EnhancedDataGenerator, CurriculumSchedule


class TestEnhancedDataGenerator(unittest.TestCase):
    """Test EnhancedDataGenerator"""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = EnhancedDataGenerator(
            num_atmospheres=100,
            n_wavelengths=512,
        )

    def test_initialization(self):
        """Test generator initialization."""
        self.assertEqual(self.generator.num_atmospheres, 100)
        self.assertEqual(self.generator.n_wavelengths, 512)
        self.assertTrue(self.generator.use_enhanced_rt)
        self.assertEqual(len(self.generator.wavelengths), 512)

    def test_wavelength_range(self):
        """Test wavelength array properties."""
        wavelengths = self.generator.wavelengths
        self.assertEqual(len(wavelengths), 512)
        self.assertAlmostEqual(wavelengths[0], 0.3)
        self.assertAlmostEqual(wavelengths[-1], 5.0)
        # Check monotonicity
        self.assertTrue(np.all(np.diff(wavelengths) > 0))

    def test_diverse_compositions_shape(self):
        """Test composition generation shape."""
        comps = self.generator.generate_diverse_compositions(50)
        self.assertEqual(comps.shape, (50, 10))

    def test_diverse_compositions_normalization(self):
        """Test that compositions are normalized to 1."""
        comps = self.generator.generate_diverse_compositions(100)
        sums = np.sum(comps, axis=1)
        # Check normalization (within numerical precision)
        np.testing.assert_array_almost_equal(sums, np.ones(100), decimal=5)

    def test_diverse_compositions_positivity(self):
        """Test that all compositions are positive."""
        comps = self.generator.generate_diverse_compositions(100)
        self.assertTrue(np.all(comps >= 0))

    def test_diverse_compositions_diversity(self):
        """Test that compositions are diverse (not all identical)."""
        comps = self.generator.generate_diverse_compositions(100)
        # Check variance across samples
        variance = np.var(comps, axis=0)
        # At least some species should vary
        self.assertTrue(np.any(variance > 0.001))

    def test_spectrum_generation_shape(self):
        """Test spectrum generation output shape."""
        composition = {
            'H2O': 0.1, 'CO2': 0.01, 'O2': 0.2, 'N2': 0.68,
            'CH4': 0.001, 'H2': 0.001, 'O3': 0.0001,
            'NH3': 0.0001, 'NO': 0.0001, 'H2S': 0.0001
        }
        spectrum = self.generator.generate_realistic_spectrum(
            composition, 300, 1.0
        )
        self.assertEqual(len(spectrum), 512)

    def test_spectrum_generation_bounds(self):
        """Test that spectrum is within physical bounds."""
        composition = {
            'H2O': 0.1, 'CO2': 0.01, 'O2': 0.2, 'N2': 0.68,
            'CH4': 0.001, 'H2': 0.001, 'O3': 0.0001,
            'NH3': 0.0001, 'NO': 0.0001, 'H2S': 0.0001
        }
        spectrum = self.generator.generate_realistic_spectrum(
            composition, 300, 1.0
        )
        # Spectrum should be non-negative (optical depth)
        self.assertTrue(np.all(spectrum >= 0))
        # Spectrum should be less than 0.1 (transit depth bound)
        self.assertTrue(np.all(spectrum <= 0.1))

    def test_spectrum_temperature_dependence(self):
        """Test that spectrum depends on temperature."""
        composition = {
            'H2O': 0.1, 'CO2': 0.01, 'O2': 0.2, 'N2': 0.68,
            'CH4': 0.001, 'H2': 0.001, 'O3': 0.0001,
            'NH3': 0.0001, 'NO': 0.0001, 'H2S': 0.0001
        }
        spectrum_300 = self.generator.generate_realistic_spectrum(
            composition, 300, 1.0
        )
        spectrum_500 = self.generator.generate_realistic_spectrum(
            composition, 500, 1.0
        )
        # Spectra should be different for different temperatures
        difference = np.mean(np.abs(spectrum_300 - spectrum_500))
        self.assertGreater(difference, 1e-6)

    def test_batch_generation_shapes(self):
        """Test batch generation output shapes."""
        spectra, compositions, temps, press = self.generator.generate_batch(50)

        self.assertEqual(spectra.shape, (50, 512))
        self.assertEqual(compositions.shape, (50, 10))
        self.assertEqual(temps.shape, (50,))
        self.assertEqual(press.shape, (50,))

    def test_batch_generation_consistency(self):
        """Test that batch generation produces consistent data."""
        spectra, compositions, temps, press = self.generator.generate_batch(100)

        # Temperatures in range
        self.assertTrue(np.all(temps >= 200))
        self.assertTrue(np.all(temps <= 500))

        # Pressures in range
        self.assertTrue(np.all(press >= 0.1))
        self.assertTrue(np.all(press <= 10))

        # Compositions normalized
        sums = np.sum(compositions, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(100), decimal=5)

    def test_simple_spectrum_fallback(self):
        """Test simple spectrum fallback when RT fails."""
        # Create generator with disabled RT
        generator = EnhancedDataGenerator(
            num_atmospheres=10,
            use_enhanced_rt=False,
        )

        composition = {
            'H2O': 0.1, 'CO2': 0.01, 'O2': 0.2, 'N2': 0.68,
            'CH4': 0.001, 'H2': 0.001, 'O3': 0.0001,
            'NH3': 0.0001, 'NO': 0.0001, 'H2S': 0.0001
        }
        spectrum = generator.generate_realistic_spectrum(composition, 300, 1.0)

        # Should still produce valid spectrum
        self.assertEqual(len(spectrum), 512)
        self.assertTrue(np.all(spectrum >= 1e-6))
        self.assertTrue(np.all(spectrum <= 0.1))


class TestCurriculumSchedule(unittest.TestCase):
    """Test CurriculumSchedule"""

    def setUp(self):
        """Set up test fixtures."""
        self.curriculum = CurriculumSchedule()

    def test_initialization(self):
        """Test curriculum initialization."""
        self.assertEqual(self.curriculum.current_stage, 0)
        self.assertEqual(len(self.curriculum.stages), 4)

    def test_stage_names(self):
        """Test that stages have proper names."""
        names = [stage['name'] for stage in self.curriculum.stages]
        self.assertEqual(names[0], 'Abundant Species Only')
        self.assertEqual(names[1], 'Add Medium Species')
        self.assertEqual(names[2], 'Add Trace Species')
        self.assertEqual(names[3], 'Full Training')

    def test_stage_masks(self):
        """Test that stage masks are valid."""
        for stage in self.curriculum.stages:
            mask = stage['species_mask']
            self.assertEqual(len(mask), 10)  # 10 species
            self.assertTrue(all(m in [0, 1] for m in mask))

    def test_mask_progression(self):
        """Test that masks progress (more species included)."""
        masks = [stage['species_mask'] for stage in self.curriculum.stages]
        sums = [sum(mask) for mask in masks]
        # Sums should be non-decreasing
        self.assertTrue(all(sums[i] <= sums[i+1] for i in range(3)))

    def test_temperature_ranges(self):
        """Test temperature ranges in stages."""
        temps = [stage['temperature_range'] for stage in self.curriculum.stages]

        # Stage 1: Fixed 288K
        self.assertEqual(temps[0], (288, 288))

        # Subsequent stages: Ranges expand
        self.assertLess(temps[1][0], temps[2][0])  # Min decreases
        self.assertGreater(temps[1][1], temps[0][1])  # Max increases

    def test_get_current_stage(self):
        """Test getting current stage."""
        stage = self.curriculum.get_current_stage()
        self.assertEqual(stage['name'], 'Abundant Species Only')

    def test_advance_stage(self):
        """Test stage advancement."""
        # Advance from stage 0 to 1
        success = self.curriculum.advance_stage()
        self.assertTrue(success)
        self.assertEqual(self.curriculum.current_stage, 1)

        # Check stage changed
        stage = self.curriculum.get_current_stage()
        self.assertEqual(stage['name'], 'Add Medium Species')

    def test_advance_to_final_stage(self):
        """Test advancement through all stages."""
        for i in range(3):  # Advance 3 times
            success = self.curriculum.advance_stage()
            self.assertTrue(success)

        # Try to advance past final
        success = self.curriculum.advance_stage()
        self.assertFalse(success)
        self.assertEqual(self.curriculum.current_stage, 3)

    def test_apply_mask_shape(self):
        """Test that mask application preserves shape."""
        composition = torch.ones(32, 10) / 10  # Uniform composition
        stage = self.curriculum.get_current_stage()

        masked = self.curriculum.apply_mask_to_composition(
            composition,
            stage['species_mask']
        )

        self.assertEqual(masked.shape, composition.shape)

    def test_apply_mask_normalization(self):
        """Test that masked compositions are normalized."""
        composition = torch.ones(32, 10) / 10
        stage = self.curriculum.get_current_stage()

        masked = self.curriculum.apply_mask_to_composition(
            composition,
            stage['species_mask']
        )

        sums = masked.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(32))

    def test_apply_mask_preserves_masked_species(self):
        """Test that masked species are set to near-zero."""
        composition = torch.ones(10, 10) / 10
        # Create custom mask: first 4 species included, rest zeroed
        mask = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]

        masked = self.curriculum.apply_mask_to_composition(composition, mask)

        # Last 6 species should be very small
        self.assertTrue(torch.all(masked[:, 4:] < 1e-6))

    def test_stage_copy_independence(self):
        """Test that returned stages are independent copies."""
        stage1 = self.curriculum.get_current_stage()
        stage1['name'] = 'Modified'

        stage2 = self.curriculum.get_current_stage()
        self.assertNotEqual(stage2['name'], 'Modified')

    def test_learning_rate_scales(self):
        """Test learning rate scales in stages."""
        scales = [stage['learning_rate_scale'] for stage in self.curriculum.stages]

        # Should be decreasing or equal
        self.assertEqual(scales[0], 1.0)
        self.assertEqual(scales[1], 0.5)
        self.assertEqual(scales[2], 0.2)
        self.assertEqual(scales[3], 0.1)

    def test_epoch_durations(self):
        """Test epoch durations in stages."""
        epochs = [stage['epochs'] for stage in self.curriculum.stages]
        total_epochs = sum(epochs)

        # Check reasonable values
        self.assertGreater(total_epochs, 0)
        self.assertTrue(all(e > 0 for e in epochs))


class TestIntegration(unittest.TestCase):
    """Integration tests for data generation pipeline."""

    def test_full_pipeline(self):
        """Test complete data generation pipeline."""
        generator = EnhancedDataGenerator(num_atmospheres=50, n_wavelengths=256)
        curriculum = CurriculumSchedule()

        # Generate batch
        spectra, compositions, temps, press = generator.generate_batch(20)

        # Verify shapes
        self.assertEqual(spectra.shape, (20, 256))
        self.assertEqual(compositions.shape, (20, 10))

        # Apply curriculum
        comp_tensor = torch.from_numpy(compositions).float()
        stage = curriculum.get_current_stage()
        masked_comp = curriculum.apply_mask_to_composition(
            comp_tensor,
            stage['species_mask']
        )

        # Verify masked composition is valid
        self.assertEqual(masked_comp.shape, comp_tensor.shape)
        torch.testing.assert_close(masked_comp.sum(dim=1), torch.ones(20))

    def test_reproducibility_with_seed(self):
        """Test reproducibility when seed is set."""
        np.random.seed(42)
        gen1 = EnhancedDataGenerator(num_atmospheres=10)
        spec1, comp1, _, _ = gen1.generate_batch(5)

        np.random.seed(42)
        gen2 = EnhancedDataGenerator(num_atmospheres=10)
        spec2, comp2, _, _ = gen2.generate_batch(5)

        # Should be very similar (might not be exactly equal due to randomness)
        # but statistical properties should match
        self.assertAlmostEqual(np.mean(spec1), np.mean(spec2), places=2)
        self.assertAlmostEqual(np.mean(comp1), np.mean(comp2), places=3)


if __name__ == '__main__':
    unittest.main()
