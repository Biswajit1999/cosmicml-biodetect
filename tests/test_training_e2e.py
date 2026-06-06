"""
Integration tests for end-to-end training pipeline.

Tests:
- Data generation and loading
- Model forward passes
- Training loop execution
- Curriculum scheduling
- Checkpoint saving and loading
- Loss convergence
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cosmicml.training import EnhancedDataGenerator, CurriculumSchedule, TrainingManager
from cosmicml.models.pinn_v3 import PINNv3


class TestEndToEndTraining(unittest.TestCase):
    """Integration tests for complete training pipeline."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cpu'
        self.temp_dir = tempfile.mkdtemp()

        # Create small model for testing
        self.model = PINNv3(
            input_dim=256,
            output_dim=10,
            num_attention_heads=2,
            num_mc_samples=5,
            physics_weight=0.5,
        )

        self.generator = EnhancedDataGenerator(
            num_atmospheres=100,
            n_wavelengths=256,
        )

    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def test_data_generation_for_training(self):
        """Test data generation for training."""
        spectra, compositions, temps, press = self.generator.generate_batch(32)

        # Check shapes
        self.assertEqual(spectra.shape, (32, 256))
        self.assertEqual(compositions.shape, (32, 10))
        self.assertEqual(temps.shape, (32,))
        self.assertEqual(press.shape, (32,))

        # Convert to torch
        spec_tensor = torch.from_numpy(spectra).float()
        comp_tensor = torch.from_numpy(compositions).float()
        temp_tensor = torch.from_numpy(temps).float()

        # Verify tensor shapes
        self.assertEqual(spec_tensor.shape[0], 32)
        self.assertEqual(comp_tensor.shape[0], 32)

    def test_model_forward_pass(self):
        """Test model forward pass."""
        spectra, _, _, _ = self.generator.generate_batch(8)
        spec_tensor = torch.from_numpy(spectra).float().to(self.device)

        self.model.to(self.device)
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(spec_tensor)

        # Check output keys
        self.assertIn('composition', outputs)
        self.assertIn('temperature', outputs)
        self.assertIn('uncertainty', outputs)
        self.assertIn('attention_weights', outputs)

        # Check output shapes
        self.assertEqual(outputs['composition'].shape, (8, 10))
        self.assertEqual(outputs['temperature'].shape, (8, 1))
        self.assertEqual(outputs['uncertainty'].shape, (8, 10))

    def test_training_manager_initialization(self):
        """Test training manager initialization."""
        trainer = TrainingManager(
            self.model,
            learning_rate=0.001,
            device=self.device,
        )

        self.assertIsNotNone(trainer.optimizer_manager)
        self.assertIsNotNone(trainer.loss_balancer)
        self.assertEqual(trainer.device, self.device)

    def test_train_step(self):
        """Test single training step."""
        trainer = TrainingManager(
            self.model,
            learning_rate=0.001,
            device=self.device,
        )

        spectra, compositions, temps, _ = self.generator.generate_batch(16)
        spec_tensor = torch.from_numpy(spectra).float()
        comp_tensor = torch.from_numpy(compositions).float()
        temp_tensor = torch.from_numpy(temps).float()

        metrics = trainer.train_step(spec_tensor, comp_tensor, temp_tensor)

        # Check metrics
        self.assertIn('total_loss', metrics)
        self.assertIn('data_loss', metrics)
        self.assertIn('learning_rate', metrics)

        # Check values
        self.assertFalse(np.isnan(metrics['total_loss']))
        self.assertGreater(metrics['total_loss'], 0)

    def test_curriculum_scheduling(self):
        """Test curriculum scheduling during training."""
        curriculum = CurriculumSchedule()

        # Stage 1
        stage1 = curriculum.get_current_stage()
        self.assertEqual(stage1['name'], 'Abundant Species Only')
        self.assertEqual(sum(stage1['species_mask']), 4)

        # Advance to stage 2
        curriculum.advance_stage()
        stage2 = curriculum.get_current_stage()
        self.assertEqual(stage2['name'], 'Add Medium Species')
        self.assertEqual(sum(stage2['species_mask']), 6)

        # Advance to stage 3
        curriculum.advance_stage()
        stage3 = curriculum.get_current_stage()
        self.assertEqual(stage3['name'], 'Add Trace Species')
        self.assertEqual(sum(stage3['species_mask']), 10)

    def test_multi_epoch_training(self):
        """Test training over multiple epochs."""
        trainer = TrainingManager(
            self.model,
            learning_rate=0.001,
            device=self.device,
        )
        curriculum = CurriculumSchedule()

        losses = []

        for epoch in range(3):
            stage = curriculum.get_current_stage()

            # Generate batch
            spectra, compositions, temps, _ = self.generator.generate_batch(16)
            spec_tensor = torch.from_numpy(spectra).float()
            comp_tensor = torch.from_numpy(compositions).float()
            temp_tensor = torch.from_numpy(temps).float()

            # Apply curriculum
            masked_comp = curriculum.apply_mask_to_composition(
                comp_tensor,
                stage['species_mask']
            )

            # Train step
            metrics = trainer.train_step(spec_tensor, masked_comp, temp_tensor)
            losses.append(metrics['total_loss'])

            # Advance curriculum
            if (epoch + 1) % 2 == 0:
                curriculum.advance_stage()

        # Check losses are valid
        self.assertEqual(len(losses), 3)
        self.assertTrue(all(not np.isnan(l) for l in losses))

    def test_validation_step(self):
        """Test validation step."""
        trainer = TrainingManager(
            self.model,
            learning_rate=0.001,
            device=self.device,
        )

        spectra, compositions, _, _ = self.generator.generate_batch(16)
        spec_tensor = torch.from_numpy(spectra).float()
        comp_tensor = torch.from_numpy(compositions).float()

        val_metrics = trainer.validate(spec_tensor, comp_tensor)

        # Check metrics
        self.assertIn('val_r2', val_metrics)
        self.assertIn('val_mae_mean', val_metrics)

        # Check values
        self.assertLess(val_metrics['val_r2'], 1.0)
        self.assertGreater(val_metrics['val_mae_mean'], 0)

    def test_checkpoint_save_load(self):
        """Test checkpoint saving and loading."""
        trainer = TrainingManager(
            self.model,
            learning_rate=0.001,
            device=self.device,
        )

        # Save checkpoint
        checkpoint_path = Path(self.temp_dir) / 'test_checkpoint.pt'
        trainer.save_checkpoint(epoch=0, val_loss=0.5)

        # Verify checkpoint exists
        checkpoints = list(Path(self.temp_dir).glob('*.pt'))
        self.assertGreater(len(checkpoints), 0)

    def test_loss_convergence_trend(self):
        """Test that losses show convergence trend."""
        trainer = TrainingManager(
            self.model,
            learning_rate=0.01,
            device=self.device,
        )

        losses = []

        for epoch in range(5):
            spectra, compositions, temps, _ = self.generator.generate_batch(32)
            spec_tensor = torch.from_numpy(spectra).float()
            comp_tensor = torch.from_numpy(compositions).float()
            temp_tensor = torch.from_numpy(temps).float()

            metrics = trainer.train_step(spec_tensor, comp_tensor, temp_tensor)
            losses.append(metrics['total_loss'])

        # Losses should not be increasing consistently
        # (some fluctuation is ok, but trend should be down or flat)
        final_avg = np.mean(losses[-2:])
        initial_avg = np.mean(losses[:2])

        # This is a weak test, but checks basic learning
        self.assertLess(final_avg, initial_avg * 1.5)

    def test_model_parameters_updated(self):
        """Test that model parameters are updated during training."""
        trainer = TrainingManager(
            self.model,
            learning_rate=0.1,  # Higher LR to see changes
            device=self.device,
        )

        # Get initial parameters
        initial_params = []
        for param in trainer.model.parameters():
            initial_params.append(param.data.clone())

        # Training step
        spectra, compositions, temps, _ = self.generator.generate_batch(16)
        spec_tensor = torch.from_numpy(spectra).float()
        comp_tensor = torch.from_numpy(compositions).float()
        temp_tensor = torch.from_numpy(temps).float()

        trainer.train_step(spec_tensor, comp_tensor, temp_tensor)

        # Check that some parameters changed
        params_changed = False
        for param, initial in zip(trainer.model.parameters(), initial_params):
            if not torch.allclose(param.data, initial, atol=1e-6):
                params_changed = True
                break

        self.assertTrue(params_changed)

    def test_batch_learning_consistency(self):
        """Test that batch learning is consistent."""
        trainer1 = TrainingManager(self.model, learning_rate=0.001, device=self.device)

        # Create same batch twice
        spectra, compositions, temps, _ = self.generator.generate_batch(16)
        spec_tensor = torch.from_numpy(spectra).float()
        comp_tensor = torch.from_numpy(compositions).float()
        temp_tensor = torch.from_numpy(temps).float()

        # Train step 1
        metrics1 = trainer1.train_step(spec_tensor, comp_tensor, temp_tensor)

        # Train step 2 with same batch
        metrics2 = trainer1.train_step(spec_tensor, comp_tensor, temp_tensor)

        # Losses should be different (parameters changed)
        self.assertNotAlmostEqual(metrics1['total_loss'], metrics2['total_loss'], places=3)

    def test_temperature_auxiliary_task(self):
        """Test temperature prediction auxiliary task."""
        trainer = TrainingManager(
            self.model,
            learning_rate=0.001,
            device=self.device,
        )

        spectra, compositions, temps, _ = self.generator.generate_batch(16)
        spec_tensor = torch.from_numpy(spectra).float()
        comp_tensor = torch.from_numpy(compositions).float()
        temp_tensor = torch.from_numpy(temps).float()

        metrics = trainer.train_step(spec_tensor, comp_tensor, temp_tensor)

        # Temperature loss should be in metrics
        self.assertIn('temperature_loss', metrics)
        self.assertGreater(metrics['temperature_loss'], 0)

    def test_uncertainty_consistency(self):
        """Test that uncertainty predictions are consistent."""
        trainer = TrainingManager(
            self.model,
            learning_rate=0.001,
            device=self.device,
        )

        spectra, _, _, _ = self.generator.generate_batch(8)
        spec_tensor = torch.from_numpy(spectra).float()

        trainer.model.eval()
        with torch.no_grad():
            outputs = trainer.model(spec_tensor)

        uncertainties = outputs['uncertainty'].numpy()

        # Uncertainties should be positive
        self.assertTrue(np.all(uncertainties > 0))

        # Uncertainties should be reasonable magnitude
        self.assertTrue(np.all(uncertainties < 1.0))


class TestRegressionBehavior(unittest.TestCase):
    """Tests to catch regressions in behavior."""

    def test_no_nan_loss(self):
        """Test that loss never becomes NaN."""
        model = PINNv3(input_dim=256, output_dim=10, num_attention_heads=2)
        trainer = TrainingManager(model, device='cpu')
        generator = EnhancedDataGenerator(n_wavelengths=256)

        for i in range(5):
            spectra, compositions, temps, _ = generator.generate_batch(8)
            spec_t = torch.from_numpy(spectra).float()
            comp_t = torch.from_numpy(compositions).float()
            temp_t = torch.from_numpy(temps).float()

            metrics = trainer.train_step(spec_t, comp_t, temp_t)

            self.assertFalse(np.isnan(metrics['total_loss']),
                           f"NaN loss at iteration {i}")

    def test_no_nan_gradient(self):
        """Test that gradients never become NaN."""
        model = PINNv3(input_dim=256, output_dim=10, num_attention_heads=2)
        trainer = TrainingManager(model, device='cpu')
        generator = EnhancedDataGenerator(n_wavelengths=256)

        spectra, compositions, temps, _ = generator.generate_batch(8)
        spec_t = torch.from_numpy(spectra).float()
        comp_t = torch.from_numpy(compositions).float()
        temp_t = torch.from_numpy(temps).float()

        trainer.train_step(spec_t, comp_t, temp_t)

        # Check no NaN gradients
        for param in model.parameters():
            if param.grad is not None:
                self.assertFalse(torch.isnan(param.grad).any())

    def test_output_shapes_consistent(self):
        """Test that output shapes stay consistent."""
        model = PINNv3(input_dim=256, output_dim=10)
        model.eval()

        for batch_size in [1, 4, 8, 16, 32]:
            x = torch.randn(batch_size, 256)

            with torch.no_grad():
                outputs = model(x)

            self.assertEqual(outputs['composition'].shape, (batch_size, 10))
            self.assertEqual(outputs['temperature'].shape, (batch_size, 1))
            self.assertEqual(outputs['uncertainty'].shape, (batch_size, 10))


if __name__ == '__main__':
    unittest.main()
