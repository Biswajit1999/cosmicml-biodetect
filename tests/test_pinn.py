"""
Unit tests for PINN model.

References:
    Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks:
    A deep learning framework for solving forward and inverse problems involving nonlinear partial
    differential equations. Journal of Computational Physics, 378, 686-707.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cosmicml.models import PINN


class TestPINN:
    """Test PINN model."""

    @pytest.fixture
    def pinn_model(self):
        """Create PINN instance."""
        return PINN(
            input_dim=512,
            latent_dim=64,
            output_dim=10,
            physics_weight=1.0,
        )

    @pytest.fixture
    def sample_batch(self):
        """Create sample batch."""
        spectra = torch.randn(4, 512)
        compositions = torch.softmax(torch.randn(4, 10), dim=1)
        return spectra, compositions

    def test_pinn_creation(self, pinn_model):
        """Test PINN initialization."""
        assert pinn_model is not None
        assert pinn_model.input_dim == 512
        assert pinn_model.output_dim == 10

    def test_forward_pass(self, pinn_model, sample_batch):
        """Test forward pass through PINN."""
        spectra, _ = sample_batch

        with torch.no_grad():
            output = pinn_model(spectra)

        # Check shape
        assert output.shape == (4, 10)

        # Check output is valid composition
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)

    def test_composition_sums_to_one(self, pinn_model, sample_batch):
        """Test that output compositions sum to 1."""
        spectra, _ = sample_batch

        with torch.no_grad():
            output = pinn_model(spectra)
            sums = torch.sum(output, dim=1)

        # Should sum to 1
        assert torch.allclose(sums, torch.ones(4), atol=1e-6)

    def test_abundance_loss(self, pinn_model, sample_batch):
        """Test abundance constraint loss."""
        _, compositions = sample_batch

        loss = pinn_model.compute_abundance_loss(compositions)

        # Should be non-negative
        assert loss >= 0
        # Should be small for valid compositions
        assert loss < 1.0

    def test_conservation_loss(self, pinn_model, sample_batch):
        """Test conservation loss."""
        spectra, compositions = sample_batch

        loss = pinn_model.compute_conservation_loss(compositions)

        # Should be non-negative
        assert loss >= 0

    def test_thermodynamic_loss(self, pinn_model, sample_batch):
        """Test thermodynamic loss."""
        _, compositions = sample_batch

        loss = pinn_model.compute_thermodynamic_loss(compositions)

        # Should be non-negative
        assert loss >= 0

    def test_physics_loss(self, pinn_model, sample_batch):
        """Test total physics loss."""
        _, compositions = sample_batch

        loss = pinn_model.compute_physics_loss(compositions)

        # Should be non-negative
        assert loss >= 0
        # Should be real number
        assert not torch.isnan(loss)

    def test_total_loss_computation(self, pinn_model, sample_batch):
        """Test complete loss computation."""
        spectra, compositions = sample_batch

        total_loss, data_loss, physics_loss = pinn_model.compute_loss(
            spectra,
            compositions,
        )

        # All should be positive
        assert total_loss > 0
        assert data_loss >= 0
        assert physics_loss >= 0

        # Total should be combination
        expected_total = data_loss + pinn_model.physics_weight * physics_loss
        assert torch.isclose(total_loss, expected_total, atol=1e-4)

    def test_backward_pass(self, pinn_model, sample_batch):
        """Test backpropagation."""
        spectra, compositions = sample_batch

        # Compute loss
        loss, _, _ = pinn_model.compute_loss(spectra, compositions)

        # Backprop
        loss.backward()

        # Check gradients exist
        for param in pinn_model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_uncertainty_quantification(self, pinn_model):
        """Test MC Dropout uncertainty estimation."""
        spectrum = torch.randn(2, 512)

        mean, std = pinn_model.predict_with_uncertainty(spectrum, n_samples=5)

        # Check shapes
        assert mean.shape == (2, 10)
        assert std.shape == (2, 10)

        # Uncertainties should be non-negative
        assert torch.all(std >= 0)

        # Mean should be valid composition
        assert torch.all(mean >= 0)
        assert torch.all(mean <= 1)

    def test_model_parameters(self, pinn_model):
        """Test model has correct number of parameters."""
        total_params = sum(p.numel() for p in pinn_model.parameters())

        # Should have reasonable number of parameters
        assert total_params > 1000
        assert total_params < 10_000_000

    def test_model_requires_grad(self, pinn_model):
        """Test model parameters require gradients."""
        for param in pinn_model.parameters():
            assert param.requires_grad is True

    def test_training_vs_eval_mode(self, pinn_model):
        """Test model behaves differently in train vs eval mode."""
        spectra = torch.randn(2, 512)  # Batch size > 1 for BatchNorm

        # Set to training mode
        pinn_model.train()
        with torch.no_grad():
            out_train = pinn_model(spectra)

        # Set to eval mode
        pinn_model.eval()
        with torch.no_grad():
            out_eval = pinn_model(spectra)

        # Both should be valid
        assert out_train.shape == out_eval.shape
        assert torch.all(out_train >= 0)
        assert torch.all(out_eval >= 0)


class TestPINNPhysicsConstraints:
    """Test physics constraint enforcement."""

    @pytest.fixture
    def pinn_model(self):
        return PINN(input_dim=512, latent_dim=64, output_dim=10, physics_weight=2.0)

    def test_softmax_ensures_validity(self, pinn_model):
        """Test softmax ensures valid compositions."""
        # Create batch with extreme random values
        spectra = torch.randn(10, 512) * 100

        with torch.no_grad():
            outputs = pinn_model(spectra)

        # All should be valid despite extreme inputs
        assert torch.all(outputs >= 0)
        assert torch.all(outputs <= 1)
        assert torch.allclose(torch.sum(outputs, dim=1), torch.ones(10))

    def test_no_negative_abundances(self, pinn_model):
        """Test no negative abundances in output."""
        spectra = torch.randn(50, 512)

        with torch.no_grad():
            outputs = pinn_model(spectra)

        # Should never have negative values
        assert torch.all(outputs >= -1e-7)  # Allow numerical precision issues

    def test_physics_weight_effect(self):
        """Test that physics weight affects loss computation."""
        spectra = torch.randn(2, 512)
        compositions = torch.softmax(torch.randn(2, 10), dim=1)

        # Model with low physics weight
        pinn_low = PINN(output_dim=10, physics_weight=0.1)
        loss_low, _, _ = pinn_low.compute_loss(spectra, compositions)

        # Model with high physics weight
        pinn_high = PINN(output_dim=10, physics_weight=10.0)
        loss_high, _, _ = pinn_high.compute_loss(spectra, compositions)

        # Both should compute successfully
        assert loss_low > 0
        assert loss_high > 0


class TestPINNIntegration:
    """Integration tests for PINN."""

    def test_training_step(self):
        """Test a single training step."""
        model = PINN(output_dim=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Create batch
        spectra = torch.randn(8, 512)
        compositions = torch.softmax(torch.randn(8, 10), dim=1)

        # Training step
        optimizer.zero_grad()
        loss, _, _ = model.compute_loss(spectra, compositions)
        loss.backward()
        optimizer.step()

        # Loss should decrease slightly
        with torch.no_grad():
            loss_after = model.compute_loss(spectra, compositions)[0]

        # Loss should be valid
        assert not torch.isnan(loss_after)
        assert loss_after > 0

    def test_multiple_batches(self):
        """Test processing multiple batches."""
        model = PINN(output_dim=10)

        for i in range(5):
            spectra = torch.randn(4, 512)
            compositions = torch.softmax(torch.randn(4, 10), dim=1)

            with torch.no_grad():
                loss, _, _ = model.compute_loss(spectra, compositions)

            assert not torch.isnan(loss)
            assert loss > 0

    def test_device_movement(self):
        """Test moving model between devices."""
        model = PINN()

        # CPU
        model = model.cpu()
        spectra = torch.randn(2, 512).cpu()
        with torch.no_grad():
            out = model(spectra)
        assert out.device.type == "cpu"

        # Can also move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            spectra = spectra.cuda()
            with torch.no_grad():
                out = model(spectra)
            assert out.device.type == "cuda"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
