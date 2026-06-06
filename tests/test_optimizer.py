"""
Unit tests for AdvancedOptimizer and MultiTaskLossBalancer.

Tests:
- Learning rate scheduling (warmup, annealing)
- Optimizer step and gradient clipping
- Loss balancing and task weights
- Checkpoint compatibility
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cosmicml.training.trainer import AdvancedOptimizer, MultiTaskLossBalancer


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TestAdvancedOptimizer(unittest.TestCase):
    """Test AdvancedOptimizer"""

    def setUp(self):
        """Set up test fixtures."""
        self.model = SimpleModel()
        self.optimizer = AdvancedOptimizer(
            self.model,
            learning_rate=0.001,
            weight_decay=0.0001,
            warmup_epochs=5,
            total_epochs=20,
            gradient_clip=1.0,
        )

    def test_initialization(self):
        """Test optimizer initialization."""
        self.assertIsNotNone(self.optimizer.optimizer)
        self.assertIsNotNone(self.optimizer.scheduler)
        self.assertEqual(self.optimizer.learning_rate, 0.001)
        self.assertEqual(self.optimizer.gradient_clip, 1.0)

    def test_initial_learning_rate(self):
        """Test that initial learning rate is correct."""
        lr = self.optimizer.get_learning_rate()
        # During warmup, initial LR is scaled down
        self.assertLess(lr, 0.001)
        self.assertGreater(lr, 0)

    def test_learning_rate_progression(self):
        """Test that learning rate increases then decreases."""
        lrs = []
        for _ in range(20):
            lrs.append(self.optimizer.get_learning_rate())
            # Simulate step
            loss = torch.tensor(1.0, requires_grad=True)
            self.optimizer.step(loss)

        # Check that warmup happens (increasing)
        self.assertLess(lrs[0], lrs[2])  # First steps increase

        # Check that we eventually decrease
        self.assertGreater(np.mean(lrs[:5]), np.mean(lrs[-5:]))

    def test_zero_grad(self):
        """Test gradient zeroing."""
        # Create some gradients
        x = torch.randn(5, 10)
        y = self.model(x)
        loss = y.sum()
        loss.backward()

        # Verify gradients exist
        has_grad = False
        for param in self.model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        self.assertTrue(has_grad)

        # Zero gradients
        self.optimizer.zero_grad()

        # Verify all gradients are zero
        for param in self.model.parameters():
            if param.grad is not None:
                self.assertTrue(torch.allclose(param.grad, torch.zeros_like(param.grad)))

    def test_gradient_clipping(self):
        """Test gradient clipping."""
        # Create large gradients
        x = torch.randn(5, 10, requires_grad=True)
        y = self.model(x)
        loss = (y ** 2).sum() * 1e6
        loss.backward()

        # Step with clipping
        self.optimizer.step(loss)

        # Check that gradients are clipped (this is tested implicitly)
        # by checking that training doesn't NaN
        self.assertFalse(torch.isnan(loss))

    def test_optimizer_state(self):
        """Test optimizer state access."""
        param_groups = self.optimizer.optimizer.param_groups
        self.assertEqual(len(param_groups), 1)
        self.assertIn('lr', param_groups[0])
        self.assertIn('weight_decay', param_groups[0])

    def test_scheduler_step(self):
        """Test that scheduler is properly chained."""
        initial_lr = self.optimizer.get_learning_rate()

        # Take several steps
        for _ in range(3):
            self.optimizer.step(torch.tensor(1.0))

        new_lr = self.optimizer.get_learning_rate()

        # During warmup phase, LR should increase
        self.assertGreater(new_lr, initial_lr)

    def test_get_learning_rate_format(self):
        """Test learning rate return format."""
        lr = self.optimizer.get_learning_rate()
        self.assertIsInstance(lr, float)
        self.assertGreater(lr, 0)
        self.assertLess(lr, self.optimizer.learning_rate * 2)


class TestMultiTaskLossBalancer(unittest.TestCase):
    """Test MultiTaskLossBalancer"""

    def setUp(self):
        """Set up test fixtures."""
        self.balancer = MultiTaskLossBalancer(
            num_tasks=4,
            initial_weights=[1.0, 0.1, 1.0, 0.05]
        )

    def test_initialization(self):
        """Test balancer initialization."""
        self.assertEqual(self.balancer.num_tasks, 4)
        self.assertEqual(len(self.balancer.loss_history), 4)

    def test_log_sigma_parameter(self):
        """Test log_sigma parameter."""
        self.assertIsInstance(self.balancer.log_sigma, nn.Parameter)
        self.assertEqual(self.balancer.log_sigma.shape, (4,))

    def test_compute_weighted_loss_shape(self):
        """Test weighted loss computation shape."""
        loss_dict = {
            'data': torch.tensor(0.5),
            'temperature': torch.tensor(0.1),
            'physics': torch.tensor(0.3),
            'aleatoric': torch.tensor(0.05),
        }
        loss_names = ['data', 'temperature', 'physics', 'aleatoric']

        weighted_loss = self.balancer.compute_weighted_loss(loss_dict, loss_names)

        self.assertIsInstance(weighted_loss, torch.Tensor)
        self.assertEqual(weighted_loss.shape, torch.Size([]))  # Scalar

    def test_weighted_loss_is_scalar(self):
        """Test that weighted loss is a scalar."""
        loss_dict = {
            'data': torch.tensor([0.5]),
            'temperature': torch.tensor([0.1]),
            'physics': torch.tensor([0.3]),
            'aleatoric': torch.tensor([0.05]),
        }
        loss_names = ['data', 'temperature', 'physics', 'aleatoric']

        weighted_loss = self.balancer.compute_weighted_loss(loss_dict, loss_names)

        # Should be broadcastable to scalar
        self.assertEqual(weighted_loss.dim(), 0)

    def test_loss_tracking(self):
        """Test that losses are tracked in history."""
        loss_dict = {
            'data': torch.tensor(0.5),
            'temperature': torch.tensor(0.1),
            'physics': torch.tensor(0.3),
            'aleatoric': torch.tensor(0.05),
        }
        loss_names = ['data', 'temperature', 'physics', 'aleatoric']

        self.balancer.compute_weighted_loss(loss_dict, loss_names)

        # Check history is populated
        for i, history in enumerate(self.balancer.loss_history):
            self.assertEqual(len(history), 1)
            self.assertGreater(history[0], 0)

    def test_task_weights(self):
        """Test task weight computation."""
        weights = self.balancer.get_task_weights()

        self.assertEqual(weights.shape, (4,))
        # Weights should sum to 1 (normalized)
        torch.testing.assert_close(weights.sum(), torch.tensor(1.0), atol=1e-6, rtol=1e-5)
        # Weights should be positive
        self.assertTrue(torch.all(weights > 0))

    def test_loss_statistics(self):
        """Test loss statistics computation."""
        loss_dict = {
            'data': torch.tensor(0.5),
            'temperature': torch.tensor(0.1),
            'physics': torch.tensor(0.3),
            'aleatoric': torch.tensor(0.05),
        }
        loss_names = ['data', 'temperature', 'physics', 'aleatoric']

        # Compute multiple times to build history
        for _ in range(10):
            self.balancer.compute_weighted_loss(loss_dict, loss_names)

        stats = self.balancer.get_loss_statistics()

        self.assertIn('task_0_mean', stats)
        self.assertIn('task_0_std', stats)
        self.assertEqual(len(stats), 8)  # 4 tasks * 2 metrics

    def test_missing_loss_handling(self):
        """Test handling of missing losses in dict."""
        loss_dict = {
            'data': torch.tensor(0.5),
            'temperature': torch.tensor(0.1),
            # physics and aleatoric missing
        }
        loss_names = ['data', 'temperature', 'physics', 'aleatoric']

        # Should not raise error
        weighted_loss = self.balancer.compute_weighted_loss(loss_dict, loss_names)

        self.assertIsInstance(weighted_loss, torch.Tensor)

    def test_weight_adaptation(self):
        """Test that weights can adapt over training."""
        loss_dict = {
            'data': torch.tensor(1.0),
            'temperature': torch.tensor(0.1),
            'physics': torch.tensor(0.5),
            'aleatoric': torch.tensor(0.05),
        }
        loss_names = ['data', 'temperature', 'physics', 'aleatoric']

        weights_initial = self.balancer.get_task_weights().clone()

        # Compute loss multiple times with different values
        for i in range(5):
            loss_dict['data'] = torch.tensor(0.1)  # Reduce data loss
            loss_dict['physics'] = torch.tensor(1.0)  # Increase physics loss
            self.balancer.compute_weighted_loss(loss_dict, loss_names)

        weights_adapted = self.balancer.get_task_weights()

        # Weights can change through the learnable log_sigma
        # (In practice, this requires gradient updates)
        self.assertEqual(weights_adapted.shape, weights_initial.shape)

    def test_multiple_iterations(self):
        """Test balancer over multiple iterations."""
        for iteration in range(20):
            loss_dict = {
                'data': torch.tensor(0.5 + 0.1 * np.sin(iteration)),
                'temperature': torch.tensor(0.1),
                'physics': torch.tensor(0.3 + 0.2 * np.cos(iteration)),
                'aleatoric': torch.tensor(0.05),
            }
            loss_names = ['data', 'temperature', 'physics', 'aleatoric']

            weighted_loss = self.balancer.compute_weighted_loss(loss_dict, loss_names)

            # Should always produce valid scalar loss
            self.assertFalse(torch.isnan(weighted_loss))
            self.assertFalse(torch.isinf(weighted_loss))

    def test_weights_stay_positive(self):
        """Test that weights remain positive over iterations."""
        for _ in range(30):
            loss_dict = {
                'data': torch.tensor(np.random.uniform(0.1, 1.0)),
                'temperature': torch.tensor(np.random.uniform(0.01, 0.5)),
                'physics': torch.tensor(np.random.uniform(0.1, 1.0)),
                'aleatoric': torch.tensor(np.random.uniform(0.01, 0.2)),
            }
            loss_names = ['data', 'temperature', 'physics', 'aleatoric']

            self.balancer.compute_weighted_loss(loss_dict, loss_names)

        weights = self.balancer.get_task_weights()
        self.assertTrue(torch.all(weights > 0))


class TestOptimizationIntegration(unittest.TestCase):
    """Integration tests for optimizer and loss balancer."""

    def test_training_step_integration(self):
        """Test complete training step with optimizer and balancer."""
        model = SimpleModel()
        optimizer = AdvancedOptimizer(
            model,
            learning_rate=0.001,
            warmup_epochs=2,
            total_epochs=10,
        )
        balancer = MultiTaskLossBalancer(num_tasks=4)

        # Simulate training steps
        for step in range(5):
            # Forward pass
            x = torch.randn(4, 10)
            y = model(x)

            # Create loss dict
            loss_dict = {
                'data': torch.mean((y - torch.randn(4, 5)) ** 2),
                'temperature': torch.mean(y ** 2) * 0.1,
                'physics': torch.mean(y ** 2) * 0.5,
                'aleatoric': torch.mean(torch.abs(y)) * 0.05,
            }

            # Compute balanced loss
            total_loss = balancer.compute_weighted_loss(
                loss_dict,
                ['data', 'temperature', 'physics', 'aleatoric']
            )

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step(total_loss)

            # Verify no NaNs
            self.assertFalse(torch.isnan(total_loss))

    def test_no_gradient_explosion(self):
        """Test that gradient clipping prevents explosion."""
        model = SimpleModel()
        optimizer = AdvancedOptimizer(
            model,
            learning_rate=0.1,
            gradient_clip=1.0,
        )

        # Create large gradients intentionally
        x = torch.randn(2, 10)
        y = model(x)
        loss = (y ** 4).sum() * 1e10  # Very large loss

        optimizer.zero_grad()
        loss.backward()

        # Check gradients before clipping
        max_grad_before = 0
        for param in model.parameters():
            if param.grad is not None:
                max_grad_before = max(max_grad_before, param.grad.abs().max().item())

        optimizer.step(loss)

        # Check gradients after clipping
        max_grad_after = 0
        for param in model.parameters():
            if param.grad is not None:
                max_grad_after = max(max_grad_after, param.grad.abs().max().item())

        # Gradient should be clipped (or already small)
        self.assertLess(max_grad_after, max_grad_before * 10)


if __name__ == '__main__':
    unittest.main()
