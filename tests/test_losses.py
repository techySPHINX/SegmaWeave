"""
Unit tests for loss functions
"""

import pytest
import torch
import torch.nn as nn
from losses import (
    DiceLoss,
    FocalLoss,
    TverskyLoss,
    CombinedLoss,
    DeepSupervisionLoss,
    create_loss_function,
    get_deep_supervision_loss,
    soft_dice_score,
    hard_dice_score
)
from config import get_quick_test_config


class TestDiceLoss:
    """Test Dice Loss implementation"""

    def test_dice_loss_basic(self):
        """Test basic Dice loss computation"""
        criterion = DiceLoss(smooth=1.0)
        pred = torch.randn(2, 3, 32, 32, 32)
        target = torch.randint(0, 3, (2, 32, 32, 32))

        loss = criterion(pred, target)
        assert loss.item() >= 0.0
        assert loss.item() <= 1.0

    def test_dice_loss_perfect_prediction(self):
        """Test Dice loss with perfect prediction"""
        criterion = DiceLoss(smooth=1.0)

        # Create one-hot target
        target = torch.zeros(1, 3, 16, 16, 16)
        target[0, 1, :, :, :] = 1.0  # Class 1 everywhere

        # Create perfect prediction (high logits for class 1)
        pred = torch.zeros(1, 3, 16, 16, 16)
        pred[0, 1, :, :, :] = 10.0  # High logits for class 1

        loss = criterion(pred, target)
        assert loss.item() < 0.1  # Should be very low

    def test_dice_loss_with_class_weights(self):
        """Test Dice loss with class weights"""
        class_weights = torch.tensor([0.5, 1.0, 2.0])
        criterion = DiceLoss(smooth=1.0, class_weights=class_weights)

        pred = torch.randn(2, 3, 16, 16, 16)
        target = torch.randint(0, 3, (2, 16, 16, 16))

        loss = criterion(pred, target)
        assert loss.item() >= 0.0

    def test_dice_loss_backward(self):
        """Test Dice loss backward pass"""
        criterion = DiceLoss()
        pred = torch.randn(1, 3, 16, 16, 16, requires_grad=True)
        target = torch.randint(0, 3, (1, 16, 16, 16))

        loss = criterion(pred, target)
        loss.backward()

        assert pred.grad is not None


class TestFocalLoss:
    """Test Focal Loss implementation"""

    def test_focal_loss_basic(self):
        """Test basic Focal loss computation"""
        criterion = FocalLoss(alpha=1.0, gamma=2.0)
        pred = torch.randn(2, 3, 16, 16, 16)
        target = torch.randint(0, 3, (2, 16, 16, 16))

        loss = criterion(pred, target)
        assert loss.item() >= 0.0

    def test_focal_loss_different_gamma(self):
        """Test Focal loss with different gamma values"""
        pred = torch.randn(2, 3, 16, 16, 16)
        target = torch.randint(0, 3, (2, 16, 16, 16))

        loss_gamma_0 = FocalLoss(gamma=0.0)(pred, target)
        loss_gamma_2 = FocalLoss(gamma=2.0)(pred, target)
        loss_gamma_5 = FocalLoss(gamma=5.0)(pred, target)

        # All should be valid losses
        assert loss_gamma_0.item() >= 0.0
        assert loss_gamma_2.item() >= 0.0
        assert loss_gamma_5.item() >= 0.0


class TestTverskyLoss:
    """Test Tversky Loss implementation"""

    def test_tversky_loss_basic(self):
        """Test basic Tversky loss computation"""
        criterion = TverskyLoss(alpha=0.5, beta=0.5)
        pred = torch.randn(2, 3, 16, 16, 16)
        target = torch.randint(0, 3, (2, 16, 16, 16))

        loss = criterion(pred, target)
        assert loss.item() >= 0.0
        assert loss.item() <= 1.0

    def test_tversky_loss_fp_emphasis(self):
        """Test Tversky loss with false positive emphasis"""
        criterion_fp = TverskyLoss(alpha=0.7, beta=0.3)
        criterion_fn = TverskyLoss(alpha=0.3, beta=0.7)

        pred = torch.randn(2, 3, 16, 16, 16)
        target = torch.randint(0, 3, (2, 16, 16, 16))

        loss_fp = criterion_fp(pred, target)
        loss_fn = criterion_fn(pred, target)

        # Both should be valid
        assert loss_fp.item() >= 0.0
        assert loss_fn.item() >= 0.0


class TestCombinedLoss:
    """Test Combined Loss implementation"""

    def test_combined_dice_ce(self):
        """Test combined Dice + CE loss"""
        criterion = CombinedLoss(
            loss_types=['dice', 'ce'],
            loss_weights=[1.0, 1.0]
        )

        pred = torch.randn(2, 3, 16, 16, 16)
        target = torch.randint(0, 3, (2, 16, 16, 16))

        loss = criterion(pred, target)
        assert loss.item() >= 0.0

    def test_combined_dice_focal(self):
        """Test combined Dice + Focal loss"""
        criterion = CombinedLoss(
            loss_types=['dice', 'focal'],
            loss_weights=[1.0, 0.5]
        )

        pred = torch.randn(2, 3, 16, 16, 16)
        target = torch.randint(0, 3, (2, 16, 16, 16))

        loss = criterion(pred, target)
        assert loss.item() >= 0.0

    def test_combined_all_losses(self):
        """Test combined Dice + CE + Focal"""
        criterion = CombinedLoss(
            loss_types=['dice', 'ce', 'focal'],
            loss_weights=[1.0, 0.5, 0.5]
        )

        pred = torch.randn(2, 3, 16, 16, 16)
        target = torch.randint(0, 3, (2, 16, 16, 16))

        loss = criterion(pred, target)
        assert loss.item() >= 0.0


class TestDeepSupervisionLoss:
    """Test Deep Supervision Loss implementation"""

    def test_deep_supervision_basic(self):
        """Test basic deep supervision loss"""
        base_loss = DiceLoss()
        criterion = DeepSupervisionLoss(
            base_loss=base_loss,
            weights=[1.0, 0.5, 0.25, 0.125]
        )

        # Create multiple outputs at different scales
        outputs = [
            torch.randn(2, 3, 32, 32, 32),  # Main output
            torch.randn(2, 3, 32, 32, 32),  # DS output 1
            torch.randn(2, 3, 32, 32, 32),  # DS output 2
            torch.randn(2, 3, 32, 32, 32),  # DS output 3
        ]
        target = torch.randint(0, 3, (2, 32, 32, 32))

        loss = criterion(outputs, target)
        assert loss.item() >= 0.0

    def test_deep_supervision_single_output(self):
        """Test deep supervision with single output"""
        base_loss = DiceLoss()
        criterion = DeepSupervisionLoss(
            base_loss=base_loss,
            weights=[1.0]
        )

        outputs = [torch.randn(2, 3, 16, 16, 16)]
        target = torch.randint(0, 3, (2, 16, 16, 16))

        loss = criterion(outputs, target)
        assert loss.item() >= 0.0

    def test_deep_supervision_backward(self):
        """Test deep supervision backward pass"""
        base_loss = DiceLoss()
        criterion = DeepSupervisionLoss(
            base_loss=base_loss,
            weights=[1.0, 0.5]
        )

        outputs = [
            torch.randn(1, 3, 16, 16, 16, requires_grad=True),
            torch.randn(1, 3, 16, 16, 16, requires_grad=True)
        ]
        target = torch.randint(0, 3, (1, 16, 16, 16))

        loss = criterion(outputs, target)
        loss.backward()

        for output in outputs:
            assert output.grad is not None


class TestLossFactory:
    """Test loss factory functions"""

    def test_create_dice_loss(self):
        """Test creating Dice loss from config"""
        # Create a config mock with loss_type and training.dice_smooth
        class TrainingMock:
            def __init__(self, dice_smooth=1.0):
                self.dice_smooth = dice_smooth

        class ConfigMock:
            def __init__(self, loss_type):
                self.loss_type = loss_type
                self.training = TrainingMock()
        config = ConfigMock('dice')
        criterion = create_loss_function(config)
        assert isinstance(criterion, DiceLoss)

    def test_create_focal_loss(self):
        """Test creating Focal loss from config"""
        class TrainingMock:
            def __init__(self, dice_smooth=1.0):
                self.dice_smooth = dice_smooth

        class ConfigMock:
            def __init__(self, loss_type):
                self.loss_type = loss_type
                self.training = TrainingMock()
        config = ConfigMock('focal')
        criterion = create_loss_function(config)
        assert isinstance(criterion, FocalLoss)

    def test_create_combined_loss(self):
        """Test creating Combined loss from config"""
        class TrainingMock:
            def __init__(self, dice_smooth=1.0):
                self.dice_smooth = dice_smooth

        class ConfigMock:
            def __init__(self, loss_type):
                self.loss_type = loss_type
                self.training = TrainingMock()
        config = ConfigMock('combined')
        criterion = create_loss_function(config)
        assert isinstance(criterion, CombinedLoss)

    def test_get_deep_supervision_loss(self):
        """Test wrapping loss with deep supervision"""
        base_loss = DiceLoss()
        criterion = get_deep_supervision_loss(base_loss)

        assert isinstance(criterion, DeepSupervisionLoss)


class TestDiceScoreUtilities:
    """Test Dice score utility functions"""

    def test_soft_dice_score(self):
        """Test soft Dice score computation"""
        pred = torch.rand(1, 3, 16, 16, 16)
        target = torch.rand(1, 3, 16, 16, 16)

        score = soft_dice_score(pred, target)
        assert 0.0 <= score.item() <= 1.0

    def test_hard_dice_score(self):
        """Test hard Dice score computation"""
        pred = torch.rand(1, 3, 16, 16, 16)
        target = torch.randint(0, 2, (1, 3, 16, 16, 16)).float()

        score = hard_dice_score(pred, target, threshold=0.5)
        assert 0.0 <= score.item() <= 1.0

    def test_perfect_dice_score(self):
        """Test Dice score with perfect prediction"""
        pred = torch.ones(1, 16, 16, 16)
        target = torch.ones(1, 16, 16, 16)

        score = soft_dice_score(pred, target)
        assert score.item() > 0.99  # Should be ~1.0

    def test_zero_dice_score(self):
        """Test Dice score with no overlap"""
        pred = torch.ones(1, 16, 16, 16)
        target = torch.zeros(1, 16, 16, 16)

        score = soft_dice_score(pred, target)
        assert score.item() < 0.1  # Should be ~0.0


class TestLossEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_target(self):
        """Test loss with empty target"""
        criterion = DiceLoss()
        pred = torch.randn(1, 3, 16, 16, 16)
        target = torch.zeros(1, 16, 16, 16, dtype=torch.long)

        loss = criterion(pred, target)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_single_class_target(self):
        """Test loss with single class in target"""
        criterion = DiceLoss()
        pred = torch.randn(2, 3, 16, 16, 16)
        target = torch.ones(2, 16, 16, 16, dtype=torch.long)  # All class 1

        loss = criterion(pred, target)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        criterion = DiceLoss(smooth=1.0)

        # Very large predictions
        pred_large = torch.randn(1, 3, 8, 8, 8) * 100
        target = torch.randint(0, 3, (1, 8, 8, 8))

        loss_large = criterion(pred_large, target)
        assert not torch.isnan(loss_large)
        assert not torch.isinf(loss_large)

        # Very small predictions
        pred_small = torch.randn(1, 3, 8, 8, 8) * 0.001
        loss_small = criterion(pred_small, target)
        assert not torch.isnan(loss_small)
        assert not torch.isinf(loss_small)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
