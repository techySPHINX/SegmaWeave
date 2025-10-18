"""
Loss functions for medical image segmentation
Optimized implementations with numerical stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
import numpy as np


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks
    Handles multi-class with optional class weighting
    """

    def __init__(self, smooth: float = 1.0, reduction: str = 'mean',
                 class_weights: Optional[torch.Tensor] = None, ignore_index: int = -100):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.class_weights = class_weights
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions [B, C, H, W, D] (logits or probabilities)
            target: Ground truth [B, C, H, W, D] (one-hot) or [B, H, W, D] (class indices)
        """
        # Handle different target formats
        if target.dim() == pred.dim() - 1:
            # Convert class indices to one-hot
            target = F.one_hot(target, num_classes=pred.size(
                1)).permute(0, -1, 1, 2, 3).float()

        # Apply softmax to predictions if needed (assuming logits)
        if pred.max() > 1 or pred.min() < 0:
            pred = F.softmax(pred, dim=1)

        # Flatten tensors
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)  # [B, C, N]
        target_flat = target.view(target.size(
            0), target.size(1), -1)  # [B, C, N]

        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum(dim=2)  # [B, C]
        pred_sum = pred_flat.sum(dim=2)  # [B, C]
        target_sum = target_flat.sum(dim=2)  # [B, C]

        # Dice coefficient per class
        dice = (2. * intersection + self.smooth) / \
            (pred_sum + target_sum + self.smooth)

        # Apply class weights if provided
        if self.class_weights is not None:
            dice = dice * self.class_weights.to(dice.device)

        # Calculate loss (1 - dice)
        dice_loss = 1 - dice

        # Apply reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions [B, C, H, W, D]
            target: Ground truth [B, H, W, D] (class indices)
        """
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice Loss
    Controls false positives and false negatives independently
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Convert to one-hot if needed
        if target.dim() == pred.dim() - 1:
            target = F.one_hot(target, num_classes=pred.size(
                1)).permute(0, -1, 1, 2, 3).float()

        # Apply softmax to predictions
        if pred.max() > 1 or pred.min() < 0:
            pred = F.softmax(pred, dim=1)

        # Flatten
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)
        target_flat = target.view(target.size(0), target.size(1), -1)

        # Calculate TP, FP, FN
        TP = (pred_flat * target_flat).sum(dim=2)
        FP = (pred_flat * (1 - target_flat)).sum(dim=2)
        FN = ((1 - pred_flat) * target_flat).sum(dim=2)

        # Tversky index
        tversky = (TP + self.smooth) / (TP + self.alpha *
                                        FP + self.beta * FN + self.smooth)

        return (1 - tversky).mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function mixing multiple loss types
    Commonly used: Dice + Cross-Entropy or Dice + Focal
    """

    def __init__(self, loss_types: List[str] = ['dice', 'ce'],
                 loss_weights: List[float] = [1.0, 1.0],
                 dice_smooth: float = 1.0,
                 focal_alpha: float = 1.0,
                 focal_gamma: float = 2.0):
        super().__init__()
        self.loss_types = loss_types
        self.loss_weights = loss_weights

        # Initialize loss functions
        self.losses = nn.ModuleDict()

        if 'dice' in loss_types:
            self.losses['dice'] = DiceLoss(smooth=dice_smooth)

        if 'ce' in loss_types:
            self.losses['ce'] = nn.CrossEntropyLoss()

        if 'focal' in loss_types:
            self.losses['focal'] = FocalLoss(
                alpha=focal_alpha, gamma=focal_gamma)

        if 'tversky' in loss_types:
            self.losses['tversky'] = TverskyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        for loss_type, weight in zip(self.loss_types, self.loss_weights):
            if loss_type == 'dice':
                loss = self.losses['dice'](pred, target)
            elif loss_type == 'ce':
                # CrossEntropy expects class indices
                if target.dim() == pred.dim():
                    target = target.argmax(dim=1)
                loss = self.losses['ce'](pred, target)
            elif loss_type == 'focal':
                if target.dim() == pred.dim():
                    target = target.argmax(dim=1)
                loss = self.losses['focal'](pred, target)
            elif loss_type == 'tversky':
                loss = self.losses['tversky'](pred, target)
            else:
                continue

            total_loss = total_loss + weight * loss

        return total_loss


class DeepSupervisionLoss(nn.Module):
    """
    Deep supervision loss for multi-scale training
    """

    def __init__(self, base_loss: nn.Module, weights: List[float] = [1.0, 0.5, 0.25, 0.125]):
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights

    def forward(self, predictions: List[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: List of predictions at different scales [main_pred, ds1, ds2, ds3]
            target: Ground truth
        """
        total_loss = torch.tensor(
            0.0, device=target.device, dtype=target.dtype)

        for i, (pred, weight) in enumerate(zip(predictions, self.weights)):
            # Resize prediction to match target if needed
            if pred.shape != target.shape:
                if len(target.shape[2:]) != 3:
                    raise ValueError(
                        f"Target spatial shape must be 3D, got {target.shape[2:]}")
                pred = F.interpolate(pred, size=target.shape[2:],
                                     mode='trilinear', align_corners=False)

            loss = self.base_loss(pred, target)
            total_loss = total_loss + weight * loss

        return total_loss


class AdaptiveLoss(nn.Module):
    """
    Adaptive loss that adjusts weights based on training progress
    Simplified version without complex buffer operations
    """

    def __init__(self, base_loss: nn.Module, adaptation_rate: float = 0.01):
        super().__init__()
        self.base_loss = base_loss
        self.adaptation_rate = adaptation_rate
        self.loss_ema = None  # Exponential moving average of loss
        self.alpha = 0.1  # EMA smoothing factor

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.base_loss(pred, target)

        # Update exponential moving average
        if self.loss_ema is None:
            self.loss_ema = loss.detach()
        else:
            self.loss_ema = self.alpha * loss.detach() + (1 - self.alpha) * self.loss_ema

        # Adaptive weighting based on recent performance
        if self.loss_ema is not None:
            current_loss = loss.detach()
            adaptation_factor = 1.0 + self.adaptation_rate * \
                (current_loss - self.loss_ema) / self.loss_ema
            loss = loss * adaptation_factor.clamp(0.5, 2.0)

        return loss


def create_loss_function(config) -> nn.Module:
    """
    Factory function to create loss function from config
    """
    if hasattr(config, 'loss_type'):
        loss_type = config.loss_type
    else:
        loss_type = 'combined'  # Default

    if loss_type == 'dice':
        return DiceLoss(smooth=getattr(config.training, 'dice_smooth', 1.0))

    elif loss_type == 'focal':
        return FocalLoss(alpha=1.0, gamma=2.0)

    elif loss_type == 'tversky':
        # Focus more on false negatives
        return TverskyLoss(alpha=0.7, beta=0.3)

    elif loss_type == 'combined':
        return CombinedLoss(
            loss_types=['dice', 'ce'],
            loss_weights=[1.0, 1.0],
            dice_smooth=getattr(config.training, 'dice_smooth', 1.0)
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def get_deep_supervision_loss(base_loss: nn.Module, weights: Optional[List[float]] = None) -> DeepSupervisionLoss:
    """
    Wrap base loss with deep supervision
    """
    if weights is None:
        weights = [1.0, 0.5, 0.25, 0.125]

    return DeepSupervisionLoss(base_loss, weights)


# Utility functions for loss computation
def compute_class_weights(dataset_stats: dict, method: str = 'inverse_freq') -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets

    Args:
        dataset_stats: Dictionary with class frequencies
        method: 'inverse_freq' or 'effective_num'

    Returns:
        Class weights tensor
    """
    if method == 'inverse_freq':
        total_samples = sum(dataset_stats.values())
        weights = []
        for class_id in sorted(dataset_stats.keys()):
            weight = total_samples / \
                (len(dataset_stats) * dataset_stats[class_id])
            weights.append(weight)

        weights = torch.tensor(weights, dtype=torch.float32)
        return weights / weights.sum() * len(weights)  # Normalize

    elif method == 'effective_num':
        beta = 0.999  # Smoothing parameter
        weights = []
        for class_id in sorted(dataset_stats.keys()):
            effective_num = (1 - beta ** dataset_stats[class_id]) / (1 - beta)
            weights.append(1.0 / effective_num)

        weights = torch.tensor(weights, dtype=torch.float32)
        return weights / weights.sum() * len(weights)

    else:
        raise ValueError(f"Unknown weighting method: {method}")


def soft_dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Compute soft Dice score (differentiable)
    """
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / \
        (pred_flat.sum() + target_flat.sum() + smooth)

    return dice


def hard_dice_score(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Compute hard Dice score (for evaluation)
    """
    pred_binary = (pred > threshold).float()
    return soft_dice_score(pred_binary, target)
