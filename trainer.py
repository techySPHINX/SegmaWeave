"""
Main trainer class for Hybrid Efficient nnU-Net
Production-ready training pipeline with all optimizations
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import wandb

from model import create_model
from losses import create_loss_function, get_deep_supervision_loss
from training_utils import (
    create_optimizer, create_scheduler, EarlyStopping, MetricsTracker,
    ModelCheckpoint, GradientClipping, AverageMeter, ProgressMeter,
    setup_logging, count_parameters, get_device_info, get_memory_usage
)


class Trainer:

    def _setup_loss_function(self):
        """Setup loss function with deep supervision"""
        base_loss = create_loss_function(self.config)
        if hasattr(self.config.training, 'deep_supervision_weights'):
            weights = self.config.training.deep_supervision_weights
            return get_deep_supervision_loss(base_loss, weights)
        return base_loss

    def _setup_wandb(self):
        """Initialize Weights & Biases logging if configured"""
        if self.config.system.wandb_project:
            wandb.init(
                project=self.config.system.wandb_project,
                entity=self.config.system.wandb_entity,
                config=self.config.__dict__,
                name=f"hybrid_nnunet_{int(time.time())}"
            )
            wandb.watch(self.model, log_freq=100)

    def _log_model_info(self):
        """Log model architecture information"""
        total_params, trainable_params = count_parameters(self.model)
        self.logger.info("=" * 80)
        self.logger.info("MODEL INFORMATION")
        self.logger.info("=" * 80)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Model size: {total_params * 4 / 1e6:.1f} MB")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(
            f"Mixed precision: {self.config.system.mixed_precision}")
        self.logger.info("=" * 80)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        dice_scores = AverageMeter('Dice', ':6.2f')
        progress = ProgressMeter(
            len(self.train_loader),
            [batch_time, data_time, losses, dice_scores],
            prefix=f"Epoch: [{self.current_epoch}]"
        )
        end = time.time()
        for batch_idx, (images, targets) in enumerate(self.train_loader):
            data_time.update(time.time() - end)
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            if batch_idx == 0 and self.current_epoch == 1:
                self.logger.info(
                    f"DEBUG: images shape = {images.shape}, targets shape = {targets.shape}")
            loss, dice = self._train_step(images, targets)
            losses.update(loss.item(), images.size(0))
            dice_scores.update(dice, images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % 50 == 0:
                progress.display(batch_idx)
                self.logger.info(get_memory_usage())
            self.global_step += 1
            if batch_idx % self.config.system.empty_cache_frequency == 0:
                torch.cuda.empty_cache()
        return {
            'loss': losses.avg,
            'dice': dice_scores.avg
        }
    """
    Production-ready trainer for medical image segmentation

    Features:
    - Mixed precision training
    - Deep supervision
    - Progressive training
    - Comprehensive logging
    - Model checkpointing
    - Early stopping
    - Gradient clipping
    - Memory optimization
    """

    def __init__(self, config, train_loader: DataLoader, val_loader: DataLoader,
                 test_loader: Optional[DataLoader] = None):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Setup device and logging
        self.device = get_device_info()
        self.logger = setup_logging(
            config.system.log_dir, config.system.log_level)

        # Initialize model
        self.model = create_model(config).to(self.device)

        # Setup training components
        self.optimizer = create_optimizer(self.model, config)
        self.scheduler = create_scheduler(self.optimizer, config)
        self.criterion = self._setup_loss_function()

        # Training utilities
        self.scaler = GradScaler() if config.system.mixed_precision else None
        self.grad_clipper = GradientClipping(config.training.grad_clip_norm)
        self.early_stopping = EarlyStopping(patience=30, min_delta=0.001)
        self.metrics_tracker = MetricsTracker(['loss', 'dice', 'iou'])
        self.checkpoint_manager = ModelCheckpoint(config.system.checkpoint_dir)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_score = float('inf')

        # Initialize Weights & Biases if configured
        self._setup_wandb()

        # Log model information
        self._log_model_info()

    def _train_step(self, images: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, float]:
        # Debug: Print target shape for diagnosis
        if self.current_epoch == 0:
            print(f"DEBUG: targets shape in _train_step: {targets.shape}")
        """Single training step with mixed precision"""

        self.optimizer.zero_grad()

        if self.scaler is not None:
            # Mixed precision training
            with autocast():
                outputs = self.model(images)

                if isinstance(outputs, tuple):
                    # Deep supervision
                    main_output, ds_outputs = outputs
                    all_outputs = [main_output] + ds_outputs
                    loss = self.criterion(all_outputs, targets)
                else:
                    loss = self.criterion(outputs, targets)

            # Backward pass with scaling
            self.scaler.scale(loss).backward()

            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            grad_norm = self.grad_clipper(self.model)

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

        else:
            # Standard precision training
            outputs = self.model(images)

            if isinstance(outputs, tuple):
                main_output, ds_outputs = outputs
                all_outputs = [main_output] + ds_outputs
                loss = self.criterion(all_outputs, targets)
            else:
                loss = self.criterion(outputs, targets)

            loss.backward()
            grad_norm = self.grad_clipper(self.model)
            self.optimizer.step()

        # Calculate Dice score for monitoring
        with torch.no_grad():
            if isinstance(outputs, tuple):
                pred = torch.sigmoid(outputs[0])
            else:
                pred = torch.sigmoid(outputs)

            dice = self._calculate_dice_score(pred, targets)

        return loss, dice

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()

        val_losses = AverageMeter('Val Loss', ':.4e')
        val_dice = AverageMeter('Val Dice', ':6.2f')
        val_iou = AverageMeter('Val IoU', ':6.2f')

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                # Forward pass (no deep supervision in validation)
                outputs = self.model(images, return_deep_supervision=False)
                loss = self.criterion(outputs, targets)

                # Calculate metrics
                pred = torch.sigmoid(outputs)
                dice = self._calculate_dice_score(pred, targets)
                iou = self._calculate_iou_score(pred, targets)

                # Update metrics
                val_losses.update(loss.item(), images.size(0))
                val_dice.update(dice, images.size(0))
                val_iou.update(iou, images.size(0))

        return {
            'loss': val_losses.avg,
            'dice': val_dice.avg,
            'iou': val_iou.avg
        }

    def _calculate_dice_score(self, pred: torch.Tensor, target: torch.Tensor,
                              threshold: float = 0.5) -> float:
        """Calculate Dice score"""
        pred_binary = (pred > threshold).float()

        # Handle different target formats
        if target.dim() == pred.dim() - 1:
            target = torch.nn.functional.one_hot(
                target, num_classes=pred.size(1))
            target = target.permute(0, -1, 1, 2, 3).float()

        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum()

        if union == 0:
            return 1.0  # Perfect score when both are empty

        dice = (2.0 * intersection) / union
        return dice.item()

    def _calculate_iou_score(self, pred: torch.Tensor, target: torch.Tensor,
                             threshold: float = 0.5) -> float:
        """Calculate IoU score"""
        pred_binary = (pred > threshold).float()

        if target.dim() == pred.dim() - 1:
            target = torch.nn.functional.one_hot(
                target, num_classes=pred.size(1))
            target = target.permute(0, -1, 1, 2, 3).float()

        intersection = (pred_binary * target).sum()
        union = pred_binary.sum() + target.sum() - intersection

        if union == 0:
            return 1.0

        iou = intersection / union
        return iou.item()

    def train(self):
        """Main training loop"""
        self.logger.info("🚀 Starting training...")
        self.logger.info(f"Training for {self.config.training.epochs} epochs")

        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch

            # Training phase
            train_metrics = self.train_epoch()
            self.metrics_tracker.update('train', **train_metrics)

            # Validation phase
            if epoch % self.config.training.val_frequency == 0:
                val_metrics = self.validate_epoch()
                self.metrics_tracker.update('val', **val_metrics)

                # Learning rate scheduling
                if hasattr(self.scheduler, 'step'):
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()

                # Check for best model
                current_score = val_metrics['loss']
                is_best = current_score < self.best_val_score
                if is_best:
                    self.best_val_score = current_score
                    self.logger.info(
                        f"🎯 New best validation score: {current_score:.4f}")

                # Save checkpoint
                if epoch % self.config.training.save_frequency == 0 or is_best:
                    self.checkpoint_manager.save(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        score=current_score,
                        metrics=val_metrics,
                        is_best=is_best
                    )

                # Early stopping
                if self.early_stopping(current_score):
                    self.logger.info(
                        f"⏹ Early stopping triggered at epoch {epoch}")
                    break

                # Log to wandb
                if self.config.system.wandb_project:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_metrics['loss'],
                        'train_dice': train_metrics['dice'],
                        'val_loss': val_metrics['loss'],
                        'val_dice': val_metrics['dice'],
                        'val_iou': val_metrics['iou'],
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    })

                # Log progress
                self.logger.info(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Train Dice: {train_metrics['dice']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val Dice: {val_metrics['dice']:.4f} | "
                    f"Val IoU: {val_metrics['iou']:.4f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )

            else:
                # Only update scheduler if no validation
                if hasattr(self.scheduler, 'step'):
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(train_metrics['loss'])
                    else:
                        self.scheduler.step()

                self.logger.info(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Train Dice: {train_metrics['dice']:.4f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                )

            # End epoch tracking
            self.metrics_tracker.end_epoch()

        self.logger.info("✅ Training completed!")

        # Final evaluation on test set if available
        if self.test_loader is not None:
            self.logger.info("🧪 Running final evaluation on test set...")
            test_metrics = self.test()
            self.logger.info(f"Final test results: {test_metrics}")

        # Close wandb
        if self.config.system.wandb_project:
            wandb.finish()

    def test(self) -> Dict[str, float]:
        """Evaluate on test set"""
        if self.test_loader is None:
            self.logger.warning("No test loader provided")
            return {}

        self.model.eval()
        test_metrics = AverageMeter('Test Dice', ':6.2f')

        with torch.no_grad():
            for images, targets in self.test_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                outputs = self.model(images)
                pred = torch.sigmoid(outputs)
                dice = self._calculate_dice_score(pred, targets)

                test_metrics.update(dice, images.size(0))

        return {'test_dice': test_metrics.avg}

    def save_final_model(self, path: str):
        """Save final trained model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'metrics': self.metrics_tracker.history,
            'best_score': self.best_val_score
        }, path)

        self.logger.info(f"💾 Final model saved to {path}")

    def load_checkpoint(self, path: str, resume_training: bool = True):
        """Load checkpoint and optionally resume training"""
        checkpoint = self.checkpoint_manager.load(
            self.model, path, self.optimizer, self.scheduler, str(self.device)
        )

        if resume_training:
            self.current_epoch = checkpoint['epoch'] + 1
            self.best_val_score = checkpoint.get('score', float('inf'))

        return checkpoint
