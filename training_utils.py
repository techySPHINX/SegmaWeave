"""
Training utilities and helper functions
Optimized for production-ready CNN training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from pathlib import Path
import time
import logging

from losses import create_loss_function, get_deep_supervision_loss


class WarmupPolyLR(_LRScheduler):
    """
    Polynomial learning rate scheduler with warmup
    Commonly used for medical image segmentation
    """

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 power: float = 0.9, min_lr: float = 1e-7, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Polynomial decay
            decay_epochs = self.total_epochs - self.warmup_epochs
            current_epoch = self.last_epoch - self.warmup_epochs
            factor = (1 - current_epoch / decay_epochs) ** self.power
            return [
                max(self.min_lr, base_lr * factor)
                for base_lr in self.base_lrs
            ]


class CosineAnnealingWarmupLR(_LRScheduler):
    """
    Cosine annealing with warmup
    Alternative to polynomial decay
    """

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 min_lr: float = 1e-7, last_epoch: int = -1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / \
                (self.total_epochs - self.warmup_epochs)
            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 *
                (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


def create_optimizer(model: nn.Module, config) -> optim.Optimizer:
    """
    Create optimizer from configuration
    """
    # Separate parameters that should not have weight decay
    no_decay = []
    decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # No weight decay for bias, normalization, and positional embeddings
        if ('bias' in name or
            'norm' in name or
            'bn' in name or
            'pos_embed' in name or
                'cls_token' in name):
            no_decay.append(param)
        else:
            decay.append(param)

    param_groups = [
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': config.training.weight_decay}
    ]

    # Create optimizer
    if hasattr(config.training, 'optimizer') and config.training.optimizer == 'adamw':
        optimizer = optim.AdamW(
            param_groups,
            lr=config.training.learning_rate,
            betas=(config.training.beta1, config.training.beta2),
            eps=1e-8
        )
    elif hasattr(config.training, 'optimizer') and config.training.optimizer == 'sgd':
        optimizer = optim.SGD(
            param_groups,
            lr=config.training.learning_rate,
            momentum=0.9,
            nesterov=True
        )
    else:
        # Default to AdamW
        optimizer = optim.AdamW(
            param_groups,
            lr=config.training.learning_rate,
            betas=(config.training.beta1, config.training.beta2),
            eps=1e-8
        )

    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config) -> _LRScheduler:
    """
    Create learning rate scheduler from configuration
    """
    if hasattr(config.training, 'scheduler') and config.training.scheduler == 'cosine':
        return CosineAnnealingWarmupLR(
            optimizer,
            warmup_epochs=config.training.warmup_epochs,
            total_epochs=config.training.epochs,
            min_lr=config.training.min_lr
        )
    else:
        # Default to polynomial
        return WarmupPolyLR(
            optimizer,
            warmup_epochs=config.training.warmup_epochs,
            total_epochs=config.training.epochs,
            power=config.training.poly_power,
            min_lr=config.training.min_lr
        )


class EarlyStopping:
    """
    Early stopping to prevent overfitting
    """

    def __init__(self, patience: int = 20, min_delta: float = 0.0001, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

        if mode == 'min':
            self.is_better = lambda score, best: score < (best - min_delta)
        else:
            self.is_better = lambda score, best: score > (best + min_delta)

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class MetricsTracker:
    """
    Track training and validation metrics
    """

    def __init__(self, metrics: List[str] = ['loss', 'dice', 'accuracy']):
        self.metrics = metrics
        self.history = {f'train_{metric}': [] for metric in metrics}
        self.history.update({f'val_{metric}': [] for metric in metrics})
        self.current_epoch = {}

    def update(self, phase: str, **kwargs):
        """Update metrics for current epoch"""
        for metric, value in kwargs.items():
            key = f"{phase}_{metric}"
            if key in self.history:
                self.current_epoch[key] = value

    def end_epoch(self):
        """Save current epoch metrics to history"""
        for key, value in self.current_epoch.items():
            if key in self.history:
                self.history[key].append(value)
        self.current_epoch = {}

    def get_best(self, metric: str, phase: str = 'val') -> Tuple[float, int]:
        """Get best value and epoch for a metric"""
        key = f"{phase}_{metric}"
        if key not in self.history or not self.history[key]:
            return float('nan'), -1
        values = self.history[key]
        if metric in ['loss']:
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        return float(values[best_idx]), int(best_idx)

    def get_last(self, metric: str, phase: str = 'val') -> float:
        """Get last value for a metric"""
        key = f"{phase}_{metric}"
        if key not in self.history or not self.history[key]:
            return float('nan')
        return float(self.history[key][-1])


class ModelCheckpoint:
    """
    Save model checkpoints during training
    """

    def __init__(self, checkpoint_dir: str, save_best: bool = True,
                 save_every: int = 20, max_checkpoints: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best = save_best
        self.save_every = save_every
        self.max_checkpoints = max_checkpoints
        self.best_score = None
        self.checkpoints = []

    def save(self, model: nn.Module, optimizer: optim.Optimizer,
             scheduler: _LRScheduler, epoch: int, score: float,
             metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'score': score,
            'metrics': metrics,
            'timestamp': time.time()
        }

        # Save best model
        if is_best and self.save_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model (score: {score:.4f})")

        # Save periodic checkpoint
        if epoch % self.save_every == 0 or is_best:
            checkpoint_path = self.checkpoint_dir / \
                f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, checkpoint_path)
            self.checkpoints.append(checkpoint_path)

            # Remove old checkpoints if too many
            if len(self.checkpoints) > self.max_checkpoints:
                old_checkpoint = self.checkpoints.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()

        # Always save latest
        latest_path = self.checkpoint_dir / 'latest_model.pth'
        torch.save(checkpoint, latest_path)

    def load(self, model: nn.Module, path: str,
             optimizer: Optional[optim.Optimizer] = None,
             scheduler: Optional[_LRScheduler] = None,
             device: str = 'cpu') -> Dict:
        """Load model checkpoint"""

        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"✓ Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint


class GradientClipping:
    """
    Gradient clipping utility
    """

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def __call__(self, model: nn.Module) -> float:
        """Apply gradient clipping and return gradient norm"""
        return torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type
        ).item()


class AverageMeter:
    """
    Computes and stores the average and current value
    """

    def __init__(self, name: str = '', fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """
    Display progress during training
    """

    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches: int):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def setup_logging(log_dir: str, log_level: str = 'INFO') -> logging.Logger:
    """
    Setup logging configuration
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Create logger
    logger = logging.getLogger('training')
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    file_handler = logging.FileHandler(str(log_path / 'training.log'))
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    return logger


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device_info():
    """
    Get device information
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🔥 Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")

    return device


def initialize_weights(model: nn.Module, init_type: str = 'kaiming'):
    """
    Initialize model weights
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            if init_type == 'kaiming':
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu')
            elif init_type == 'xavier':
                nn.init.xavier_normal_(module.weight)
            elif init_type == 'normal':
                nn.init.normal_(module.weight, std=0.02)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, (nn.InstanceNorm3d, nn.GroupNorm, nn.BatchNorm3d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


def freeze_model(model: nn.Module, freeze: bool = True):
    """
    Freeze/unfreeze model parameters
    """
    for param in model.parameters():
        param.requires_grad = not freeze

    status = "frozen" if freeze else "unfrozen"
    print(f"Model parameters {status}")


def get_memory_usage():
    """
    Get current GPU memory usage
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        return f"GPU Memory: {allocated:.2f}GB allocated, {cached:.2f}GB cached"
    return "CPU mode - no GPU memory tracking"
