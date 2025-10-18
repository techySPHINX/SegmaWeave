"""
Configuration file for Hybrid Efficient nnU-Net Training
Contains all hyperparameters, paths, and training configurations
"""

import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    in_channels: int = 4
    num_classes: int = 3
    base_features: int = 32
    use_transformer: bool = True
    use_attention: bool = True
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1

    # Transformer specific
    num_heads: int = 8
    mlp_ratio: float = 4.0

    # MBConv specific
    expand_ratio: int = 4
    se_ratio: float = 0.25


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Basic training params
    epochs: int = 400
    batch_size: int = 2  # Adjust based on GPU memory
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    warmup_epochs: int = 10

    # Loss configuration
    dice_smooth: float = 1.0
    # [1.0, 0.5, 0.25, 0.125]
    deep_supervision_weights: Optional[List[float]] = None

    # Optimization
    grad_clip_norm: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.999

    # Scheduler
    poly_power: float = 0.9
    min_lr: float = 1e-7

    # Validation
    val_frequency: int = 5  # Validate every N epochs
    save_frequency: int = 20  # Save checkpoint every N epochs

    def __post_init__(self):
        if self.deep_supervision_weights is None:
            self.deep_supervision_weights = [1.0, 0.5, 0.25, 0.125]


@dataclass
class DataConfig:
    """Data loading and augmentation configuration"""
    # Data paths
    data_root: str = "data/"
    train_dir: str = "train/"
    val_dir: str = "val/"
    test_dir: str = "test/"

    # Data properties
    patch_size: Tuple[int, int, int] = (128, 128, 128)
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Data loading
    num_workers: int = 0  # Set to 0 for Windows compatibility
    pin_memory: bool = True
    prefetch_factor: int = 2

    # Augmentation parameters
    rotation_range: float = 15.0
    scale_range: Tuple[float, float] = (0.85, 1.15)
    noise_std: float = 0.1
    blur_sigma: Tuple[float, float] = (0.5, 1.5)
    contrast_range: Tuple[float, float] = (0.75, 1.25)

    # Normalization
    normalize_intensity: bool = True
    intensity_clipping: Tuple[float, float] = (0.5, 99.5)  # percentiles


@dataclass
class SystemConfig:
    """System and hardware configuration"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True  # Use automatic mixed precision
    compile_model: bool = False  # Use torch.compile (PyTorch 2.0+)
    deterministic: bool = False  # For reproducible results
    benchmark: bool = True  # CuDNN benchmark for speed

    # Paths
    output_dir: str = "outputs/"
    checkpoint_dir: str = "checkpoints/"
    log_dir: str = "logs/"
    pretrained_path: Optional[str] = None

    # Logging
    log_level: str = "INFO"
    wandb_project: Optional[str] = None  # Set to use Weights & Biases
    wandb_entity: Optional[str] = None

    # Memory management
    max_memory_cache: float = 0.8  # 80% of GPU memory for caching
    empty_cache_frequency: int = 10  # Clear cache every N batches


class Config:
    """Main configuration class combining all configs"""

    def __init__(self,
                 model_config: Optional[ModelConfig] = None,
                 training_config: Optional[TrainingConfig] = None,
                 data_config: Optional[DataConfig] = None,
                 system_config: Optional[SystemConfig] = None):

        self.model = model_config or ModelConfig()
        self.training = training_config or TrainingConfig()
        self.data = data_config or DataConfig()
        self.system = system_config or SystemConfig()

        # Create output directories
        self._create_directories()

        # Set random seeds for reproducibility
        if self.system.deterministic:
            self._set_deterministic()

    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.system.output_dir,
            self.system.checkpoint_dir,
            self.system.log_dir,
            self.data.data_root
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def _set_deterministic(self):
        """Set deterministic behavior for reproducibility"""
        import random
        import numpy as np

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def get_device(self) -> torch.device:
        """Get the training device"""
        return torch.device(self.system.device)

    def print_config(self):
        """Print configuration summary"""
        print("=" * 80)
        print("HYBRID EFFICIENT nnU-NET CONFIGURATION")
        print("=" * 80)

        print(f"\n🔧 MODEL CONFIG:")
        print(f"  - Input channels: {self.model.in_channels}")
        print(f"  - Number of classes: {self.model.num_classes}")
        print(f"  - Base features: {self.model.base_features}")
        print(f"  - Use Transformer: {self.model.use_transformer}")
        print(f"  - Use Attention: {self.model.use_attention}")
        print(f"  - Dropout rate: {self.model.dropout_rate}")

        print(f"\n🚀 TRAINING CONFIG:")
        print(f"  - Epochs: {self.training.epochs}")
        print(f"  - Batch size: {self.training.batch_size}")
        print(f"  - Learning rate: {self.training.learning_rate}")
        print(f"  - Weight decay: {self.training.weight_decay}")
        print(f"  - Warmup epochs: {self.training.warmup_epochs}")
        print(f"  - Grad clip norm: {self.training.grad_clip_norm}")

        print(f"\n📊 DATA CONFIG:")
        print(f"  - Patch size: {self.data.patch_size}")
        print(f"  - Num workers: {self.data.num_workers}")
        print(f"  - Pin memory: {self.data.pin_memory}")
        print(f"  - Normalize intensity: {self.data.normalize_intensity}")

        print(f"\n💻 SYSTEM CONFIG:")
        print(f"  - Device: {self.system.device}")
        print(f"  - Mixed precision: {self.system.mixed_precision}")
        print(f"  - Compile model: {self.system.compile_model}")
        print(f"  - Deterministic: {self.system.deterministic}")

        print("=" * 80)


# Default configuration instance
default_config = Config()


# Configuration presets for different scenarios
def get_quick_test_config() -> Config:
    """Configuration for quick testing"""
    model_config = ModelConfig(base_features=16, use_transformer=False)
    training_config = TrainingConfig(epochs=5, batch_size=1, val_frequency=1)
    data_config = DataConfig(patch_size=(64, 64, 64), num_workers=0)
    system_config = SystemConfig(mixed_precision=False, compile_model=False)

    return Config(model_config, training_config, data_config, system_config)


def get_production_config() -> Config:
    """Configuration for production training"""
    model_config = ModelConfig(
        base_features=32, use_transformer=True, use_attention=True)
    training_config = TrainingConfig(
        epochs=400, batch_size=2, learning_rate=0.001)
    data_config = DataConfig(patch_size=(128, 128, 128), num_workers=0)
    system_config = SystemConfig(mixed_precision=True, compile_model=True)

    return Config(model_config, training_config, data_config, system_config)


def get_fine_tuning_config(pretrained_path: str) -> Config:
    """Configuration for fine-tuning pretrained model"""
    model_config = ModelConfig(base_features=32, use_transformer=True)
    training_config = TrainingConfig(
        epochs=100, learning_rate=0.0001, warmup_epochs=5)
    data_config = DataConfig(patch_size=(128, 128, 128))
    system_config = SystemConfig(
        mixed_precision=True, pretrained_path=pretrained_path)

    return Config(model_config, training_config, data_config, system_config)
