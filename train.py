"""
Main training script for Hybrid Efficient nnU-Net
Production-ready CNN training pipeline with comprehensive features
"""

from training_utils import set_seed, get_device_info
from trainer import Trainer
from config import Config, get_quick_test_config, get_production_config, get_fine_tuning_config
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import cast, Sized
import numpy as np
from pathlib import Path
import warnings

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for testing and demonstration
    Replace with your actual medical image dataset
    """

    def __init__(self, num_samples: int = 100, patch_size: tuple = (128, 128, 128)):
        self.num_samples = num_samples
        if not (isinstance(patch_size, tuple) and len(patch_size) == 3):
            raise ValueError(
                f"patch_size must be a 3D tuple (D, H, W), got {patch_size}")
        self.patch_size = patch_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate synthetic 4-channel medical image (T1, T1Gd, T2, T2-FLAIR)
        image = torch.randn(4, *self.patch_size)

        # Generate synthetic segmentation mask (3 classes: background, tumor core, whole tumor)
        target = torch.zeros(*self.patch_size, dtype=torch.long)

        # Add some random structures
        center = [s//2 for s in self.patch_size]
        for i in range(3):
            # Random tumor-like structures
            start_h = max(0, center[0] - 20 + np.random.randint(-10, 10))
            end_h = min(self.patch_size[0], start_h + 40)
            start_w = max(0, center[1] - 20 + np.random.randint(-10, 10))
            end_w = min(self.patch_size[1], start_w + 40)
            start_d = max(0, center[2] - 20 + np.random.randint(-10, 10))
            end_d = min(self.patch_size[2], start_d + 40)

            target[start_h:end_h, start_w:end_w, start_d:end_d] = i

        # Debug: Verify shapes before returning
        assert image.shape == (
            4, *self.patch_size), f"Image shape mismatch: {image.shape} vs expected (4, {self.patch_size})"
        assert target.shape == self.patch_size, f"Target shape mismatch: {target.shape} vs expected {self.patch_size}"

        return image.float(), target


def create_dataloaders(config):
    """
    Create data loaders for training, validation, and testing
    Replace with your actual data loading pipeline
    """
    print("📊 Creating synthetic datasets...")
    print("⚠️  Replace SyntheticDataset with your actual medical image dataset")

    # Create datasets
    train_dataset = SyntheticDataset(
        num_samples=200,
        patch_size=config.data.patch_size
    )
    val_dataset = SyntheticDataset(
        num_samples=50,
        patch_size=config.data.patch_size
    )
    test_dataset = SyntheticDataset(
        num_samples=30,
        patch_size=config.data.patch_size
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Test one at a time
        shuffle=False,
        num_workers=0  # Avoid multiprocessing issues during testing
    )

    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Validation samples: {len(val_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    print(f"✓ Batch size: {config.training.batch_size}")

    return train_loader, val_loader, test_loader


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(
        description='Hybrid Efficient nnU-Net Training')
    parser.add_argument('--config', type=str, default='production',
                        choices=['quick_test', 'production', 'fine_tuning'],
                        help='Configuration preset to use')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained model for fine-tuning')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs to train (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Weights & Biases project name')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--patch_size', type=int, nargs=3, default=None,
                        help='Patch size as three integers (D H W)')

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Create configuration
    print("⚙️  Setting up configuration...")

    # Assign config based on args.config
    if args.config == 'quick_test':
        config = get_quick_test_config()
        print("🚀 Using quick test configuration (5 epochs, small model)")
    elif args.config == 'production':
        config = get_production_config()
        print("🏭 Using production configuration (400 epochs, full model)")
    elif args.config == 'fine_tuning':
        if not args.pretrained:
            print("❌ Fine-tuning requires --pretrained argument")
            sys.exit(1)
        config = get_fine_tuning_config(args.pretrained)
        print("🔧 Using fine-tuning configuration")
    else:
        raise ValueError(f"Unknown config preset: {args.config}")

    # Override config with command line arguments
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.learning_rate = args.lr
    if args.wandb_project is not None:
        config.system.wandb_project = args.wandb_project
    if args.output_dir != 'outputs':
        config.system.output_dir = args.output_dir
        config.system.checkpoint_dir = os.path.join(
            args.output_dir, 'checkpoints')
        config.system.log_dir = os.path.join(args.output_dir, 'logs')
    if args.patch_size is not None:
        if len(args.patch_size) != 3:
            raise ValueError(
                f"patch_size must be three integers (D H W), got {args.patch_size}")
        config.data.patch_size = tuple(args.patch_size)

    # Print configuration
    config.print_config()

    # Setup device and check requirements
    device = get_device_info()

    # Check CUDA memory if using GPU
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        if total_memory < 8:
            print("⚠️  Warning: Less than 8GB GPU memory detected.")
            print("   Consider reducing batch size or using CPU training.")
            if config.training.batch_size > 1:
                config.training.batch_size = 1
                print(f"   Automatically reduced batch size to 1")

    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # Create trainer
    print("\n🎯 Initializing trainer...")
    trainer = Trainer(
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"📂 Resuming training from {args.resume}")
        trainer.load_checkpoint(args.resume, resume_training=True)

    # Start training
    print("\n" + "="*80)
    print("🚀 STARTING TRAINING")
    print(f"📊 Training samples: {len(cast(Sized, train_loader.dataset))}")
    print(f"📊 Validation samples: {len(cast(Sized, val_loader.dataset))}")
    print(f"📊 Test samples: {len(cast(Sized, test_loader.dataset))}")
    print(f"📊 Test samples: {len(cast(Sized, test_loader.dataset))}")
    print(
        f"⏱️  Estimated time per epoch: {len(train_loader) * config.training.batch_size * 2:.0f}s")
    print(f"🎯 Target epochs: {config.training.epochs}")
    print("="*80)

    try:
        # Run training
        trainer.train()

        # Save final model
        final_model_path = os.path.join(
            config.system.output_dir, 'final_model.pth')
        trainer.save_final_model(final_model_path)

        print("\n" + "="*80)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"💾 Final model saved to: {final_model_path}")
        print(f"📊 Best validation score: {trainer.best_val_score:.4f}")
        print(f"📂 Outputs directory: {config.system.output_dir}")
        print("="*80)

    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        # Save current state
        interrupt_path = os.path.join(
            config.system.checkpoint_dir, 'interrupted_model.pth')
        trainer.save_final_model(interrupt_path)
        print(f"💾 Current model saved to: {interrupt_path}")

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def demo_inference():
    """
    Demonstration of model inference
    """
    print("\n🔮 Running inference demonstration...")

    # Load a simple config for demo
    config = get_quick_test_config()

    # Create model
    from model import create_lightweight_model
    model = create_lightweight_model(num_classes=3)
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 4, 64, 64, 64)

    print(f"Input shape: {dummy_input.shape}")

    # Run inference
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")

        # Apply sigmoid for probabilities
        probs = torch.sigmoid(output)
        print(f"Output range: [{probs.min():.3f}, {probs.max():.3f}]")

        # Get predictions
        predictions = (probs > 0.5).float()
        print(f"Prediction shape: {predictions.shape}")
        print(
            f"Unique values in prediction: {torch.unique(predictions).tolist()}")

    print("✅ Inference demonstration completed!")


if __name__ == "__main__":
    print("🏥 HYBRID EFFICIENT nnU-NET TRAINING PIPELINE")
    print("=" * 80)
    print("🔬 Advanced Medical Image Segmentation with CNN-Transformer Hybrid")
    print("📋 Features:")
    print("   ✓ EfficientNet-style MBConv blocks")
    print("   ✓ Hybrid CNN-Transformer architecture")
    print("   ✓ Shuffle attention mechanism")
    print("   ✓ Deep supervision training")
    print("   ✓ Mixed precision training")
    print("   ✓ Advanced loss functions")
    print("   ✓ Comprehensive logging & monitoring")
    print("   ✓ Model checkpointing & early stopping")
    print("=" * 80)

    # Check if this is a demo run
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        demo_inference()
    else:
        main()


"""
🚀 USAGE EXAMPLES:

1. Quick Test (5 epochs, small model):
   python train.py --config quick_test

2. Production Training (400 epochs, full model):
   python train.py --config production --epochs 400 --batch_size 2

3. Fine-tuning from pretrained:
   python train.py --config fine_tuning --pretrained path/to/model.pth

4. Resume training from checkpoint:
   python train.py --resume checkpoints/latest_model.pth

5. Custom settings:
   python train.py --config production --lr 0.0005 --batch_size 1 --wandb_project my_project

6. Demo inference:
   python train.py demo

📋 DATASET INTEGRATION:
- Replace SyntheticDataset with your medical image dataset
- Implement proper data augmentation pipeline
- Add data normalization and preprocessing
- Handle different image formats (NIfTI, DICOM, etc.)

🔧 CUSTOMIZATION:
- Modify config.py for different architectures
- Add new loss functions in losses.py
- Extend trainer.py for custom training logic
- Add evaluation metrics in training_utils.py

💡 OPTIMIZATION TIPS:
- Use mixed precision training for 2x speedup
- Adjust batch size based on GPU memory
- Use gradient accumulation for larger effective batch sizes
- Enable torch.compile for additional speedup (PyTorch 2.0+)
"""
