"""
Enhanced Training Script with High Epochs and Best Practices
Optimized for Google Colab with BraTS 2020 Dataset
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
from nibabel.nifti1 import Nifti1Image
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Optional, Tuple
import matplotlib.pyplot as plt

from config import Config, get_production_config
from model import create_production_model
from trainer import Trainer
from visualization import visualize_test_results, BrainSegmentationVisualizer


class BraTSDataset(Dataset):
    """
    Enhanced BraTS Dataset with proper preprocessing
    """

    def __init__(self,
                 data_dir: str,
                 mode: str = 'train',
                 patch_size: Tuple[int, int, int] = (128, 128, 128),
                 normalize: bool = True):

        self.data_dir = Path(data_dir) / mode
        self.patch_size = patch_size
        self.normalize = normalize
        self.mode = mode

        # Find all sample directories
        self.samples = sorted(list(self.data_dir.glob("sample_*")))

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in {self.data_dir}")

        print(f"✅ Loaded {len(self.samples)} samples for {mode} set")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

        # Load all 4 modalities
        modalities = ['t1', 't1ce', 't2', 'flair']
        images = []

        for mod in modalities:
            img_path = sample_dir / f"{mod}.nii.gz"
            if img_path.exists():
                img = Nifti1Image.from_filename(str(img_path)).get_fdata()
                images.append(img)
            else:
                # If modality missing, use zeros
                images.append(np.zeros(self.patch_size))

        # Stack modalities: [4, H, W, D]
        image = np.stack(images, axis=0).astype(np.float32)

        # Load segmentation
        seg_path = sample_dir / "seg.nii.gz"
        if seg_path.exists():
            segmentation = Nifti1Image.from_filename(
                str(seg_path)).get_fdata().astype(np.int64)
        else:
            segmentation = np.zeros(self.patch_size, dtype=np.int64)

        # Normalize intensity
        if self.normalize:
            for i in range(image.shape[0]):
                channel = image[i]
                if channel.max() > 0:
                    # Z-score normalization
                    mean = channel.mean()
                    std = channel.std()
                    if std > 0:
                        image[i] = (channel - mean) / std
                    else:
                        image[i] = channel - mean

        # Resize if needed
        if image.shape[1:] != self.patch_size:
            image = self._resize_volume(image, self.patch_size)
            segmentation = self._resize_volume(
                segmentation[np.newaxis, ...], self.patch_size)[0]

        # Convert to torch tensors
        image = torch.from_numpy(image).float()
        segmentation = torch.from_numpy(segmentation).long()

        return image, segmentation

    def _resize_volume(self, volume, target_size):
        """Resize 3D volume to target size"""
        # Simple resize - in production use proper resampling
        if len(volume.shape) == 4:  # [C, H, W, D]
            return np.array([self._resize_single(volume[i], target_size)
                             for i in range(volume.shape[0])])
        else:  # [H, W, D]
            return self._resize_single(volume, target_size)

    def _resize_single(self, volume, target_size):
        """Resize single channel"""
        try:
            from scipy.ndimage import zoom
        except ImportError:
            raise ImportError(
                "scipy is required for resizing. Please install with 'pip install scipy'.")

        factors = [t/s for t, s in zip(target_size, volume.shape)]
        return zoom(volume, factors, order=1)


def create_dataloaders(config: Config):
    """
    Create train, validation, and test dataloaders
    """
    print("\n" + "="*80)
    print("📂 CREATING DATALOADERS")
    print("="*80)

    # Create datasets
    train_dataset = BraTSDataset(
        data_dir=config.data.data_root,
        mode='train',
        patch_size=config.data.patch_size,
        normalize=config.data.normalize_intensity
    )

    val_dataset = BraTSDataset(
        data_dir=config.data.data_root,
        mode='val',
        patch_size=config.data.patch_size,
        normalize=config.data.normalize_intensity
    )

    test_dataset = BraTSDataset(
        data_dir=config.data.data_root,
        mode='test',
        patch_size=config.data.patch_size,
        normalize=config.data.normalize_intensity
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
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
        batch_size=1,  # Test one at a time for visualization
        shuffle=False,
        num_workers=0
    )

    print(f"\n✅ Dataloaders created:")
    print(
        f"   - Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"   - Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(
        f"   - Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    print("="*80)

    return train_loader, val_loader, test_loader


def train_enhanced(config: Optional[Config] = None,
                   resume_from: Optional[str] = None):
    """
    Enhanced training function with high epochs
    """
    # Use production config if not provided
    if config is None:
        config = get_production_config()

        # Override for high-performance training
        config.training.epochs = 500  # High epochs for best results
        config.training.batch_size = 2  # Adjust based on GPU memory
        config.training.learning_rate = 0.0001
        config.training.warmup_epochs = 20
        config.training.val_frequency = 5
        config.training.save_frequency = 10

        # Enable all optimizations
        config.system.mixed_precision = True
        config.system.compile_model = False  # Disable for Colab compatibility

    print("\n" + "="*80)
    print("🚀 HYBRID EFFICIENT nnU-NET - ENHANCED TRAINING")
    print("="*80)
    config.print_config()

    # Setup device
    device = config.get_device()
    print(f"\n💻 Using device: {device}")

    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(config)

    # Create model
    print("\n" + "="*80)
    print("🏗️  CREATING MODEL")
    print("="*80)

    # Pass correct arguments to create_production_model
    model = create_production_model(num_classes=config.model.num_classes)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)

    print(f"\n📊 Model Statistics:")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Model size: ~{total_params * 4 / 1e6:.2f} MB")

    # Create trainer
    # Trainer constructor: pass only required arguments
    trainer = Trainer(
        config,
        train_loader,
        val_loader,
        test_loader
    )

    # Resume from checkpoint if provided
    if resume_from:
        print(f"\n🔄 Resuming from checkpoint: {resume_from}")
        trainer.load_checkpoint(resume_from)

    # Train model
    print("\n" + "="*80)
    print("🎯 STARTING TRAINING")
    print("="*80)
    print(
        f"\n⏱️  Expected training time: ~{config.training.epochs * len(train_loader) * 2 / 3600:.1f} hours")
    print(f"   (Estimated based on ~2 seconds per batch)\n")

    history = trainer.train()

    # Plot training history
    print("\n" + "="*80)
    print("📊 PLOTTING TRAINING HISTORY")
    print("="*80)

    visualizer = BrainSegmentationVisualizer(
        save_dir=config.system.output_dir + "/visualizations/"
    )
    if history is not None:
        visualizer.plot_training_history(history, save=True)

    # Test and visualize results
    print("\n" + "="*80)
    print("🧪 TESTING MODEL")
    print("="*80)

    test_metrics = trainer.test()

    print(f"\n📊 Test Results:")
    print(f"   - Test Loss: {test_metrics['test_loss']:.4f}")
    print(f"   - Test Dice: {test_metrics['test_dice']:.4f}")

    # Generate visualizations
    print("\n" + "="*80)
    print("🎨 GENERATING VISUALIZATIONS")
    print("="*80)

    visualize_test_results(
        model=model,
        test_loader=test_loader,
        device=device,
        config=config,
        num_cases=min(10, len(test_loader))  # Visualize up to 10 cases
    )

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\n📁 All outputs saved to: {config.system.output_dir}")
    print(f"   - Checkpoints: {config.system.checkpoint_dir}")
    print(f"   - Logs: {config.system.log_dir}")
    print(f"   - Visualizations: {config.system.output_dir}/visualizations/")
    print("\n" + "="*80)

    return model, history, test_metrics


def quick_test():
    """
    Quick test with reduced epochs for validation
    """
    from config import get_quick_test_config

    config = get_quick_test_config()
    config.training.epochs = 10  # Quick test
    config.training.val_frequency = 2

    print("\n⚡ RUNNING QUICK TEST (10 epochs)")
    return train_enhanced(config)


if __name__ == "__main__":
    import sys

    # Check if running quick test
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        print("\n⚡ Quick Test Mode")
        model, history, metrics = quick_test()
    else:
        print("\n🚀 Full Training Mode (High Epochs)")
        model, history, metrics = train_enhanced()
