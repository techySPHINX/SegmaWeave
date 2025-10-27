# Brain Tumor Segmentation - Transfer Learning Guide

## 🎯 What's New in the Notebook

The `Brain_Tumor_transfer_learning.ipynb` notebook now includes **three advanced features** for maximizing transfer learning performance:

### 1. 🔷 3D Patch-Based Training with 2D Weight Inflation

**Why?** Leverages pretrained 2D ImageNet weights for 3D volumetric segmentation.

**How it works:**

- Takes a 2D pretrained model (e.g., EfficientNet trained on ImageNet)
- Inflates Conv2d → Conv3d by replicating weights along depth dimension
- Enables true 3D context while benefiting from 2D pretraining

**Usage:**

```python
# Convert any 2D model to 3D
model_3d = inflate_2d_model_to_3d(model)

# Use 3D patch dataset
train_ds = BraTS3DPatchDataset(data_root, split='train', patch_size=(32,64,64))
```

**When to use:** When you need full 3D context and have sufficient GPU memory.

---

### 2. 🔷 Production-Ready SMP (Segmentation Models PyTorch)

**Why?** Battle-tested encoder-decoder architectures with 100+ pretrained backbones.

**Features:**

- Multiple architectures: Unet, UnetPlusPlus, FPN, DeepLabV3Plus, etc.
- Pretrained encoders: EfficientNet, ResNet, ResNeXt, DenseNet, MobileNet, etc.
- Production-ready code, well-maintained library

**Usage:**

```python
model = create_smp_model(
    architecture='UnetPlusPlus',
    encoder_name='efficientnet-b2',
    encoder_weights='imagenet',
    in_channels=4,
    num_classes=4
)
```

**When to use:** For production deployments or when you want proven architectures.

---

### 3. 🔷 Automated Encoder Freezing/Unfreezing Schedule

**Why?** Progressive unfreezing improves transfer learning by preventing catastrophic forgetting.

**Strategy:**

- **Phase 1** (epochs 0-5): Freeze encoder, train decoder only
- **Phase 2** (epochs 5-10): Unfreeze top encoder layers with low LR
- **Phase 3** (epochs 10+): Unfreeze all layers with differential LR

**Usage:**

```python
freezer = EncoderFreezer(model, freeze_epochs=5, unfreeze_epochs=10)
optimizer = Adam(freezer.get_param_groups(base_lr=1e-4))

# In training loop:
for epoch in range(epochs):
    freezer.step(epoch)  # Auto-manages freezing
    # ... train ...
```

**When to use:** Always! Especially with limited training data.

---

## 🚀 Complete Pipeline (All Features Combined)

```python
# One function to rule them all
model, history = run_complete_advanced_pipeline(
    use_3d=False,      # Set True for 3D patch training
    use_smp=True,      # Set True for SMP models
    use_freezing=True  # Set True for freezing schedule
)
```

---

## 📊 Performance Expectations

| Configuration              | Expected Dice Score | GPU Memory | Training Time |
| -------------------------- | ------------------- | ---------- | ------------- |
| Baseline (2D, no freezing) | 0.65-0.70           | ~4 GB      | 1-2 hrs       |
| + Freezing schedule        | 0.70-0.75           | ~4 GB      | 1-2 hrs       |
| + SMP EfficientNet-B2      | 0.75-0.80           | ~6 GB      | 2-3 hrs       |
| + Augmentations            | 0.80-0.85           | ~6 GB      | 2-3 hrs       |
| + 3D patches + large model | 0.85-0.90           | ~10 GB     | 4-6 hrs       |

---

## 🎓 Recommended Workflow for BraTS

### Quick Test (Colab Free Tier)

```python
IMG_SIZE = 128
EPOCHS = 10
model = create_smp_model('Unet', 'efficientnet-b0', in_channels=4, num_classes=4)
freezer = EncoderFreezer(model, freeze_epochs=3, unfreeze_epochs=7)
train_ds = BraTSSlicesAugmented(data_root, split='train', use_augmentation=True)
```

### Production Quality (Colab Pro / Local GPU)

```python
IMG_SIZE = 224
EPOCHS = 50
model = create_smp_model('UnetPlusPlus', 'efficientnet-b3', in_channels=4, num_classes=4)
freezer = EncoderFreezer(model, freeze_epochs=10, unfreeze_epochs=25)
train_ds = BraTSSlicesAugmented(data_root, split='train', use_augmentation=True)
# Add ReduceLROnPlateau, early stopping
```

### Maximum Performance (Multi-GPU)

```python
IMG_SIZE = 256
EPOCHS = 100
use_3d = True
model = create_smp_model('DeepLabV3Plus', 'efficientnet-b5', in_channels=4, num_classes=4)
# 3D inflation if needed
# Multi-GPU training with DDP
```

---

## 💡 Tips & Tricks

### Memory Optimization

- Reduce `IMG_SIZE`: 224 → 128 → 96
- Reduce batch size to 1
- Use gradient accumulation (4 steps = effective batch of 4)
- Use smaller backbone: b2 → b0
- Enable gradient checkpointing (SMP models support this)

### Training Stability

- Always use freezing schedule with pretrained encoders
- Start with low learning rate (1e-4 or 1e-5)
- Use warmup for first few epochs
- Monitor validation Dice, save best model only

### Data Quality

- Use all 4 modalities (T1, T1ce, T2, FLAIR) for best results
- Normalize per-channel (already implemented)
- Enable augmentations (spatial + intensity)
- Balance classes if severe imbalance

### Architecture Selection

- **Fast prototyping**: Unet + efficientnet-b0
- **Best speed/accuracy**: UnetPlusPlus + efficientnet-b2
- **Maximum accuracy**: DeepLabV3Plus + efficientnet-b4/b5
- **Limited memory**: FPN + mobilenet_v2

---

## 📦 Installation Commands

```bash
# Core dependencies
pip install timm nibabel nilearn matplotlib tqdm scikit-image scipy

# Advanced features
pip install segmentation-models-pytorch  # SMP models
pip install albumentations  # Augmentations
pip install torchinfo  # Model summaries

# Optional (for GIF visualization)
pip install gif_your_nifti
```

---

## 🔍 Troubleshooting

**Issue**: "Out of memory" error

- **Solution**: Reduce IMG_SIZE or batch_size, use gradient accumulation

**Issue**: Model not improving after freezing phase

- **Solution**: Check that encoder is actually unfrozen (print param.requires_grad)

**Issue**: SMP model gives error with in_channels=4

- **Solution**: Some encoders don't support in_channels != 3. Use timm-based encoders or custom model.

**Issue**: 3D inflation crashes

- **Solution**: 3D requires much more memory. Start with small patch_size (32,64,64)

---

## 📚 Additional Resources

- **SMP Documentation**: https://smp.readthedocs.io/
- **Timm Models**: https://github.com/rwightman/pytorch-image-models
- **BraTS Dataset**: https://www.med.upenn.edu/cbica/brats2020/
- **Medical Segmentation Best Practices**: https://github.com/MIC-DKFZ/nnUNet

---

## ✅ Next Steps

1. Run the notebook sections 1-4 to prepare data
2. Choose your configuration based on available resources
3. Train the model with your chosen advanced features
4. Evaluate on test set (Section 17)
5. Visualize predictions (Section 16)
6. Save best checkpoint and use for inference

Good luck with your brain tumor segmentation! 🧠✨
