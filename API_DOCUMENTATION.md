# API Documentation

## Complete API Reference for Hybrid Efficient nnU-Net

---

## Table of Contents

1. [Model API](#1-model-api)
2. [Configuration API](#2-configuration-api)
3. [Loss Functions API](#3-loss-functions-api)
4. [Training API](#4-training-api)
5. [Utilities API](#5-utilities-api)

---

## 1. Model API

### 1.1 HybridEfficientnnUNet

Main model class for 3D medical image segmentation.

```python
from model import HybridEfficientnnUNet

model = HybridEfficientnnUNet(
    in_channels=4,
    num_classes=3,
    base_features=32,
    use_transformer=True,
    use_attention=True,
    dropout_rate=0.1,
    drop_path_rate=0.1,
    deep_supervision=True
)
```

#### Parameters

| Parameter          | Type  | Default | Description                                        |
| ------------------ | ----- | ------- | -------------------------------------------------- |
| `in_channels`      | int   | 4       | Number of input channels                           |
| `num_classes`      | int   | 3       | Number of output classes                           |
| `base_features`    | int   | 32      | Base number of features (multiplied at each stage) |
| `use_transformer`  | bool  | True    | Enable Transformer blocks in deeper layers         |
| `use_attention`    | bool  | True    | Enable Shuffle Attention                           |
| `dropout_rate`     | float | 0.1     | Dropout probability                                |
| `drop_path_rate`   | float | 0.1     | Stochastic depth rate                              |
| `deep_supervision` | bool  | True    | Enable deep supervision during training            |

#### Methods

##### forward()

```python
output = model(x, return_features=False)
```

**Parameters:**

- `x` (torch.Tensor): Input tensor of shape `[B, C, H, W, D]`
- `return_features` (bool): Whether to return intermediate features

**Returns:**

- If `training=True` and `deep_supervision=True`: `(main_output, [ds_output1, ds_output2, ds_output3])`
- If `training=False` or `deep_supervision=False`: `output`
- If `return_features=True`: `(output, features_dict)`

**Example:**

```python
import torch
x = torch.randn(2, 4, 128, 128, 128)
output = model(x)  # Shape: [2, 3, 128, 128, 128]
```

##### load_pretrained_encoder()

```python
model.load_pretrained_encoder(pretrained_path, strict=False)
```

**Parameters:**

- `pretrained_path` (str): Path to pretrained checkpoint
- `strict` (bool): Whether to strictly enforce key matching

**Example:**

```python
model.load_pretrained_encoder("checkpoints/encoder_pretrained.pth")
```

##### freeze_encoder()

```python
model.freeze_encoder(freeze=True)
```

**Parameters:**

- `freeze` (bool): If True, freezes encoder parameters; if False, unfreezes

**Example:**

```python
# Freeze encoder for fine-tuning decoder only
model.freeze_encoder(True)
```

##### get_feature_maps()

```python
features = model.get_feature_maps(x)
```

**Returns:**

- `dict`: Dictionary with keys `['stem', 'enc1', 'enc2', 'enc3', 'enc4', 'bottleneck']`

**Example:**

```python
x = torch.randn(1, 4, 128, 128, 128)
features = model.get_feature_maps(x)
print(features['bottleneck'].shape)  # [1, 512, 8, 8, 8]
```

---

### 1.2 Model Factory Functions

#### create_model()

```python
from model import create_model
from config import get_production_config

config = get_production_config()
model = create_model(config)
```

**Parameters:**

- `config` (Config): Configuration object

**Returns:**

- `HybridEfficientnnUNet`: Initialized model

---

#### create_lightweight_model()

```python
from model import create_lightweight_model

model = create_lightweight_model(num_classes=3)
```

**Returns:**

- Lightweight model (16 base features, no Transformer)

---

#### create_production_model()

```python
from model import create_production_model

model = create_production_model(num_classes=3)
```

**Returns:**

- Production model (32 base features, full Transformer)

---

#### create_large_model()

```python
from model import create_large_model

model = create_large_model(num_classes=3)
```

**Returns:**

- Large model (48 base features, full Transformer)

---

## 2. Configuration API

### 2.1 Config Classes

#### ModelConfig

```python
from config import ModelConfig

model_cfg = ModelConfig(
    in_channels=4,
    num_classes=3,
    base_features=32,
    use_transformer=True,
    use_attention=True,
    dropout_rate=0.1
)
```

**Parameters:**

| Parameter         | Type  | Default | Description        |
| ----------------- | ----- | ------- | ------------------ |
| `in_channels`     | int   | 4       | Input channels     |
| `num_classes`     | int   | 3       | Output classes     |
| `base_features`   | int   | 32      | Base feature count |
| `use_transformer` | bool  | True    | Use Transformer    |
| `use_attention`   | bool  | True    | Use attention      |
| `dropout_rate`    | float | 0.1     | Dropout rate       |

---

#### TrainingConfig

```python
from config import TrainingConfig

training_cfg = TrainingConfig(
    epochs=400,
    batch_size=2,
    learning_rate=0.001,
    weight_decay=1e-4,
    warmup_epochs=10
)
```

**Parameters:**

| Parameter        | Type  | Default | Description               |
| ---------------- | ----- | ------- | ------------------------- |
| `epochs`         | int   | 400     | Number of training epochs |
| `batch_size`     | int   | 2       | Batch size                |
| `learning_rate`  | float | 0.001   | Initial learning rate     |
| `weight_decay`   | float | 1e-4    | Weight decay              |
| `warmup_epochs`  | int   | 10      | Warmup epochs             |
| `grad_clip_norm` | float | 1.0     | Gradient clipping norm    |

---

#### DataConfig

```python
from config import DataConfig

data_cfg = DataConfig(
    data_root="data/",
    patch_size=(128, 128, 128),
    num_workers=4
)
```

**Parameters:**

| Parameter     | Type                 | Default         | Description        |
| ------------- | -------------------- | --------------- | ------------------ |
| `data_root`   | str                  | "data/"         | Data directory     |
| `patch_size`  | Tuple[int, int, int] | (128, 128, 128) | 3D patch size      |
| `num_workers` | int                  | 0               | DataLoader workers |
| `pin_memory`  | bool                 | True            | Pin memory for GPU |

---

#### SystemConfig

```python
from config import SystemConfig

system_cfg = SystemConfig(
    device="cuda",
    mixed_precision=True,
    compile_model=False
)
```

**Parameters:**

| Parameter         | Type | Default        | Description          |
| ----------------- | ---- | -------------- | -------------------- |
| `device`          | str  | "cuda"         | Device (cuda/cpu)    |
| `mixed_precision` | bool | True           | Use AMP              |
| `compile_model`   | bool | False          | Use torch.compile    |
| `checkpoint_dir`  | str  | "checkpoints/" | Checkpoint directory |
| `log_dir`         | str  | "logs/"        | Log directory        |

---

### 2.2 Configuration Presets

#### get_quick_test_config()

```python
from config import get_quick_test_config

config = get_quick_test_config()
```

**Returns:** Config for quick testing (5 epochs, small model)

---

#### get_production_config()

```python
from config import get_production_config

config = get_production_config()
```

**Returns:** Config for production training (400 epochs, full model)

---

#### get_fine_tuning_config()

```python
from config import get_fine_tuning_config

config = get_fine_tuning_config("checkpoints/pretrained.pth")
```

**Parameters:**

- `pretrained_path` (str): Path to pretrained model

**Returns:** Config for fine-tuning (100 epochs, low LR)

---

## 3. Loss Functions API

### 3.1 DiceLoss

```python
from losses import DiceLoss

criterion = DiceLoss(
    smooth=1.0,
    reduction='mean',
    class_weights=None,
    ignore_index=-100
)
```

**Parameters:**

| Parameter       | Type   | Default | Description       |
| --------------- | ------ | ------- | ----------------- |
| `smooth`        | float  | 1.0     | Smoothing factor  |
| `reduction`     | str    | 'mean'  | Reduction method  |
| `class_weights` | Tensor | None    | Per-class weights |
| `ignore_index`  | int    | -100    | Index to ignore   |

**Example:**

```python
pred = torch.randn(2, 3, 128, 128, 128)
target = torch.randint(0, 3, (2, 128, 128, 128))
loss = criterion(pred, target)
```

---

### 3.2 FocalLoss

```python
from losses import FocalLoss

criterion = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')
```

**Parameters:**

- `alpha` (float): Balancing factor
- `gamma` (float): Focusing parameter
- `reduction` (str): Reduction method

---

### 3.3 TverskyLoss

```python
from losses import TverskyLoss

criterion = TverskyLoss(alpha=0.5, beta=0.5, smooth=1.0)
```

**Parameters:**

- `alpha` (float): False positive weight
- `beta` (float): False negative weight
- `smooth` (float): Smoothing factor

---

### 3.4 CombinedLoss

```python
from losses import CombinedLoss

criterion = CombinedLoss(
    loss_types=['dice', 'ce'],
    loss_weights=[1.0, 1.0]
)
```

**Parameters:**

- `loss_types` (List[str]): Loss types to combine
- `loss_weights` (List[float]): Weight for each loss

---

### 3.5 DeepSupervisionLoss

```python
from losses import DeepSupervisionLoss, DiceLoss

base_loss = DiceLoss()
criterion = DeepSupervisionLoss(
    base_loss=base_loss,
    weights=[1.0, 0.5, 0.25, 0.125]
)
```

**Example:**

```python
# During training with deep supervision
outputs = [main_out, ds_out1, ds_out2, ds_out3]
loss = criterion(outputs, target)
```

---

### 3.6 Loss Factory

#### create_loss_function()

```python
from losses import create_loss_function

criterion = create_loss_function(config)
```

**Returns:** Loss function based on config

---

#### get_deep_supervision_loss()

```python
from losses import get_deep_supervision_loss, DiceLoss

base_loss = DiceLoss()
criterion = get_deep_supervision_loss(base_loss, weights=[1.0, 0.5, 0.25])
```

---

## 4. Training API

### 4.1 Trainer

```python
from trainer import Trainer

trainer = Trainer(
    config=config,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader
)
```

#### Methods

##### train()

```python
trainer.train()
```

**Description:** Main training loop with automatic validation, checkpointing, and early stopping

---

##### train_epoch()

```python
metrics = trainer.train_epoch()
```

**Returns:** `Dict[str, float]` with keys `['loss', 'dice']`

---

##### validate_epoch()

```python
metrics = trainer.validate_epoch()
```

**Returns:** `Dict[str, float]` with keys `['loss', 'dice', 'iou']`

---

##### test()

```python
test_metrics = trainer.test()
```

**Returns:** `Dict[str, float]` with test metrics

---

##### save_final_model()

```python
trainer.save_final_model("outputs/final_model.pth")
```

---

##### load_checkpoint()

```python
checkpoint = trainer.load_checkpoint(
    path="checkpoints/best_model.pth",
    resume_training=True
)
```

**Parameters:**

- `path` (str): Checkpoint path
- `resume_training` (bool): Whether to resume training state

---

## 5. Utilities API

### 5.1 Optimizer Creation

#### create_optimizer()

```python
from training_utils import create_optimizer

optimizer = create_optimizer(model, config)
```

**Returns:** AdamW or SGD optimizer

---

### 5.2 Scheduler Creation

#### create_scheduler()

```python
from training_utils import create_scheduler

scheduler = create_scheduler(optimizer, config)
```

**Returns:** WarmupPolyLR or CosineAnnealingWarmupLR scheduler

---

### 5.3 Metrics and Tracking

#### MetricsTracker

```python
from training_utils import MetricsTracker

tracker = MetricsTracker(['loss', 'dice', 'iou'])
tracker.update('train', loss=0.5, dice=0.85)
tracker.end_epoch()
best_dice, best_epoch = tracker.get_best('dice', 'val')
```

---

#### AverageMeter

```python
from training_utils import AverageMeter

losses = AverageMeter('Loss', ':.4e')
losses.update(0.5, n=2)  # Update with value and batch size
print(losses)  # Loss 0.5000
```

---

### 5.4 Model Utilities

#### count_parameters()

```python
from training_utils import count_parameters

total, trainable = count_parameters(model)
print(f"Total: {total:,}, Trainable: {trainable:,}")
```

---

#### set_seed()

```python
from training_utils import set_seed

set_seed(42)  # For reproducibility
```

---

#### get_device_info()

```python
from training_utils import get_device_info

device = get_device_info()
print(device)  # cuda or cpu
```

---

#### get_memory_usage()

```python
from training_utils import get_memory_usage

memory_info = get_memory_usage()
print(memory_info)
```

---

### 5.5 Logging

#### setup_logging()

```python
from training_utils import setup_logging

logger = setup_logging(log_dir="logs/", log_level="INFO")
logger.info("Training started")
```

---

## 6. Complete Training Example

```python
# 1. Setup configuration
from config import get_production_config
config = get_production_config()

# 2. Create data loaders
from train import create_dataloaders
train_loader, val_loader, test_loader = create_dataloaders(config)

# 3. Create model
from model import create_model
model = create_model(config)

# 4. Setup trainer
from trainer import Trainer
trainer = Trainer(config, train_loader, val_loader, test_loader)

# 5. Train
trainer.train()

# 6. Test
test_results = trainer.test()
print(f"Test Dice: {test_results['test_dice']:.4f}")

# 7. Save model
trainer.save_final_model("outputs/final_model.pth")
```

---

## 7. Inference Example

```python
import torch
from model import create_production_model

# Load model
model = create_production_model(num_classes=3)
checkpoint = torch.load("checkpoints/best_model.pth")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare input
x = torch.randn(1, 4, 128, 128, 128).cuda()

# Inference
with torch.no_grad():
    output = model(x)
    pred = torch.argmax(output, dim=1)  # [1, 128, 128, 128]

print(f"Prediction shape: {pred.shape}")
```

---

## 8. Custom Dataset Example

```python
from torch.utils.data import Dataset, DataLoader

class MedicalImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # Load your data paths here

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        # Load image and label
        image = load_nifti(self.data_paths[idx])  # [4, H, W, D]
        label = load_nifti(self.label_paths[idx])  # [H, W, D]

        if self.transform:
            image, label = self.transform(image, label)

        return image, label

# Create dataloader
dataset = MedicalImageDataset("data/train")
loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
```

---

## 9. Error Handling

All functions include proper error handling and logging. Common exceptions:

```python
try:
    model = create_model(config)
except Exception as e:
    print(f"Error creating model: {e}")

try:
    trainer.train()
except KeyboardInterrupt:
    print("Training interrupted")
    trainer.save_final_model("outputs/interrupted_model.pth")
```

---

## 10. Type Hints

All functions include complete type hints:

```python
def create_model(config: Config) -> HybridEfficientnnUNet:
    ...

def train_epoch(self) -> Dict[str, float]:
    ...

def get_device_info() -> torch.device:
    ...
```

---

For more examples and use cases, see the [Training Guide](docs/TRAINING_GUIDE.md) and [examples/](examples/) directory.
