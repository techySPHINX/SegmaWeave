import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class DepthwiseSeparableConv3D(nn.Module):
    """Depthwise Separable Convolution: 3x3x3 depthwise + 1x1x1 pointwise"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MultiHeadSelfAttention3D(nn.Module):
    """3D Multi-Head Self-Attention for hybrid CNN-Transformer architecture"""
    def __init__(self, channels, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert channels % num_heads == 0, "Channels must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Conv3d(channels, channels * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv3d(channels, channels, 1)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, C, H, W, D = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W * D)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, num_heads, HWD, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(2, 3).reshape(B, C, H, W, D)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TransformerBlock3D(nn.Module):
    """Transformer block with 3D convolutions for local-global feature learning"""
    def __init__(self, channels, num_heads=8, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.attn = MultiHeadSelfAttention3D(channels, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=channels)
        mlp_hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, mlp_hidden, 1),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv3d(mlp_hidden, channels, 1),
            nn.Dropout(drop)
        )
    
    def forward(self, x):
        # Self-attention with residual
        x = x + self.attn(self.norm1(x))
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x


class ShuffleAttention3D(nn.Module):
    """3D Shuffle Attention Module as described in the paper"""
    def __init__(self, channels, groups=4):
        super().__init__()
        self.groups = groups
        assert channels % (2 * groups) == 0, "Channels must be divisible by 2*groups"
        
        self.group_channels = channels // (2 * groups)
        
        # Channel attention branch
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc_channel = nn.Conv3d(self.group_channels, self.group_channels, 1, bias=True)
        
        # Spatial attention branch
        self.gn_spatial = nn.GroupNorm(num_groups=self.group_channels, num_channels=self.group_channels)
        self.fc_spatial = nn.Conv3d(self.group_channels, self.group_channels, 1, bias=True)
    
    def forward(self, x):
        b, c, h, w, d = x.size()
        
        # Split into groups
        x = x.view(b, self.groups, -1, h, w, d)
        
        # Split each group into two branches
        x_0, x_1 = x.chunk(2, dim=2)
        x_0 = x_0.contiguous().view(b, -1, h, w, d)
        x_1 = x_1.contiguous().view(b, -1, h, w, d)
        
        # Channel attention
        s = self.gap(x_0)
        s = self.fc_channel(s)
        x_0 = torch.sigmoid(s) * x_0
        
        # Spatial attention
        x_1 = self.gn_spatial(x_1)
        x_1 = self.fc_spatial(x_1)
        x_1 = torch.sigmoid(x_1) * x_1
        
        # Concatenate branches
        out = torch.cat([x_0, x_1], dim=1)
        
        # Channel shuffle
        out = out.view(b, self.groups, -1, h, w, d)
        out = out.transpose(1, 2).contiguous()
        out = out.view(b, -1, h, w, d)
        
        return out


class MBConvBlock(nn.Module):
    """MobileNet-style Inverted Residual Block (inspired by EfficientNet)"""
    def __init__(self, in_channels, out_channels, expand_ratio=4, stride=1, se_ratio=0.25):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        
        hidden_dim = in_channels * expand_ratio
        
        layers = []
        
        # Expansion phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv3d(in_channels, hidden_dim, 1, bias=False),
                nn.InstanceNorm3d(hidden_dim, affine=True),
                nn.SiLU(inplace=True)
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv3d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.InstanceNorm3d(hidden_dim, affine=True),
            nn.SiLU(inplace=True)
        ])
        
        # Squeeze-and-Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(hidden_dim, se_channels, 1),
                nn.SiLU(inplace=True),
                nn.Conv3d(se_channels, hidden_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.se = None
        
        # Projection phase
        layers.extend([
            nn.Conv3d(hidden_dim, out_channels, 1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        
        if self.se is not None:
            # Apply SE before final projection
            se_weight = self.se(out)
            out = out * se_weight
        
        if self.use_residual:
            out = out + x
        
        return out


class HybridEncoderBlock(nn.Module):
    """Hybrid encoder block combining CNN (MBConv) and Transformer layers"""
    def __init__(self, in_channels, out_channels, use_transformer=True, num_heads=8, use_attention=True):
        super().__init__()
        
        # MBConv layers for local feature extraction
        self.mbconv1 = MBConvBlock(in_channels, out_channels, expand_ratio=4)
        self.mbconv2 = MBConvBlock(out_channels, out_channels, expand_ratio=4)
        
        # Optional transformer for global context
        self.use_transformer = use_transformer
        if use_transformer:
            self.transformer = TransformerBlock3D(out_channels, num_heads=num_heads)
        
        # Optional shuffle attention
        self.use_attention = use_attention
        if use_attention:
            self.attention = ShuffleAttention3D(out_channels)
    
    def forward(self, x):
        x = self.mbconv1(x)
        x = self.mbconv2(x)
        
        if self.use_transformer:
            x = self.transformer(x)
        
        if self.use_attention:
            x = self.attention(x)
        
        return x


class DecoderBlock(nn.Module):
    """Decoder block with transpose conv and hybrid blocks"""
    def __init__(self, in_channels, skip_channels, out_channels, use_transformer=False, use_attention=True):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.hybrid_block = HybridEncoderBlock(
            out_channels + skip_channels, 
            out_channels, 
            use_transformer=use_transformer,
            use_attention=use_attention
        )
    
    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.hybrid_block(x)
        return x


class HybridEfficientnnUNet(nn.Module):
    """
    Enhanced nnU-Net with:
    - Transfer learning-ready encoder with EfficientNet-style MBConv blocks
    - Hybrid CNN-Transformer architecture for local-global feature learning
    - Shuffle attention for enhanced feature representation
    - Deep supervision for better gradient flow
    """
    def __init__(self, in_channels=4, num_classes=3, base_features=32, 
                 use_transformer=True, use_attention=True, pretrained_path=None):
        super().__init__()
        
        # Initial projection to match base_features
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_features, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(base_features, affine=True),
            nn.SiLU(inplace=True)
        )
        
        # Encoder pathway with hybrid blocks
        # Use transformer in deeper layers where spatial resolution is smaller
        self.enc1 = HybridEncoderBlock(base_features, base_features, use_transformer=False, use_attention=use_attention)
        self.down1 = nn.Conv3d(base_features, base_features * 2, kernel_size=3, stride=2, padding=1)
        
        self.enc2 = HybridEncoderBlock(base_features * 2, base_features * 2, use_transformer=False, use_attention=use_attention)
        self.down2 = nn.Conv3d(base_features * 2, base_features * 4, kernel_size=3, stride=2, padding=1)
        
        self.enc3 = HybridEncoderBlock(base_features * 4, base_features * 4, use_transformer=use_transformer, use_attention=use_attention)
        self.down3 = nn.Conv3d(base_features * 4, base_features * 8, kernel_size=3, stride=2, padding=1)
        
        self.enc4 = HybridEncoderBlock(base_features * 8, base_features * 8, use_transformer=use_transformer, use_attention=use_attention)
        self.down4 = nn.Conv3d(base_features * 8, base_features * 10, kernel_size=3, stride=2, padding=1)
        
        # Bottleneck with transformer for maximum global context
        self.bottleneck = HybridEncoderBlock(base_features * 10, base_features * 10, use_transformer=True, use_attention=use_attention)
        
        # Decoder pathway with deep supervision
        self.dec4 = DecoderBlock(base_features * 10, base_features * 8, base_features * 8, use_transformer=use_transformer, use_attention=use_attention)
        self.dec3 = DecoderBlock(base_features * 8, base_features * 4, base_features * 4, use_transformer=False, use_attention=use_attention)
        self.dec2 = DecoderBlock(base_features * 4, base_features * 2, base_features * 2, use_transformer=False, use_attention=use_attention)
        self.dec1 = DecoderBlock(base_features * 2, base_features, base_features, use_transformer=False, use_attention=use_attention)
        
        # Output heads for deep supervision
        self.out_main = nn.Sequential(
            nn.Conv3d(base_features, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
        self.out_ds1 = nn.Sequential(
            nn.Conv3d(base_features * 2, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
        self.out_ds2 = nn.Sequential(
            nn.Conv3d(base_features * 4, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
        self.out_ds3 = nn.Sequential(
            nn.Conv3d(base_features * 8, num_classes, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Load pretrained weights if provided
        if pretrained_path:
            self.load_pretrained_encoder(pretrained_path)
    
    def _initialize_weights(self):
        """Initialize weights with proper initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.InstanceNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def load_pretrained_encoder(self, pretrained_path):
        """Load pretrained encoder weights (e.g., from pre-training on large dataset)"""
        print(f"Loading pretrained weights from {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location='cpu')
        
        # Load only encoder weights
        encoder_keys = [k for k in state_dict.keys() if k.startswith(('stem', 'enc', 'down', 'bottleneck'))]
        encoder_state_dict = {k: state_dict[k] for k in encoder_keys}
        
        # Load with strict=False to allow partial loading
        self.load_state_dict(encoder_state_dict, strict=False)
        print(f"Loaded {len(encoder_keys)} encoder layers from pretrained checkpoint")
    
    def forward(self, x, return_deep_supervision=True):
        # Stem
        x = self.stem(x)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        
        # Bottleneck
        b = self.bottleneck(self.down4(e4))
        
        # Decoder
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        
        # Output
        out = self.out_main(d1)
        
        if return_deep_supervision and self.training:
            # Deep supervision outputs
            ds1 = self.out_ds1(d2)
            ds2 = self.out_ds2(d3)
            ds3 = self.out_ds3(d4)
            return out, ds1, ds2, ds3
        
        return out


class DiceBCELoss(nn.Module):
    """Combined Dice and Binary Cross-Entropy Loss"""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCELoss()
    
    def dice_loss(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce(pred, target)
        return dice + bce


def get_optimizer_and_scheduler(model, initial_lr=0.01, total_epochs=400, warmup_epochs=10):
    """Setup optimizer and learning rate scheduler with warmup"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Warmup + polynomial decay scheduler
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return (1 - (epoch - warmup_epochs) / (total_epochs - warmup_epochs)) ** 0.9
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    return optimizer, scheduler


def train_step(model, batch, optimizer, criterion):
    """Single training step with deep supervision"""
    images, labels = batch
    
    optimizer.zero_grad()
    
    # Forward pass
    if model.training:
        out, ds1, ds2, ds3 = model(images, return_deep_supervision=True)
        
        # Main loss
        loss = criterion(out, labels)
        
        # Deep supervision losses with weights
        ds1_resized = F.interpolate(ds1, size=labels.shape[2:], mode='trilinear', align_corners=False)
        ds2_resized = F.interpolate(ds2, size=labels.shape[2:], mode='trilinear', align_corners=False)
        ds3_resized = F.interpolate(ds3, size=labels.shape[2:], mode='trilinear', align_corners=False)
        
        loss += 0.5 * criterion(ds1_resized, labels)
        loss += 0.25 * criterion(ds2_resized, labels)
        loss += 0.125 * criterion(ds3_resized, labels)
    else:
        out = model(images, return_deep_supervision=False)
        loss = criterion(out, labels)
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss.item()


# Example usage
if __name__ == "__main__":
    # Initialize hybrid model with transfer learning capabilities
    model = HybridEfficientnnUNet(
        in_channels=4,  # T1, T1Gd, T2, T2-FLAIR
        num_classes=3,  # Whole tumor, tumor core, enhancing tumor
        base_features=32,
        use_transformer=True,  # Enable hybrid CNN-Transformer
        use_attention=True,     # Enable shuffle attention
        pretrained_path=None    # Set path to pretrained weights if available
    )
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 4, 128, 128, 128)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input, return_deep_supervision=False)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Setup training components
    criterion = DiceBCELoss()
    optimizer, scheduler = get_optimizer_and_scheduler(model, initial_lr=0.001, total_epochs=400)
    
    print("\n" + "="*70)
    print("HYBRID EFFICIENT nnU-NET - Enhanced Architecture")
    print("="*70)
    print("✓ EfficientNet-style MBConv blocks with SE attention")
    print("✓ Hybrid CNN-Transformer for local-global features")
    print("✓ Transfer learning ready (load pretrained encoder)")
    print("✓ Shuffle attention mechanism")
    print("✓ Deep supervision with weighted losses")
    print("✓ AdamW optimizer with warmup + polynomial decay")
    print("="*70)
    print("\nTraining configuration:")
    print("  - Patch size: 128×128×128")
    print("  - Total epochs: 400")
    print("  - Warmup epochs: 10")
    print("  - Adjust batch size based on GPU memory")
    print("\nTo use pretrained weights:")
    print("  model = HybridEfficientnnUNet(pretrained_path='path/to/weights.pth')")
    print("="*70)
    
    
# Key Enhancements:
# 1. Transfer Learning Ready:
# ●load_pretrained_encoder() method to load pretrained weights
# ●EfficientNet-style MBConv blocks with Squeeze-Excitation
# ●Easy to freeze encoder layers for fine-tuning
# 2. Hybrid CNN-Transformer Architecture:
# ●MultiHeadSelfAttention3D for global context
# ●TransformerBlock3D with MLP layers
# ●Strategically placed in deeper layers (lower spatial resolution = more efficient)
# 3. Advanced Components:
# ●MBConv blocks with inverted residuals (from EfficientNet)
# ●SE (Squeeze-Excitation) attention in MBConv
# ●Shuffle Attention from original paper
# ●Better optimizer: AdamW with warmup + polynomial decay
# 4. Computational Efficiency:
# ●Transformers only in deeper layers to manage memory
# ●Depthwise separable convolutions maintained
# ●Gradient clipping for training stability

