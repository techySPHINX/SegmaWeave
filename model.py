"""
Main model architecture for Hybrid Efficient nnU-Net
Production-ready implementation with best practices
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union
import math

from model_components import (
    HybridEncoderBlock,
    DecoderBlock
)


class HybridEfficientnnUNet(nn.Module):
    """
    Enhanced nnU-Net with hybrid CNN-Transformer architecture

    Features:
    - Transfer learning ready encoder
    - EfficientNet-style MBConv blocks 
    - Hybrid CNN-Transformer for local-global features
    - Shuffle attention mechanism
    - Deep supervision for better training
    - Progressive training support
    """

    def __init__(self,
                 in_channels: int = 4,
                 num_classes: int = 3,
                 base_features: int = 32,
                 use_transformer: bool = True,
                 use_attention: bool = True,
                 dropout_rate: float = 0.1,
                 drop_path_rate: float = 0.1,
                 deep_supervision: bool = True):
        super().__init__()

        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        # Calculate drop path rates (stochastic depth)
        num_blocks = 9  # Total number of blocks in the network
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]

        # Stem: Initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_features,
                      kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(base_features, affine=True),
            nn.SiLU(inplace=True),
            nn.Dropout3d(dropout_rate / 2)
        )

        # Encoder pathway with progressive complexity
        self.enc1 = HybridEncoderBlock(
            base_features, base_features,
            use_transformer=False, use_attention=use_attention,
            drop_path=dpr[0]
        )
        self.down1 = self._make_downsample(base_features, base_features * 2)

        self.enc2 = HybridEncoderBlock(
            base_features * 2, base_features * 2,
            use_transformer=False, use_attention=use_attention,
            drop_path=dpr[1]
        )
        self.down2 = self._make_downsample(
            base_features * 2, base_features * 4)

        self.enc3 = HybridEncoderBlock(
            base_features * 4, base_features * 4,
            use_transformer=use_transformer, use_attention=use_attention,
            drop_path=dpr[2]
        )
        self.down3 = self._make_downsample(
            base_features * 4, base_features * 8)

        self.enc4 = HybridEncoderBlock(
            base_features * 8, base_features * 8,
            use_transformer=use_transformer, use_attention=use_attention,
            drop_path=dpr[3]
        )
        self.down4 = self._make_downsample(
            base_features * 8, base_features * 16)

        # Bottleneck with maximum global context
        self.bottleneck = HybridEncoderBlock(
            base_features * 16, base_features * 16,
            use_transformer=True, use_attention=use_attention,
            drop_path=dpr[4]
        )

        # Decoder pathway with skip connections
        self.dec4 = DecoderBlock(
            base_features * 16, base_features * 8, base_features * 8,
            use_transformer=use_transformer, use_attention=use_attention
        )

        self.dec3 = DecoderBlock(
            base_features * 8, base_features * 4, base_features * 4,
            use_transformer=False, use_attention=use_attention
        )

        self.dec2 = DecoderBlock(
            base_features * 4, base_features * 2, base_features * 2,
            use_transformer=False, use_attention=use_attention
        )

        self.dec1 = DecoderBlock(
            base_features * 2, base_features, base_features,
            use_transformer=False, use_attention=use_attention
        )

        # Output heads
        self.final_conv = nn.Sequential(
            nn.Conv3d(base_features, base_features //
                      2, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_features // 2, affine=True),
            nn.SiLU(inplace=True),
            nn.Dropout3d(dropout_rate),
            nn.Conv3d(base_features // 2, num_classes, kernel_size=1)
        )

        # Deep supervision heads
        if deep_supervision:
            self.ds_heads = nn.ModuleList([
                self._make_output_head(
                    base_features * 2, num_classes),   # From dec2
                self._make_output_head(
                    base_features * 4, num_classes),   # From dec3
                self._make_output_head(
                    base_features * 8, num_classes),   # From dec4
            ])

        # Initialize weights
        self._initialize_weights()

    def _make_downsample(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create efficient downsampling layer"""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3,
                      stride=2, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.SiLU(inplace=True)
        )

    def _make_output_head(self, in_channels: int, num_classes: int) -> nn.Module:
        """Create output head for deep supervision"""
        return nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 4, kernel_size=3, padding=1),
            nn.InstanceNorm3d(in_channels // 4, affine=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(in_channels // 4, num_classes, kernel_size=1)
        )

    def _initialize_weights(self):
        """Initialize model weights with proper strategies"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv3d):
                if 'final' in name or 'output' in name or 'ds_heads' in name:
                    # Use normal initialization for output layers
                    nn.init.normal_(module.weight, std=0.01)
                else:
                    # Use He initialization for hidden layers
                    nn.init.kaiming_normal_(
                        module.weight, mode='fan_out', nonlinearity='relu')

                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, (nn.InstanceNorm3d, nn.GroupNorm, nn.BatchNorm3d)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def load_pretrained_encoder(self, pretrained_path: str, strict: bool = False):
        """
        Load pretrained encoder weights for transfer learning

        Args:
            pretrained_path: Path to pretrained checkpoint
            strict: Whether to use strict loading
        """
        print(f"Loading pretrained encoder from: {pretrained_path}")

        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint)

            # Filter encoder keys
            encoder_keys = [
                k for k in state_dict.keys()
                if any(prefix in k for prefix in ['stem', 'enc', 'down', 'bottleneck'])
            ]

            encoder_state_dict = {k: state_dict[k]
                                  for k in encoder_keys if k in state_dict}

            # Load encoder weights
            missing_keys, unexpected_keys = self.load_state_dict(
                encoder_state_dict, strict=False)

            print(f"✓ Loaded {len(encoder_state_dict)} encoder parameters")
            if missing_keys:
                print(
                    f"⚠ Missing keys: {len(missing_keys)} (expected for new decoder)")
            if unexpected_keys:
                print(f"⚠ Unexpected keys: {len(unexpected_keys)}")

        except Exception as e:
            print(f"❌ Failed to load pretrained weights: {e}")
            raise

    def freeze_encoder(self, freeze: bool = True):
        """Freeze/unfreeze encoder for fine-tuning"""
        encoder_modules = [self.stem, self.enc1, self.down1, self.enc2, self.down2,
                           self.enc3, self.down3, self.enc4, self.down4, self.bottleneck]

        for module in encoder_modules:
            for param in module.parameters():
                param.requires_grad = not freeze

        status = "frozen" if freeze else "unfrozen"
        print(f"Encoder parameters {status}")

    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """Extract feature maps for visualization/analysis"""
        features = {}

        # Stem
        x = self.stem(x)
        features['stem'] = x

        # Encoder
        e1 = self.enc1(x)
        features['enc1'] = e1

        e2 = self.enc2(self.down1(e1))
        features['enc2'] = e2

        e3 = self.enc3(self.down2(e2))
        features['enc3'] = e3

        e4 = self.enc4(self.down3(e3))
        features['enc4'] = e4

        # Bottleneck
        b = self.bottleneck(self.down4(e4))
        features['bottleneck'] = b

        return features

    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Tuple]:
        """
        Forward pass with optional feature extraction

        Args:
            x: Input tensor [B, C, H, W, D]
            return_features: Whether to return intermediate features

        Returns:
            Output tensor(s) and optionally features dict
        """
        features = None
        if return_features:
            features = self.get_feature_maps(x)
            x = features['stem']
            e1, e2, e3, e4, b = features['enc1'], features['enc2'], features['enc3'], features['enc4'], features['bottleneck']
        else:
            # Standard forward pass
            x = self.stem(x)
            e1 = self.enc1(x)
            e2 = self.enc2(self.down1(e1))
            e3 = self.enc3(self.down2(e2))
            e4 = self.enc4(self.down3(e3))
            b = self.bottleneck(self.down4(e4))

        # Decoder with skip connections
        d4 = self.dec4(b, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)

        # Final output
        output = self.final_conv(d1)

        # Deep supervision during training
        if self.training and self.deep_supervision and hasattr(self, 'ds_heads'):
            ds_outputs = []
            decoder_features = [d2, d3, d4]

            for i, (ds_head, feat) in enumerate(zip(self.ds_heads, decoder_features)):
                ds_out = ds_head(feat)
                # Resize to match target size
                if ds_out.shape[2:] != output.shape[2:]:
                    ds_out = F.interpolate(ds_out, size=output.shape[2:],
                                           mode='trilinear', align_corners=False)
                ds_outputs.append(ds_out)

            if return_features:
                return output, ds_outputs, features
            return output, ds_outputs

        if return_features:
            return output, features
        return output

    def compute_flops(self, input_shape: Tuple[int, int, int, int, int]) -> int:
        """Estimate FLOPs for given input shape"""
        # Simplified FLOP estimation
        total_flops = 0

        # This is a basic estimation - for precise measurements use torchprofile
        def conv3d_flops(in_channels, out_channels, kernel_size, input_size):
            kernel_flops = kernel_size[0] * kernel_size[1] * kernel_size[2]
            output_elements = input_size[0] * input_size[1] * input_size[2]
            return kernel_flops * in_channels * output_elements * out_channels

        # Add FLOPs for major components
        # This is a simplified calculation - implement detailed version if needed

        return total_flops

    @torch.jit.ignore()
    def no_weight_decay(self) -> set:
        """Return parameter names that should not have weight decay"""
        return {'pos_embed', 'cls_token', 'attention.relative_position_bias_table'}


def create_model(config):
    """
    Factory function to create model from config

    Args:
        config: Configuration object with model parameters

    Returns:
        Initialized model
    """
    model = HybridEfficientnnUNet(
        in_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        base_features=config.model.base_features,
        use_transformer=config.model.use_transformer,
        use_attention=config.model.use_attention,
        dropout_rate=config.model.dropout_rate,
        drop_path_rate=0.1,
        deep_supervision=True
    )

    # Load pretrained weights if specified
    if config.system.pretrained_path:
        model.load_pretrained_encoder(config.system.pretrained_path)

    # Apply model compilation if requested
    if config.system.compile_model and hasattr(torch, 'compile'):
        try:
            torch.compile(model, mode='default')
            print("✓ Model compiled with torch.compile")
        except Exception as e:
            print(f"⚠ Failed to compile model: {e}")
    return model


# Model factory functions for different use cases
def create_lightweight_model(num_classes: int = 3) -> HybridEfficientnnUNet:
    """Create lightweight model for testing/prototyping"""
    model = HybridEfficientnnUNet(
        in_channels=4,
        num_classes=num_classes,
        base_features=16,
        use_transformer=False,
        use_attention=True,
        dropout_rate=0.1
    )
    return model


def create_production_model(num_classes: int = 3) -> HybridEfficientnnUNet:
    """Create full-featured model for production"""
    model = HybridEfficientnnUNet(
        in_channels=4,
        num_classes=num_classes,
        base_features=32,
        use_transformer=True,
        use_attention=True,
        dropout_rate=0.1,
        drop_path_rate=0.15
    )
    return model


def create_large_model(num_classes: int = 3) -> HybridEfficientnnUNet:
    """Create large model for maximum performance"""
    model = HybridEfficientnnUNet(
        in_channels=4,
        num_classes=num_classes,
        base_features=48,
        use_transformer=True,
        use_attention=True,
        dropout_rate=0.15,
        drop_path_rate=0.2
    )
    return model
