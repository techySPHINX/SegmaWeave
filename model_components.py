"""
Model components for Hybrid Efficient nnU-Net
Organized, optimized, and production-ready PyTorch modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class DepthwiseSeparableConv3D(nn.Module):
    """
    Efficient Depthwise Separable Convolution for 3D data
    Reduces parameters while maintaining performance
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv3d(
            in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class SqueezeExcitation3D(nn.Module):
    """
    3D Squeeze-and-Excitation module for channel attention
    Adaptively recalibrates channel-wise feature responses
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_channels = max(1, channels // reduction)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, reduced_channels, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv3d(reduced_channels, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class MultiHeadSelfAttention3D(nn.Module):
    """
    Efficient 3D Multi-Head Self-Attention with optimized memory usage
    """

    def __init__(self, channels: int, num_heads: int = 8,
                 qkv_bias: bool = False, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        assert channels % num_heads == 0, f"Channels {channels} must be divisible by num_heads {num_heads}"

        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv3d(channels, channels * 3, 1, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv3d(channels, channels, 1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W, D = x.shape
        N = H * W * D

        # Generate Q, K, V with memory-efficient reshaping
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, N)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv.unbind(0)

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention and project
        out = (attn @ v).transpose(-2, -1).reshape(B, C, H, W, D)
        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class TransformerBlock3D(nn.Module):
    """
    3D Transformer block with layer normalization and MLP
    """

    def __init__(self, channels: int, num_heads: int = 8, mlp_ratio: float = 4.,
                 drop: float = 0., attn_drop: float = 0.):
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups=min(
            32, channels), num_channels=channels)
        self.attn = MultiHeadSelfAttention3D(
            channels, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop
        )

        self.norm2 = nn.GroupNorm(num_groups=min(
            32, channels), num_channels=channels)
        mlp_hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, mlp_hidden, 1),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv3d(mlp_hidden, channels, 1),
            nn.Dropout(drop)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm with residual connections
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ShuffleAttention3D(nn.Module):
    """
    3D Shuffle Attention mechanism for enhanced feature representation
    Combines channel and spatial attention with feature shuffling
    """

    def __init__(self, channels: int, groups: int = 4):
        super().__init__()
        assert channels % (
            2 * groups) == 0, f"Channels {channels} must be divisible by 2*groups {2*groups}"

        self.groups = groups
        self.group_channels = channels // (2 * groups)

        # Channel attention pathway
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.channel_fc = nn.Sequential(
            nn.Conv3d(self.group_channels,
                      max(1, self.group_channels // 4), 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv3d(max(1, self.group_channels // 4),
                      self.group_channels, 1, bias=False)
        )

        # Spatial attention pathway
        self.spatial_norm = nn.GroupNorm(
            num_groups=max(1, self.group_channels // 8),
            num_channels=self.group_channels
        )
        self.spatial_fc = nn.Conv3d(
            self.group_channels, self.group_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, d = x.size()

        # Reshape and split into groups
        x = x.view(b, self.groups, -1, h, w, d)
        x_0, x_1 = x.chunk(2, dim=2)

        # Channel attention branch
        # x_0 shape: (b, self.groups, self.group_channels, h, w, d)
        x_0 = x_0.contiguous().view(b * self.groups, self.group_channels, h, w, d)
        attn_c = self.avg_pool(x_0)
        attn_c = self.channel_fc(attn_c)
        x_0 = torch.sigmoid(attn_c) * x_0
        x_0 = x_0.view(b, -1, h, w, d)

        # Spatial attention branch
        x_1 = x_1.contiguous().view(b * self.groups, self.group_channels, h, w, d)
        attn_s = self.spatial_norm(x_1)
        attn_s = self.spatial_fc(attn_s)
        x_1 = torch.sigmoid(attn_s) * x_1
        x_1 = x_1.view(b, -1, h, w, d)

        # Concatenate and shuffle
        out = torch.cat([x_0, x_1], dim=1)
        out = out.view(b, self.groups, -1, h, w, d)
        out = out.transpose(1, 2).contiguous()
        out = out.view(b, -1, h, w, d)

        return out


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Bottleneck Convolution Block (MBConv)
    Efficient building block inspired by EfficientNet
    """

    def __init__(self, in_channels: int, out_channels: int,
                 expand_ratio: int = 4, stride: int = 1, se_ratio: float = 0.25,
                 drop_path: float = 0.):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.drop_path_rate = drop_path

        hidden_dim = in_channels * expand_ratio

        # Build layers
        layers = []

        # Expansion phase (if needed)
        if expand_ratio != 1:
            layers.extend([
                nn.Conv3d(in_channels, hidden_dim, 1, bias=False),
                nn.InstanceNorm3d(hidden_dim, affine=True),
                nn.SiLU(inplace=True)
            ])

        # Depthwise convolution
        layers.extend([
            nn.Conv3d(hidden_dim, hidden_dim, 3, stride=stride, padding=1,
                      groups=hidden_dim, bias=False),
            nn.InstanceNorm3d(hidden_dim, affine=True),
            nn.SiLU(inplace=True)
        ])

        self.conv = nn.Sequential(*layers)

        # Squeeze-and-Excitation
        self.se = SqueezeExcitation3D(hidden_dim, reduction=int(
            1/se_ratio)) if se_ratio > 0 else None

        # Output projection
        self.project = nn.Sequential(
            nn.Conv3d(hidden_dim, out_channels, 1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True)
        )

        # Drop path for regularization
        self.drop_path = DropPath(
            drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Main convolution pathway
        x = self.conv(x)

        # Apply SE if present
        if self.se is not None:
            x = self.se(x)

        # Output projection
        x = self.project(x)

        # Apply drop path and residual connection
        if self.use_residual:
            x = residual + self.drop_path(x)

        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample for regularization
    """

    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + \
            torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class HybridEncoderBlock(nn.Module):
    """
    Hybrid encoder block combining efficient convolutions with attention mechanisms
    """

    def __init__(self, in_channels: int, out_channels: int,
                 use_transformer: bool = True, use_attention: bool = True,
                 num_heads: int = 8, drop_path: float = 0.):
        super().__init__()

        # Efficient convolution layers
        self.mbconv1 = MBConvBlock(
            in_channels, out_channels, expand_ratio=4, drop_path=drop_path)
        self.mbconv2 = MBConvBlock(
            out_channels, out_channels, expand_ratio=4, drop_path=drop_path)

        # Optional transformer for global context
        self.transformer = TransformerBlock3D(
            out_channels, num_heads=num_heads) if use_transformer else None

        # Optional shuffle attention
        self.attention = ShuffleAttention3D(
            out_channels) if use_attention else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mbconv1(x)
        x = self.mbconv2(x)

        if self.transformer is not None:
            x = self.transformer(x)

        if self.attention is not None:
            x = self.attention(x)

        return x


class DecoderBlock(nn.Module):
    """
    Efficient decoder block with upsampling and feature fusion
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 use_transformer: bool = False, use_attention: bool = True):
        super().__init__()

        # Upsampling
        self.upconv = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=2, stride=2)

        # Feature fusion and processing
        self.hybrid_block = HybridEncoderBlock(
            out_channels + skip_channels,
            out_channels,
            use_transformer=use_transformer,
            use_attention=use_attention
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)

        # Handle size mismatch if any
        if x.shape != skip.shape:
            x = F.interpolate(
                x, size=skip.shape[2:], mode='trilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.hybrid_block(x)
        return x
