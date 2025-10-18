"""
Unit tests for model architecture
"""

import pytest
import torch
import torch.nn as nn
from model import (
    HybridEfficientnnUNet,
    create_model,
    create_lightweight_model,
    create_production_model,
    create_large_model
)
from model_components import (
    DepthwiseSeparableConv3D,
    SqueezeExcitation3D,
    MultiHeadSelfAttention3D,
    TransformerBlock3D,
    ShuffleAttention3D,
    MBConvBlock,
    HybridEncoderBlock,
    DecoderBlock
)
from config import get_quick_test_config


class TestModelComponents:
    """Test individual model components"""

    def test_depthwise_separable_conv(self):
        """Test DepthwiseSeparableConv3D"""
        module = DepthwiseSeparableConv3D(32, 64)
        x = torch.randn(2, 32, 16, 16, 16)
        output = module(x)
        assert output.shape == (2, 64, 16, 16, 16)

    def test_squeeze_excitation(self):
        """Test SqueezeExcitation3D"""
        module = SqueezeExcitation3D(64, reduction=16)
        x = torch.randn(2, 64, 16, 16, 16)
        output = module(x)
        assert output.shape == x.shape

    def test_multi_head_attention(self):
        """Test MultiHeadSelfAttention3D"""
        module = MultiHeadSelfAttention3D(128, num_heads=8)
        x = torch.randn(2, 128, 8, 8, 8)
        output = module(x)
        assert output.shape == x.shape

    def test_transformer_block(self):
        """Test TransformerBlock3D"""
        module = TransformerBlock3D(128, num_heads=8)
        x = torch.randn(2, 128, 8, 8, 8)
        output = module(x)
        assert output.shape == x.shape

    def test_shuffle_attention(self):
        """Test ShuffleAttention3D"""
        module = ShuffleAttention3D(64, groups=4)
        x = torch.randn(2, 64, 16, 16, 16)
        output = module(x)
        assert output.shape == x.shape

    def test_mbconv_block(self):
        """Test MBConvBlock"""
        module = MBConvBlock(64, 64, expand_ratio=4)
        x = torch.randn(2, 64, 16, 16, 16)
        output = module(x)
        assert output.shape == x.shape

    def test_hybrid_encoder_block(self):
        """Test HybridEncoderBlock"""
        module = HybridEncoderBlock(
            64, 64, use_transformer=True, use_attention=True)
        x = torch.randn(2, 64, 16, 16, 16)
        output = module(x)
        assert output.shape == x.shape

    def test_decoder_block(self):
        """Test DecoderBlock"""
        module = DecoderBlock(
            128, 64, 64, use_transformer=False, use_attention=True)
        x = torch.randn(2, 128, 8, 8, 8)
        skip = torch.randn(2, 64, 16, 16, 16)
        output = module(x, skip)
        assert output.shape == skip.shape


class TestHybridEfficientnnUNet:
    """Test main model architecture"""

    @pytest.fixture
    def small_model(self):
        """Create small model for testing"""
        return HybridEfficientnnUNet(
            in_channels=4,
            num_classes=3,
            base_features=16,
            use_transformer=False,
            use_attention=True,
            deep_supervision=True
        )

    @pytest.fixture
    def full_model(self):
        """Create full model for testing"""
        return HybridEfficientnnUNet(
            in_channels=4,
            num_classes=3,
            base_features=32,
            use_transformer=True,
            use_attention=True,
            deep_supervision=True
        )

    def test_forward_pass_no_deep_supervision(self, small_model):
        """Test forward pass without deep supervision"""
        small_model.eval()
        x = torch.randn(1, 4, 64, 64, 64)
        with torch.no_grad():
            output = small_model(x)
        assert output.shape == (1, 3, 64, 64, 64)

    def test_forward_pass_with_deep_supervision(self, small_model):
        """Test forward pass with deep supervision"""
        small_model.train()
        x = torch.randn(1, 4, 64, 64, 64)
        output = small_model(x)

        # Should return tuple: (main_output, [ds_outputs])
        assert isinstance(output, tuple)
        main_output, ds_outputs = output
        assert main_output.shape == (1, 3, 64, 64, 64)
        assert len(ds_outputs) == 3  # Three deep supervision outputs

    def test_forward_pass_with_features(self, small_model):
        """Test forward pass with feature extraction"""
        small_model.eval()
        x = torch.randn(1, 4, 64, 64, 64)
        with torch.no_grad():
            output, features = small_model(x, return_features=True)

        assert output.shape == (1, 3, 64, 64, 64)
        assert isinstance(features, dict)
        assert 'stem' in features
        assert 'bottleneck' in features

    def test_get_feature_maps(self, small_model):
        """Test feature map extraction"""
        x = torch.randn(1, 4, 64, 64, 64)
        features = small_model.get_feature_maps(x)

        assert 'stem' in features
        assert 'enc1' in features
        assert 'enc2' in features
        assert 'enc3' in features
        assert 'enc4' in features
        assert 'bottleneck' in features

    def test_freeze_encoder(self, small_model):
        """Test encoder freezing"""
        # Freeze encoder
        small_model.freeze_encoder(True)

        # Check that encoder parameters don't require grad
        for name, param in small_model.named_parameters():
            if any(prefix in name for prefix in ['stem', 'enc', 'down', 'bottleneck']):
                assert not param.requires_grad

        # Unfreeze encoder
        small_model.freeze_encoder(False)

        # Check that encoder parameters require grad again
        for name, param in small_model.named_parameters():
            if any(prefix in name for prefix in ['stem', 'enc', 'down', 'bottleneck']):
                assert param.requires_grad

    def test_model_output_shape_consistency(self, full_model):
        """Test output shape consistency across different input sizes"""
        full_model.eval()

        sizes = [64, 96, 128]
        for size in sizes:
            x = torch.randn(1, 4, size, size, size)
            with torch.no_grad():
                output = full_model(x)
            assert output.shape == (1, 3, size, size, size)

    def test_model_device_consistency(self, small_model):
        """Test model works on different devices"""
        x = torch.randn(1, 4, 64, 64, 64)

        # CPU test
        small_model.cpu()
        x_cpu = x.cpu()
        output_cpu = small_model(x_cpu)
        assert output_cpu.device.type == 'cpu'

        # GPU test (if available)
        if torch.cuda.is_available():
            small_model.cuda()
            x_cuda = x.cuda()
            output_cuda = small_model(x_cuda)
            assert output_cuda.device.type == 'cuda'

    def test_parameter_count(self, small_model, full_model):
        """Test parameter counts are reasonable"""
        small_params = sum(p.numel() for p in small_model.parameters())
        full_params = sum(p.numel() for p in full_model.parameters())

        # Small model should have fewer parameters
        assert small_params < full_params

        # Reasonable ranges
        assert 5_000_000 < small_params < 20_000_000  # 5M - 20M
        assert 15_000_000 < full_params < 30_000_000  # 15M - 30M

    def test_gradient_flow(self, small_model):
        """Test that gradients flow properly"""
        small_model.train()
        x = torch.randn(2, 4, 64, 64, 64, requires_grad=True)
        target = torch.randint(0, 3, (2, 64, 64, 64))

        output = small_model(x)
        if isinstance(output, tuple):
            output = output[0]

        # Simple loss
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        for param in small_model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestModelFactory:
    """Test model factory functions"""

    def test_create_lightweight_model(self):
        """Test lightweight model creation"""
        model = create_lightweight_model(num_classes=3)
        assert isinstance(model, HybridEfficientnnUNet)
        assert model.num_classes == 3

        # Test forward pass
        x = torch.randn(1, 4, 64, 64, 64)
        output = model(x)
        assert output.shape == (1, 3, 64, 64, 64)

    def test_create_production_model(self):
        """Test production model creation"""
        model = create_production_model(num_classes=3)
        assert isinstance(model, HybridEfficientnnUNet)

        x = torch.randn(1, 4, 64, 64, 64)
        output = model(x)
        assert output.shape == (1, 3, 64, 64, 64)

    def test_create_large_model(self):
        """Test large model creation"""
        model = create_large_model(num_classes=3)
        assert isinstance(model, HybridEfficientnnUNet)

        x = torch.randn(1, 4, 64, 64, 64)
        output = model(x)
        assert output.shape == (1, 3, 64, 64, 64)

    def test_create_model_from_config(self):
        """Test model creation from config"""
        config = get_quick_test_config()
        model = create_model(config)
        assert isinstance(model, HybridEfficientnnUNet)

        x = torch.randn(1, 4, 64, 64, 64)
        output = model(x)
        assert output.shape == (1, 3, 64, 64, 64)


class TestModelEdgeCases:
    """Test edge cases and error handling"""

    def test_invalid_input_channels(self):
        """Test model with wrong number of input channels"""
        model = HybridEfficientnnUNet(
            in_channels=4, num_classes=3, base_features=16)
        model.eval()

        # Wrong number of channels
        x = torch.randn(1, 3, 64, 64, 64)
        with pytest.raises(RuntimeError):
            model(x)

    def test_batch_size_one(self):
        """Test model with batch size of 1"""
        model = HybridEfficientnnUNet(
            in_channels=4, num_classes=3, base_features=16)
        model.eval()

        x = torch.randn(1, 4, 64, 64, 64)
        output = model(x)
        assert output.shape == (1, 3, 64, 64, 64)

    def test_large_batch_size(self):
        """Test model with larger batch size"""
        model = HybridEfficientnnUNet(
            in_channels=4, num_classes=3, base_features=16)
        model.eval()

        x = torch.randn(4, 4, 64, 64, 64)
        output = model(x)
        assert output.shape == (4, 3, 64, 64, 64)

    def test_different_num_classes(self):
        """Test model with different number of classes"""
        for num_classes in [2, 3, 5, 10]:
            model = HybridEfficientnnUNet(
                in_channels=4, num_classes=num_classes, base_features=16)
            model.eval()

            x = torch.randn(1, 4, 64, 64, 64)
            output = model(x)
            assert output.shape == (1, num_classes, 64, 64, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
