"""
Unit tests for SimCLR implementation
====================================

Tests for the core SimCLR components including model, loss function, and utilities.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock

# Import our implementation
from main import SimCLR, SimCLRTransform, nt_xent_loss, Config, LinearEvaluator

class TestSimCLR(unittest.TestCase):
    """Test cases for SimCLR model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device('cpu')
        self.batch_size = 4
        self.input_size = (3, 32, 32)
        
    def test_simclr_initialization(self):
        """Test SimCLR model initialization"""
        model = SimCLR(backbone='resnet18', projection_dim=128, hidden_dim=512)
        
        # Check model components
        self.assertIsInstance(model.encoder, nn.Module)
        self.assertIsInstance(model.projector, nn.Module)
        
        # Check projection head structure
        self.assertEqual(len(model.projector), 5)  # 3 Linear + 2 ReLU layers
        
    def test_simclr_forward(self):
        """Test SimCLR forward pass"""
        model = SimCLR(backbone='resnet18', projection_dim=128, hidden_dim=512)
        model.eval()
        
        # Create dummy input
        x = torch.randn(self.batch_size, *self.input_size)
        
        with torch.no_grad():
            output = model(x)
            
        # Check output shape and normalization
        self.assertEqual(output.shape, (self.batch_size, 128))
        self.assertTrue(torch.allclose(torch.norm(output, dim=1), torch.ones(self.batch_size), atol=1e-6))
        
    def test_simclr_encode(self):
        """Test SimCLR encode method"""
        model = SimCLR(backbone='resnet18', projection_dim=128, hidden_dim=512)
        model.eval()
        
        x = torch.randn(self.batch_size, *self.input_size)
        
        with torch.no_grad():
            features = model.encode(x)
            
        # Check feature shape (ResNet-18 output is 512-dim)
        self.assertEqual(features.shape, (self.batch_size, 512))
        
    def test_different_backbones(self):
        """Test different backbone architectures"""
        for backbone in ['resnet18', 'resnet50']:
            with self.subTest(backbone=backbone):
                model = SimCLR(backbone=backbone, projection_dim=128, hidden_dim=512)
                x = torch.randn(self.batch_size, *self.input_size)
                
                with torch.no_grad():
                    output = model(x)
                    
                self.assertEqual(output.shape, (self.batch_size, 128))

class TestSimCLRTransform(unittest.TestCase):
    """Test cases for SimCLR data augmentation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.transform = SimCLRTransform(image_size=32, strong_augment=True)
        
    def test_transform_output(self):
        """Test transform output format"""
        # Create dummy PIL image
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        
        x1, x2 = self.transform(img)
        
        # Check output types and shapes
        self.assertIsInstance(x1, torch.Tensor)
        self.assertIsInstance(x2, torch.Tensor)
        self.assertEqual(x1.shape, (3, 32, 32))
        self.assertEqual(x2.shape, (3, 32, 32))
        
    def test_transform_normalization(self):
        """Test transform normalization"""
        from PIL import Image
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        
        x1, x2 = self.transform(img)
        
        # Check normalization (should be roughly in [-2, 2] range)
        self.assertTrue(torch.all(x1 >= -3))
        self.assertTrue(torch.all(x1 <= 3))
        self.assertTrue(torch.all(x2 >= -3))
        self.assertTrue(torch.all(x2 <= 3))

class TestNTXentLoss(unittest.TestCase):
    """Test cases for NT-Xent loss function"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.batch_size = 4
        self.projection_dim = 128
        self.temperature = 0.5
        
    def test_loss_computation(self):
        """Test NT-Xent loss computation"""
        # Create dummy projections
        z1 = torch.randn(self.batch_size, self.projection_dim)
        z2 = torch.randn(self.batch_size, self.projection_dim)
        
        # Normalize projections
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.nn.functional.normalize(z2, dim=1)
        
        loss = nt_xent_loss(z1, z2, self.temperature)
        
        # Check loss properties
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss.item() >= 0)
        
    def test_loss_symmetry(self):
        """Test loss symmetry"""
        z1 = torch.randn(self.batch_size, self.projection_dim)
        z2 = torch.randn(self.batch_size, self.projection_dim)
        
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.nn.functional.normalize(z2, dim=1)
        
        loss1 = nt_xent_loss(z1, z2, self.temperature)
        loss2 = nt_xent_loss(z2, z1, self.temperature)
        
        self.assertAlmostEqual(loss1.item(), loss2.item(), places=6)
        
    def test_loss_scaling(self):
        """Test loss scaling with temperature"""
        z1 = torch.randn(self.batch_size, self.projection_dim)
        z2 = torch.randn(self.batch_size, self.projection_dim)
        
        z1 = torch.nn.functional.normalize(z1, dim=1)
        z2 = torch.nn.functional.normalize(z2, dim=1)
        
        loss_low_temp = nt_xent_loss(z1, z2, temperature=0.1)
        loss_high_temp = nt_xent_loss(z1, z2, temperature=1.0)
        
        # Lower temperature should give higher loss
        self.assertGreater(loss_low_temp.item(), loss_high_temp.item())

class TestConfig(unittest.TestCase):
    """Test cases for configuration management"""
    
    def test_default_config(self):
        """Test default configuration loading"""
        config = Config()
        
        # Check default values
        self.assertEqual(config.get('model.backbone'), 'resnet18')
        self.assertEqual(config.get('model.projection_dim'), 128)
        self.assertEqual(config.get('training.batch_size'), 256)
        
    def test_config_override(self):
        """Test configuration override"""
        config_dict = {
            'model': {
                'backbone': 'resnet50',
                'projection_dim': 256
            },
            'training': {
                'batch_size': 128
            }
        }
        
        config = Config()
        config.config = config_dict
        
        self.assertEqual(config.get('model.backbone'), 'resnet50')
        self.assertEqual(config.get('model.projection_dim'), 256)
        self.assertEqual(config.get('training.batch_size'), 128)
        
    def test_nested_config_access(self):
        """Test nested configuration access"""
        config = Config()
        
        # Test nested access
        backbone = config.get('model.backbone')
        self.assertEqual(backbone, 'resnet18')
        
        # Test non-existent key
        non_existent = config.get('non.existent.key', 'default')
        self.assertEqual(non_existent, 'default')

class TestLinearEvaluator(unittest.TestCase):
    """Test cases for linear evaluation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = SimCLR(backbone='resnet18', projection_dim=128, hidden_dim=512)
        self.device = torch.device('cpu')
        self.evaluator = LinearEvaluator(self.model, self.device)
        
    @patch('main.DataLoader')
    def test_extract_features(self, mock_dataloader):
        """Test feature extraction"""
        # Mock dataloader
        mock_batch = [
            (torch.randn(2, 3, 32, 32), torch.randint(0, 10, (2,))),
            (torch.randn(2, 3, 32, 32), torch.randint(0, 10, (2,)))
        ]
        mock_dataloader.return_value = mock_batch
        
        features, labels = self.evaluator.extract_features(mock_dataloader)
        
        # Check output shapes
        self.assertEqual(features.shape[0], 4)  # 2 batches * 2 samples
        self.assertEqual(labels.shape[0], 4)
        self.assertEqual(features.shape[1], 512)  # ResNet-18 feature dim

class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_full_training_step(self):
        """Test a complete training step"""
        # Create model and dummy data
        model = SimCLR(backbone='resnet18', projection_dim=128, hidden_dim=512)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Create dummy batch
        x1 = torch.randn(4, 3, 32, 32)
        x2 = torch.randn(4, 3, 32, 32)
        
        # Forward pass
        z1 = model(x1)
        z2 = model(x2)
        
        # Compute loss
        loss = nt_xent_loss(z1, z2)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that loss is finite
        self.assertTrue(torch.isfinite(loss))
        
    def test_model_gradient_flow(self):
        """Test that gradients flow properly through the model"""
        model = SimCLR(backbone='resnet18', projection_dim=128, hidden_dim=512)
        
        x1 = torch.randn(4, 3, 32, 32)
        x2 = torch.randn(4, 3, 32, 32)
        
        z1 = model(x1)
        z2 = model(x2)
        loss = nt_xent_loss(z1, z2)
        
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertFalse(torch.all(param.grad == 0))

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
