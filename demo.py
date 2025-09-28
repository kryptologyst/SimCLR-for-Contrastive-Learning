#!/usr/bin/env python3
"""
Demo script for SimCLR Contrastive Learning
===========================================

This script demonstrates the key features of the SimCLR implementation
with a quick training run and evaluation.
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time

# Import our implementation
from main import (
    Config, SimCLR, SimCLRTransform, nt_xent_loss, 
    SimCLRTrainer, LinearEvaluator, get_dataloaders, get_dataset, visualize_augmentations
)

def demo_data_augmentation():
    """Demonstrate data augmentation"""
    print("🎨 Demonstrating data augmentation...")
    
    # Create config
    config = Config()
    
    # Get dataset
    train_dataset, _ = get_dataset(config)
    
    # Visualize augmentations
    visualize_augmentations(train_dataset, num_samples=4)
    print("✅ Data augmentation visualization saved as 'augmentations.png'")

def demo_model_architecture():
    """Demonstrate model architecture"""
    print("🏗️ Demonstrating model architecture...")
    
    # Create model
    model = SimCLR(backbone='resnet18', projection_dim=128, hidden_dim=512)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    dummy_input = torch.randn(4, 3, 32, 32).to(device)
    
    with torch.no_grad():
        output = model(dummy_input)
        features = model.encode(dummy_input)
    
    print(f"✅ Forward pass successful:")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Features shape: {features.shape}")
    print(f"   Output normalized: {torch.allclose(torch.norm(output, dim=1), torch.ones(4), atol=1e-6)}")

def demo_loss_function():
    """Demonstrate NT-Xent loss function"""
    print("📉 Demonstrating NT-Xent loss function...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy projections
    batch_size = 8
    projection_dim = 128
    
    z1 = torch.randn(batch_size, projection_dim).to(device)
    z2 = torch.randn(batch_size, projection_dim).to(device)
    
    # Normalize projections
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)
    
    # Test different temperatures
    temperatures = [0.1, 0.5, 1.0, 2.0]
    
    print("📊 Loss values at different temperatures:")
    for temp in temperatures:
        loss = nt_xent_loss(z1, z2, temperature=temp)
        print(f"   Temperature {temp}: {loss.item():.4f}")
    
    # Test loss symmetry
    loss1 = nt_xent_loss(z1, z2, temperature=0.5)
    loss2 = nt_xent_loss(z2, z1, temperature=0.5)
    
    print(f"✅ Loss symmetry test: {torch.allclose(loss1, loss2, atol=1e-6)}")

def demo_training_step():
    """Demonstrate a single training step"""
    print("🚀 Demonstrating training step...")
    
    # Create config with small batch size for demo
    config = Config()
    config.config['training']['batch_size'] = 8
    config.config['training']['epochs'] = 1
    
    # Get data loaders
    train_loader, test_loader = get_dataloaders(config)
    
    # Create model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimCLR(backbone='resnet18', projection_dim=128, hidden_dim=512).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Training step
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    print("🔄 Running training step...")
    start_time = time.time()
    
    for batch_idx, (images, _) in enumerate(train_loader):
        if batch_idx >= 3:  # Limit to 3 batches for demo
            break
            
        x1, x2 = images
        x1, x2 = x1.to(device), x2.to(device)
        
        # Forward pass
        z1 = model(x1)
        z2 = model(x2)
        
        # Compute loss
        loss = nt_xent_loss(z1, z2, temperature=0.5)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        print(f"   Batch {batch_idx + 1}: Loss = {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    elapsed_time = time.time() - start_time
    
    print(f"✅ Training step completed:")
    print(f"   Average loss: {avg_loss:.4f}")
    print(f"   Time elapsed: {elapsed_time:.2f} seconds")
    print(f"   Batches processed: {num_batches}")

def demo_evaluation():
    """Demonstrate linear evaluation"""
    print("📊 Demonstrating linear evaluation...")
    
    # Create config
    config = Config()
    config.config['training']['batch_size'] = 32
    
    # Get data loaders
    train_loader, test_loader = get_dataloaders(config)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimCLR(backbone='resnet18', projection_dim=128, hidden_dim=512).to(device)
    
    # Create evaluator
    evaluator = LinearEvaluator(model, device)
    
    print("🔍 Extracting features...")
    
    # Extract features (limited for demo)
    train_features, train_labels = evaluator.extract_features(train_loader)
    test_features, test_labels = evaluator.extract_features(test_loader)
    
    print(f"✅ Feature extraction completed:")
    print(f"   Train features shape: {train_features.shape}")
    print(f"   Test features shape: {test_features.shape}")
    print(f"   Train labels shape: {train_labels.shape}")
    print(f"   Test labels shape: {test_labels.shape}")

def main():
    """Run all demonstrations"""
    print("🧠 SimCLR Contrastive Learning Demo")
    print("=" * 50)
    
    try:
        # Run demonstrations
        demo_model_architecture()
        print()
        
        demo_loss_function()
        print()
        
        demo_training_step()
        print()
        
        demo_evaluation()
        print()
        
        demo_data_augmentation()
        print()
        
        print("🎉 All demonstrations completed successfully!")
        print("\n📚 Next steps:")
        print("   1. Run 'python main.py' to start full training")
        print("   2. Run 'streamlit run streamlit_app.py' for interactive UI")
        print("   3. Run 'python test_simclr.py' to run the test suite")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        print("Please check your installation and try again.")

if __name__ == "__main__":
    main()
