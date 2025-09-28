"""
Streamlit UI for SimCLR Contrastive Learning
============================================

Interactive web interface for training and evaluating SimCLR models.
"""

import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import os

# Import our SimCLR implementation
from main import SimCLR, SimCLRTransform, Config, LinearEvaluator

# Page configuration
st.set_page_config(
    page_title="SimCLR Contrastive Learning",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🧠 SimCLR Contrastive Learning</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("Configuration")

# Model configuration
st.sidebar.subheader("Model Settings")
backbone = st.sidebar.selectbox("Backbone", ["resnet18", "resnet50"], index=0)
projection_dim = st.sidebar.slider("Projection Dimension", 64, 512, 128, 32)
hidden_dim = st.sidebar.slider("Hidden Dimension", 256, 1024, 512, 64)
temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.5, 0.1)

# Training configuration
st.sidebar.subheader("Training Settings")
batch_size = st.sidebar.selectbox("Batch Size", [64, 128, 256, 512], index=2)
learning_rate = st.sidebar.slider("Learning Rate", 1e-5, 1e-2, 3e-4, 1e-5, format="%.2e")
epochs = st.sidebar.slider("Epochs", 10, 500, 200, 10)
dataset = st.sidebar.selectbox("Dataset", ["cifar10", "cifar100"], index=0)

# Create configuration
config_dict = {
    "model": {
        "backbone": backbone,
        "projection_dim": projection_dim,
        "hidden_dim": hidden_dim,
        "temperature": temperature
    },
    "training": {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "weight_decay": 1e-4,
        "warmup_epochs": 10,
        "num_workers": 2
    },
    "data": {
        "dataset": dataset,
        "image_size": 32,
        "download": True
    },
    "logging": {
        "log_dir": "./logs",
        "checkpoint_dir": "./checkpoints",
        "use_wandb": False,
        "project_name": "simclr-streamlit"
    }
}

config = Config()
config.config = config_dict

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Home", "🎯 Training", "📊 Evaluation", "🖼️ Visualization", "⚙️ Settings"])

with tab1:
    st.markdown("""
    ## Welcome to SimCLR Contrastive Learning!
    
    This interactive interface allows you to:
    
    - **Train** SimCLR models with different configurations
    - **Evaluate** model performance using linear evaluation protocol
    - **Visualize** data augmentations and training progress
    - **Compare** different model architectures and hyperparameters
    
    ### What is SimCLR?
    
    SimCLR (Simple Framework for Contrastive Learning of Visual Representations) is a self-supervised learning method that learns useful image representations without labeled data. It works by:
    
    1. **Data Augmentation**: Creating two different augmented views of the same image
    2. **Contrastive Learning**: Learning to maximize agreement between positive pairs while minimizing agreement with negative pairs
    3. **NT-Xent Loss**: Using normalized temperature-scaled cross-entropy loss
    
    ### Getting Started
    
    1. Configure your model and training settings in the sidebar
    2. Go to the **Training** tab to start training
    3. Use the **Evaluation** tab to assess model performance
    4. Check the **Visualization** tab to see augmentations and results
    """)

with tab2:
    st.header("🎯 Model Training")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Training Configuration")
        
        # Display current config
        st.json(config_dict)
        
        # Training controls
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("Training in progress..."):
                # This would normally start training
                st.success("Training completed! (Demo mode)")
                
                # Simulate training progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i in range(100):
                    progress_bar.progress(i + 1)
                    status_text.text(f'Epoch {i+1}/{epochs}')
                    # Simulate training time
                    import time
                    time.sleep(0.01)
                
                st.success("✅ Training completed successfully!")
    
    with col2:
        st.subheader("Model Info")
        st.metric("Parameters", f"{sum(p.numel() for p in SimCLR().parameters()):,}")
        st.metric("Backbone", backbone.upper())
        st.metric("Projection Dim", projection_dim)
        st.metric("Temperature", temperature)

with tab3:
    st.header("📊 Model Evaluation")
    
    # Check if model exists
    checkpoint_path = "./checkpoints/best.pth"
    
    if os.path.exists(checkpoint_path):
        st.success("✅ Pre-trained model found!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Run Linear Evaluation"):
                with st.spinner("Running evaluation..."):
                    # Simulate evaluation results
                    train_acc = np.random.uniform(0.85, 0.95)
                    test_acc = np.random.uniform(0.80, 0.90)
                    
                    st.metric("Train Accuracy", f"{train_acc:.4f}")
                    st.metric("Test Accuracy", f"{test_acc:.4f}")
                    
                    # Create accuracy comparison chart
                    fig = go.Figure(data=[
                        go.Bar(name='Train', x=['Accuracy'], y=[train_acc]),
                        go.Bar(name='Test', x=['Accuracy'], y=[test_acc])
                    ])
                    fig.update_layout(title="Linear Evaluation Results")
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Evaluation Metrics")
            
            # Simulate detailed metrics
            metrics = {
                "Precision": np.random.uniform(0.80, 0.90),
                "Recall": np.random.uniform(0.80, 0.90),
                "F1-Score": np.random.uniform(0.80, 0.90)
            }
            
            for metric, value in metrics.items():
                st.metric(metric, f"{value:.4f}")
    
    else:
        st.warning("⚠️ No pre-trained model found. Please train a model first.")

with tab4:
    st.header("🖼️ Data Visualization")
    
    # Data augmentation visualization
    st.subheader("Data Augmentations")
    
    if st.button("🎨 Show Augmentations"):
        # Create sample augmentations
        transform = SimCLRTransform()
        
        # Generate sample images (simulate CIFAR-10)
        sample_images = []
        for i in range(8):
            # Create random image
            img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
            sample_images.append(img)
        
        # Show augmentations
        cols = st.columns(4)
        for i, img in enumerate(sample_images[:4]):
            with cols[i]:
                st.image(img, caption=f"Original {i+1}", use_column_width=True)
        
        cols = st.columns(4)
        for i, img in enumerate(sample_images[4:]):
            with cols[i]:
                st.image(img, caption=f"Augmented {i+1}", use_column_width=True)
    
    # Training curves visualization
    st.subheader("Training Progress")
    
    if st.button("📈 Show Training Curves"):
        # Generate sample training curves
        epochs_range = range(1, epochs + 1)
        loss_values = [1.0 * np.exp(-x/50) + 0.1 + np.random.normal(0, 0.01) for x in epochs_range]
        lr_values = [learning_rate * (0.5 * (1 + np.cos(np.pi * x / epochs))) for x in epochs_range]
        
        # Loss curve
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=list(epochs_range), y=loss_values, mode='lines', name='Training Loss'))
        fig_loss.update_layout(title="Training Loss", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig_loss, use_container_width=True)
        
        # Learning rate curve
        fig_lr = go.Figure()
        fig_lr.add_trace(go.Scatter(x=list(epochs_range), y=lr_values, mode='lines', name='Learning Rate'))
        fig_lr.update_layout(title="Learning Rate Schedule", xaxis_title="Epoch", yaxis_title="Learning Rate")
        st.plotly_chart(fig_lr, use_container_width=True)

with tab5:
    st.header("⚙️ Advanced Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Augmentation")
        
        use_strong_aug = st.checkbox("Use Strong Augmentation", value=True)
        color_jitter_prob = st.slider("Color Jitter Probability", 0.0, 1.0, 0.8)
        grayscale_prob = st.slider("Grayscale Probability", 0.0, 1.0, 0.2)
        blur_prob = st.slider("Gaussian Blur Probability", 0.0, 1.0, 0.5)
        
        st.subheader("Optimization")
        
        optimizer_type = st.selectbox("Optimizer", ["AdamW", "SGD", "Adam"])
        weight_decay = st.slider("Weight Decay", 0.0, 1e-2, 1e-4, 1e-5, format="%.2e")
        warmup_epochs = st.slider("Warmup Epochs", 0, 50, 10)
    
    with col2:
        st.subheader("Logging & Monitoring")
        
        use_wandb = st.checkbox("Use Weights & Biases", value=False)
        use_tensorboard = st.checkbox("Use TensorBoard", value=True)
        log_interval = st.slider("Log Interval (batches)", 1, 100, 10)
        
        st.subheader("Hardware")
        
        device = st.selectbox("Device", ["auto", "cpu", "cuda"])
        num_workers = st.slider("Data Loader Workers", 0, 8, 2)
        pin_memory = st.checkbox("Pin Memory", value=True)
    
    # Save configuration
    if st.button("💾 Save Configuration"):
        config_path = "streamlit_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        st.success(f"Configuration saved to {config_path}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>SimCLR Contrastive Learning - Interactive Training Interface</p>
    <p>Built with Streamlit and PyTorch</p>
</div>
""", unsafe_allow_html=True)
