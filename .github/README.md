# SimCLR for Contrastive Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A modern, comprehensive implementation of SimCLR (Simple Framework for Contrastive Learning of Visual Representations) with advanced features, interactive UI, and extensive evaluation tools.

## Features

- **Modern PyTorch Implementation**: Latest best practices and optimizations
- **Multiple Architectures**: Support for ResNet-18 and ResNet-50 backbones
- **Advanced Data Augmentation**: Gaussian blur, color jittering, and more
- **Comprehensive Logging**: TensorBoard and Weights & Biases integration
- **Interactive UI**: Streamlit-based web interface for training and visualization
- **Linear Evaluation**: Standard protocol for evaluating learned representations
- **Model Checkpointing**: Resume training and save best models
- **Multiple Datasets**: CIFAR-10 and CIFAR-100 support
- **Unit Tests**: Comprehensive test suite for all components

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Interactive UI](#interactive-ui)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/kryptologyst/SimCLR-for-Contrastive-Learning.git
cd SimCLR-for-Contrastive-Learning

# Install requirements
pip install -r requirements.txt
```

### Optional: Weights & Biases Setup

For experiment tracking with Weights & Biases:

```bash
pip install wandb
wandb login
```

## Quick Start

### Command Line Training

```bash
# Train with default configuration
python main.py

# Train with custom configuration
python main.py --config config.json

# Resume training from checkpoint
python main.py --resume ./checkpoints/latest.pth

# Visualize data augmentations
python main.py --visualize

# Run evaluation only
python main.py --eval-only --resume ./checkpoints/best.pth
```

### Interactive Web Interface

```bash
# Launch Streamlit UI
streamlit run streamlit_app.py
```

Then open your browser to `http://localhost:8501`

## Usage

### Basic Training

```python
from main import Config, SimCLR, SimCLRTrainer, get_dataloaders

# Load configuration
config = Config('config.json')

# Create model
model = SimCLR(
    backbone='resnet18',
    projection_dim=128,
    hidden_dim=512
)

# Get data loaders
train_loader, test_loader = get_dataloaders(config)

# Create trainer
trainer = SimCLRTrainer(config, model, train_loader, test_loader, device)

# Start training
trainer.train()
```

### Custom Configuration

```python
# Create custom configuration
config_dict = {
    'model': {
        'backbone': 'resnet50',
        'projection_dim': 256,
        'hidden_dim': 1024,
        'temperature': 0.1
    },
    'training': {
        'batch_size': 512,
        'learning_rate': 1e-3,
        'epochs': 300,
        'weight_decay': 1e-4
    },
    'data': {
        'dataset': 'cifar100',
        'image_size': 32
    }
}

config = Config()
config.config = config_dict
```

### Linear Evaluation

```python
from main import LinearEvaluator

# Load trained model
checkpoint = torch.load('./checkpoints/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Create evaluator
evaluator = LinearEvaluator(model, device)

# Run evaluation
results = evaluator.evaluate(train_loader, test_loader)
print(f"Test Accuracy: {results['test_accuracy']:.4f}")
```

## Configuration

The configuration system supports JSON files with the following structure:

```json
{
  "model": {
    "backbone": "resnet18",
    "projection_dim": 128,
    "hidden_dim": 512,
    "temperature": 0.5
  },
  "training": {
    "batch_size": 256,
    "learning_rate": 3e-4,
    "epochs": 200,
    "weight_decay": 1e-4,
    "warmup_epochs": 10,
    "num_workers": 4
  },
  "data": {
    "dataset": "cifar10",
    "image_size": 32,
    "download": true
  },
  "logging": {
    "log_dir": "./logs",
    "checkpoint_dir": "./checkpoints",
    "use_wandb": true,
    "project_name": "simclr-cifar10"
  }
}
```

### Configuration Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `model.backbone` | Backbone architecture | `resnet18` | `resnet18`, `resnet50` |
| `model.projection_dim` | Projection head output dimension | `128` | Any positive integer |
| `model.hidden_dim` | Projection head hidden dimension | `512` | Any positive integer |
| `model.temperature` | NT-Xent loss temperature | `0.5` | Positive float |
| `training.batch_size` | Training batch size | `256` | Any positive integer |
| `training.learning_rate` | Learning rate | `3e-4` | Positive float |
| `training.epochs` | Number of training epochs | `200` | Any positive integer |
| `data.dataset` | Dataset to use | `cifar10` | `cifar10`, `cifar100` |

## Interactive UI

The Streamlit interface provides:

- **Model Configuration**: Adjust hyperparameters through the sidebar
- **Training Control**: Start/stop training with real-time progress
- **Visualization**: View data augmentations and training curves
- **Evaluation**: Run linear evaluation and view results
- **Settings**: Advanced configuration options

### Launching the UI

```bash
streamlit run streamlit_app.py
```

## Evaluation

### Linear Evaluation Protocol

The standard evaluation protocol for self-supervised learning:

1. **Freeze Encoder**: Keep the trained encoder weights frozen
2. **Extract Features**: Get representations for all training/test images
3. **Train Linear Classifier**: Train a logistic regression classifier on features
4. **Evaluate**: Test the classifier on test set

### Running Evaluation

```bash
# Evaluate best model
python main.py --eval-only --resume ./checkpoints/best.pth

# Evaluate specific checkpoint
python main.py --eval-only --resume ./checkpoints/epoch_100.pth
```

### Expected Results

| Dataset | Backbone | Epochs | Test Accuracy |
|---------|----------|--------|---------------|
| CIFAR-10 | ResNet-18 | 200 | ~85-90% |
| CIFAR-10 | ResNet-50 | 200 | ~88-92% |
| CIFAR-100 | ResNet-18 | 200 | ~60-65% |
| CIFAR-100 | ResNet-50 | 200 | ~65-70% |

*Results may vary based on hyperparameters and random initialization*

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python test_simclr.py

# Run specific test class
python -m unittest test_simclr.TestSimCLR

# Run with verbose output
python test_simclr.py -v
```

### Test Coverage

- ✅ Model initialization and forward pass
- ✅ Data augmentation transforms
- ✅ NT-Xent loss computation
- ✅ Configuration management
- ✅ Linear evaluation protocol
- ✅ Integration tests
- ✅ Gradient flow verification

## 📁 Project Structure

```
simclr-contrastive-learning/
├── main.py                 # Main SimCLR implementation
├── streamlit_app.py       # Interactive web interface
├── test_simclr.py         # Unit tests
├── config.json           # Default configuration
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── logs/                # Training logs and TensorBoard files
├── checkpoints/         # Model checkpoints
└── data/               # Dataset storage
```

## Technical Details

### SimCLR Architecture

1. **Data Augmentation**: Two random augmentations of each image
2. **Encoder**: ResNet backbone (ResNet-18 or ResNet-50)
3. **Projection Head**: 3-layer MLP with ReLU activations
4. **Contrastive Loss**: NT-Xent (Normalized Temperature-scaled Cross Entropy)

### Key Improvements

- **Modern PyTorch**: Uses latest PyTorch 2.0+ features
- **Advanced Augmentation**: Gaussian blur, improved color jittering
- **Better Optimization**: AdamW optimizer with cosine annealing
- **Comprehensive Logging**: TensorBoard + Weights & Biases
- **Robust Evaluation**: Linear evaluation protocol
- **Production Ready**: Error handling, checkpointing, resume training

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

```bash
# Clone repository
git clone https://github.com/kryptologyst/SimCLR-for-Contrastive-Learning.git
cd SimCLR-for-Contrastive-Learning

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Run tests
python test_simclr.py

# Format code
black *.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original SimCLR paper: [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- PyTorch team for the excellent deep learning framework
- Streamlit team for the amazing web interface framework
- Weights & Biases for experiment tracking

## References

1. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. ICML.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.
3. Chen, T., Kornblith, S., Swersky, K., Norouzi, M., & Hinton, G. (2020). Big self-supervised models are strong semi-supervised learners. NeurIPS.


# SimCLR-for-Contrastive-Learning
