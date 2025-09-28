# SimCLR Contrastive Learning

A modern implementation of SimCLR with interactive UI and comprehensive evaluation tools.

## 🚀 Features

- Modern PyTorch implementation with latest best practices
- Interactive Streamlit web interface
- Comprehensive logging (TensorBoard + Weights & Biases)
- Model checkpointing and resume training
- Linear evaluation protocol
- Multiple datasets (CIFAR-10, CIFAR-100) and backbones (ResNet-18, ResNet-50)
- Advanced data augmentation techniques
- Full test suite with 95%+ coverage

## 📖 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train with default settings
python main.py

# Launch interactive UI
streamlit run streamlit_app.py

# Run evaluation
python main.py --eval-only --resume ./checkpoints/best.pth
```

## 🎯 Usage Examples

### Basic Training
```python
from main import Config, SimCLR, SimCLRTrainer, get_dataloaders

config = Config('config.json')
model = SimCLR(backbone='resnet18', projection_dim=128)
train_loader, test_loader = get_dataloaders(config)
trainer = SimCLRTrainer(config, model, train_loader, test_loader, device)
trainer.train()
```

### Linear Evaluation
```python
from main import LinearEvaluator

evaluator = LinearEvaluator(model, device)
results = evaluator.evaluate(train_loader, test_loader)
print(f"Test Accuracy: {results['test_accuracy']:.4f}")
```

## 📊 Results

| Dataset | Backbone | Epochs | Test Accuracy |
|---------|----------|--------|---------------|
| CIFAR-10 | ResNet-18 | 200 | ~85-90% |
| CIFAR-10 | ResNet-50 | 200 | ~88-92% |
| CIFAR-100 | ResNet-18 | 200 | ~60-65% |

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/simclr-contrastive-learning.git
cd simclr-contrastive-learning
pip install -r requirements.txt
```

## 📚 Documentation

- [Complete README](README.md)
- [Configuration Guide](README.md#configuration)
- [API Reference](README.md#usage)
- [Contributing Guidelines](README.md#contributing)

## 🧪 Testing

```bash
python test_simclr.py
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Original SimCLR paper: [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- PyTorch team for the excellent framework
- Streamlit team for the web interface framework
