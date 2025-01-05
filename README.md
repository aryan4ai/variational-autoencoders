# Latent Attention VAE Implementation

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A TensorFlow implementation of a Variational Autoencoder (VAE) with latent attention mechanism, originally designed for MNIST dataset but adaptable to custom images.

## 🔍 Overview

This project implements a Variational Autoencoder with a focus on latent space attention, using TensorFlow. The model is designed to learn meaningful latent representations while incorporating attention mechanisms in the encoding/decoding process.

## 📁 Project Structure

```
.
├── input_data.py     # Data loading and preprocessing
├── main.py          # Main training script and model definition
├── ops.py           # Neural network operations and utilities
├── utils.py         # Helper functions for visualization
├── prepare_image.py # Custom image preprocessing script
├── MNIST_data/      # Directory for dataset storage
├── results/         # Output directory for generated images
└── training/        # Directory for model checkpoints
```

## ✨ Key Features

- **Latent Space Attention**: Custom attention mechanism in the latent space
- **Convolutional Architecture**: Uses convolutional layers for both encoder and decoder
- **Batch Normalization**: Implements custom batch normalization for training stability
- **Visualization**: Includes utilities for visualizing generated samples
- **Custom Image Support**: Can process and reconstruct custom grayscale images

## 🔧 Technical Details

### Model Architecture

```python
# Encoder (Recognition Network)
Input: 28x28x1 images
Conv1: 28x28x1 → 14x14x16
Conv2: 14x14x16 → 7x7x32
Dense: Mean and STD of latent space

# Decoder (Generation Network)
Input: Latent vector (20 dimensions)
Dense: Reshape layer
Deconv1: 7x7x32 → 14x14x16
Deconv2: 14x14x16 → 28x28x1
Output: Sigmoid activation
```

## 📦 Requirements

```bash
tensorflow>=2.0.0
numpy>=1.19.2
matplotlib>=3.3.2
scipy>=1.5.2
Pillow>=8.0.0  # For image processing
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/latent-attention-vae.git
cd latent-attention-vae

# Create necessary directories
mkdir MNIST_data results training

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### For Custom Images

```python
# 1. Prepare your image
python prepare_image.py

# 2. Run the model
python main.py
```

#### For MNIST Dataset

```python
# 1. Prepare MNIST data
from input_data import read_data_sets
mnist = read_data_sets("MNIST_data/", one_hot=True)

# 2. Train model
python main.py
```

## ⚙️ Configuration

```python
CONFIGURATION = {
    'batch_size': 100,
    'latent_dim': 20,
    'hidden_units': 500,
    'learning_rate': 0.001,
    'epochs': 10,
    'save_interval': 1
}
```

## 📝 Implementation Notes

### Custom Image Processing
- Images automatically resized to 28x28 pixels
- Recommended adjustments for complex patterns:
  ```python
  n_hidden = 1000  # Increased from 500
  n_z = 40        # Increased from 20
  learning_rate = 0.0005  # Decreased for stability
  ```

### Output Files
```plaintext
results/
├── base.jpg        # Original preprocessed image
├── result.jpg      # VAE reconstruction
└── epoch_{n}.jpg   # Per-epoch samples
```

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| Complex Image | Increase `n_hidden` and `n_z` |
| Poor Reconstruction | Lower learning rate or increase epochs |
| Memory Issues | Reduce batch size |
| Blurry Output | Adjust loss function weights |

## 🤝 Contributing

```bash
# Fork and clone
git clone https://github.com/yourusername/latent-attention-vae.git

# Create feature branch
git checkout -b feature/amazing-feature

# Commit changes
git commit -m 'Add amazing feature'

# Push to branch
git push origin feature/amazing-feature

# Open Pull Request
```

## 📋 Example Output

```plaintext
epoch 0: genloss 156.123 latloss 12.456
epoch 1: genloss 142.234 latloss 10.345
...
epoch 9: genloss 98.765 latloss 8.901
```

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- MNIST dataset by Yann LeCun and Corinna Cortes
- Original VAE paper and architecture
- TensorFlow framework

---

<div align="center">
  Made with ❤️ and Python
  
  [![Made with Python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
</div>
