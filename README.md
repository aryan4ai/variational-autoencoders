# Variational Autoencoder (VAE) Implementation

This repository provides a simple implementation of Variational Autoencoders (VAEs) using Python and PyTorch.
## Overview

Variational Autoencoders (VAEs) are a type of generative model that encode input data into a latent space and then decode it back, aiming to approximate the original data. VAEs introduce a probabilistic approach to latent variable modeling by incorporating regularization to improve the learning process and ensure meaningful latent spaces.

## Repository Contents

This repository contains the following files:

### 1. `vae.py`
- **Description**: Implements the core VAE model in PyTorch, including the encoder and decoder networks, and the VAE loss function.
- **Key Functions**:
  - `Encoder`: Encodes the input data into a latent distribution.
  - `Decoder`: Decodes latent variables back to the original data space.
  - `loss_function`: Combines reconstruction loss and KL divergence.
- **Usage**: Import this file to integrate the VAE model into your projects.

### 2. `train.py`
- **Description**: Script for training the VAE model on a dataset.
- **Key Features**:
  - Loads dataset (e.g., MNIST).
  - Configures the optimizer and training parameters.
  - Logs training loss for evaluation.
- **Usage**:
  ```bash
  python train.py
  ```

### 3. `data_loader.py`
- **Description**: Handles data loading and preprocessing for the VAE.
- **Key Features**:
  - Prepares datasets such as MNIST for training and testing.
  - Normalizes input images to the range [0, 1].
- **Usage**: Automatically invoked in `train.py`.

### 4. `utils.py`
- **Description**: Contains utility functions for visualization and latent space analysis.
- **Key Features**:
  - Plot reconstructed images.
  - Visualize latent space using scatter plots.
  - Save models and checkpoints.

### 5. `requirements.txt`
- **Description**: Specifies the required Python packages for running the project.
- **Dependencies**:
  ```
  torch
  torchvision
  numpy
  matplotlib
  ```
- **Usage**:
  Install dependencies using:
  ```bash
  pip install -r requirements.txt
  ```

## Getting Started

### Prerequisites
- Python 3.8+
- PyTorch 1.9.0+

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/aryan4ai/variational-autoencoders.git
   cd variational-autoencoder
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project
1. Train the VAE:
   ```bash
   python train.py
   ```
2. Visualize reconstructions:
   Run the visualization functions in `utils.py` to inspect the output of the VAE.

### Results
- The trained VAE can generate realistic samples and visualize the latent space, providing insights into the underlying data distribution.

## References
- VAE: [Variational Autoencoders Documentation](https://pyro.ai/examples/vae.html)