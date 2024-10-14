# Handwritten-Digit-Recognition
Neural Networks for Handwritten Digit Recognition using TensorFlow

This project implements a neural network using TensorFlow/Keras to recognize handwritten digits (0-9) from the MNIST dataset. The project includes training, validation, and testing phases, with visualizations of the model's loss over time and a function to display random test images along with their predictions.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup](#setup)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Introduction
The purpose of this project is to implement a simple feed-forward neural network for digit recognition using the MNIST dataset. The network consists of two hidden layers with ReLU activations and an output layer with softmax activation. The model is trained, validated, and tested using a split of the dataset, and performance metrics such as accuracy and loss are reported.

Additionally, this project includes functions to visualize training loss, validation loss, and the model's predictions on random samples of test images.

## Dataset
The dataset used in this project is a subset of the MNIST dataset:

- **Training set**: 5,000 images of handwritten digits (0-9), each represented as a 784-dimensional vector (28x28 pixel images).
- **Testing set**: 10,000 images of handwritten digits (0-9), also 28x28 pixels in size.

Each image is a grayscale pixel intensity matrix that is flattened into a single vector.

## Model Architecture
The neural network architecture consists of the following layers:
- **Input Layer**: 784 units (28x28 pixels flattened)
- **Hidden Layer 1**: 25 units with ReLU activation
- **Hidden Layer 2**: 15 units with ReLU activation
- **Output Layer**: 10 units (one for each digit, 0-9), with softmax activation to generate probabilities for each class.

## Setup

### Requirements
Ensure you have the following dependencies installed:

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Scikit-learn

To install the required packages, run:

```bash
pip install -r requirements.txt
