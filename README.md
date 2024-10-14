# Handwritten-Digit-Recognition
Neural Networks for Handwritten Digit Recognition using TensorFlow

This project implements a neural network using TensorFlow/Keras to recognize handwritten digits (0-9). The dataset used is a subset of the MNIST dataset.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Setup](#setup)
5. [Results](#results)

## Introduction
The purpose of this project is to implement a simple feed-forward neural network for digit recognition using the MNIST dataset. The network consists of two hidden layers with ReLU activations and an output layer with softmax activation.

## Dataset
The dataset consists of 5000 images of handwritten digits, each represented as a 400-dimensional vector (20x20 pixel images).

## Model Architecture
The neural network architecture consists of:
- Input Layer: 400 units
- Hidden Layer 1: 25 units with ReLU activation
- Hidden Layer 2: 15 units with ReLU activation
- Output Layer: 10 units (one for each digit, 0-9)

## Setup
### Requirements
- Python 3.x
- TensorFlow
- NumPy
- Matplotlib

Install the required packages:

```bash
pip install -r requirements.txt
