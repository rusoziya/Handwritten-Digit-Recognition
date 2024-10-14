## Handwritten-Digit-Recognition
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

Training set: 5,000 images of handwritten digits (0-9), each represented as a 784-dimensional vector (28x28 pixel images).
Testing set: 10,000 images of handwritten digits (0-9), also 28x28 pixels in size.
Each image is a grayscale pixel intensity matrix that is flattened into a single vector.

## Model Architecture
The neural network architecture consists of the following layers:

Input Layer: 784 units (28x28 pixels flattened)
Hidden Layer 1: 25 units with ReLU activation
Hidden Layer 2: 15 units with ReLU activation
Output Layer: 10 units (one for each digit, 0-9), with softmax activation to generate probabilities for each class.
## Setup
Requirements
Ensure you have the following dependencies installed:

Python 3.x
TensorFlow
NumPy
Matplotlib
Scikit-learn

Ensure you have the following dataset files in the project directory:
train-images.idx3-ubyte
train-labels.idx1-ubyte
t10k-images.idx3-ubyte
t10k-labels.idx1-ubyte
These files can be downloaded from the Kaggle MNIST Dataset page.

## Usage
To train the model, evaluate its performance, and display the results, simply run:

The script will:

The script will:

Load the MNIST dataset.
Split the training set into 75% training and 15% validation.
Train the neural network for 10 epochs.
Evaluate the trained model on the test set.
Display plots for the loss over epochs and random image predictions along with their labels.
Main Features
Training and Validation: The model is trained using 75% of the dataset and validated on 15%.
Error Display: The number of misclassified images is displayed after evaluating the model.
Loss Plot: A plot of training and validation loss is generated to track model performance over epochs.
Random Image Prediction: Displays a grid of randomly selected test images along with the predicted label and true label.
Results
Test Accuracy
After training the model, the test accuracy is printed.

Loss Plot
A plot showing the loss over training epochs is generated. This plot illustrates how the training and validation loss decreases over time, which helps to monitor whether the model is learning efficiently and not overfitting.


Prediction Visualization
A grid of random test images is displayed with true and predicted labels. This allows us to visually inspect the model's performance on random test samples and see how often it predicts the correct digit.


Sample Output
Sample Output
Training Loss and Validation Loss plot:
This plot shows how the training loss and validation loss change over time (epochs). A decreasing trend indicates that the model is learning effectively, and the gap between training and validation loss provides insight into overfitting.


Random Prediction Grid:
This grid displays a random selection of images from the test set along with the predicted label and the true label for each image. The images are shown in grayscale with labels above each image.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
