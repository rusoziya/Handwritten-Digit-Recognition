import numpy as np
import struct
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import Sequential
from tensorflow import Dense
from model import build_model
from autils import load_mnist_images
from autils import load_mnist_labels
from autils import display_random_predictions
from autils import plot_loss
from autils import display_errors

# Load dataset
train_images_path = r'C:\Users\ziyar\Github\Handwritten Digit Recognition using Neural Networks\train-images.idx3-ubyte'
train_labels_path = r'C:\Users\ziyar\Github\Handwritten Digit Recognition using Neural Networks\train-labels.idx1-ubyte'
test_images_path = r'C:\Users\ziyar\Github\Handwritten Digit Recognition using Neural Networks\t10k-images.idx3-ubyte'
test_labels_path = r'C:\Users\ziyar\Github\Handwritten Digit Recognition using Neural Networks\t10k-labels.idx1-ubyte'

train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)
test_images = load_mnist_images(test_images_path)
test_labels = load_mnist_labels(test_labels_path)

# Split the training data into 75% training and 15% validation
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.1667, random_state=42
)

# Initialize and compile the model
model = build_model()
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Train the model using the training and validation sets
history = model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

# Evaluate the model using the test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_accuracy}")

# Function to count the number of prediction errors

errors = display_errors(model, test_images, test_labels)
print(f"{errors} errors out of {len(test_images)} images")

# Plotting the loss over epochs
# Call the plotting function
plot_loss(history)

# Display random images and their predictions
# Call the display function for random predictions
display_random_predictions(model, test_images, test_labels)
