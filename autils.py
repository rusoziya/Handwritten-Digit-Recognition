import numpy as np
import struct
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import Sequential
from tensorflow import Dense
from model import build_model

def load_mnist_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)
        images = images / 255.0  # Normalize to [0,1]
    return images

def load_mnist_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

def plot_loss(history):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

def display_errors(model, X, y):
    y_pred = model.predict(X)
    y_pred_labels = np.argmax(tf.nn.softmax(y_pred), axis=1)  # Convert logits to label predictions

    # Count how many predictions were incorrect
    errors = np.sum(y_pred_labels != y)
    return errors

def display_random_predictions(model, X, y, num_images=64):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    m, n = X.shape  # m: number of examples, n: number of features (28*28)

    fig, axes = plt.subplots(8, 8, figsize=(5, 5))
    fig.tight_layout(pad=0.13, rect=[0, 0.03, 1, 0.91])  # Adjust layout

    for i, ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)
        
        # Select rows corresponding to the random indices and
        # reshape the image to a 28x28 pixel grid
        X_random_reshaped = X[random_index].reshape((28, 28))  # MNIST images are 28x28
        
        # Display the image
        ax.imshow(X_random_reshaped, cmap='gray')
        
        # Predict using the Neural Network
        prediction = model.predict(X[random_index].reshape(1, 784))  # Flattened image is 28*28=784
        prediction_p = tf.nn.softmax(prediction)
        yhat = np.argmax(prediction_p)
        
        # Display the label above the image
        ax.set_title(f"True: {y[random_index]}, Pred: {yhat}", fontsize=10)
        ax.set_axis_off()

    fig.suptitle("Label (True), Prediction", fontsize=14)
    plt.show()