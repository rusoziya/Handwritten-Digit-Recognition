import numpy as np

def load_data():
    """
    Loads the MNIST handwritten digit dataset (modified to 20x20 images).
    Returns:
    - X: input features (5000 samples of 400 features)
    - y: corresponding labels for each image (0-9)
    """
    # Use a function (not provided here) to load your dataset
    X = np.random.randn(5000, 400)  # Replace this with actual loading
    y = np.random.randint(0, 10, (5000, 1))  # Random labels as placeholders
    return X, y

def display_errors(model, X, y):
    """
    Checks and displays the number of errors in the model predictions.
    """
    predictions = model.predict(X)  # Get model predictions
    predicted_labels = np.argmax(predictions, axis=1)  # Convert logits to class labels
    actual_labels = y.reshape(-1)  # Flatten the labels array
    
    # Compare predictions and actual labels
    errors = np.sum(predicted_labels != actual_labels)
    
    return errors  # Return the number of errors
