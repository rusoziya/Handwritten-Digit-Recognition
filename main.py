from model import build_model  # Import model builder function
from autils import load_data, display_errors  # Import helper functions

import tensorflow as tf

# Load dataset (X: input images, y: corresponding labels)
X, y = load_data()

# Build the model (defined in model.py)
model = build_model()

# Compile the model with loss and optimizer
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
)

# Train the model for 40 epochs (40 times over the data)
history = model.fit(X, y, epochs=40)

# Check how many errors the model made
print(f"Errors: {display_errors(model, X, y)}")

