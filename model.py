from tensorflow import Sequential
from tensorflow import Dense

def build_model():
    model = Sequential([
        Dense(25, activation='relu', input_shape=(784,), name='L1'),  # Layer 1 with 25 neurons
        Dense(15, activation='relu', name='L2'),                      # Layer 2 with 15 neurons
        Dense(10, activation='linear', name='L3')                     # Output layer with 10 neurons
    ])
    
    model.summary()  # Show the model architecture
    
    return model  # Return the constructed model
