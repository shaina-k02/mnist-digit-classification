import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

def build_model():
    # Initialize a sequential model
    model = Sequential([
        # Add a 2D convolutional layer with 32 filters, a kernel size of 3x3, and ReLU activation
        # The input shape is (28, 28, 1) for 28x28 pixel grayscale images
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        
        # Add a max-pooling layer with a pool size of 2x2 to reduce the spatial dimensions
        MaxPooling2D(pool_size=(2, 2)),
        
        # Add another 2D convolutional layer with 64 filters, a kernel size of 3x3, and ReLU activation
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        
        # Add another max-pooling layer with a pool size of 2x2
        MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten the 3D output to 1D to prepare for the fully connected layers
        Flatten(),
        
        # Add a dense (fully connected) layer with 128 units and ReLU activation
        Dense(128, activation='relu'),
        
        # Add the output dense layer with 10 units (one for each class) and softmax activation
        # Softmax activation is used to get the probability distribution of the classes
        Dense(10, activation='softmax')
    ])
    
    # Compile the model with the Adam optimizer, sparse categorical crossentropy loss, and accuracy as the metric
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    # Build the model
    model = build_model()
    
    # Print the model summary to verify the architecture
    model.summary()
