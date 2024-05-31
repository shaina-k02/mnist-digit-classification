import tensorflow as tf
from tensorflow.keras.datasets import mnist

def load_and_preprocess_data():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize the data to be in the range [0, 1]
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    
    # Reshape data to match the expected input shape for the model
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    print("Training data shape:", x_train.shape)
    print("Test data shape:", x_test.shape)
