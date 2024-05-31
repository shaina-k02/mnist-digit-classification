# Import necessary modules
from data_preparation import load_and_preprocess_data
from model import build_model

def main():
    # Load and preprocess the MNIST dataset
    (x_train, y_train), _ = load_and_preprocess_data()

    # Build the neural network model
    model = build_model()

    # Train the model on the training data
    # - x_train: Training images
    # - y_train: Training labels
    # - epochs: Number of times to iterate over the training data
    model.fit(x_train, y_train, epochs=3)

    # Save the trained model to the 'models' directory
    model.save('../models/epic_num_reader.keras')

if __name__ == '__main__':
    main()
